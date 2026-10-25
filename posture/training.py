"""Train an MLP using train-only normalization and validation-only selection."""

import csv
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import CLASS_NAMES
from .features import FEATURE_DIM, FEATURE_NAMES
from .metrics import classification_metrics
from .model import PostureMLP


def select_device(device: str = "auto") -> torch.device:
    """Select an available backend and reject unavailable explicit requests."""
    name = str(device).lower()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if mps else "cpu")
    if name not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be one of auto, cpu, mps, cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "mps" and not mps:
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(name)


def fit_standardizer(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit on training features only; constant dimensions retain unit scale."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("standardizer requires a nonempty, finite feature matrix")
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    return mean, std


def _load_cache(cache_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as cache:
        required = {"features", "labels", "splits", "sample_ids", "views"}
        missing = required.difference(cache.files)
        if missing:
            raise ValueError("cache is missing fields: {}".format(", ".join(sorted(missing))))
        features = np.asarray(cache["features"], dtype=np.float32)
        raw_labels = np.asarray(cache["labels"])
        splits = np.asarray(cache["splits"]).astype(str)
        sample_ids = np.asarray(cache["sample_ids"])
        views = np.asarray(cache["views"])
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError("cache features must have shape (N, {})".format(FEATURE_DIM))
    n_samples = len(features)
    for name, values in (("labels", raw_labels), ("splits", splits),
                         ("sample_ids", sample_ids), ("views", views)):
        if values.shape != (n_samples,):
            raise ValueError("cache {} must have shape (N,)".format(name))
    # NaN rows are intentional extraction rejections, including in test data.
    if (not np.issubdtype(raw_labels.dtype, np.number)
            or not np.isfinite(raw_labels).all()
            or np.any(raw_labels != np.floor(raw_labels))):
        raise ValueError("cache labels must be integers")
    labels = raw_labels.astype(np.int64)
    if np.any((labels < 0) | (labels >= len(CLASS_NAMES))):
        raise ValueError("cache labels must be in [0, 3]")
    if not set(splits.tolist()).issubset({"train", "val", "test"}):
        raise ValueError("cache splits must be train, val, or test")
    if not np.any(splits == "train") or not np.any(splits == "val"):
        raise ValueError("nonempty train and val splits are required")
    if len(set(sample_ids.tolist())) != n_samples:
        raise ValueError("cache sample_ids must be unique")
    return features, labels, splits


def train(
    cache_path: str,
    config: Dict[str, Any],
    output_dir: str,
    seed: int = 42,
    device: str = "auto",
) -> Dict[str, Any]:
    """Train, save the validation-selected checkpoint, and return a summary.

    Test rows are never normalized, passed through the model, or evaluated by
    this function. A separate evaluation step must consume checkpoint.pt.
    """
    settings = config.get("training", {})
    learning_rate = float(settings.get("learning_rate", settings.get("lr", 1e-3)))
    batch_size = int(settings.get("batch_size", 64))
    max_epochs = int(settings.get("max_epochs", 100))
    patience = int(settings.get("patience", 10))
    hidden_dims = tuple(int(size) for size in settings.get("hidden_dims", (64, 32)))
    dropout = float(settings.get("dropout", 0.2))
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if min(batch_size, max_epochs, patience) < 1:
        raise ValueError("batch_size, max_epochs, and patience must be positive")
    target = select_device(device)
    features, labels, splits = _load_cache(cache_path)
    train_mask, val_mask = splits == "train", splits == "val"
    finite = np.isfinite(features).all(axis=1)
    usable_train = train_mask & finite
    usable_val = finite[val_mask]
    if not usable_train.any() or not usable_val.any():
        raise ValueError("train and val must each have at least one valid extracted pose")
    mean, std = fit_standardizer(features[usable_train])
    train_x = (features[usable_train] - mean) / std
    val_x = (features[val_mask][usable_val] - mean) / std
    train_y, val_y = labels[usable_train], labels[val_mask]
    if not np.isfinite(train_x).all() or not np.isfinite(val_x).all():
        raise ValueError("normalization produced non-finite features")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = PostureMLP(FEATURE_DIM, len(CLASS_NAMES), hidden_dims, dropout).to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    val_features = torch.from_numpy(val_x).to(target)
    val_labels = torch.from_numpy(val_y[usable_val]).to(target)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checkpoint_path = destination / "checkpoint.pt"
    history = []
    best_score, best_epoch, stale_epochs = -1.0, 0, 0
    best_val_metrics = {}
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        predictions, targets = [], []
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(target)
            batch_labels = batch_labels.to(target)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(batch_labels)
            predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
            targets.extend(batch_labels.detach().cpu().tolist())
        training_metrics = classification_metrics(np.asarray(targets), np.asarray(predictions))
        model.eval()
        with torch.no_grad():
            logits = model(val_features)
            val_loss = float(criterion(logits, val_labels).cpu())
            val_predictions = np.full(len(val_y), -1, dtype=np.int64)
            val_predictions[usable_val] = logits.argmax(dim=1).cpu().numpy()
        val_metrics = classification_metrics(val_y, val_predictions)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / len(train_y),
            "train_accuracy": training_metrics["accuracy"],
            "train_macro_f1": training_metrics["macro_f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })
        if val_metrics["macro_f1"] > best_score:
            best_score, best_epoch, stale_epochs = val_metrics["macro_f1"], epoch, 0
            best_val_metrics = val_metrics
            torch.save({
                "state_dict": {name: value.detach().cpu().clone()
                               for name, value in model.state_dict().items()},
                "input_dim": FEATURE_DIM,
                "num_classes": len(CLASS_NAMES),
                "hidden_dims": list(hidden_dims),
                "dropout": dropout,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "class_names": list(CLASS_NAMES),
                "feature_names": list(FEATURE_NAMES),
                "seed": seed,
                "best_epoch": best_epoch,
            }, checkpoint_path)
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    with (destination / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "seed": seed,
        "device": str(target),
        "train_samples": int(train_mask.sum()),
        "train_used_samples": int(usable_train.sum()),
        "train_rejected_samples": int((train_mask & ~finite).sum()),
        "val_samples": int(val_mask.sum()),
        "val_rejected_samples": int((~usable_val).sum()),
        "input_dim": FEATURE_DIM,
        "class_names": list(CLASS_NAMES),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_score,
        "best_val_metrics": best_val_metrics,
        "epochs_completed": len(history),
        "training_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path.resolve()),
        "cache_path": str(Path(cache_path).resolve()),
        "cache_sha256": hashlib.sha256(Path(cache_path).read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),
        "loss_scope": "train/val loss uses valid poses; val classification metrics include rejections",
        "training": {
            "learning_rate": learning_rate, "batch_size": batch_size,
            "max_epochs": max_epochs, "patience": patience,
            "hidden_dims": list(hidden_dims), "dropout": dropout,
        },
    }
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    return summary
