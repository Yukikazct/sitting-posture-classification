"""Offline classification evaluation and explicitly scoped classifier profiling."""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import CLASS_NAMES
from .features import FEATURE_DIM, FEATURE_NAMES
from .metrics import classification_metrics
from .model import PostureMLP
from .training import select_device


def load_classifier(checkpoint_path, device="auto"):
    target = select_device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("checkpoint class_names do not match project labels")
    if checkpoint.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("checkpoint feature schema does not match this framework")
    if checkpoint.get("input_dim") != FEATURE_DIM or checkpoint.get("num_classes") != 4:
        raise ValueError("checkpoint has incompatible input or output dimensions")
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    if (mean.shape != (FEATURE_DIM,) or std.shape != (FEATURE_DIM,)
            or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or np.any(std <= 0)):
        raise ValueError("checkpoint has invalid normalization statistics")
    model = PostureMLP(FEATURE_DIM, 4, checkpoint["hidden_dims"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(target).eval(), mean, std, checkpoint, target


def evaluate(cache_path, config, output_dir, checkpoint_path=None,
             split="test", method="mlp", device="auto"):
    """Argmax classification metrics; NaN feature rows remain rejected predictions.

    The reminder's confidence threshold and temporal filter are evaluated on
    continuous video, not applied to these static-image classification metrics.
    """
    if split not in ("train", "val", "test") or method not in ("mlp", "rules"):
        raise ValueError("invalid evaluation split or method")
    with np.load(cache_path, allow_pickle=False) as cache:
        features = np.asarray(cache["features"], dtype=np.float32)
        raw_labels = np.asarray(cache["labels"])
        splits = np.asarray(cache["splits"]).astype(str)
        sample_ids = np.asarray(cache["sample_ids"]).astype(str)
        views = np.asarray(cache["views"]).astype(str)
    n = len(features)
    if features.shape != (n, FEATURE_DIM) or any(
        values.shape != (n,) for values in (raw_labels, splits, sample_ids, views)
    ):
        raise ValueError("cache arrays have inconsistent shapes")
    if (not np.issubdtype(raw_labels.dtype, np.number)
            or not np.isfinite(raw_labels).all()
            or np.any(raw_labels != np.floor(raw_labels))):
        raise ValueError("cache labels must be integer class indices")
    labels = raw_labels.astype(np.int64)
    if np.any((labels < 0) | (labels >= 4)):
        raise ValueError("cache labels must be in [0, 3]")
    if len(set(sample_ids.tolist())) != n:
        raise ValueError("cache sample_ids must be unique")
    selected = splits == split
    if not selected.any():
        raise ValueError("cache has no samples in split '{}'".format(split))
    features, labels = features[selected], labels[selected]
    sample_ids, views = sample_ids[selected], views[selected]
    valid = np.isfinite(features).all(axis=1)
    predictions = np.full(len(labels), -1, dtype=np.int64)
    if method == "mlp":
        if checkpoint_path is None:
            raise ValueError("--checkpoint is required for MLP evaluation")
        model, mean, std, _, target = load_classifier(checkpoint_path, device)
        valid_indices = np.flatnonzero(valid)
        with torch.no_grad():
            for start in range(0, len(valid_indices), 256):
                indices = valid_indices[start:start + 256]
                batch = torch.from_numpy((features[indices] - mean) / std).to(target)
                predictions[indices] = model(batch).argmax(1).cpu().numpy()
    else:
        from .baseline import RuleClassifier
        predictions = RuleClassifier(config).predict(features)
    metrics = classification_metrics(labels, predictions)
    metrics.update({
        "method": method,
        "split": split,
        "class_names": list(CLASS_NAMES),
        "prediction_policy": "argmax; invalid features rejected",
        "by_view": {
            view: classification_metrics(labels[views == view], predictions[views == view])
            for view in sorted(set(views.tolist()))
        },
    })
    if method == "rules":
        metrics["baseline_thresholds"] = config["baseline"]
        metrics["baseline_note"] = "Heuristic defaults; calibrate using validation data only."
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    with (destination / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "prediction", "view", "rejected"])
        for sample, label, prediction, view in zip(sample_ids, labels, predictions, views):
            writer.writerow([sample, CLASS_NAMES[label],
                             CLASS_NAMES[prediction] if prediction >= 0 else "unknown",
                             view, int(prediction < 0)])
    with (destination / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/predicted"] + list(CLASS_NAMES) + ["unknown"])
        for name, row in zip(CLASS_NAMES, metrics["confusion_matrix"]):
            writer.writerow([name] + row)
    return metrics


def profile_classifier(checkpoint_path, device="auto", iterations=100):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    model, _, _, _, target = load_classifier(checkpoint_path, device)
    params = sum(parameter.numel() for parameter in model.parameters())
    macs = sum(layer.in_features * layer.out_features
               for layer in model.modules() if isinstance(layer, nn.Linear))
    sample = torch.zeros(1, FEATURE_DIM, device=target)

    def sync():
        if target.type == "mps":
            torch.mps.synchronize()
        elif target.type == "cuda":
            torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(20):
            model(sample)
        sync()
        started = time.perf_counter()
        for _ in range(iterations):
            model(sample)
        sync()
    return {
        "scope": "classifier forward only; excludes pose, normalization and image I/O",
        "device": str(target), "torch_version": str(torch.__version__),
        "input_shape": [1, FEATURE_DIM], "params": params,
        "linear_macs": macs, "linear_flops": 2 * macs,
        "counting_convention": "1 MAC = 2 FLOPs; Linear weights only, excludes bias/activations",
        "iterations": iterations,
        "mean_latency_ms": (time.perf_counter() - started) * 1000 / iterations,
    }


def plot_history(history_path, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install plotting dependencies: pip install -e '.[plots]'") from exc
    with Path(history_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("history has no epochs")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Avoid inheriting a machine-specific font absent on another computer.
    with plt.rc_context({"font.family": "DejaVu Sans", "font.sans-serif": ["DejaVu Sans"]}):
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        epochs = [int(row["epoch"]) for row in rows]
        for column, metric, title in ((0, "loss", "Loss"), (1, "macro_f1", "Macro-F1")):
            for split in ("train", "val"):
                axes[column].plot(epochs, [float(row[split + "_" + metric]) for row in rows], label=split)
            axes[column].set(xlabel="Epoch", ylabel=title)
            axes[column].legend()
        if any((parent / "SMOKE_ONLY.json").is_file() for parent in Path(history_path).resolve().parents):
            figure.suptitle("SYNTHETIC DATA - pipeline check only")
        figure.tight_layout()
        figure.savefig(output, dpi=160)
        plt.close(figure)
    return {"figure": str(output.resolve())}
