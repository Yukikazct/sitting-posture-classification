"""Command-line entrypoints. Optional vision dependencies are imported on demand."""

import argparse
import copy
import importlib.util
import json
import platform
import sys
from pathlib import Path

from .config import load_config


def check_environment(config):
    import torch
    from .training import select_device
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": str(torch.__version__), "device": str(select_device()),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "cuda_available": torch.cuda.is_available(),
        "vision_dependencies": {
            name: importlib.util.find_spec(name) is not None for name in ("ultralytics", "cv2")
        },
        "pose_weights": config["pose"]["weights"],
        "pose_weights_exist": Path(config["pose"]["weights"]).is_file(),
        "note": "Environment inspection only; no pretrained inference or data acceptance performed.",
    }


def smoke(config, output_dir, device="cpu"):
    """Synthetic feature integration check, never a real experiment."""
    import numpy as np
    from .features import FEATURE_DIM
    from .training import train
    from .evaluation import evaluate
    from .temporal import ReminderState
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    n = 80
    features = rng.normal(size=(n, FEATURE_DIM)).astype(np.float32)
    splits = np.array(["train"] * 48 + ["val"] * 16 + ["test"] * 16)
    labels = np.arange(n, dtype=np.int64) % 4
    features[48] = np.nan  # Exercise validation rejection, not optimistic filtering.
    features[-1] = np.nan
    cache = destination / "synthetic_features.npz"
    np.savez_compressed(cache, features=features, labels=labels, splits=splits,
                        sample_ids=np.array(["synthetic-{}".format(i) for i in range(n)]),
                        views=np.array(["synthetic"] * n))
    settings = copy.deepcopy(config)
    settings["training"].update({"max_epochs": 2, "patience": 2, "batch_size": 16})
    train_result = train(str(cache), settings, str(destination / "train"), seed=42, device=device)
    evaluate(str(cache), settings, str(destination / "evaluate"),
             checkpoint_path=train_result["checkpoint"], device=device)
    state = ReminderState(settings)
    state.update(np.array([0.0, 1.0, 0.0, 0.0]), 0.0)
    triggered = state.update(np.array([0.0, 1.0, 0.0, 0.0]), 3.0)
    if not triggered["alert"]:
        raise RuntimeError("smoke reminder did not trigger")
    result = {
        "status": "passed", "data_kind": "SYNTHETIC_ONLY",
        "scope": "cache -> normalization -> MLP training -> checkpoint -> evaluation; reminder timing",
        "not_validated": ["real labels", "YOLO inference", "camera", "real accuracy", "end-to-end FPS"],
        "output_dir": str(destination.resolve()),
    }
    (destination / "SMOKE_ONLY.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Sitting-posture course project")
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, help_text):
        item = sub.add_parser(name, help=help_text)
        item.add_argument("--config", help="JSON config; omitted keys use shared defaults")
        return item

    command("check", "Inspect environment without downloading or opening camera")
    p = command("audit", "Validate a labeled manifest and identify leakage limitations")
    p.add_argument("--manifest", required=True)
    p.add_argument("--require-split", action="store_true")
    p = command("split", "Create group-isolated train/val/test assignments")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p = command("extract", "Extract features with local YOLO11n-pose weights")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "mps", "cuda"])
    p = command("train", "Train MLP using train/val only")
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p = command("evaluate", "Evaluate MLP or rules including rejected samples")
    p.add_argument("--cache", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--method", choices=["mlp", "rules"], default="mlp")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p = command("demo", "Run single-person webcam or recorded-video demo")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--source", default="0", help="camera index or video path")
    p.add_argument("--save", help="optional annotated video path")
    p.add_argument("--raw", action="store_true", help="disable probability smoothing for comparison")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p = command("smoke", "Run SYNTHETIC-ONLY core pipeline check")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu", choices=["auto", "cpu", "mps", "cuda"])
    p = command("profile", "Measure classifier-only parameters, MACs and latency")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--iterations", type=int, default=100)
    p = command("plot", "Plot actual train/val history")
    p.add_argument("--history", required=True)
    p.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "check":
            result = check_environment(config)
        elif args.command == "audit":
            from .data import read_manifest, validate_manifest
            result = validate_manifest(read_manifest(args.manifest), config["class_names"], args.require_split)
        elif args.command == "split":
            from .data import split_manifest
            result = split_manifest(args.manifest, args.output, config, args.seed)
        elif args.command == "extract":
            from .extraction import extract_manifest
            result = extract_manifest(args.manifest, args.output, config, args.device)
        elif args.command == "train":
            from .training import train
            result = train(args.cache, config, args.output, args.seed, args.device)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(args.cache, config, args.output, args.checkpoint,
                              args.split, args.method, args.device)
        elif args.command == "demo":
            from .demo import run_demo
            result = run_demo(args.checkpoint, config, args.source, args.device,
                              args.save, smoothed=not args.raw)
        elif args.command == "smoke":
            result = smoke(config, args.output, args.device)
        elif args.command == "profile":
            from .evaluation import profile_classifier
            result = profile_classifier(args.checkpoint, args.device, args.iterations)
        else:
            from .evaluation import plot_history
            result = plot_history(args.history, args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, ImportError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 1
