"""Measure real local pose inference; no training or implicit model downloads.

Run from the repository root: python -m scripts.benchmark_pose --help
"""

import argparse
import hashlib
import json
import platform
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np

from posture.config import load_config
from posture.data import read_manifest
from posture.features import extract_features
from posture.training import select_device


def benchmark(manifest_path, output_dir, config, device="mps", repeats=10, warmup=3):
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    target = select_device(device)
    rows = read_manifest(manifest_path, require_split=True)
    pose = config["pose"]
    weights = Path(pose["weights"]).resolve()
    if not weights.is_file():
        raise FileNotFoundError("local official weights required: {}".format(weights))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    images_dir = destination / "overlays"
    images_dir.mkdir()

    def synchronize():
        if target.type == "mps":
            torch.mps.synchronize()
        elif target.type == "cuda":
            torch.cuda.synchronize()

    def mps_memory():
        if target.type != "mps":
            return None
        return {
            "allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_bytes": int(torch.mps.driver_allocated_memory()),
        }

    rss_samples = []
    stopped = threading.Event()
    try:
        import psutil
        process = psutil.Process()
        def sample_memory():
            while not stopped.is_set():
                rss_samples.append(process.memory_info().rss)
                stopped.wait(0.01)
        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
    except ImportError:
        sampler = None

    started = time.perf_counter()
    memory_before = mps_memory()
    observations = []
    timing_values = []
    mps_observations = []
    try:
        model_started = time.perf_counter()
        model = YOLO(str(weights))
        model.model.requires_grad_(False)
        model.model.eval()
        model.to(str(target))
        synchronize()
        load_seconds = time.perf_counter() - model_started
        full_pose_params = sum(p.numel() for p in model.model.parameters())
        first_inference_seconds = None

        def predict(image):
            return model.predict(
                source=image, imgsz=int(pose["imgsz"]),
                conf=float(pose["person_confidence"]), device=str(target), verbose=False,
            )

        for index, row in enumerate(rows):
            image = cv2.imread(row["image_path"])
            if image is None:
                raise ValueError("cannot decode real image: {}".format(row["image_path"]))
            synchronize()
            first_started = time.perf_counter()
            results = predict(image)
            synchronize()
            per_image_first = time.perf_counter() - first_started
            if first_inference_seconds is None:
                first_inference_seconds = per_image_first
            for _ in range(warmup):
                predict(image)
            synchronize()
            durations = []
            for _ in range(repeats):
                synchronize()
                iteration_started = time.perf_counter()
                results = predict(image)
                synchronize()
                durations.append(time.perf_counter() - iteration_started)
                memory = mps_memory()
                if memory is not None:
                    mps_observations.append(memory)
            timing_values.extend(durations)
            result = results[0]
            people = 0 if result.boxes is None else len(result.boxes)
            features = None
            points = None
            reason = "multiple_people" if people > 1 else "no_person"
            if people == 1 and result.keypoints is not None:
                points = result.keypoints.data[0].detach().cpu().numpy()
                features = extract_features(points, float(pose["min_keypoint_confidence"]))
                reason = "valid" if features is not None else "required_keypoints_unreliable"
            overlay = images_dir / "{:03d}_{}.jpg".format(index, row["sample_id"])
            if not cv2.imwrite(str(overlay), result.plot()):
                raise RuntimeError("failed to save overlay: {}".format(overlay))
            observations.append({
                "sample_id": row["sample_id"], "image_path": row["image_path"],
                "manifest_label": row["label"], "view": row["view"],
                "label_note": "trial annotation only; no classifier accuracy inferred",
                "image_shape": list(image.shape), "people": people, "status": reason,
                "feature_dim": None if features is None else int(features.size),
                "features": None if features is None else features.tolist(),
                "required_keypoint_confidence": None if points is None else {
                    name: float(points[joint, 2]) for name, joint in (
                        ("nose", 0), ("left_shoulder", 5), ("right_shoulder", 6),
                        ("left_hip", 11), ("right_hip", 12))
                },
                "first_call_seconds": per_image_first,
                "steady_mean_ms": float(np.mean(durations) * 1000),
                "steady_median_ms": float(np.median(durations) * 1000),
                "steady_p95_ms": float(np.percentile(durations, 95) * 1000),
                "overlay": str(overlay.resolve()),
            })
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
        report = {
            "status": "completed", "scope": "real pose feasibility; not posture classification accuracy",
            "device": str(target), "python": platform.python_version(),
            "torch": str(torch.__version__), "ultralytics": str(ultralytics.__version__),
            "platform": platform.platform(), "config": config,
            "manifest": str(Path(manifest_path).resolve()),
            "manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "pose_parameters_before_fusion": int(full_pose_params),
            "model_load_and_device_transfer_seconds": load_seconds,
            "first_inference_seconds": first_inference_seconds,
            "steady_protocol": {
                "warmup_per_image": warmup, "repeats_per_image": repeats,
                "predecoded_images": True, "synchronize_device_each_iteration": True,
                "includes": "preprocessing + pose forward + postprocessing through YOLO.predict",
                "excludes": "disk image read, plotting, feature/MLP/reminder, camera and display",
            },
            "steady_mean_ms": float(np.mean(timing_values) * 1000),
            "steady_median_ms": float(np.median(timing_values) * 1000),
            "process_peak_rss_bytes": peak_rss_bytes,
            "sampled_runtime_peak_rss_bytes": max(rss_samples) if rss_samples else None,
            "rss_sampling_interval_seconds": 0.01 if sampler else None,
            "mps_memory_before": memory_before,
            "mps_peak_observed": None if not mps_observations else {
                key: max(item[key] for item in mps_observations)
                for key in ("allocated_bytes", "driver_bytes")
            },
            "memory_note": "MPS values sampled after synchronized predictions, not a true device peak. Unified memory overlaps process memory; do not add them.",
            "elapsed_seconds": time.perf_counter() - started,
            "samples": observations,
        }
        (destination / "measurements.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        return report
    except Exception as exc:
        (destination / "failure.json").write_text(json.dumps({
            "device": str(target), "type": type(exc).__name__, "error": str(exc),
            "completed_samples": observations,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise
    finally:
        stopped.set()
        if sampler is not None:
            sampler.join(timeout=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--repeats", default=10, type=int)
    parser.add_argument("--warmup", default=3, type=int)
    args = parser.parse_args()
    report = benchmark(args.manifest, args.output, load_config(args.config), args.device, args.repeats, args.warmup)
    print(json.dumps({key: report[key] for key in (
        "status", "device", "first_inference_seconds", "steady_mean_ms", "process_peak_rss_bytes"
    )}, indent=2))


if __name__ == "__main__":
    main()
