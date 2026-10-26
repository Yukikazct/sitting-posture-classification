"""Measure pose availability on unlabeled review candidates, never class accuracy.

Run as python -m scripts.check_candidate_pose from the repository root.
"""

import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import platform
import time

import numpy as np

from posture.features import FEATURE_DIM, extract_features


def check(candidates_path, output, weights, device="cuda", count=24, batch_size=4):
    import torch
    import ultralytics
    from ultralytics import YOLO

    if count < 1 or batch_size < 1:
        raise ValueError("count and batch_size must be positive")
    candidates_path = Path(candidates_path).resolve()
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError("Prepare the official weights locally first")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    pool = json.loads(candidates_path.read_text(encoding="utf-8"))
    strata = {}
    for entry in pool["entries"]:
        if entry.get("status") == "downloaded":
            strata.setdefault(entry.get("selection_stratum", "unknown"), []).append(entry)
    entries = [entry for row in zip_longest(*strata.values()) for entry in row if entry is not None][:count]
    if not entries:
        raise ValueError("No downloaded candidates")
    paths = []
    for entry in entries:
        path = Path(entry.get("crop_path") or entry.get("local_path") or entry["original_path"])
        if not path.is_absolute():
            path = candidates_path.parent / path
        path = path.resolve()
        expected = entry.get("crop_sha256") if entry.get("crop_path") else entry.get("original_sha256")
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Candidate image changed: " + str(path))
        paths.append(str(path))
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "overlays").mkdir()
    observations = []
    try:
        def sync():
            if device == "cuda":
                torch.cuda.synchronize()

        model = YOLO(str(weights))
        model.model.requires_grad_(False)
        model.to(device)
        model.model.eval()
        features = np.full((len(entries), FEATURE_DIM), np.nan, dtype=np.float32)
        sync()
        started = time.perf_counter()
        for _ in range(3):
            model.predict(source=paths[:batch_size], device=device, imgsz=640, conf=0.25, verbose=False)
        sync()
        warmup_seconds = time.perf_counter() - started
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        batch_seconds = []
        for start in range(0, len(entries), batch_size):
            current_paths = paths[start:start + batch_size]
            sync()
            started = time.perf_counter()
            results = model.predict(source=current_paths, device=device, imgsz=640, conf=0.25, verbose=False)
            sync()
            batch_seconds.append(time.perf_counter() - started)
            if len(results) != len(current_paths):
                raise RuntimeError("Pose output count differs from candidate count")
            for offset, result in enumerate(results):
                index = start + offset
                entry = entries[index]
                people = 0 if result.boxes is None else len(result.boxes)
                reason = "multiple_people" if people > 1 else "no_person"
                if people == 1:
                    reason = "missing_keypoints"
                    if result.keypoints is not None and len(result.keypoints.data) == 1:
                        points = result.keypoints.data[0].detach().cpu().numpy()
                        vector = extract_features(points, min_confidence=0.3)
                        reason = "unusable_keypoints" if vector is None else "valid"
                        if vector is not None:
                            features[index] = vector
                sample_id = entry.get("sample_id", str(index))
                overlay = destination / "overlays" / ("{:04d}.jpg".format(index))
                if index < 12:
                    # PIL handles non-ASCII Windows paths reliably.
                    from PIL import Image
                    Image.fromarray(result.plot()[..., ::-1]).save(overlay)
                observations.append({
                    "sample_id": sample_id, "group_id": entry.get("group_id"),
                    "image_path": paths[index], "people": people, "status": reason,
                    "overlay": str(overlay) if index < 12 else None,
                })
        with (destination / "unlabeled_features.npz").open("xb") as handle:
            np.savez_compressed(handle, features=features,
                                sample_ids=np.asarray([o["sample_id"] for o in observations]))
        report = {
            "scope": "unlabeled development candidates; pose availability only; no class labels or accuracy",
            "device": device, "python": platform.python_version(), "platform": platform.platform(),
            "torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "candidates_sha256": hashlib.sha256(candidates_path.read_bytes()).hexdigest(),
            "count": len(entries), "status_counts": dict(Counter(o["status"] for o in observations)),
            "selection_protocol": "deterministic round-robin over candidate selection strata",
            "selection_counts": dict(Counter(e.get("selection_stratum", "unknown") for e in entries)),
            "imgsz": 640, "person_confidence": 0.25, "keypoint_confidence": 0.3,
            "batch_size": batch_size, "warmup_batches": 3, "warmup_seconds": warmup_seconds,
            "batch_seconds": batch_seconds,
            "mean_ms_per_image": 1000 * sum(batch_seconds) / len(entries),
            "timing_scope": "synchronized YOLO.predict including file loading, preprocess and postprocess; excludes overlays/features/MLP/reminders; not application FPS",
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None,
            "memory_scope": "PyTorch allocator during measured calls after warmup; not total device peak",
            "samples": observations,
        }
        (destination / "measurements.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return {key: report[key] for key in ("device", "count", "status_counts", "mean_ms_per_image", "cuda_peak_allocated_bytes")}
    except Exception as error:
        (destination / "failure.json").write_text(json.dumps({
            "error": str(error), "type": type(error).__name__, "completed_samples": observations,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="models/yolo11n-pose.pt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(check(args.candidates, args.output, args.weights, args.device,
                           args.count, args.batch_size), ensure_ascii=False, indent=2))
