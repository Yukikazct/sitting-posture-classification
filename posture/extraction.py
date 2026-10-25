"""Optional YOLO pose extraction into pickle-free NumPy caches."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import numpy as np

from .data import CLASS_NAMES, read_manifest, validate_manifest


def extract_manifest(
    manifest_path: Union[str, Path],
    output_path: Union[str, Path],
    config: Mapping[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Cache one feature row for every manifest sample, including rejections.

    Rejections are NaN feature rows, not removed samples. Only an explicitly
    prepared local model file is accepted: this entrypoint never downloads a
    model implicitly. No GPU is needed to validate manifests or train the MLP.
    """
    manifest = Path(manifest_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    metadata_path = destination.with_suffix(".metadata.json")
    if destination.suffix.lower() != ".npz":
        raise ValueError("Feature cache output must end in .npz.")
    for candidate in (destination, metadata_path):
        if candidate.exists():
            raise FileExistsError("Refusing to overwrite extraction output: {}".format(candidate))
    classes = tuple(config.get("class_names", CLASS_NAMES))
    rows = read_manifest(manifest, require_split=True)
    manifest_report = validate_manifest(rows, classes, require_split=True)
    pose = config.get("pose", {})
    weights = Path(pose.get("weights", "yolo11n-pose.pt")).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError("Prepare official YOLO11n-pose weights locally and set pose.weights to that file: {}. Automatic model downloads are disabled.".format(weights))
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Pose extraction needs optional vision dependencies. Run: pip install -e '.[vision]'") from error
    from .features import FEATURE_DIM, FEATURE_NAMES, extract_features
    from .training import select_device

    min_confidence = float(pose.get("min_keypoint_confidence", 0.3))
    person_confidence = float(pose.get("person_confidence", 0.25))
    image_size = int(pose.get("imgsz", 640))
    if not np.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("pose.min_keypoint_confidence must be between 0 and 1.")
    if not np.isfinite(person_confidence) or not 0 <= person_confidence <= 1:
        raise ValueError("pose.person_confidence must be between 0 and 1.")
    if image_size <= 0:
        raise ValueError("pose.imgsz must be a positive integer.")
    selected_device = str(select_device(device))

    model = YOLO(str(weights))
    all_features = np.full((len(rows), FEATURE_DIM), np.nan, dtype=np.float32)
    rejected = []
    valid_count = 0
    for index, row in enumerate(rows):
        reason = None
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            reason = "image_missing"
        else:
            try:
                results = model.predict(
                    source=str(image_path),
                    device=selected_device,
                    imgsz=image_size,
                    conf=person_confidence,
                    verbose=False,
                )
                if len(results) != 1:
                    reason = "unexpected_result_count"
                else:
                    result = results[0]
                    person_count = 0 if result.boxes is None else len(result.boxes)
                    if person_count == 0:
                        reason = "no_person"
                    elif person_count != 1:
                        reason = "multiple_people"
                    elif result.keypoints is None or len(result.keypoints.data) != 1:
                        reason = "missing_keypoints"
                    else:
                        keypoints = result.keypoints.data[0].detach().cpu().numpy()
                        feature = extract_features(keypoints, min_confidence=min_confidence)
                        if feature is None:
                            reason = "unusable_keypoints"
                        else:
                            feature = np.asarray(feature, dtype=np.float32)
                            if feature.shape != (FEATURE_DIM,) or not np.all(np.isfinite(feature)):
                                raise ValueError("Feature extractor must produce {} finite values.".format(FEATURE_DIM))
                            all_features[index] = feature
                            valid_count += 1
            except OSError as error:
                # Per-image I/O failures are recorded, but invalid devices,
                # model/configuration errors and runtime failures must abort
                # extraction rather than masquerading as rejected poses.
                reason = "{}: {}".format(type(error).__name__, str(error))
        if reason is not None:
            rejected.append({"sample_id": row["sample_id"], "reason": reason})
    metadata = {
        "schema_version": 1,
        "manifest_path": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "model_path": str(weights),
        "model_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "device": selected_device,
        "requested_device": device,
        "pose_config": dict(pose),
        "class_names": list(classes),
        "keypoint_schema": "COCO 17 keypoints; each keypoint is [x_pixels, y_pixels, confidence]",
        "feature_count": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "sample_count": len(rows),
        "valid_count": valid_count,
        "rejected_count": len(rejected),
        "rejected_samples": rejected,
        "manifest_report": manifest_report,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another process starting this job.
    with metadata_path.open("x", encoding="utf-8") as metadata_handle:
        try:
            with destination.open("xb") as cache_handle:
                np.savez_compressed(
                    cache_handle,
                    features=all_features,
                    labels=np.asarray([classes.index(row["label"]) for row in rows], dtype=np.int64),
                    splits=np.asarray([row["split"] for row in rows], dtype=str),
                    sample_ids=np.asarray([row["sample_id"] for row in rows], dtype=str),
                    views=np.asarray([row["view"] for row in rows], dtype=str),
                )
            json.dump(metadata, metadata_handle, indent=2, ensure_ascii=False)
            metadata_handle.write("\n")
        except Exception:
            # Remove only the metadata file created by this call; leave caches
            # for inspection if writing their contents failed.
            metadata_path.unlink(missing_ok=True)
            raise
    return {"cache_path": str(destination), "metadata_path": str(metadata_path), "sample_count": len(rows), "valid_count": valid_count, "rejected_count": len(rejected)}
