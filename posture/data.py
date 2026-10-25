"""Manifest validation and group-safe data splitting.

Images are never copied or relabelled by this module. Relative image paths are
always interpreted relative to the CSV that contains them.
"""

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union


CLASS_NAMES = ("upright", "head_down", "forward_lean", "side_lean")
SPLITS = ("train", "val", "test")
MANIFEST_FIELDS = (
    "sample_id", "image_path", "label", "split", "subject_id", "video_id",
    "group_id", "view", "source", "timestamp_sec",
)
PathLike = Union[str, Path]


def read_manifest(path: PathLike, require_split: bool = False) -> List[Dict[str, str]]:
    """Read a UTF-8 CSV and validate its standard four-class schema.

    Returned image paths are absolute. Missing images are deliberately left for
    the extraction/audit stage so that they can be recorded as rejected samples.
    """
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not fields:
            raise ValueError("Manifest is empty; add a CSV header and manually labelled samples first.")
        missing = [field for field in MANIFEST_FIELDS if field not in fields]
        if missing:
            raise ValueError("Manifest is missing CSV columns: " + ", ".join(missing))
        if len(fields) != len(set(fields)):
            raise ValueError("Manifest contains duplicate CSV column names.")
        rows = []
        for line_number, raw in enumerate(reader, start=2):
            if None in raw:
                raise ValueError("Too many CSV values on line {}.".format(line_number))
            row = {key: (value or "").strip() for key, value in raw.items()}
            if not any(row.values()):
                continue
            if row["image_path"]:
                image_path = Path(row["image_path"]).expanduser()
                if not image_path.is_absolute():
                    image_path = manifest_path.parent / image_path
                row["image_path"] = str(image_path.resolve())
            rows.append(row)
    validate_manifest(rows, CLASS_NAMES, require_split=require_split)
    return rows


def validate_manifest(
    rows: Sequence[Mapping[str, Any]],
    class_names: Sequence[str] = CLASS_NAMES,
    require_split: bool = False,
) -> Dict[str, Any]:
    """Reject invalid labels and known leakage; disclose unknown identities.

    Identity columns are separate namespaces. A subject, video, capture group,
    or image can occur several times within one split, but never in two splits.
    Blank identities do not establish that subjects are independent.
    """
    if not rows:
        raise ValueError("Manifest is empty; add real, manually labelled samples first.")
    classes = tuple(class_names)
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("class_names must contain unique class labels.")
    seen_ids = set()
    split_counts = Counter()
    class_counts = Counter()
    per_split = {split: Counter() for split in SPLITS}
    owners = {field: {} for field in ("image_path", "subject_id", "video_id", "group_id")}
    missing_identity = Counter()
    image_counts = Counter()
    for index, row in enumerate(rows, start=1):
        sample_id = str(row.get("sample_id", "")).strip()
        image_path = str(row.get("image_path", "")).strip()
        label = str(row.get("label", "")).strip()
        split = str(row.get("split", "")).strip()
        if not sample_id or not image_path:
            raise ValueError("Row {} requires sample_id and image_path.".format(index))
        if sample_id in seen_ids:
            raise ValueError("Duplicate sample_id: {}".format(sample_id))
        seen_ids.add(sample_id)
        if label not in classes:
            raise ValueError("Invalid label {!r} in sample {}. Expected {}.".format(label, sample_id, classes))
        if split not in ("",) + SPLITS:
            raise ValueError("Invalid split {!r} in sample {}.".format(split, sample_id))
        if require_split and not split:
            raise ValueError("Sample {} has no split; assign train/val/test first.".format(sample_id))
        timestamp = str(row.get("timestamp_sec", "")).strip()
        if timestamp:
            try:
                timestamp_value = float(timestamp)
            except ValueError:
                raise ValueError("Invalid timestamp_sec in sample {}.".format(sample_id)) from None
            if not math.isfinite(timestamp_value) or timestamp_value < 0:
                raise ValueError("timestamp_sec must be finite and nonnegative in sample {}.".format(sample_id))
        split_counts[split or "unassigned"] += 1
        class_counts[label] += 1
        if split:
            per_split[split][label] += 1
        for field, field_owners in owners.items():
            value = image_path if field == "image_path" else str(row.get(field, "")).strip()
            if field == "image_path":
                value = str(Path(value).expanduser().resolve())
                image_counts[value] += 1
            elif not value:
                missing_identity[field] += 1
            if value and split:
                previous = field_owners.get(value)
                if previous is not None and previous != split:
                    raise ValueError("Data leakage: {} {!r} occurs in {} and {}.".format(field, value, previous, split))
                field_owners[value] = split
    limitations = []
    for field in ("subject_id", "video_id", "group_id"):
        count = missing_identity[field]
        if count:
            limitations.append("{} is missing for {}/{} samples; isolation by this identity cannot be verified.".format(field, count, len(rows)))
    duplicate_images = sum(count - 1 for count in image_counts.values() if count > 1)
    if duplicate_images:
        limitations.append("{} repeated image references exist within the same split or unassigned data; review duplicates before training.".format(duplicate_images))
    limitations.append("Path and identity checks do not detect visually duplicated files; audit image content and crop provenance separately.")
    for split in SPLITS:
        if split_counts[split]:
            missing_classes = [label for label in classes if not per_split[split][label]]
            if missing_classes:
                limitations.append("{} has no samples for: {}.".format(split, ", ".join(missing_classes)))
    return {
        "sample_count": len(rows),
        "class_counts": {label: class_counts[label] for label in classes},
        "split_counts": {split: split_counts[split] for split in SPLITS + ("unassigned",)},
        "class_counts_by_split": {split: {label: per_split[split][label] for label in classes} for split in SPLITS},
        "missing_identity_counts": {field: missing_identity[field] for field in ("subject_id", "video_id", "group_id")},
        "limitations": limitations,
    }


def _connected_groups(rows: Sequence[Mapping[str, str]]) -> List[List[int]]:
    """Union every shared identity, including indirect subject/video links."""
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        for field in ("subject_id", "video_id", "group_id", "image_path"):
            value = row.get(field, "")
            if not value:
                continue
            key = (field, value)
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    return list(groups.values())


def split_manifest(
    input_path: PathLike,
    output_path: PathLike,
    config: Mapping[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """Make an approximate 70/15/15 split without splitting connected groups.

    Existing official assignments must be preserved rather than silently
    regenerated. Output paths are absolute to preserve image references even
    when the new CSV is saved in a different directory.
    """
    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("Refusing to overwrite manifest: {}".format(destination))
    rows = read_manifest(input_path)
    classes = tuple(config.get("class_names", CLASS_NAMES))
    validate_manifest(rows, classes)
    if any(row["split"] for row in rows):
        raise ValueError("Input already contains split assignments. Preserve reviewed official splits; use an explicitly unassigned manifest to create a new split.")
    ratios_config = config.get("data", {}).get("split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15})
    if isinstance(ratios_config, Mapping):
        ratios = [float(ratios_config[split]) for split in SPLITS]
    else:
        ratios = [float(value) for value in ratios_config]
    if len(ratios) != 3 or any(not math.isfinite(value) or value <= 0 for value in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError("data.split_ratios must give three positive train/val/test fractions summing to 1.")
    groups = _connected_groups(rows)
    if len(groups) < 3:
        raise ValueError("Need at least 3 independent identity/image groups for train/val/test; found {}.".format(len(groups)))
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    targets = [len(rows) * ratio for ratio in ratios]
    counts = [0, 0, 0]
    group_counts = [0, 0, 0]
    for position, group in enumerate(groups):
        empty = [index for index, count in enumerate(group_counts) if count == 0]
        remaining = len(groups) - position
        choices = empty if len(empty) == remaining else list(range(3))
        rng.shuffle(choices)

        def cost(candidate: int) -> float:
            return sum(((count + (len(group) if index == candidate else 0) - targets[index]) ** 2) / max(targets[index], 1.0) for index, count in enumerate(counts))

        selected = min(choices, key=cost)
        for row_index in group:
            rows[row_index]["split"] = SPLITS[selected]
        counts[selected] += len(group)
        group_counts[selected] += 1
    report = validate_manifest(rows, classes, require_split=True)
    report.update({
        "seed": seed,
        "group_count": len(groups),
        "group_counts_by_split": dict(zip(SPLITS, group_counts)),
        "target_ratios": dict(zip(SPLITS, ratios)),
        "actual_ratios": {split: counts[index] / len(rows) for index, split in enumerate(SPLITS)},
        "output_path": str(destination),
    })
    report["limitations"].append("Group sizes may prevent exact split proportions and class balance; inspect class_counts_by_split before training.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return report
