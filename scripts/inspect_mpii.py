"""Inventory official MPII metadata without downloading images or assigning labels."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


def scalar(value):
    values = np.asarray(value).reshape(-1)
    return values[0].item() if values.size == 1 else None


def inspect(annotations, mapping, output):
    annotations, mapping = Path(annotations), Path(mapping)
    release = loadmat(annotations, simplify_cells=True)["RELEASE"]
    keyframes = loadmat(mapping, simplify_cells=True)["annolist_keyframes"]
    images, activities, train = release["annolist"], release["act"], release["img_train"]
    if not len(images) == len(activities) == len(train) == len(keyframes):
        raise ValueError("MPII metadata arrays differ in length")
    videos = release["video_list"]
    counts = Counter()
    groups = defaultdict(set)
    candidates = []
    missing_video = Counter()
    for image, activity, is_train, keyframe in zip(images, activities, train, keyframes):
        split = "train" if bool(is_train) else "test"
        filename = image["image"]["name"]
        name = activity.get("act_name")
        name = name if isinstance(name, str) else ""
        counts[(name, split)] += 1
        video_index = scalar(image.get("vididx", []))
        video_id = None
        if video_index is not None and int(video_index) == video_index and 1 <= video_index <= len(videos):
            video_id = str(videos[int(video_index) - 1])
            groups[video_id].add(split)
        else:
            missing_video[split] += 1
        keyframe_path = keyframe["image"]["name"]
        if keyframe_path.split("/")[0] != Path(filename).stem:
            raise ValueError("Image-to-keyframe mapping order mismatch")
        # Activity proposes candidates only. Mixed standing/reclining activities
        # are not selected in this first inventory; they remain in activity.csv.
        if not re.search(r"\b(sitting|reading|typing)\b", name, re.I):
            continue
        if re.search(r"\b(standing|reclining)\b", name, re.I):
            continue
        candidates.append({
            "image_filename": filename, "original_split": split,
            "activity": name, "activity_id": int(activity["act_id"]),
            "video_id": video_id, "timestamp_sec": scalar(image.get("frame_sec", [])),
            "keyframe_path": keyframe_path,
            "group_id": "mpii_video_" + video_id if video_id else "mpii_image_" + Path(filename).stem,
            "image_downloaded": False, "posture_label": None,
        })
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "activities.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["activity", "original_split", "image_count"])
        writer.writerows((name, split, count) for (name, split), count in sorted(counts.items()))
    report = {
        "scope": "official metadata inventory only; activities are not four-class posture labels",
        "image_count": len(images),
        "original_split_counts": {"train": int(np.count_nonzero(train)), "test": int(np.count_nonzero(np.asarray(train) == 0))},
        "images_without_activity": {s: counts[("", s)] for s in ("train", "test")},
        "listed_video_count": len(videos), "images_missing_video_id": dict(missing_video),
        "videos_crossing_original_splits": sorted(k for k, v in groups.items() if len(v) > 1),
        "video_isolation_limitation": "Only records with known video IDs can be compared. Missing test IDs prevent verifying train/test video isolation; an empty crossing list is not proof of isolation.",
        "candidate_image_count": len(candidates),
        "candidate_split_counts": dict(Counter(c["original_split"] for c in candidates)),
        "candidate_video_group_count": len({c["video_id"] for c in candidates if c["video_id"]}),
        "candidate_activity_counts": dict(sorted(Counter(c["activity"] for c in candidates).items())),
        "annotations_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "mapping_sha256": hashlib.sha256(mapping.read_bytes()).hexdigest(),
        "candidate_rule": "activity contains sitting/reading/typing and does not contain standing/reclining; manual image review still required",
        "images_downloaded": 0, "candidates": candidates,
    }
    (destination / "inventory.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return {k: v for k, v in report.items() if k not in ("candidates", "candidate_activity_counts", "videos_crossing_original_splits")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/raw/mpii/mpii_human_pose_v1_u12_1.mat")
    parser.add_argument("--mapping", default="data/raw/mpii/mpii_human_pose_v1_sequences_keyframes.mat")
    parser.add_argument("--output", default="runs/stage2/mpii_inventory")
    args = parser.parse_args()
    print(json.dumps(inspect(args.annotations, args.mapping, args.output), ensure_ascii=False, indent=2))
