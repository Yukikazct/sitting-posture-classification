"""Prepare bounded, unlabelled COCO person crops for development-only review."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
from urllib.parse import unquote, urlparse

import numpy as np


SPLITS = ("train2017", "val2017")
ALLOWED_LICENSES = (1, 2, 4, 5)
MAX_CANDIDATES = 240
SCOPE = "development_candidates_only"
SELECTION_MODES = ("sitting-focused", "broad-audit")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _image_filename(image):
    """Validate metadata without resolving 100,000+ annotation-only paths."""
    image_id = image["id"]
    if isinstance(image_id, bool) or not isinstance(image_id, int) or image_id < 0:
        raise ValueError("COCO image id must be a nonnegative integer")
    filename = image.get("file_name", "")
    if filename != "{:012d}.jpg".format(image_id):
        raise ValueError("Unsafe or noncanonical COCO file_name: {!r}".format(filename))
    return filename


def _safe_image_path(images_root, split, image):
    """Only canonical COCO filenames may become filesystem paths."""
    filename = _image_filename(image)
    root = Path(images_root).expanduser().resolve()
    target = (root / split / filename).resolve()
    if root not in target.parents:
        raise ValueError("Image path escapes images_root")
    return target


def _load_document(path, split):
    """Reject a requested split contradicted by filenames or image URLs."""
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("images"), list) or not isinstance(data.get("annotations"), list):
        raise ValueError("COCO document requires images and annotations arrays")
    name_splits = set(re.findall(r"(?:train|val)2017", path.stem))
    if name_splits and name_splits != {split}:
        raise ValueError("Annotation filename does not match requested split {}".format(split))
    seen = set()
    for image in data["images"]:
        _image_filename(image)
        if image["id"] in seen:
            raise ValueError("Duplicate COCO image id: {}".format(image["id"]))
        seen.add(image["id"])
        url = image.get("coco_url")
        if url:
            components = unquote(urlparse(url).path).split("/")
            url_splits = {part for part in components if part in SPLITS}
            if url_splits != {split} or components[-1] != image["file_name"]:
                raise ValueError("coco_url does not match requested split or filename: {}".format(url))
        elif not name_splits:
            raise ValueError("Cannot verify split: use a split-named annotation file or coco_url for every image")
    if not data["images"] and not name_splits:
        raise ValueError("Cannot verify split of empty document without a split-named filename")
    return path, data


def _valid_bbox(bbox):
    try:
        values = np.asarray(bbox, dtype=float)
    except (TypeError, ValueError):
        return False
    return values.shape == (4,) and bool(np.isfinite(values).all()) and bool(np.all(values[2:] > 0))


def _geometry(points):
    scores = []
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        if np.any(points[[hip, knee, ankle], 2] < 1):
            continue
        first = points[hip, :2] - points[knee, :2]
        second = points[ankle, :2] - points[knee, :2]
        scale = np.linalg.norm(first) * np.linalg.norm(second)
        if scale < 1:
            continue
        angle = float(np.degrees(np.arccos(np.clip(np.dot(first, second) / scale, -1, 1))))
        if 45 < angle < 145:
            scores.append(1 - abs(angle - 90) / 90)
    return max(scores, default=0.0)


def _associated_seats(bbox, points, seats):
    visible_hips = points[[11, 12]][points[[11, 12], 2] > 0]
    x, y, width, height = bbox
    hip_x, hip_y = visible_hips[:, :2].mean(axis=0) if len(visible_hips) else (x + width / 2, y + height * 0.6)
    associated = []
    for seat in seats:
        sx, sy, sw, sh = seat["bbox"]
        horizontal_overlap = max(0, min(x + width, sx + sw) - max(x, sx))
        if (sx - sw * 0.2 <= hip_x <= sx + sw * 1.2
                and sy - sh * 0.2 <= hip_y <= sy + sh * 1.2
                and horizontal_overlap >= min(width, sw) * 0.2):
            associated.append(seat["id"])
    return associated


def _candidate_pool(data, split, instances=None):
    images = {item["id"]: item for item in data["images"]}
    licenses = {item["id"]: item for item in data.get("licenses", [])}
    person_categories = {item["id"] for item in data.get("categories", []) if item.get("name") == "person"} or {1}
    seats = {}
    if instances is not None:
        seat_categories = {item["id"] for item in instances.get("categories", []) if item.get("name") in ("chair", "couch")}
        instance_images = {item["id"]: item for item in instances["images"]}
        for image_id, image in images.items():
            other = instance_images.get(image_id)
            if other is None or any(other.get(field) != image.get(field) for field in ("file_name", "width", "height")):
                raise ValueError("Instances image metadata does not match keypoints for image {}".format(image_id))
        for item in instances["annotations"]:
            if item.get("category_id") in seat_categories and not item.get("iscrowd", 0) and _valid_bbox(item.get("bbox")):
                seats.setdefault(item["image_id"], []).append(item)
    pool = []
    skipped = Counter()
    seen_annotations = set()
    for annotation in data["annotations"]:
        annotation_id = annotation.get("id")
        if isinstance(annotation_id, bool) or not isinstance(annotation_id, int) or annotation_id < 0:
            raise ValueError("COCO annotation id must be a nonnegative integer")
        if annotation_id in seen_annotations:
            raise ValueError("Duplicate COCO annotation id: {}".format(annotation_id))
        seen_annotations.add(annotation_id)
        image = images.get(annotation.get("image_id"))
        if image is None:
            raise ValueError("Annotation refers to an absent image")
        if annotation.get("category_id", 1) not in person_categories or annotation.get("iscrowd", 0):
            skipped["not_individual_person"] += 1
            continue
        if image.get("license") not in ALLOWED_LICENSES:
            skipped["license_not_allowed"] += 1
            continue
        if image["license"] not in licenses:
            raise ValueError("Missing license metadata for image {}".format(image["id"]))
        if not _valid_bbox(annotation.get("bbox")):
            skipped["invalid_bbox"] += 1
            continue
        bbox = list(map(float, annotation["bbox"]))
        if bbox[2] < 60 or bbox[3] < 100:
            skipped["small_person_bbox"] += 1
            continue
        try:
            points = np.asarray(annotation.get("keypoints", []), dtype=float).reshape(17, 3)
        except (TypeError, ValueError):
            skipped["invalid_keypoints"] += 1
            continue
        if not np.isfinite(points).all() or np.count_nonzero(points[:, 2] > 0) < 5:
            skipped["too_few_or_invalid_keypoints"] += 1
            continue
        geometry_score = _geometry(points)
        seat_ids = _associated_seats(bbox, points, seats.get(image["id"], []))
        reasons = []
        if seat_ids:
            reasons.append("chair_or_couch_association")
        if geometry_score:
            reasons.append("bent_leg_geometry")
        pool.append({
            "sample_id": "coco_{}_i{:012d}_a{}".format(split, image["id"], annotation_id),
            "image_id": image["id"], "annotation_id": annotation_id,
            "source": "coco", "original_split": split,
            "group_id": "coco_{}_image_{:012d}".format(split, image["id"]),
            "subject_id": "", "scope": SCOPE, "human_confirmed": False,
            "image": dict(image), "bbox": bbox,
            "keypoints": annotation["keypoints"], "license": dict(licenses[image["license"]]),
            "candidate_reason": reasons, "seat_annotation_ids": seat_ids,
            "geometry_score": geometry_score,
        })
    return pool, dict(skipped)


def _select_candidates(pool, count, seed, selection_mode="sitting-focused"):
    """Use seat-plus-geometry candidates by default, with no unrelated fill."""
    if selection_mode not in SELECTION_MODES:
        raise ValueError("Unknown selection_mode: {}".format(selection_mode))
    rng = random.Random(seed)
    ordered = sorted(pool, key=lambda item: (item["image_id"], item["annotation_id"]))
    if selection_mode == "sitting-focused":
        focused = [item for item in ordered if item["seat_annotation_ids"] and item["geometry_score"] > 0]
        rng.shuffle(focused)
        selected = []
        for candidate in focused[:count]:
            entry = dict(candidate)
            entry["candidate_reason"] = list(candidate["candidate_reason"])
            entry["selection_stratum"] = "sitting_focused"
            selected.append(entry)
        return selected
    # The broader pool is available only as a separately requested audit.
    # Reserve its random stratum before applying any geometric ranking.
    shuffled = ordered[:]
    rng.shuffle(shuffled)
    selected, used = [], set()

    def take(items, target, stratum):
        taken = 0
        for candidate in items:
            if len(selected) >= count or taken >= target:
                break
            if candidate["annotation_id"] in used:
                continue
            entry = dict(candidate)
            entry["selection_stratum"] = stratum
            entry["candidate_reason"] = list(candidate["candidate_reason"])
            if stratum == "random_audit":
                entry["candidate_reason"].append("seeded_random_audit")
            selected.append(entry)
            used.add(candidate["annotation_id"])
            taken += 1

    take(shuffled, max(1, count // 4), "random_audit")
    chairs = [item for item in shuffled if item["seat_annotation_ids"]]
    take(chairs, count // 2, "chair_association")
    geometry = sorted((item for item in ordered if item["geometry_score"] > 0), key=lambda item: -item["geometry_score"])
    take(geometry, count - len(selected), "seated_geometry")
    take(shuffled, count - len(selected), "random_audit")
    return selected


def _download_original(image, images_root, split):
    import requests
    from PIL import Image

    target = _safe_image_path(images_root, split, image)
    url = "https://s3.amazonaws.com/images.cocodataset.org/{}/{}".format(split, image["file_name"])
    record = {"image_id": image["id"], "original_path": str(target), "requested_url": url}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        record["download_status"] = "cached" if target.exists() else "fetched"
        if target.exists():
            content = target.read_bytes()
        else:
            response = requests.get(url, timeout=(15, 60))
            response.raise_for_status()
            record["final_url"] = response.url
            content = response.content
        with Image.open(io.BytesIO(content)) as decoded:
            dimensions = decoded.size
            decoded.verify()
        if dimensions != (image["width"], image["height"]):
            raise ValueError("Image dimensions differ from COCO metadata")
        if record["download_status"] == "fetched":
            with target.open("xb") as handle:
                handle.write(content)
        record.update(status="downloaded", bytes=len(content), dimensions=list(dimensions),
                      original_sha256=hashlib.sha256(content).hexdigest())
    except Exception as exc:
        record.update(status="failed", download_status="failed", error="{}: {}".format(type(exc).__name__, exc))
    return record


def _create_crop(entry, destination, padding):
    from PIL import Image

    target = (destination / "crops" / (entry["sample_id"] + ".jpg")).resolve()
    if destination not in target.parents:
        raise ValueError("Crop path escapes output directory")
    with Image.open(entry["original_path"]) as original:
        original = original.convert("RGB")
        x, y, width, height = entry["bbox"]
        rect = [max(0, math.floor(x - width * padding)), max(0, math.floor(y - height * padding)),
                min(original.width, math.ceil(x + width * (1 + padding))),
                min(original.height, math.ceil(y + height * (1 + padding)))]
        if rect[2] <= rect[0] or rect[3] <= rect[1]:
            raise ValueError("Person bbox does not intersect the image")
        crop = original.crop(rect)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            crop.save(handle, format="JPEG", quality=95)
    entry.update(crop_path=str(target), image_path=str(target), crop_rect=rect,
                 target_bbox_in_crop=[x - rect[0], y - rect[1], width, height],
                 crop_sha256=_sha256(target))


def _contact_sheets(entries, destination, page_size=24):
    from PIL import Image, ImageDraw, ImageOps

    successful = [item for item in entries if item["status"] == "downloaded"]
    paths = []
    columns, cell_width, cell_height = 4, 320, 280
    for start in range(0, len(successful), page_size):
        page = successful[start:start + page_size]
        canvas = Image.new("RGB", (columns * cell_width, math.ceil(len(page) / columns) * cell_height), "white")
        draw = ImageDraw.Draw(canvas)
        for index, entry in enumerate(page):
            with Image.open(entry["crop_path"]) as opened:
                crop = opened.convert("RGB")
            target_draw = ImageDraw.Draw(crop)
            x, y, width, height = entry["target_bbox_in_crop"]
            target_draw.rectangle((x, y, x + width, y + height), outline="red", width=3)
            thumbnail = ImageOps.contain(crop, (cell_width - 8, cell_height - 48))
            cx, cy = (index % columns) * cell_width, (index // columns) * cell_height
            canvas.paste(thumbnail, (cx + 4, cy + 3))
            draw.text((cx + 3, cy + cell_height - 42), "image {} / person {}".format(entry["image_id"], entry["annotation_id"]), fill="black")
            draw.text((cx + 3, cy + cell_height - 24), entry["selection_stratum"] + " | UNLABELLED", fill="black")
        path = destination / "contact_sheets" / "page_{:03d}.jpg".format(start // page_size + 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            canvas.save(handle, format="JPEG", quality=90)
        paths.append(str(path.resolve()))
    return paths


def prepare(annotations, output_dir, count=240, *, split, images_root="data/raw/coco2017/images", instances=None, seed=42, padding=0.12, selection_mode="sitting-focused"):
    if split not in SPLITS:
        raise ValueError("split must be train2017 or val2017")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_CANDIDATES:
        raise ValueError("count must be between 1 and {}".format(MAX_CANDIDATES))
    if not math.isfinite(padding) or not 0 <= padding <= 0.5:
        raise ValueError("padding must be finite and between 0 and 0.5")
    if selection_mode not in SELECTION_MODES:
        raise ValueError("Unknown selection_mode: {}".format(selection_mode))
    if selection_mode == "sitting-focused" and not instances:
        raise ValueError("sitting-focused requires matching --instances annotations for chair/couch association; use broad-audit explicitly for a separate general-person audit")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError("Refusing to overwrite existing output directory: {}".format(destination))
    annotations_path, data = _load_document(annotations, split)
    instance_path, instance_data = _load_document(instances, split) if instances else (None, None)
    pool, skipped = _candidate_pool(data, split, instance_data)
    selected = _select_candidates(pool, count, seed, selection_mode)
    eligible_count = len(pool)
    if selection_mode == "sitting-focused":
        eligible_count = sum(bool(item["seat_annotation_ids"]) and item["geometry_score"] > 0 for item in pool)
        skipped["not_seat_and_geometry"] = len(pool) - eligible_count
    for entry in selected:
        _safe_image_path(images_root, split, entry["image"])
    destination.mkdir(parents=True, exist_ok=False)
    originals = {item["image_id"]: item["image"] for item in selected}
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(lambda image: _download_original(image, images_root, split), originals.values()))
    by_image = {record["image_id"]: record for record in records}
    entries = []
    for candidate in selected:
        entry = dict(candidate)
        entry.update(by_image[entry["image_id"]])
        if entry["status"] == "downloaded":
            try:
                _create_crop(entry, destination, padding)
            except Exception as exc:
                entry.update(status="failed", error="{}: {}".format(type(exc).__name__, exc))
        entries.append(entry)
    report = {
        "schema_version": 1, "scope": SCOPE, "human_confirmed_count": 0,
        "note": "Candidates have no posture labels. Machine screening is not proof of sitting: visual prescreening is required before any candidate enters the main human-labeling queue. Sitting-focused requires both chair/couch association and bent-leg geometry, and never fills shortages with unrelated people. Broad-audit is a separately requested general-person audit. Do not use this development pool as a final held-out test set.",
        "identity_limitation": "COCO person annotation IDs are not cross-image subject identities. All crops of an original image share a group_id.",
        "annotations": str(annotations_path), "annotation_sha256": _sha256(annotations_path),
        "instances": str(instance_path) if instance_path else None,
        "instances_sha256": _sha256(instance_path) if instance_path else None,
        "original_split": split, "seed": seed, "padding": padding,
        "selection_mode": selection_mode,
        "allowed_license_ids": list(ALLOWED_LICENSES), "images_root": str(Path(images_root).expanduser().resolve()),
        "person_pool_count": len(pool), "eligible_candidate_count": eligible_count, "skipped_counts": skipped,
        "requested_count": count, "selected_count": len(entries),
        "requested_unique_image_count": len(originals),
        "downloaded_unique_image_count": sum(record["status"] == "downloaded" for record in records),
        "downloaded_count": sum(entry["status"] == "downloaded" for entry in entries),
        "selection_counts": dict(Counter(item["selection_stratum"] for item in entries)),
        "entries": entries,
    }
    report["contact_sheets"] = _contact_sheets(entries, destination)
    with (destination / "downloads.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in ("eligible_candidate_count", "requested_count", "selected_count", "downloaded_count", "requested_unique_image_count")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--instances", help="Matching COCO instances JSON; required for sitting-focused")
    parser.add_argument("--selection-mode", choices=SELECTION_MODES, default="sitting-focused",
                        help="Default requires seat association AND bent-leg geometry; broad-audit is a separate general-person audit")
    parser.add_argument("--images-root", default="data/raw/coco2017/images")
    parser.add_argument("--output", default="runs/dataset_review/coco_candidates")
    parser.add_argument("--count", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding", type=float, default=0.12)
    args = parser.parse_args()
    if not 1 <= args.count <= MAX_CANDIDATES:
        parser.error("--count must be between 1 and {}".format(MAX_CANDIDATES))
    if args.selection_mode == "sitting-focused" and not args.instances:
        parser.error("--selection-mode sitting-focused requires --instances")
    prepare(args.annotations, args.output, args.count, split=args.split, images_root=args.images_root,
            instances=args.instances, seed=args.seed, padding=args.padding, selection_mode=args.selection_mode)
