"""Propose and download a small official COCO RGB review pool, never labels."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def prepare(annotations, output_dir, count=24):
    import requests
    from PIL import Image, ImageDraw, ImageOps

    annotations = Path(annotations)
    data = json.loads(annotations.read_text())
    images = {image["id"]: image for image in data["images"]}
    licenses = {license["id"]: license for license in data["licenses"]}
    counts = {}
    for annotation in data["annotations"]:
        counts[annotation["image_id"]] = counts.get(annotation["image_id"], 0) + 1
    candidates = []
    for annotation in data["annotations"]:
        if annotation["num_keypoints"] < 10 or counts[annotation["image_id"]] != 1:
            continue
        points = np.asarray(annotation["keypoints"], dtype=float).reshape(17, 3)
        if np.any(points[[0, 5, 6, 11, 12], 2] == 0):
            continue
        leg_scores = []
        for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
            if np.any(points[[hip, knee, ankle], 2] < 1):
                continue
            first = points[hip, :2] - points[knee, :2]
            second = points[ankle, :2] - points[knee, :2]
            scale = np.linalg.norm(first) * np.linalg.norm(second)
            if scale < 1:
                continue
            angle = np.degrees(np.arccos(np.clip(np.dot(first, second) / scale, -1, 1)))
            if 45 < angle < 140:
                leg_scores.append(1 - abs(angle - 90) / 90)
        if not leg_scores:
            continue
        _, _, width, height = annotation["bbox"]
        if width < 90 or height < 150:
            continue
        image = images[annotation["image_id"]]
        # Retain the exact license; avoid NoDerivs images for this crop/review pool.
        if image["license"] not in (1, 2, 4, 5):
            continue
        candidates.append({
            "image": image, "annotation_id": annotation["id"],
            "bbox": annotation["bbox"], "keypoints": annotation["keypoints"],
            "license": licenses[image["license"]],
            "candidate_score": max(leg_scores) * math.log(width * height),
        })
    candidates.sort(key=lambda item: item["candidate_score"], reverse=True)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected = candidates[:count]

    def download(candidate):
        entry = dict(candidate)
        image = candidate["image"]
        target = Path("data/raw/coco2017/images/val2017") / image["file_name"]
        target.parent.mkdir(parents=True, exist_ok=True)
        url = "https://s3.amazonaws.com/images.cocodataset.org/val2017/" + image["file_name"]
        entry.update(requested_url=url, local_path=str(target.resolve()))
        try:
            if not target.exists():
                response = requests.get(url, timeout=(15, 60))
                response.raise_for_status()
                entry["final_url"] = response.url
                with Image.open(__import__("io").BytesIO(response.content)) as decoded:
                    decoded.verify()
                target.write_bytes(response.content)
            with Image.open(target) as decoded:
                entry["dimensions"] = list(decoded.size)
            entry.update(status="downloaded", bytes=target.stat().st_size,
                         sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        except Exception as exc:
            entry.update(status="failed", error="{}: {}".format(type(exc).__name__, exc))
        return entry

    entries = list(ThreadPoolExecutor(max_workers=4).map(download, selected))
    report = {
        "note": "Geometry proposes review candidates only; no posture labels were generated.",
        "annotation_sha256": hashlib.sha256(annotations.read_bytes()).hexdigest(),
        "transport_note": "Canonical images.cocodataset.org certificate name mismatch was logged; this uses the same official public S3 bucket over verified HTTPS.",
        "annotations": str(annotations.resolve()), "eligible_candidate_count": len(candidates),
        "requested_count": count, "downloaded_count": sum(item["status"] == "downloaded" for item in entries),
        "entries": entries,
    }
    (destination / "downloads.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    successful = [entry for entry in entries if entry["status"] == "downloaded"]
    columns, cell_width, cell_height = 4, 320, 280
    canvas = Image.new("RGB", (columns * cell_width, math.ceil(len(successful) / columns) * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, entry in enumerate(successful):
        image = Image.open(entry["local_path"]).convert("RGB")
        thumbnail = ImageOps.contain(image, (cell_width - 8, cell_height - 35))
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        canvas.paste(thumbnail, (x, y))
        draw.text((x + 3, y + cell_height - 30), entry["image"]["file_name"], fill="black")
    if successful:
        canvas.save(destination / "review_contact_sheet.jpg")
    print(json.dumps({key: report[key] for key in ("eligible_candidate_count", "requested_count", "downloaded_count")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", default="data/raw/coco2017/annotations/person_keypoints_val2017.json")
    parser.add_argument("--output", default="runs/stage1/coco_candidates")
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    prepare(args.annotations, args.output, args.count)
