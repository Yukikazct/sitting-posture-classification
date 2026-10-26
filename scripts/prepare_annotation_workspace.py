"""Build a portable, local annotation workspace from archived metadata and images.

Only local files are read. Original images and human decisions are never changed.
The new directory contains image copies, relative paths and migration provenance.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "mpii": "annotations/mpii/priority_candidates_v2.json",
    "coco": "annotations/archive/coco/downloads.json",
    "classmate": "annotations/archive/classmate/candidates.json",
}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_image(source, entry, images_dir):
    """Derive paths only from source filenames, never archived absolute paths."""
    if source == "mpii":
        filename = entry["image_filename"]
        if not re.fullmatch(r"[0-9]+\.(?:jpg|jpeg|png)", filename, re.I):
            raise ValueError("Invalid MPII image filename")
        choices = [images_dir / filename, images_dir / "images" / filename]
    elif source == "coco":
        filename = entry["image"]["file_name"]
        if not re.fullmatch(r"[0-9]{12}\.jpg", filename):
            raise ValueError("Invalid COCO image filename")
        split = entry.get("original_split", "train2017")
        if split not in ("train2017", "val2017"):
            raise ValueError("Invalid COCO split")
        choices = [images_dir / split / filename, images_dir / filename]
    else:
        relative = PurePosixPath(entry["relative_path"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
            raise ValueError("Invalid classmate relative image path")
        choices = [images_dir.joinpath(*relative.parts), images_dir / relative.name]
    for path in choices:
        path = path.resolve()
        if images_dir not in path.parents:
            raise ValueError("Image path escapes images directory")
        if path.is_file():
            return path
    return None


def prepare(source, images_dir, output, resume_v1=False, metadata_root=ROOT):
    if source not in SOURCES:
        raise ValueError("Unknown source")
    if resume_v1 and source == "mpii":
        raise ValueError("MPII has no archived V1 human review")
    images_dir = Path(images_dir).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    if not images_dir.is_dir():
        raise ValueError("Images directory does not exist")
    metadata = Path(metadata_root) / SOURCES[source]
    raw = metadata.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    entries = payload["candidates" if source == "mpii" else "entries"]
    if source != "mpii":
        entries = [entry for entry in entries if entry.get("status") == "downloaded"]
    found, missing = [], []
    for entry in entries:
        key = entry.get("sample_id") or "mpii_" + Path(entry["image_filename"]).stem
        path = locate_image(source, entry, images_dir)
        if path is None:
            missing.append({"sample_id": key, "image_filename": entry.get("image_filename") or entry.get("relative_path") or entry["image"]["file_name"]})
            continue
        digest = sha256(path)
        expected = entry.get("original_sha256") or entry.get("sha256")
        if expected and digest != expected:
            raise ValueError("Image SHA256 mismatch: {}".format(key))
        found.append((entry, key, path, digest))
    if not found:
        raise ValueError("No matching local images; download or transfer the source images first")
    if resume_v1 and missing:
        raise ValueError("V1 resume requires every archived candidate image; {} missing".format(len(missing)))
    archived_review = None
    prescreen = None
    if resume_v1:
        review_path = metadata.with_name("review.json")
        archived_review = json.loads(review_path.read_text(encoding="utf-8-sig"))
        if archived_review.get("label_schema_version", "posture_v1") != "posture_v1":
            raise ValueError("Archived review is not posture_v1")
        if archived_review.get("candidate_pool_sha256") != hashlib.sha256(raw).hexdigest():
            raise ValueError("Archived review does not match the archived candidate pool")
        if [item["sample_id"] for item in archived_review["candidates"]] != [item[1] for item in found]:
            raise ValueError("Archived review candidate order does not match metadata")
        screening = metadata.with_name("screening.json")
        if screening.exists():
            prescreen = json.loads(screening.read_text(encoding="utf-8-sig"))
            if prescreen.get("source_pool_sha256") != hashlib.sha256(raw).hexdigest():
                raise ValueError("Archived screening does not match source metadata")

    # Reserve the directory only after checking all existing inputs. An existing
    # workspace, its annotations and all archived source files remain untouched.
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    candidates = []
    for entry, key, path, digest in found:
        relative = "images/{}{}".format(digest, path.suffix.lower())
        destination = output / relative
        if not destination.exists():
            shutil.copyfile(path, destination)
        candidate = copy.deepcopy(entry)
        for field in ("original_path", "crop_path", "image_path", "local_path", "provided_crop_path", "crop_rect", "crop_sha256", "target_bbox_in_crop"):
            candidate.pop(field, None)
        candidate.update(sample_id=key, original_path=relative, crop_path=relative,
                         image_path=relative, original_sha256=digest, crop_sha256=digest,
                         status="downloaded", source=entry.get("source") or source,
                         dev_only=True, training_ready=False, split="")
        if source == "mpii":
            candidate["image_downloaded"] = True
            candidate["downloaded"] = True
            candidate["posture_label"] = ""
            candidate["target_bbox_unknown"] = True
        if candidate.get("bbox"):
            candidate["target_bbox_in_crop"] = copy.deepcopy(candidate["bbox"])
        candidates.append(candidate)
    pool = {"scope": "development_only_portable_annotation", "entries": candidates}
    pool_bytes = json_bytes(pool)
    (output / "candidates.json").write_bytes(pool_bytes)
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source, "source_metadata": SOURCES[source],
        "source_metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "candidate_pool_sha256": hashlib.sha256(pool_bytes).hexdigest(),
        "available_candidates": len(candidates), "missing_candidates": missing,
        "label_schema_version": "posture_v1" if resume_v1 else "posture_v2",
        "dev_only": True, "training_ready": False,
        "note": "Images copied locally; whole originals shown in both panes. Existing target boxes retained. Missing targets require human confirmation. V2 starts without copied V1 decisions.",
    }
    if archived_review is not None:
        migrated = copy.deepcopy(archived_review)
        migrated["candidate_file"] = "candidates.json"
        migrated["candidate_pool_sha256"] = provenance["candidate_pool_sha256"]
        migrated["candidates"] = copy.deepcopy(candidates)
        migrated["migration"] = {
            "created_at": provenance["created_at"],
            "source_review_sha256": sha256(metadata.with_name("review.json")),
            "source_candidate_pool_sha256": provenance["source_metadata_sha256"],
            "records_and_history_preserved": True,
            "display_change": "whole original replaces old crop; target bbox retained where available",
        }
        (output / "review.json").write_bytes(json_bytes(migrated))
        provenance["source_review_sha256"] = migrated["migration"]["source_review_sha256"]
        if prescreen is not None:
            prescreen["source_pool_sha256"] = provenance["candidate_pool_sha256"]
            (output / "screening.json").write_bytes(json_bytes(prescreen))
    (output / "preparation.json").write_bytes(json_bytes(provenance))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=tuple(SOURCES))
    parser.add_argument("--images-dir", required=True, help="Local original-image directory; no downloads are performed")
    parser.add_argument("--output", required=True, help="New workspace directory; existing directories are refused")
    parser.add_argument("--resume-v1", action="store_true", help="Restore archived four-class review history, requiring all original images")
    args = parser.parse_args()
    try:
        result = prepare(args.source, args.images_dir, args.output, args.resume_v1)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print("Prepared {} candidates; {} missing. {}".format(result["available_candidates"], len(result["missing_candidates"]), result["label_schema_version"]))
    print("Next (from repository root):")
    print('python scripts/review_candidates.py --candidates "{}" --output "{}" --label-schema {} --open-browser'.format(
        Path(args.output) / "candidates.json", Path(args.output) / "review.json", result["label_schema_version"]))


if __name__ == "__main__":
    main()
