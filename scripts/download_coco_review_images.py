"""Restore archived COCO originals from the official mirror, verifying SHA256.

Standard library only. Per-image licensing is in the archived metadata.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
MAX_IMAGE_BYTES = 25 * 1024 * 1024


def restore(pool_path, output, limit=None):
    pool = json.loads(Path(pool_path).read_text(encoding="utf-8-sig"))
    output = Path(output).expanduser().resolve()
    entries = [entry for entry in pool["entries"] if entry.get("status") == "downloaded"]
    if limit is not None:
        entries = entries[:limit]
    seen = set()
    fetched = cached = 0
    for entry in entries:
        split = entry["original_split"]
        filename = entry["image"]["file_name"]
        expected = entry["original_sha256"]
        if split not in ("train2017", "val2017") or not re.fullmatch(r"\d{12}\.jpg", filename):
            raise ValueError("Invalid COCO split or filename")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Missing or invalid original image hash")
        target = output / split / filename
        if target in seen:
            continue
        seen.add(target)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError("Existing image differs from archive; refusing overwrite: " + str(target))
            cached += 1
            continue
        url = "https://s3.amazonaws.com/images.cocodataset.org/{}/{}".format(split, filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            temporary = None
            try:
                with urlopen(url, timeout=30) as response:
                    content = response.read(MAX_IMAGE_BYTES + 1)
                if len(content) > MAX_IMAGE_BYTES:
                    raise ValueError("COCO image exceeds download limit")
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError("COCO image hash differs from archived original: " + filename)
                descriptor, temporary = tempfile.mkstemp(prefix=".coco-", suffix=".part", dir=str(target.parent))
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content)
                if target.exists():
                    raise ValueError("Image appeared during download; rerun to verify: " + str(target))
                os.rename(temporary, target)
                temporary = None
                fetched += 1
                print("Downloaded {} (SHA256 verified)".format(filename), flush=True)
                break
            except (OSError, ValueError):
                if attempt == 2:
                    raise
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
    return {"unique_images": len(seen), "downloaded": fetched, "verified_existing": cached}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default=str(ROOT / "annotations/archive/coco/downloads.json"))
    parser.add_argument("--output", default="data/raw/coco2017/images")
    parser.add_argument("--limit", type=int, help="optional small download trial")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        print(json.dumps(restore(args.pool, args.output, args.limit), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, "Restore failed: {}\n".format(error))


if __name__ == "__main__":
    main()
