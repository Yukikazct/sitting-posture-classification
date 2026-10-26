"""Serve a local human-review page without generating labels or training data."""

import argparse
from pathlib import Path
import sys
import threading
import webbrowser


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from posture.review import LABEL_SCHEMAS, ReviewStore, create_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="runs/stage2/coco_pilot/downloads.json")
    parser.add_argument("--output", default="runs/stage2/coco_pilot/review.json")
    parser.add_argument("--screening", help="optional seating-prescreen JSON; auto-discovers screening.json beside candidates when present")
    parser.add_argument("--label-schema", choices=tuple(LABEL_SCHEMAS), default="posture_v1", help="posture_v2 adds back_lean for fresh human reviews; existing V1 decisions require a new output and re-review")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true", help="open the local review URL once after startup")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        store = ReviewStore(args.candidates, args.output, screening_path=args.screening, label_schema=args.label_schema)
        server = create_server(store, args.port)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    url = "http://127.0.0.1:{}/".format(server.server_port)
    print("Human review: {}".format(url), flush=True)
    print("Audit file: {}".format(store.output_path), flush=True)
    print("Label schema: {} ({} classes). Training pipeline remains posture_v1.".format(store.label_schema, len(store.class_names)), flush=True)
    if store.label_schema == "posture_v2":
        print("V2 requires fresh human review. Old V1 decisions are not copied or marked migrated.", flush=True)
    if store.screening is not None:
        counts = store.snapshot()["screening"]["counts"]
        print("AI seating prescreen: {} kept, {} awaiting human review; {} original candidates retained.".format(
            counts["keep_seated"], counts["unreviewed_seated"], counts["original_candidates"]), flush=True)
    print("Development-only pool; no final-test assignments or automatic labels. Ctrl+C stops the server.", flush=True)
    browser_timer = None
    if args.open_browser:
        browser_timer = threading.Timer(0.5, webbrowser.open, args=(url,))
        browser_timer.daemon = True
        browser_timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if browser_timer is not None:
            browser_timer.cancel()
        server.server_close()


if __name__ == "__main__":
    main()
