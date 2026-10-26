"""Protect human truth, revision persistence, and local-only review routes."""

import base64
import csv
import hashlib
import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from posture.data import MANIFEST_FIELDS
from posture.review import MAX_BODY_BYTES, ReviewStore, create_server


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aVxkAAAAASUVORK5CYII=")


class _ReviewFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "image.png"
        self.image.write_bytes(PNG)
        self.candidates = self.root / "downloads.json"
        self.output = self.root / "review.json"
        self.entries = [self.candidate("s{}".format(i)) for i in range(4)]
        self.write_candidates()
        self.store = ReviewStore(self.candidates, self.output)

    def candidate(self, sample_id):
        return {
            "sample_id": sample_id, "status": "downloaded", "image_id": sample_id,
            "annotation_id": 1, "source": "coco", "original_split": "train2017",
            "group_id": "group-" + sample_id, "original_path": str(self.image),
            "crop_path": str(self.image), "license": {"name": "CC BY 4.0"},
        }

    def write_candidates(self):
        self.candidates.write_text(json.dumps({"entries": self.entries}), encoding="utf-8")

    def annotation(self, sample_id="s0", **updates):
        result = {
            "sample_id": sample_id, "status": "accepted", "label": "upright",
            "view": "oblique", "annotator": "人工标注者", "human_confirmed": True,
            "expected_revision": 0, "reason": "", "note": "",
        }
        result.update(updates)
        return result


class ReviewTests(_ReviewFixture):
    def test_unreviewed_candidates_have_no_default_labels(self):
        self.assertEqual(self.store.snapshot()["records"], {})
        self.assertTrue(all(c["dev_only"] for c in self.store.snapshot()["candidates"]))
        rows = list(csv.DictReader(io.StringIO(self.store.export_csv().decode("utf-8-sig"))))
        self.assertEqual(rows, [])
        self.assertFalse(self.output.exists())

    def test_revisions_survive_restart_without_losing_previous_decision(self):
        self.store.save(self.annotation(note="初标"))
        self.store.save(self.annotation(label="head_down", expected_revision=1, note="人工复核修改"))
        restored = ReviewStore(self.candidates, self.output)
        record = restored.snapshot()["records"]["s0"]
        self.assertEqual(record["latest"]["label"], "head_down")
        self.assertEqual([r["revision"] for r in record["history"]], [1, 2])
        self.assertEqual(record["history"][0]["note"], "初标")
        self.assertTrue(record["latest"]["saved_at"])
        self.assertEqual(record["latest"]["annotator_type"], "human_declared")
        self.assertFalse(list(self.root.glob(".review-*.tmp")))

    def test_export_contains_only_latest_human_accepted_labels_and_blank_split(self):
        self.store.save(self.annotation("s0"))
        self.store.save(self.annotation("s1", status="excluded", label="", reason="not_seated"))
        self.store.save(self.annotation("s2", status="pending", label="", reason="uncertain"))
        stream = io.StringIO(self.store.export_csv().decode("utf-8-sig"))
        reader = csv.DictReader(stream)
        self.assertEqual(tuple(reader.fieldnames), MANIFEST_FIELDS)
        rows = list(reader)
        self.assertEqual([r["sample_id"] for r in rows], ["s0"])
        self.assertEqual(rows[0]["split"], "")
        self.assertEqual(rows[0]["image_path"], str(self.image.resolve()))
        self.store.save(self.annotation("s0", status="pending", label="", expected_revision=1))
        self.assertEqual(len(list(csv.DictReader(io.StringIO(self.store.export_csv().decode("utf-8-sig"))))), 0)
        self.assertEqual(len(self.store.snapshot()["candidates"]), 4)

    def test_invalid_labels_confirmation_names_views_and_splits_cannot_be_saved(self):
        bad = (
            {"label": "slouch"}, {"label": ""}, {"annotator": " "},
            {"human_confirmed": False}, {"human_confirmed": "true"},
            {"view": "camera-a"}, {"status": "maybe"}, {"split": "test"},
            {"status": "excluded", "label": "", "reason": ""},
            {"status": "pending"}, {"reason": "uncertain"},
            {"reason": "invalid"}, {"note": "x" * 4001},
            {"expected_revision": True}, {"sample_id": "missing"},
        )
        for updates in bad:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                self.store.save(self.annotation(**updates))
        self.assertEqual(self.store.snapshot()["records"], {})
        self.assertFalse(self.output.exists())

    def test_stale_tab_and_external_file_changes_are_not_overwritten(self):
        self.store.save(self.annotation())
        with self.assertRaisesRegex(ValueError, "another tab"):
            self.store.save(self.annotation())
        self.output.write_text("external edit", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "externally"):
            self.store.save(self.annotation(expected_revision=1))
        self.assertEqual(self.output.read_text(encoding="utf-8"), "external edit")
        self.assertEqual(self.store.snapshot()["records"]["s0"]["latest"]["revision"], 1)

    def test_candidate_pool_change_requires_a_new_output(self):
        self.store.save(self.annotation())
        original = self.output.read_bytes()
        self.entries.append(self.candidate("s4"))
        self.write_candidates()
        with self.assertRaisesRegex(ValueError, "different candidate pool"):
            ReviewStore(self.candidates, self.output)
        self.assertEqual(self.output.read_bytes(), original)

    def test_corrupt_revision_history_is_rejected(self):
        self.store.save(self.annotation())
        data = self.store.snapshot()
        data["records"]["s0"]["latest"]["label"] = "bad"
        self.output.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(ValueError):
            ReviewStore(self.candidates, self.output)

    def test_failed_atomic_replace_preserves_both_memory_and_disk(self):
        self.store.save(self.annotation())
        original = self.output.read_bytes()
        with patch("posture.review.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                self.store.save(self.annotation(expected_revision=1, label="head_down"))
        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(self.store.snapshot()["records"]["s0"]["latest"]["label"], "upright")
        self.assertFalse(list(self.root.glob(".review-*.tmp")))

    def test_duplicate_ids_and_non_image_paths_are_rejected(self):
        self.entries.append(self.entries[0])
        self.write_candidates()
        with self.assertRaisesRegex(ValueError, "duplicated"):
            ReviewStore(self.candidates, self.output)
        secret = self.root / "private.txt"
        secret.write_text("private", encoding="utf-8")
        self.entries = [dict(self.candidate("s0"), original_path=str(secret))]
        self.write_candidates()
        with self.assertRaisesRegex(ValueError, "unsupported"):
            ReviewStore(self.candidates, self.output)

    def test_old_local_path_metadata_is_supported_and_failed_downloads_are_skipped(self):
        self.entries = [
            {"status": "downloaded", "image": {"id": 123}, "annotation_id": 4, "local_path": "image.png"},
            {"status": "failed", "sample_id": "failed"},
        ]
        self.write_candidates()
        legacy = ReviewStore(self.candidates, self.output)
        self.assertEqual(len(legacy.candidates), 1)
        self.assertEqual(legacy.candidates[0]["sample_id"], "coco_123_a4")
        self.assertEqual(legacy.candidates[0]["crop_path"], str(self.image.resolve()))


class ReviewV2Tests(_ReviewFixture):
    def test_v2_back_lean_survives_restart_and_has_separate_csv_schema(self):
        from posture.data import CLASS_NAMES
        with self.assertRaises(ValueError):
            self.store.save(self.annotation(label="back_lean"))
        v2 = ReviewStore(self.candidates, self.output, label_schema="posture_v2")
        revision = v2.save(self.annotation(label="back_lean"))
        self.assertEqual(revision["label_schema_version"], "posture_v2")
        restored = ReviewStore(self.candidates, self.output, label_schema="posture_v2")
        self.assertEqual(restored.snapshot()["label_schema_version"], "posture_v2")
        self.assertEqual(restored.snapshot()["records"]["s0"]["latest"]["label"], "back_lean")
        rows = list(csv.DictReader(io.StringIO(restored.export_csv().decode("utf-8-sig"))))
        self.assertEqual(rows[0]["label"], "back_lean")
        self.assertEqual(rows[0]["label_schema_version"], "posture_v2")
        self.assertEqual(rows[0]["split"], "")
        self.assertNotIn("back_lean", CLASS_NAMES)
        with self.assertRaisesRegex(ValueError, "different label schema"):
            ReviewStore(self.candidates, self.output)

    def test_v1_records_are_preserved_and_new_v2_requires_fresh_human_decisions(self):
        self.store.save(self.annotation(note="旧人工记录"))
        original = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "new output.*re-review"):
            ReviewStore(self.candidates, self.output, label_schema="posture_v2")
        self.assertEqual(self.output.read_bytes(), original)
        v2 = ReviewStore(self.candidates, self.root / "review.v2.json", label_schema="posture_v2")
        self.assertEqual(v2.snapshot()["records"], {})
        self.assertEqual(list(csv.DictReader(io.StringIO(v2.export_csv().decode("utf-8-sig")))), [])
        v2.save(self.annotation(label="back_lean", note="已按五类重新查看"))
        self.assertEqual(self.output.read_bytes(), original)
        self.assertEqual(ReviewStore(self.candidates, self.output).snapshot()["records"]["s0"]["latest"]["note"], "旧人工记录")

    def test_changing_only_v1_top_level_schema_cannot_claim_migration(self):
        self.store.save(self.annotation())
        state = self.store.snapshot()
        state["label_schema_version"] = "posture_v2"
        self.output.write_text(json.dumps(state), encoding="utf-8")
        original = self.output.read_bytes()
        with self.assertRaisesRegex(ValueError, "fresh V2 human decisions"):
            ReviewStore(self.candidates, self.output, label_schema="posture_v2")
        self.assertEqual(self.output.read_bytes(), original)


class ReviewScreeningTests(_ReviewFixture):
    def screening(self, decisions=None):
        decisions = decisions or {"s0": "keep_seated", "s1": "uncertain", "s2": "exclude_not_seated", "s3": "keep_seated"}
        return {
            "schema_version": 1,
            "reviewer": "assistant_visual_prescreen",
            "source_pool_sha256": hashlib.sha256(self.candidates.read_bytes()).hexdigest(),
            "records": [
                {"sample_id": sample_id, "decision": decision, "reason": "Seating-only visual review", "page": 1}
                for sample_id, decision in decisions.items()
            ],
        }

    def write_screening(self, payload=None, filename="screening.json"):
        path = self.root / filename
        path.write_text(json.dumps(payload if payload is not None else self.screening()), encoding="utf-8")
        return path

    def test_auto_discovery_prioritizes_unreviewed_seated_without_creating_labels(self):
        original_snapshot = self.store.snapshot()
        self.assertNotIn("screening", original_snapshot)
        self.write_screening()
        screened = ReviewStore(self.candidates, self.output)
        snapshot = screened.snapshot()
        self.assertEqual(snapshot["records"], {})
        self.assertEqual(snapshot["candidates"], original_snapshot["candidates"])
        self.assertEqual(snapshot["candidate_pool_sha256"], original_snapshot["candidate_pool_sha256"])
        self.assertEqual(snapshot["screening"]["unreviewed_seated_sample_ids"], ["s0", "s3"])
        self.assertEqual(snapshot["screening"]["unreviewed_uncertain_sample_ids"], ["s1"])
        self.assertEqual(snapshot["screening"]["counts"], {"original_candidates": 4, "screened": 4, "keep_seated": 2, "unreviewed_seated": 2, "accepted_disagreement": 0})
        self.assertTrue(snapshot["screening"]["read_only"])
        self.assertNotIn("screening", screened.state)
        self.assertTrue(all("label" not in record and "human_confirmed" not in record for record in snapshot["screening"]["records"]))
        self.assertEqual(list(csv.DictReader(io.StringIO(screened.export_csv().decode("utf-8-sig")))), [])
        self.assertFalse(self.output.exists())

    def test_existing_human_history_and_exports_survive_conflicting_ai_screening(self):
        self.store.save(self.annotation("s0"))
        self.store.save(self.annotation("s0", expected_revision=1, label="head_down", note="人工修改"))
        self.store.save(self.annotation("s1", status="pending", label="", reason="uncertain"))
        original_bytes = self.output.read_bytes()
        original_records = self.store.snapshot()["records"]
        self.write_screening(self.screening({"s0": "exclude_not_seated", "s1": "keep_seated", "s2": "keep_seated", "s3": "uncertain"}))
        screened = ReviewStore(self.candidates, self.output)
        snapshot = screened.snapshot()
        self.assertEqual(self.output.read_bytes(), original_bytes)
        self.assertEqual(snapshot["records"], original_records)
        self.assertEqual(snapshot["screening"]["seated_sample_ids"], ["s1", "s2"])
        self.assertEqual(snapshot["screening"]["unreviewed_seated_sample_ids"], ["s2"])
        self.assertEqual(snapshot["screening"]["unreviewed_uncertain_sample_ids"], ["s3"])
        self.assertEqual(snapshot["screening"]["accepted_disagreement_sample_ids"], ["s0"])
        self.assertEqual(snapshot["screening"]["counts"]["accepted_disagreement"], 1)
        rows = list(csv.DictReader(io.StringIO(screened.export_csv().decode("utf-8-sig"))))
        self.assertEqual([row["sample_id"] for row in rows], ["s0"])
        self.assertEqual(rows[0]["label"], "head_down")
        screened.save(self.annotation("s2", label="side_lean"))
        persisted = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertNotIn("screening", persisted)
        self.assertEqual(persisted["records"]["s0"], original_records["s0"])
        self.assertEqual(persisted["records"]["s1"], original_records["s1"])
        self.assertEqual(persisted["candidate_pool_sha256"], self.store.pool_sha256)
        self.assertEqual(screened.snapshot()["screening"]["unreviewed_seated_sample_ids"], [])

    def test_invalid_pool_ids_duplicates_decisions_or_label_injection_do_not_touch_reviews(self):
        self.store.save(self.annotation())
        original_bytes = self.output.read_bytes()
        cases = []
        wrong_hash = self.screening()
        wrong_hash["source_pool_sha256"] = "0" * 64
        cases.append(wrong_hash)
        unknown = self.screening()
        unknown["records"][0]["sample_id"] = "unknown"
        cases.append(unknown)
        duplicate = self.screening()
        duplicate["records"].append(duplicate["records"][0])
        cases.append(duplicate)
        wrong_decision = self.screening()
        wrong_decision["records"][0]["decision"] = "head_down"
        cases.append(wrong_decision)
        label_injection = self.screening()
        label_injection["records"][0]["label"] = "upright"
        cases.append(label_injection)
        fake_human = self.screening()
        fake_human["reviewer"] = "human"
        cases.append(fake_human)
        for payload in cases:
            with self.subTest(payload=payload):
                self.write_screening(payload)
                with self.assertRaises(ValueError):
                    ReviewStore(self.candidates, self.output)
                self.assertEqual(self.output.read_bytes(), original_bytes)

    def test_explicit_screening_path_and_missing_sidecar_compatibility(self):
        self.assertNotIn("screening", ReviewStore(self.candidates, self.output).snapshot())
        alternate = self.write_screening(filename="alternate.json")
        self.assertNotIn("screening", ReviewStore(self.candidates, self.output).snapshot())
        explicit = ReviewStore(self.candidates, self.output, screening_path=alternate)
        self.assertEqual(explicit.snapshot()["screening"]["seated_sample_ids"], ["s0", "s3"])
        with self.assertRaises(FileNotFoundError):
            ReviewStore(self.candidates, self.output, screening_path=self.root / "missing.json")

    def test_partial_screening_never_drops_original_candidates(self):
        self.store.save(self.annotation("s0"))
        self.write_screening(self.screening({"s2": "keep_seated"}))
        snapshot = ReviewStore(self.candidates, self.output).snapshot()
        self.assertEqual(len(snapshot["candidates"]), 4)
        self.assertEqual(snapshot["screening"]["seated_sample_ids"], ["s2"])
        self.assertEqual(snapshot["screening"]["counts"]["screened"], 1)
        self.assertEqual(snapshot["screening"]["accepted_disagreement_sample_ids"], [])


class ReviewHTTPTests(_ReviewFixture):
    def setUp(self):
        super().setUp()
        self.server = create_server(self.store, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.origin = "http://127.0.0.1:{}".format(self.server.server_port)
        status, body, _ = self.request("GET", "/api/state")
        self.assertEqual(status, 200)
        self.token = json.loads(body)["csrf_token"]

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def request(self, method, path, payload=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        body = json.dumps(payload).encode() if payload is not None else None
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        result = response.status, response.read(), dict(response.getheaders())
        connection.close()
        return result

    def post_headers(self, **updates):
        headers = {"Content-Type": "application/json", "Origin": self.origin, "X-Review-Token": self.token}
        headers.update(updates)
        return headers

    def test_only_white_listed_images_and_routes_are_readable(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        for path in ("/image/0/original", "/image/0/crop"):
            status, body, headers = self.request("GET", path)
            self.assertEqual((status, body), (200, PNG))
            self.assertEqual(headers["Content-Type"], "image/png")
        for path in ("/private.txt", "/../private.txt", "/image/0/../../private.txt", "/image/%2e%2e/private.txt", "/image/99/crop", "/image/0/crop/private.txt"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 404)
        self.assertEqual(self.request("GET", "/api/state", headers={"Host": "evil.example"})[0], 403)

    def test_same_origin_token_and_content_type_are_required(self):
        requests = (
            self.post_headers(Origin="https://evil.example"),
            self.post_headers(**{"X-Review-Token": "wrong"}),
            {"Content-Type": "application/json", "X-Review-Token": self.token},
        )
        for headers in requests:
            self.assertEqual(self.request("POST", "/api/review", self.annotation(), headers)[0], 403)
        self.assertEqual(self.request("POST", "/api/review", self.annotation(), self.post_headers(**{"Content-Type": "text/plain"}))[0], 415)
        self.assertEqual(self.store.snapshot()["records"], {})

    def test_post_persists_and_exports_have_no_test_assignment(self):
        status, body, _ = self.request("POST", "/api/review", self.annotation(), self.post_headers())
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["saved"])
        self.assertTrue(self.output.is_file())
        status, body, headers = self.request("GET", "/api/export.csv")
        self.assertEqual(status, 200)
        self.assertIn("dev-only.csv", headers["Content-Disposition"])
        self.assertEqual(list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))[0]["split"], "")
        audit = json.loads(self.request("GET", "/api/export.json")[1])
        self.assertTrue(audit["dev_only"])
        self.assertEqual(len(audit["candidates"]), 4)
        self.assertEqual(len(audit["records"]["s0"]["history"]), 1)

    def test_large_body_and_test_split_are_rejected(self):
        status, _, _ = self.request("POST", "/api/review", {"note": "x" * MAX_BODY_BYTES}, self.post_headers())
        self.assertEqual(status, 413)
        status, _, _ = self.request("POST", "/api/review", self.annotation(split="test"), self.post_headers())
        self.assertEqual(status, 400)

    def test_schema_specific_page_and_v2_http_save(self):
        original_html = self.request("GET", "/")[1].decode("utf-8")
        self.assertNotIn('value="back_lean"', original_html)
        v2 = ReviewStore(self.candidates, self.root / "review.v2.json", label_schema="posture_v2")
        server = create_server(v2, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            def request(method, path, payload=None, headers=None):
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
                connection.request(method, path, json.dumps(payload) if payload is not None else None, headers or {})
                response = connection.getresponse()
                result = response.status, response.read()
                connection.close()
                return result
            html = request("GET", "/")[1].decode("utf-8")
            self.assertIn('value="back_lean"', html)
            self.assertIn('back_lean:"后仰"', html)
            self.assertIn('"5":"back_lean"', html)
            self.assertIn("posture_v2 五类新复核", html)
            token = json.loads(request("GET", "/api/state")[1])["csrf_token"]
            headers = {"Content-Type": "application/json", "Origin": "http://127.0.0.1:{}".format(server.server_port), "X-Review-Token": token}
            self.assertEqual(request("POST", "/api/review", self.annotation(label="back_lean"), headers)[0], 200)
            row = list(csv.DictReader(io.StringIO(request("GET", "/api/export.csv")[1].decode("utf-8-sig"))))[0]
            self.assertEqual((row["label"], row["label_schema_version"]), ("back_lean", "posture_v2"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
