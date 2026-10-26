"""Candidate generation must not create labels, leakage, or unsafe writes."""

import copy
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from scripts import prepare_coco_candidates as candidates


def fixture(split="train2017", people=1):
    points = [[90, 50, 2] for _ in range(17)]
    # Straight legs deliberately provide no bent-knee candidate signal.
    for hip, knee, ankle in ((11, 13, 15), (12, 14, 16)):
        points[hip], points[knee], points[ankle] = [90, 100, 2], [90, 150, 2], [90, 200, 2]
    return {
        "images": [{"id": 1, "file_name": "000000000001.jpg", "width": 320, "height": 240,
                    "license": 4, "coco_url": "http://images.cocodataset.org/{}/000000000001.jpg".format(split)}],
        "licenses": [{"id": 4, "name": "Attribution License", "url": "https://creativecommons.org/licenses/by/2.0/"}],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [{"id": 10 + index, "image_id": 1, "category_id": 1,
                         "bbox": [10 + 150 * index, 10, 100, 180], "num_keypoints": 17,
                         "keypoints": [value for point in points for value in point]}
                        for index in range(people)],
    }


def write_document(root, data, name="person_keypoints_train2017.json"):
    path = root / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def image_response():
    payload = io.BytesIO()
    Image.new("RGB", (320, 240), (30, 80, 150)).save(payload, format="JPEG")
    return types.SimpleNamespace(content=payload.getvalue(), url="https://s3.amazonaws.com/images.cocodataset.org/train2017/000000000001.jpg", raise_for_status=lambda: None)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        # All transport is mocked, including import, so offline tests never
        # depend on a separately installed HTTP client or reach the network.
        transport = types.SimpleNamespace(get=Mock(side_effect=AssertionError("Unexpected network request")))
        patcher = patch.dict(sys.modules, {"requests": transport})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_split_mismatch_fails_before_network_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                (fixture(), "person_keypoints_train2017.json", "val2017"),
                (fixture("val2017"), "person_keypoints_train2017.json", "train2017"),
            ]
            for index, (data, name, split) in enumerate(cases):
                with self.subTest(index=index), patch("requests.get") as get:
                    source = write_document(root, data, name)
                    output = root / "output{}".format(index)
                    with self.assertRaisesRegex(ValueError, "split"):
                        candidates.prepare(source, output, selection_mode="broad-audit", split=split, images_root=root / "images")
                    self.assertFalse(output.exists())
                    get.assert_not_called()

    def test_unverifiable_split_and_wrong_url_filename_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture()
            data["images"][0].pop("coco_url")
            source = write_document(root, data, "renamed.json")
            with self.assertRaisesRegex(ValueError, "Cannot verify split"):
                candidates.prepare(source, root / "output", selection_mode="broad-audit", split="train2017")
            data = fixture()
            data["images"][0]["coco_url"] = "https://images.cocodataset.org/train2017/000000000002.jpg"
            source = write_document(root, data)
            with self.assertRaisesRegex(ValueError, "filename"):
                candidates.prepare(source, root / "output", selection_mode="broad-audit", split="train2017")

    def test_unsafe_metadata_filenames_cannot_escape_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("../outside.jpg", "..\\outside.jpg", "C:\\outside.jpg", "/outside.jpg", "000000000001.jpg:stream"):
                with self.subTest(filename=filename), patch("requests.get") as get:
                    data = fixture()
                    data["images"][0]["file_name"] = filename
                    source = write_document(root, data)
                    with self.assertRaisesRegex(ValueError, "Unsafe"):
                        candidates.prepare(source, root / "output", selection_mode="broad-audit", split="train2017", images_root=root / "images")
                    self.assertFalse((root / "output").exists())
                    self.assertFalse((root / "images").exists())
                    get.assert_not_called()

    def test_zero_candidates_writes_empty_report_without_network_or_contact_sheet(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture()
            data["images"][0]["license"] = 3
            source = write_document(root, data)
            with patch("requests.get") as get:
                report = candidates.prepare(source, root / "output", selection_mode="broad-audit", split="train2017", images_root=root / "images")
            get.assert_not_called()
            self.assertEqual(report["entries"], [])
            self.assertEqual(report["downloaded_count"], 0)
            self.assertEqual(report["contact_sheets"], [])
            self.assertEqual(report["skipped_counts"], {"license_not_allowed": 1})
            self.assertTrue((root / "output/downloads.json").exists())

    def test_two_people_share_one_download_and_group_with_distinct_crops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_document(root, fixture(people=2))
            response = image_response()
            with patch("requests.get", return_value=response) as get:
                report = candidates.prepare(source, root / "output", 2, selection_mode="broad-audit", split="train2017", images_root=root / "images")
            get.assert_called_once()
            self.assertEqual(report["requested_unique_image_count"], 1)
            self.assertEqual(report["downloaded_count"], 2)
            self.assertEqual(report["human_confirmed_count"], 0)
            self.assertEqual(len({entry["group_id"] for entry in report["entries"]}), 1)
            self.assertEqual(len({entry["crop_path"] for entry in report["entries"]}), 2)
            for entry in report["entries"]:
                self.assertNotIn("label", entry)
                self.assertFalse(entry["human_confirmed"])
                self.assertEqual(entry["original_split"], "train2017")
                self.assertEqual(entry["scope"], "development_candidates_only")
                self.assertEqual(entry["subject_id"], "")
                self.assertEqual(entry["original_sha256"], hashlib.sha256(response.content).hexdigest())
                self.assertTrue(Path(entry["original_path"]).is_file())
                self.assertTrue(Path(entry["crop_path"]).is_file())
                self.assertEqual(entry["crop_sha256"], hashlib.sha256(Path(entry["crop_path"]).read_bytes()).hexdigest())
            self.assertEqual(len(report["contact_sheets"]), 1)
            # A separate development batch can reuse valid originals, never redownload.
            with patch("requests.get") as cached_get:
                cached = candidates.prepare(source, root / "again", 2, selection_mode="broad-audit", split="train2017", images_root=root / "images")
            cached_get.assert_not_called()
            self.assertTrue(all(entry["download_status"] == "cached" for entry in cached["entries"]))

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            marker = output / "downloads.json"
            marker.write_text("keep this evidence", encoding="utf-8")
            with patch("requests.get") as get, self.assertRaises(FileExistsError):
                candidates.prepare(root / "missing.json", output, selection_mode="broad-audit", split="train2017")
            get.assert_not_called()
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep this evidence")

    def test_random_audit_is_seeded_before_geometry_and_does_not_require_bent_legs(self):
        data = fixture()
        pool, _ = candidates._candidate_pool(data, "train2017")
        self.assertEqual(pool[0]["geometry_score"], 0)
        expanded = []
        for index in range(40):
            entry = copy.deepcopy(pool[0])
            entry.update(annotation_id=index, image_id=index, geometry_score=index / 40)
            expanded.append(entry)
        first = candidates._select_candidates(expanded, 12, 42, selection_mode="broad-audit")
        second = candidates._select_candidates(list(reversed(expanded)), 12, 42, selection_mode="broad-audit")
        self.assertEqual(first, second)
        random_order = list(range(40))
        random.Random(42).shuffle(random_order)
        self.assertEqual([item["annotation_id"] for item in first[:3]], random_order[:3])
        self.assertTrue(all(item["selection_stratum"] == "random_audit" for item in first[:3]))
        self.assertEqual(len({item["annotation_id"] for item in first}), 12)
        only_standing = candidates._select_candidates(pool, 1, 42, selection_mode="broad-audit")
        self.assertEqual(only_standing[0]["candidate_reason"], ["seeded_random_audit"])

    def test_chair_association_can_nominate_people_without_bent_knee_signal(self):
        data = fixture()
        instance_data = {"images": data["images"], "categories": [{"id": 62, "name": "chair"}],
                         "annotations": [{"id": 800, "image_id": 1, "category_id": 62, "bbox": [40, 90, 140, 130]}]}
        pool, _ = candidates._candidate_pool(data, "train2017", instance_data)
        self.assertEqual(pool[0]["geometry_score"], 0)
        self.assertEqual(pool[0]["seat_annotation_ids"], [800])
        self.assertEqual(pool[0]["candidate_reason"], ["chair_or_couch_association"])
        instance_data["images"] = copy.deepcopy(data["images"])
        instance_data["images"][0]["width"] = 321
        with self.assertRaisesRegex(ValueError, "metadata does not match"):
            candidates._candidate_pool(data, "train2017", instance_data)

    def test_default_focused_mode_requires_instances_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with self.assertRaisesRegex(ValueError, "sitting-focused requires matching --instances"):
                candidates.prepare("not_read.json", output, split="train2017")
            self.assertFalse(output.exists())

    def test_focused_selection_never_fills_shortage_with_seat_only_or_geometry_only_people(self):
        pool, _ = candidates._candidate_pool(fixture(), "train2017")
        examples = []
        for index, (seats, score) in enumerate((([800], 0.6), ([], 0.9), ([801], 0.0), ([], 0.0))):
            item = copy.deepcopy(pool[0])
            item.update(annotation_id=index, seat_annotation_ids=seats, geometry_score=score)
            examples.append(item)
        selected = candidates._select_candidates(examples, 240, 42)
        self.assertEqual([item["annotation_id"] for item in selected], [0])
        self.assertEqual(selected[0]["selection_stratum"], "sitting_focused")
        self.assertNotIn("label", selected[0])
        self.assertEqual(candidates._select_candidates(examples[1:], 240, 42), [])
        # Explicit broad audit remains available and admits its separate random layer.
        self.assertEqual(len(candidates._select_candidates(examples, 240, 42, "broad-audit")), 4)

    def test_focused_report_counts_only_joint_seat_geometry_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = fixture(people=2)
            # Only the first person has a bent leg; both can associate with a chair.
            for ankle in (15, 16):
                data["annotations"][0]["keypoints"][ankle * 3:ankle * 3 + 2] = [140, 150]
            source = write_document(root, data)
            instance_data = {
                "images": data["images"], "categories": [{"id": 62, "name": "chair"}],
                "annotations": [{"id": 800, "image_id": 1, "category_id": 62, "bbox": [40, 90, 140, 130]}],
            }
            instance_path = write_document(root, instance_data, "instances_train2017.json")
            with patch("requests.get", return_value=image_response()) as get:
                report = candidates.prepare(source, root / "output", 240, split="train2017",
                                            images_root=root / "images", instances=instance_path)
            get.assert_called_once()
            self.assertEqual(report["selection_mode"], "sitting-focused")
            self.assertEqual(report["person_pool_count"], 2)
            self.assertEqual(report["eligible_candidate_count"], 1)
            self.assertEqual(report["selected_count"], 1)
            self.assertEqual(report["requested_count"], 240)
            self.assertEqual(report["skipped_counts"]["not_seat_and_geometry"], 1)
            self.assertIn("visual prescreening is required", report["note"])
            self.assertNotIn("label", report["entries"][0])

    def test_corrupt_cached_image_is_recorded_and_never_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_document(root, fixture())
            original = root / "images/train2017/000000000001.jpg"
            original.parent.mkdir(parents=True)
            original.write_bytes(b"preserve invalid original for audit")
            with patch("requests.get") as get:
                report = candidates.prepare(source, root / "output", 1, selection_mode="broad-audit", split="train2017", images_root=root / "images")
            get.assert_not_called()
            self.assertEqual(original.read_bytes(), b"preserve invalid original for audit")
            self.assertEqual(report["entries"][0]["status"], "failed")
            self.assertEqual(report["downloaded_count"], 0)
            self.assertEqual(report["contact_sheets"], [])

    def test_network_failure_is_a_record_not_a_labelled_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = write_document(root, fixture())
            with patch("requests.get", side_effect=RuntimeError("network unavailable")):
                report = candidates.prepare(source, root / "output", 1, selection_mode="broad-audit", split="train2017", images_root=root / "images")
            self.assertEqual(report["entries"][0]["status"], "failed")
            self.assertIn("network unavailable", report["entries"][0]["error"])
            self.assertNotIn("label", report["entries"][0])
            self.assertEqual(report["contact_sheets"], [])

    def test_duplicate_annotation_ids_and_out_of_budget_counts_are_rejected(self):
        data = fixture(people=2)
        data["annotations"][1]["id"] = data["annotations"][0]["id"]
        with self.assertRaisesRegex(ValueError, "Duplicate COCO annotation"):
            candidates._candidate_pool(data, "train2017")
        for count in (0, 241, True):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "count"):
                candidates.prepare("not_read.json", "not_created", count, selection_mode="broad-audit", split="train2017")


if __name__ == "__main__":
    unittest.main()
