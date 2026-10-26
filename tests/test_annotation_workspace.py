"""Verify migration preserves human decisions without retaining machine paths."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_annotation_workspace import prepare, json_bytes
from posture.review import ReviewStore


class AnnotationWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "another-machine" / "photos"
        self.images.mkdir(parents=True)
        self.image = self.images / "000000000001.jpg"
        self.image.write_bytes(b"jpeg fixture for path and hash tests")
        self.entry = {"sample_id": "coco_1", "source": "coco", "image_id": 1,
                      "image": {"file_name": self.image.name}, "status": "downloaded",
                      "original_split": "train2017", "bbox": [1, 2, 20, 30],
                      "group_id": "coco_image_1", "video_id": "video1",
                      "original_path": "Z:\\old-device\\photo.jpg", "crop_path": "Z:\\old-device\\crop.jpg",
                      "original_sha256": hashlib.sha256(self.image.read_bytes()).hexdigest()}
        self.metadata = self.root / "annotations/archive/coco/downloads.json"
        self.metadata.parent.mkdir(parents=True)
        self.raw = json_bytes({"entries": [self.entry]})
        self.metadata.write_bytes(self.raw)
        self.revision = {"status": "accepted", "label": "upright", "view": "side",
                         "annotator": "reviewer", "note": "original human note", "reason": "",
                         "human_confirmed": True, "revision": 1, "saved_at": "2026-10-25T01:00:00Z"}
        self.records = {"coco_1": {"latest": self.revision, "history": [self.revision]}}
        self.review = {"schema_version": 1, "candidate_file": "Z:\\old\\downloads.json",
                       "candidate_pool_sha256": hashlib.sha256(self.raw).hexdigest(),
                       "dev_only": True, "candidates": [self.entry], "records": self.records}
        self.review_path = self.metadata.with_name("review.json")
        self.review_path.write_bytes(json_bytes(self.review))
        self.output = self.root / "workspace"

    def test_resume_rebinds_paths_and_pool_but_preserves_history(self):
        old_review = self.review_path.read_bytes()
        prepare("coco", self.images, self.output, True, self.root)
        migrated = json.loads((self.output / "review.json").read_text())
        self.assertEqual(migrated["records"], self.records)
        self.assertNotIn("Z:\\", (self.output / "candidates.json").read_text())
        self.assertNotEqual(migrated["candidate_pool_sha256"], self.review["candidate_pool_sha256"])
        candidate = json.loads((self.output / "candidates.json").read_text())["entries"][0]
        self.assertFalse(Path(candidate["original_path"]).is_absolute())
        self.assertEqual(candidate["target_bbox_in_crop"], self.entry["bbox"])
        self.assertEqual((self.output / candidate["original_path"]).read_bytes(), self.image.read_bytes())
        restored = ReviewStore(self.output / "candidates.json", self.output / "review.json")
        self.assertEqual(restored.snapshot()["records"], self.records)
        self.assertEqual(self.review_path.read_bytes(), old_review)
        self.assertEqual(self.metadata.read_bytes(), self.raw)

    def test_v2_starts_fresh_and_never_overwrites_workspace(self):
        result = prepare("coco", self.images, self.output, metadata_root=self.root)
        self.assertEqual(result["label_schema_version"], "posture_v2")
        self.assertFalse((self.output / "review.json").exists())
        marker = self.output / "review.json"
        marker.write_text("keep my new annotation")
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare("coco", self.images, self.output, metadata_root=self.root)
        self.assertEqual(marker.read_text(), "keep my new annotation")

    def test_hash_mismatch_and_missing_resume_fail_before_creating_output(self):
        self.image.write_bytes(b"wrong image")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            prepare("coco", self.images, self.output, True, self.root)
        self.assertFalse(self.output.exists())
        self.image.unlink()
        with self.assertRaisesRegex(ValueError, "No matching local images"):
            prepare("coco", self.images, self.output, True, self.root)
        self.assertFalse(self.output.exists())

    def test_mpii_only_prepares_available_images_and_keeps_group(self):
        metadata = self.root / "annotations/mpii/priority_candidates_v2.json"
        metadata.parent.mkdir(parents=True)
        metadata.write_bytes(json_bytes({"candidates": [
            {"image_filename": "000000001.jpg", "video_id": "v1", "group_id": "mpii_video_v1", "posture_label": ""},
            {"image_filename": "000000002.jpg", "video_id": "v1", "group_id": "mpii_video_v1", "posture_label": ""}]}))
        (self.images / "000000001.jpg").write_bytes(b"image1")
        result = prepare("mpii", self.images, self.output, metadata_root=self.root)
        self.assertEqual(result["available_candidates"], 1)
        self.assertEqual(len(result["missing_candidates"]), 1)
        candidate = json.loads((self.output / "candidates.json").read_text())["entries"][0]
        self.assertEqual(candidate["sample_id"], "mpii_000000001")
        self.assertEqual(candidate["group_id"], "mpii_video_v1")
        self.assertEqual(candidate["posture_label"], "")
        self.assertFalse((self.output / "review.json").exists())

    def test_resume_refuses_partial_pool_and_wrong_binding(self):
        second = dict(self.entry, sample_id="coco_2", image={"file_name": "000000000002.jpg"})
        self.metadata.write_bytes(json_bytes({"entries": [self.entry, second]}))
        with self.assertRaisesRegex(ValueError, "every archived candidate"):
            prepare("coco", self.images, self.output, True, self.root)
        self.assertFalse(self.output.exists())
        self.metadata.write_bytes(self.raw)
        self.review["candidate_pool_sha256"] = "0" * 64
        self.review_path.write_bytes(json_bytes(self.review))
        with self.assertRaisesRegex(ValueError, "does not match"):
            prepare("coco", self.images, self.output, True, self.root)
        self.assertFalse(self.output.exists())

    def test_classmate_relative_paths_cannot_escape_images_directory(self):
        metadata = self.root / "annotations/archive/classmate/candidates.json"
        metadata.parent.mkdir(parents=True)
        entry = dict(self.entry, relative_path="../outside.jpg")
        metadata.write_bytes(json_bytes({"entries": [entry]}))
        with self.assertRaisesRegex(ValueError, "relative image path"):
            prepare("classmate", self.images, self.output, metadata_root=self.root)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
