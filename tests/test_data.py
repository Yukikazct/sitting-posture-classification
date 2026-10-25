"""Protect identity isolation and keep rejected samples in evaluation caches."""

import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

from posture.data import CLASS_NAMES, MANIFEST_FIELDS, read_manifest, split_manifest, validate_manifest
from posture.extraction import extract_manifest
from posture.training import select_device


def row(sample_id, **overrides):
    result = dict.fromkeys(MANIFEST_FIELDS, "")
    result.update(sample_id=str(sample_id), image_path="images/{}.jpg".format(sample_id), label="upright")
    result.update(overrides)
    return result


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class ManifestTests(unittest.TestCase):
    def test_relative_paths_follow_manifest_and_require_assigned_split(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "samples.csv"
            write_manifest(manifest, [row("first")])
            records = read_manifest(manifest)
            self.assertEqual(records[0]["image_path"], str((Path(directory) / "images/first.jpg").resolve()))
            with self.assertRaisesRegex(ValueError, "no split"):
                read_manifest(manifest, require_split=True)

    def test_empty_and_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_manifest([], CLASS_NAMES)
        with self.assertRaisesRegex(ValueError, "Duplicate sample_id"):
            validate_manifest([row("duplicate"), row("duplicate")], CLASS_NAMES)

    def test_invalid_labels_splits_and_timestamps_are_rejected(self):
        for overrides in ({"label": "slouch"}, {"split": "validation"}, {"timestamp_sec": "nan"}, {"timestamp_sec": "-1"}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                validate_manifest([row("invalid", **overrides)], CLASS_NAMES)

    def test_every_known_identity_blocks_leakage(self):
        for field in ("image_path", "subject_id", "video_id", "group_id"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Data leakage"):
                validate_manifest([row("a", split="train", **{field: "same"}), row("b", split="test", **{field: "same"})], CLASS_NAMES)

    def test_missing_subjects_disclose_limitations(self):
        report = validate_manifest([row("a", split="train")], CLASS_NAMES)
        self.assertEqual(report["missing_identity_counts"]["subject_id"], 1)
        self.assertTrue(any("isolation" in limitation for limitation in report["limitations"]))

    def test_transitive_identity_groups_stay_together_and_split_is_repeatable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                row("a", subject_id="person-a", video_id="video-1"),
                row("b", video_id="video-1", group_id="capture-1"),
                row("c", group_id="capture-1", subject_id="person-c"),
                row("d", subject_id="person-c"),
            ] + [row(str(index), group_id="independent-{}".format(index), label=CLASS_NAMES[index % 4]) for index in range(20)]
            source = root / "source.csv"
            write_manifest(source, rows)
            summary = split_manifest(source, root / "split.csv", {}, seed=42)
            split_manifest(source, root / "again.csv", {"data": {"split_ratios": [0.7, 0.15, 0.15]}}, seed=42)
            result = read_manifest(root / "split.csv", require_split=True)
            self.assertEqual(len(set(item["split"] for item in result[:4])), 1)
            self.assertEqual(set(item["split"] for item in result), {"train", "val", "test"})
            self.assertEqual(summary["sample_count"], len(rows))
            self.assertEqual((root / "split.csv").read_text(), (root / "again.csv").read_text())
            with self.assertRaises(FileExistsError):
                split_manifest(source, root / "split.csv", {})

    def test_insufficient_groups_and_existing_assignments_are_not_resplit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            write_manifest(source, [row("a", subject_id="same"), row("b", subject_id="same")])
            with self.assertRaisesRegex(ValueError, "at least 3"):
                split_manifest(source, root / "output.csv", {})
            write_manifest(source, [row("a", split="train"), row("b", split="val"), row("c", split="test")])
            with self.assertRaisesRegex(ValueError, "already contains split"):
                split_manifest(source, root / "output.csv", {})


class _Tensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class ExtractionTests(unittest.TestCase):
    def test_cache_keeps_rejections_without_object_arrays_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "samples.csv"
            weights = root / "local.pt"
            weights.write_bytes(b"mock-model")
            rows = [row("good", split="train"), row("many", split="val"), row("missing", split="test"), row("weak", split="test")]
            (root / "images").mkdir()
            for sample_id in ("good", "many", "weak"):
                (root / "images" / (sample_id + ".jpg")).write_bytes(b"mock-image")
            write_manifest(source, rows)
            points = np.zeros((17, 3), dtype=np.float32)
            points[:, 2] = 0.9
            points[0, :2] = [100, 20]
            points[5, :2], points[6, :2] = [80, 50], [120, 50]
            points[11, :2], points[12, :2] = [80, 150], [120, 150]

            def predict(source, **kwargs):
                self.assertIn(kwargs["device"], ("cpu", "mps", "cuda"))
                name = Path(source).stem
                if name == "many":
                    return [types.SimpleNamespace(boxes=[1, 2], keypoints=None)]
                selected = points.copy()
                if name == "weak":
                    selected[11, 2] = 0.1
                return [types.SimpleNamespace(boxes=[1], keypoints=types.SimpleNamespace(data=[_Tensor(selected)]))]

            fake_ultralytics = types.SimpleNamespace(YOLO=lambda path: types.SimpleNamespace(predict=predict))
            config = {"pose": {"weights": str(weights), "min_keypoint_confidence": 0.3}}
            with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                report = extract_manifest(source, root / "features.npz", config, device="auto")
            self.assertEqual(report["valid_count"], 1)
            self.assertEqual(report["rejected_count"], 3)
            with np.load(root / "features.npz", allow_pickle=False) as cache:
                self.assertEqual(cache["features"].shape, (4, 25))
                self.assertTrue(np.isfinite(cache["features"][0]).all())
                self.assertTrue(np.isnan(cache["features"][1:]).all())
                np.testing.assert_array_equal(cache["sample_ids"], ["good", "many", "missing", "weak"])
                for field in cache.files:
                    self.assertNotEqual(cache[field].dtype.kind, "O")
            metadata = json.loads((root / "features.metadata.json").read_text())
            self.assertEqual(metadata["requested_device"], "auto")
            self.assertEqual(metadata["device"], str(select_device("auto")))
            self.assertEqual(metadata["manifest_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual({item["reason"] for item in metadata["rejected_samples"]}, {"multiple_people", "image_missing", "unusable_keypoints"})
            with self.assertRaises(FileExistsError):
                extract_manifest(source, root / "features.npz", config)

    def test_missing_model_is_a_local_prerequisite_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "samples.csv"
            write_manifest(source, [row("a", split="train")])
            with self.assertRaisesRegex(FileNotFoundError, "Automatic model downloads are disabled"):
                extract_manifest(source, root / "features.npz", {"pose": {"weights": str(root / "missing.pt")}})
            self.assertFalse((root / "features.npz").exists())

    def test_invalid_device_error_propagates_without_writing_a_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "samples.csv"
            weights = root / "local.pt"
            weights.write_bytes(b"mock-model")
            image = root / "image.jpg"
            image.write_bytes(b"mock-image")
            write_manifest(source, [row("a", image_path=str(image), split="train")])

            def predict(**kwargs):
                raise ValueError("Invalid CUDA device requested")

            fake_ultralytics = types.SimpleNamespace(YOLO=lambda path: types.SimpleNamespace(predict=predict))
            with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                with self.assertRaisesRegex(ValueError, "Invalid CUDA device"):
                    extract_manifest(source, root / "features.npz", {"pose": {"weights": str(weights)}}, device="cpu")
            self.assertFalse((root / "features.npz").exists())
            self.assertFalse((root / "features.metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
