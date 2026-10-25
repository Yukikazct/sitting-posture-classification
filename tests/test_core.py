"""Tests for data leakage, rejection accounting, and feature geometry."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from posture.features import FEATURE_DIM, FEATURE_NAMES, extract_features
from posture.metrics import classification_metrics
from posture.training import select_device, train


def upright_keypoints():
    points = np.zeros((17, 3), dtype=np.float64)
    points[:, 2] = 0.9
    points[0, :2] = [100, 30]
    points[3, :2], points[4, :2] = [90, 35], [110, 35]
    points[5, :2], points[6, :2] = [80, 60], [120, 60]
    points[11, :2], points[12, :2] = [85, 160], [115, 160]
    return points


class FeatureTests(unittest.TestCase):
    def test_translation_and_scale_invariance(self):
        points = upright_keypoints()
        expected = extract_features(points)
        transformed = points.copy()
        transformed[:, :2] = points[:, :2] * 2.5 + [700, -210]
        np.testing.assert_allclose(extract_features(transformed), expected, atol=1e-6)
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIM)
        self.assertEqual(expected.shape, (25,))
        np.testing.assert_allclose(expected[-4:], [0, 0, -0.3, 0], atol=1e-6)

    def test_missing_essential_point_rejects_but_optional_ear_is_zero(self):
        points = upright_keypoints()
        points[3, 2] = 0.1
        features = extract_features(points)
        np.testing.assert_array_equal(features[3:6], np.zeros(3))
        points[11, 2] = 0.1
        self.assertIsNone(extract_features(points))

    def test_degenerate_and_nonfinite_poses_reject(self):
        points = upright_keypoints()
        points[[11, 12], 1] = 60
        self.assertIsNone(extract_features(points))
        points = upright_keypoints()
        points[0, 0] = np.nan
        self.assertIsNone(extract_features(points))


class MetricTests(unittest.TestCase):
    def test_rejections_are_false_negatives(self):
        metrics = classification_metrics([0, 0, 1, 2, 3], [0, -1, 1, 2, 3])
        self.assertEqual(metrics["confusion_matrix"][0][4], 1)
        self.assertEqual(metrics["per_class_recall"][0], 0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.8)
        self.assertAlmostEqual(metrics["coverage"], 0.8)
        self.assertAlmostEqual(metrics["macro_f1"], (2 / 3 + 3) / 4)

    def test_empty_metrics_are_finite(self):
        metrics = classification_metrics([], [])
        self.assertEqual(metrics["accuracy"], 0)
        self.assertEqual(metrics["coverage"], 0)
        self.assertEqual(np.asarray(metrics["confusion_matrix"]).shape, (4, 5))
        self.assertTrue(np.isfinite(metrics["per_class_recall"]).all())


class TrainingTests(unittest.TestCase):
    def test_checkpoint_normalization_uses_train_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = np.vstack([
                np.full((4, 25), 2.0), np.full((4, 25), 4.0),
                np.full((4, 25), 50.0), np.full((4, 25), 1000.0),
            ]).astype(np.float32)
            np.savez(root / "features.npz", features=features,
                     labels=np.tile(np.arange(4), 4),
                     splits=np.array(["train"] * 8 + ["val"] * 4 + ["test"] * 4),
                     sample_ids=np.array(["sample_{}".format(i) for i in range(16)]),
                     views=np.array(["front"] * 16))
            summary = train(str(root / "features.npz"),
                            {"training": {"max_epochs": 2, "batch_size": 8}},
                            str(root / "run"), device="cpu")
            checkpoint = torch.load(root / "run" / "checkpoint.pt", weights_only=True)
            np.testing.assert_allclose(checkpoint["mean"], np.full(25, 3.0))
            np.testing.assert_allclose(checkpoint["std"], np.ones(25))
            self.assertEqual(summary["train_samples"], 8)
            self.assertFalse(any("test" in key for key in summary))
            self.assertTrue((root / "run" / "history.csv").is_file())
            self.assertTrue((root / "run" / "summary.json").is_file())

    def test_invalid_device_is_rejected(self):
        with self.assertRaises(ValueError):
            select_device("magic")


if __name__ == "__main__":
    unittest.main()
