"""Integration contracts: rejected samples, checkpoint provenance, safe outputs."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from posture.config import load_config
from posture.evaluation import evaluate, profile_classifier
from posture.training import train


class PipelineTests(unittest.TestCase):
    def test_rejections_survive_training_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(5)
            features = rng.normal(size=(20, 25)).astype(np.float32)
            features[[0, 10, 19]] = np.nan
            cache = root / "features.npz"
            np.savez(cache, features=features, labels=np.arange(20) % 4,
                     splits=np.array(["train"] * 8 + ["val"] * 8 + ["test"] * 4),
                     sample_ids=np.array([str(i) for i in range(20)]),
                     views=np.array(["front"] * 20))
            config = load_config()
            config["training"].update(max_epochs=1, batch_size=8)
            summary = train(cache, config, root / "train", device="cpu")
            checkpoint = torch.load(summary["checkpoint"], weights_only=True)
            np.testing.assert_allclose(checkpoint["mean"], features[1:8].mean(axis=0), atol=1e-6)
            self.assertEqual(summary["train_rejected_samples"], 1)
            self.assertEqual(summary["train_used_samples"], 7)
            self.assertEqual(summary["val_rejected_samples"], 1)
            self.assertAlmostEqual(summary["best_val_metrics"]["coverage"], 7 / 8)
            metrics = evaluate(cache, config, root / "eval", summary["checkpoint"], device="cpu")
            self.assertAlmostEqual(metrics["coverage"], 3 / 4)
            self.assertEqual(np.asarray(metrics["confusion_matrix"])[:, -1].sum(), 1)
            self.assertTrue((root / "eval" / "predictions.csv").is_file())
            self.assertTrue((root / "train" / "config.json").is_file())
            before = Path(summary["checkpoint"]).read_bytes()
            with self.assertRaises(FileExistsError):
                train(cache, config, root / "train", device="cpu")
            self.assertEqual(before, Path(summary["checkpoint"]).read_bytes())
            profile = profile_classifier(summary["checkpoint"], device="cpu", iterations=2)
            self.assertEqual(profile["params"], 3876)
            self.assertEqual(profile["linear_macs"], 3776)
            self.assertGreater(profile["mean_latency_ms"], 0)

    def test_all_rejected_train_or_val_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root / "bad.npz", features=np.full((4, 25), np.nan),
                     labels=np.arange(4), splits=np.array(["train", "train", "val", "test"]),
                     sample_ids=np.array(["a", "b", "c", "d"]), views=np.array(["front"] * 4))
            with self.assertRaisesRegex(ValueError, "valid extracted pose"):
                train(root / "bad.npz", load_config(), root / "run", device="cpu")
            self.assertFalse((root / "run").exists())


if __name__ == "__main__":
    unittest.main()
