"""Train-to-demo checkpoint integration without camera or vision dependencies."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from posture.demo import _load_classifier
from posture.features import FEATURE_NAMES
from posture.training import train


class DemoCheckpointTests(unittest.TestCase):
    def test_real_training_checkpoint_loads_and_rejects_schema_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = np.random.default_rng(7).normal(size=(16, 25)).astype(np.float32)
            cache = root / "smoke.npz"
            np.savez(
                cache, features=features, labels=np.arange(16) % 4,
                splits=np.array(["train"] * 8 + ["val"] * 4 + ["test"] * 4),
                sample_ids=np.array(["smoke-{}".format(i) for i in range(16)]),
                views=np.array(["front"] * 16),
            )
            result = train(str(cache), {"training": {
                "hidden_dims": [8, 4], "dropout": 0.0, "batch_size": 8,
                "max_epochs": 1, "patience": 1,
            }}, str(root / "run"), seed=7, device="cpu")
            checkpoint = Path(result["checkpoint"])
            loaded_torch, model, mean, std, device = _load_classifier(checkpoint, "cpu")
            self.assertIs(loaded_torch, torch)
            self.assertEqual(str(device), "cpu")
            self.assertFalse(model.training)
            np.testing.assert_allclose(mean, features[:8].mean(axis=0), atol=1e-6)
            with torch.inference_mode():
                logits = model(torch.from_numpy((features[:1] - mean) / std))
            self.assertEqual(tuple(logits.shape), (1, 4))
            self.assertTrue(torch.isfinite(logits).all())

            invalid = torch.load(checkpoint, map_location="cpu", weights_only=True)
            invalid["feature_names"] = list(reversed(FEATURE_NAMES))
            invalid_path = root / "invalid-schema.pt"
            torch.save(invalid, invalid_path)
            with self.assertRaisesRegex(ValueError, "feature schema"):
                _load_classifier(invalid_path, "cpu")


if __name__ == "__main__":
    unittest.main()
