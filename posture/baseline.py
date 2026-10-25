"""A transparent geometric baseline; thresholds require validation-set tuning.

These image-space heuristics are camera-dependent and are not a medical
assessment of posture. Labels use the shared order upright, head_down,
forward_lean, side_lean. Rejected inputs receive -1.
"""

from typing import Any, Mapping, Optional

import numpy as np


class RuleClassifier:
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        config = config or {}
        settings = config.get("baseline", {})
        self.min_keypoint_confidence = float(config.get("pose", {}).get("min_keypoint_confidence", 0.3))
        self.side_lean_angle = np.deg2rad(float(settings.get("side_lean_angle_deg", 15.0)))
        self.forward_lean_head_offset_x = float(settings.get("forward_lean_head_offset_x", 0.35))
        self.head_down_head_offset_y = float(settings.get("head_down_head_offset_y", -0.15))
        if not 0 <= self.min_keypoint_confidence <= 1:
            raise ValueError("min_keypoint_confidence must be in [0, 1].")
        if not np.isfinite(self.side_lean_angle) or not 0 < self.side_lean_angle < np.pi / 2:
            raise ValueError("side_lean_angle_deg must be between 0 and 90 degrees.")
        if not np.isfinite(self.forward_lean_head_offset_x) or self.forward_lean_head_offset_x <= 0:
            raise ValueError("forward_lean_head_offset_x must be positive.")
        if not np.isfinite(self.head_down_head_offset_y):
            raise ValueError("head_down_head_offset_y must be finite.")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Classify raw, unstandardized N x 25 feature rows.

        A visibly tilted torso takes precedence, followed by horizontal head
        displacement and a lowered nose. The fixed ordering makes compound
        poses unsuitable training/evaluation labels; exclude them at annotation.
        """
        rows = np.asarray(features, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 25:
            raise ValueError("RuleClassifier expects an N x 25 feature matrix.")
        predictions = np.full(rows.shape[0], -1, dtype=np.int64)
        # Nose, both shoulders and both hips must be visible. Ears may be hidden.
        required_confidence_columns = [2, 11, 14, 17, 20]
        valid = np.all(np.isfinite(rows), axis=1)
        valid &= np.all(rows[:, required_confidence_columns] >= self.min_keypoint_confidence, axis=1)
        for index in np.flatnonzero(valid):
            torso_angle, head_x, head_y, _ = rows[index, 21:25]
            if abs(torso_angle) >= self.side_lean_angle:
                predictions[index] = 3
            elif abs(head_x) >= self.forward_lean_head_offset_x:
                predictions[index] = 2
            elif head_y >= self.head_down_head_offset_y:
                predictions[index] = 1
            else:
                predictions[index] = 0
        return predictions
