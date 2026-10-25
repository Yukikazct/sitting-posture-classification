"""View-dependent 2D features from COCO's 17-keypoint pose convention.

Coordinates use the image convention: x points right, y points down. Features
are invariant to translation and uniform image scaling, not camera viewpoint.
"""

from typing import Optional

import numpy as np

SELECTED_KEYPOINTS = (0, 3, 4, 5, 6, 11, 12)
KEYPOINT_NAMES = (
    "nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
)
FEATURE_NAMES = [
    "{}_{}".format(name, component)
    for name in KEYPOINT_NAMES
    for component in ("x", "y", "confidence")
] + ["torso_angle", "head_offset_x", "head_offset_y", "shoulder_angle"]
FEATURE_DIM = len(FEATURE_NAMES)


def extract_features(
    keypoints: np.ndarray, min_confidence: float = 0.3
) -> Optional[np.ndarray]:
    """Return 25 float32 features, or None when the pose is not usable.

    Nose, both shoulders, and both hips must be visible. Low-confidence ears
    are represented by three zeros. Torso angle is signed radians from the
    upward image axis; shoulder angle is signed radians from image horizontal
    after orienting the shoulder segment from screen-left to screen-right.
    """
    if not np.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    points = np.asarray(keypoints, dtype=np.float64)
    if points.shape != (17, 3) or not np.isfinite(points).all():
        return None
    if np.any((points[:, 2] < 0.0) | (points[:, 2] > 1.0)):
        return None
    required = [0, 5, 6, 11, 12]
    if np.any(points[required, 2] < min_confidence):
        return None

    shoulders = points[[5, 6], :2].mean(axis=0)
    hips = points[[11, 12], :2].mean(axis=0)
    torso = shoulders - hips
    scale = float(np.linalg.norm(torso))
    if scale <= 1e-6:
        return None

    selected = points[list(SELECTED_KEYPOINTS)].copy()
    selected[:, :2] = (selected[:, :2] - hips) / scale
    selected[selected[:, 2] < min_confidence] = 0.0
    head_offset = (points[0, :2] - shoulders) / scale
    shoulder_delta = points[6, :2] - points[5, :2]
    if shoulder_delta[0] < 0.0:
        shoulder_delta = -shoulder_delta
    geometry = np.array([
        np.arctan2(torso[0], -torso[1]),
        head_offset[0],
        head_offset[1],
        np.arctan2(shoulder_delta[1], shoulder_delta[0]),
    ])
    features = np.concatenate((selected.reshape(-1), geometry)).astype(np.float32)
    return features if np.isfinite(features).all() else None
