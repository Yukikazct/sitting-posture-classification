"""Shared defaults. Relative command-line paths are relative to the working directory."""

import copy
import json
from pathlib import Path


DEFAULT_CONFIG = {
    "class_names": ["upright", "head_down", "forward_lean", "side_lean"],
    "data": {"split_ratios": [0.7, 0.15, 0.15]},
    "pose": {
        "weights": "models/yolo11n-pose.pt",
        "imgsz": 640,
        "min_keypoint_confidence": 0.3,
        "person_confidence": 0.25,
    },
    "training": {
        "learning_rate": 0.001,
        "batch_size": 64,
        "max_epochs": 100,
        "patience": 10,
        "hidden_dims": [64, 32],
        "dropout": 0.2,
    },
    "baseline": {
        "side_lean_angle_deg": 15.0,
        "forward_lean_head_offset_x": 0.35,
        "head_down_head_offset_y": -0.15,
    },
    "reminder": {
        "ema_alpha": 0.3,
        "min_class_confidence": 0.6,
        "abnormal_seconds": 3.0,
        "recovery_seconds": 1.0,
        "cooldown_seconds": 10.0,
    },
}


def _merge(target, source):
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


def load_config(path=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with Path(path).open(encoding="utf-8") as handle:
            supplied = json.load(handle)
        if not isinstance(supplied, dict):
            raise ValueError("config must contain a JSON object")
        _merge(config, supplied)
    if config["class_names"] != DEFAULT_CONFIG["class_names"]:
        raise ValueError("class_names and their order must match the four project labels")
    return config
