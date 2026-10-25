"""Probability smoothing and timestamp-based posture reminders.

All durations use seconds, so replaying a video at another processing speed does
not change reminder timing. The normal class is index 0 by default.
"""

import math
from typing import Any, Dict, Mapping, Optional

import numpy as np


class ReminderState:
    """Track continuous abnormal posture and emit individual reminder events.

    ``alert`` is true only on an emitted event; ``active`` remains true until
    normal posture has lasted ``recovery_seconds``. Cooldown is measured from
    the timestamp of the last emitted event, including across a normal recovery.
    A continuously abnormal posture emits again when cooldown expires. After a
    short normal interruption, the abnormal-duration threshold must be met again.
    Unknown observations clear EMA, pending/recovery timers, active state and
    cooldown history. The last timestamp is retained to reject time going back.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, Any]] = None,
        *,
        smoothed: bool = True,
        ema_alpha: Optional[float] = None,
        min_class_confidence: Optional[float] = None,
        abnormal_seconds: Optional[float] = None,
        recovery_seconds: Optional[float] = None,
        cooldown_seconds: Optional[float] = None,
        normal_label: int = 0,
    ) -> None:
        settings = dict((config or {}).get("reminder", config or {}))
        self.ema_alpha = float(settings.get("ema_alpha", 0.3) if ema_alpha is None else ema_alpha)
        self.min_class_confidence = float(
            settings.get("min_class_confidence", 0.6)
            if min_class_confidence is None else min_class_confidence
        )
        self.abnormal_seconds = float(
            settings.get("abnormal_seconds", 3.0) if abnormal_seconds is None else abnormal_seconds
        )
        self.recovery_seconds = float(
            settings.get("recovery_seconds", 1.0) if recovery_seconds is None else recovery_seconds
        )
        self.cooldown_seconds = float(
            settings.get("cooldown_seconds", 10.0) if cooldown_seconds is None else cooldown_seconds
        )
        if not math.isfinite(self.ema_alpha) or not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1].")
        if not math.isfinite(self.min_class_confidence) or not 0 <= self.min_class_confidence <= 1:
            raise ValueError("min_class_confidence must be in [0, 1].")
        for name in ("abnormal_seconds", "recovery_seconds", "cooldown_seconds"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError("{} must be a finite nonnegative number.".format(name))
        if normal_label < 0:
            raise ValueError("normal_label must be nonnegative.")
        self.smoothed = bool(smoothed)
        self.normal_label = int(normal_label)
        self._last_timestamp = None  # type: Optional[float]
        self._clear_observation()

    def _clear_observation(self) -> None:
        self._ema = None  # type: Optional[np.ndarray]
        self._abnormal_since = None  # type: Optional[float]
        self._normal_since = None  # type: Optional[float]
        self._last_alert = None  # type: Optional[float]
        self._active = False

    def reset(self) -> None:
        """Start a new independent video or camera session, including its clock."""
        self._last_timestamp = None
        self._clear_observation()

    def update(
        self, probabilities: Optional[np.ndarray], timestamp: float
    ) -> Dict[str, Any]:
        """Process one observation; timestamps must be finite and nondecreasing.

        Pass ``None`` when no reliable single-person pose is available. Invalid
        probability vectors raise ValueError; low classification confidence is
        an unknown observation and resets the reminder state.
        """
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite seconds.")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("Timestamps must be nondecreasing; reset for a new video.")

        candidate = None
        if probabilities is not None:
            candidate = np.asarray(probabilities, dtype=np.float64)
            if (candidate.ndim != 1 or candidate.size <= self.normal_label
                    or not np.all(np.isfinite(candidate))
                    or np.any(candidate < 0) or not math.isfinite(float(candidate.sum()))
                    or candidate.sum() <= 0):
                raise ValueError("probabilities must be a finite nonnegative class vector with positive sum.")
            if self._ema is not None and candidate.shape != self._ema.shape:
                raise ValueError("The number of classes must stay constant within a session.")
            candidate = candidate / candidate.sum()

        self._last_timestamp = timestamp
        if candidate is None:
            return self._unknown()
        if self.smoothed and self._ema is not None:
            candidate = self.ema_alpha * candidate + (1.0 - self.ema_alpha) * self._ema
        confidence = float(candidate.max())
        if confidence < self.min_class_confidence:
            return self._unknown()
        self._ema = candidate.copy()
        label = int(candidate.argmax())
        alert = False
        duration = 0.0

        if label == self.normal_label:
            self._abnormal_since = None
            if self._normal_since is None:
                self._normal_since = timestamp
            if self._active and timestamp - self._normal_since >= self.recovery_seconds:
                self._active = False
            status = "recovering" if self._active else "normal"
        else:
            self._normal_since = None
            if self._abnormal_since is None:
                self._abnormal_since = timestamp
            duration = timestamp - self._abnormal_since
            if duration >= self.abnormal_seconds:
                self._active = True
                if self._last_alert is None or timestamp - self._last_alert >= self.cooldown_seconds:
                    alert = True
                    self._last_alert = timestamp
            status = "alert" if alert else ("active" if self._active else "pending")

        return {
            "label": label, "confidence": confidence, "alert": alert,
            "active": self._active, "status": status, "duration": float(duration),
        }

    def _unknown(self) -> Dict[str, Any]:
        self._clear_observation()
        return {
            "label": -1, "confidence": 0.0, "alert": False,
            "active": False, "status": "unknown", "duration": 0.0,
        }
