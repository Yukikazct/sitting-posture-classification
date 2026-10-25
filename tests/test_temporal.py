"""Timing behavior is independent of FPS and wall-clock processing speed."""

import unittest

import numpy as np

from posture.temporal import ReminderState


class ReminderStateTests(unittest.TestCase):
    def setUp(self):
        self.normal = np.array([0.95, 0.02, 0.02, 0.01])
        self.bad = np.array([0.02, 0.95, 0.02, 0.01])

    def test_seconds_threshold_cooldown_and_recovery(self):
        state = ReminderState(smoothed=False)
        self.assertFalse(state.update(self.bad, 0.0)["alert"])
        self.assertFalse(state.update(self.bad, 2.99)["active"])
        self.assertTrue(state.update(self.bad, 3.0)["alert"])
        self.assertFalse(state.update(self.bad, 12.99)["alert"])
        self.assertTrue(state.update(self.bad, 13.0)["alert"])
        self.assertTrue(state.update(self.normal, 14.0)["active"])
        self.assertTrue(state.update(self.normal, 14.99)["active"])
        self.assertFalse(state.update(self.normal, 15.0)["active"])

    def test_unknown_clears_all_reminder_state(self):
        state = ReminderState(smoothed=False)
        state.update(self.bad, 0.0)
        self.assertTrue(state.update(self.bad, 3.0)["alert"])
        result = state.update(None, 3.1)
        self.assertEqual(result["label"], -1)
        self.assertFalse(result["active"])
        state.update(self.bad, 3.2)
        self.assertFalse(state.update(self.bad, 6.19)["alert"])
        self.assertTrue(state.update(self.bad, 6.2)["alert"])

    def test_low_confidence_clears_ema_and_pending_timer(self):
        state = ReminderState(ema_alpha=1.0)
        state.update(self.bad, 0.0)
        self.assertEqual(state.update(np.full(4, 0.25), 2.0)["label"], -1)
        state.update(self.normal, 2.1)
        self.assertEqual(state.update(self.normal, 2.2)["label"], 0)

    def test_short_normal_interruption_does_not_recover(self):
        state = ReminderState(smoothed=False)
        state.update(self.bad, 0.0)
        state.update(self.bad, 3.0)
        state.update(self.normal, 4.0)
        result = state.update(self.bad, 4.5)
        self.assertTrue(result["active"])
        self.assertEqual(result["duration"], 0.0)
        self.assertFalse(result["alert"])

    def test_ema_applies_to_probabilities(self):
        state = ReminderState(ema_alpha=0.3, min_class_confidence=0.0)
        state.update(self.normal, 0.0)
        result = state.update(self.bad, 0.5)
        self.assertEqual(result["label"], 0)
        self.assertAlmostEqual(result["confidence"], 0.7 * 0.95 + 0.3 * 0.02)

    def test_invalid_times_and_probability_vectors(self):
        state = ReminderState()
        state.update(self.bad, 2.0)
        with self.assertRaises(ValueError):
            state.update(self.bad, 1.0)
        with self.assertRaises(ValueError):
            state.update(self.bad, float("nan"))
        with self.assertRaises(ValueError):
            state.update(np.array([-1, 2, 0, 0]), 3.0)
        state.reset()
        self.assertFalse(state.update(self.bad, 0.0)["alert"])


if __name__ == "__main__":
    unittest.main()
