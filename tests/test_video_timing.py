"""Timing regression checks; no model inference is mocked as accuracy evidence."""

import unittest
from src.video_timing import CfrResampler


class Writer:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame)


class TimingTests(unittest.TestCase):
    def test_constant_rate_preserves_frame_order_and_count(self):
        w = Writer()
        r = CfrResampler(w, 5)
        for i, t in enumerate([0, 0.2, 0.4, 0.6, 0.8]):
            r.push(i, t)
        result = r.finish()
        self.assertEqual(w.frames, [0, 1, 2, 3, 4])
        self.assertFalse(result["variable_intervals_detected"])

    def test_variable_rate_preserves_timeline_instead_of_speeding_up(self):
        w = Writer()
        r = CfrResampler(w, 10)
        for frame, t in [("A", 0), ("B", 0.1), ("C", 0.4)]:
            r.push(frame, t)
        result = r.finish()
        self.assertEqual(w.frames, ["A", "B", "B", "B", "C", "C"])
        self.assertTrue(result["variable_intervals_detected"])
        self.assertAlmostEqual(result["output_duration_seconds"], 0.6)
        self.assertEqual(result["duplicated_frames"], 3)

    def test_short_frames_drop_without_reordering(self):
        w = Writer()
        r = CfrResampler(w, 10)
        for frame, t in [("A", 0), ("B", 0.02), ("C", 0.04), ("D", 0.1)]:
            r.push(frame, t)
        result = r.finish()
        self.assertEqual(w.frames, ["A", "D"])
        self.assertEqual(result["dropped_frames"], 2)
        self.assertLess(abs(result["duration_rounding_error_seconds"]), 0.1)

    def test_duplicate_or_backwards_timestamp_is_still_rejected(self):
        r = CfrResampler(Writer(), 10)
        r.push("A", 0)
        with self.assertRaises(ValueError):
            r.push("B", 0)

    def test_cancellation_remains_responsive_during_long_gap(self):
        def cancel():
            raise RuntimeError("cancelled")

        r = CfrResampler(Writer(), 30, cancel)
        r.push("masked", 0)
        with self.assertRaises(RuntimeError):
            r.push("next", 10)


if __name__ == "__main__":
    unittest.main()
