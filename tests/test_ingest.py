"""CLI extraction integration with synthetic video; privacy detector is explicitly mocked."""

from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from src.ingest import process, parse_timestamp, load_gps


class IngestTests(unittest.TestCase):
    def create_inputs(self, root):
        video = root / "route.avi"
        gps = root / "route.csv"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48)
        )
        for _ in range(3):
            writer.write(np.full((48, 64, 3), 180, np.uint8))
        writer.release()
        gps.write_text(
            "timestamp,latitude,longitude\n2026-10-27T08:00:00+03:00,41,29\n2026-10-27T08:00:00.2+03:00,41.001,29.001\n"
        )
        return video, gps

    def test_route_scoping_redacted_frames_and_unknown_gps(self):
        class Privacy:
            def process(self, frame):
                output = frame.copy()
                output[:24, :32] = 0
                return output, []

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video, gps = self.create_inputs(root)
            with patch("src.ingest.PrivacyProcessor", return_value=Privacy()):
                count = process(video, gps, root / "out.jsonl", 1, root / "frames")
            self.assertEqual(count, 3)
            rows = [
                json.loads(x) for x in (root / "out.jsonl").read_text().splitlines()
            ]
            self.assertIsNone(rows[-1]["latitude"])
            self.assertEqual(rows[-1]["privacy_status"], "review_required")
            frame = cv2.imread(rows[0]["image_path"])
            self.assertLess(frame[:20, :28].mean(), 5)
            self.assertFalse(list(root.glob("*.partial")))
            with self.assertRaises(ValueError):
                process(video, gps, root / "out.jsonl", 1, root / "frames")

    def test_failure_removes_partial_files(self):
        class Privacy:
            def __init__(self):
                self.calls = 0

            def process(self, frame):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("detector failed")
                return frame, []

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video, gps = self.create_inputs(root)
            with patch("src.ingest.PrivacyProcessor", return_value=Privacy()):
                with self.assertRaises(RuntimeError):
                    process(video, gps, root / "out.jsonl", 1, root / "frames")
            self.assertFalse((root / "out.jsonl").exists())
            self.assertFalse(list(root.rglob("*.partial")))
            self.assertFalse(list((root / "frames").rglob("*.jpg")))
            video.unlink()  # On Windows this also checks capture release.

    def test_timezone_required(self):
        with self.assertRaises(ValueError):
            parse_timestamp("2026-10-27T08:00:00")


if __name__ == "__main__":
    unittest.main()
