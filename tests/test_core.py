"""Deterministic logic checks. These are not model accuracy benchmarks."""

import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.audit_project import validate_label, audit_dataset
from src.gps import parse_gps, gps_at
from src.http_upload import parse_form, UploadError
from src.privacy import mask_regions
from src.tracking import BoxTracker, EventAccumulator
from src.video_pipeline import process_video, Cancelled
from src.video_validation import validate_video


def detection(x=10, kind="road_damage", cls="pothole"):
    return {
        "kind": kind,
        "class_name": cls,
        "confidence": 0.8,
        "box": [x, 10, x + 20, 30],
    }


class TrackingTests(unittest.TestCase):
    def test_ids_do_not_repeat_after_all_tracks_expire(self):
        t = BoxTracker(max_missing=0)
        self.assertEqual(t.update([detection()])[0]["track_id"], 1)
        t.update([])
        self.assertEqual(t.update([detection()])[0]["track_id"], 2)

    def test_privacy_types_never_merge(self):
        t = BoxTracker()
        t.update([detection(kind="face", cls="face")])
        tracks = t.update([detection(kind="plate", cls="plate")])
        self.assertEqual(len(tracks), 2)
        self.assertEqual({x["track_id"] for x in tracks}, {1, 2})

    def test_two_objects_one_to_one(self):
        t = BoxTracker()
        tracks = t.update([detection(10), detection(50)])
        tracks = t.update([detection(11), detection(51)])
        self.assertEqual([x["track_id"] for x in tracks], [1, 2])

    def test_events_merge_and_do_not_increment_for_stale_track(self):
        t = BoxTracker()
        events = EventAccumulator()
        for seconds in range(20):
            for track in t.update([detection()]):
                events.observe("road_damage", track, seconds)
        self.assertEqual(len(events.events), 1)
        event = next(iter(events.events.values()))
        self.assertEqual(event["observations"], 20)
        self.assertIsNone(event["latitude"])
        for track in t.update([]):
            events.observe("road_damage", track, 21)
        self.assertEqual(event["end_time"], 19)


class PrivacyTests(unittest.TestCase):
    def test_large_and_crowded_regions_are_not_discarded(self):
        frame = np.full((100, 100, 3), 255, np.uint8)
        regions = [{"box": [0, 0, 60, 100]}, {"box": [60, 0, 100, 100]}]
        self.assertFalse(mask_regions(frame, regions).any())

    def test_current_box_covered_despite_smoothed_lag(self):
        frame = np.full((100, 100, 3), 255, np.uint8)
        result = mask_regions(
            frame, [{"box": [0, 0, 20, 20], "observed_box": [50, 50, 70, 70]}]
        )
        self.assertFalse(result[50:70, 50:70].any())

    def test_nonfinite_box_fails_closed(self):
        self.assertFalse(
            mask_regions(
                np.ones((10, 10, 3), np.uint8), [{"box": [0, 0, float("nan"), 1]}]
            ).any()
        )


class GpsTests(unittest.TestCase):
    def test_sorted_absolute_origin_and_interpolation(self):
        rows = parse_gps(
            io.BytesIO(
                b"timestamp,latitude,longitude\n2026-10-27T08:00:10+03:00,42,30\n2026-10-27T08:00:00+03:00,40,28\n"
            )
        )
        self.assertEqual(gps_at(rows, 5), {"latitude": 41, "longitude": 29})
        self.assertIsNone(gps_at(rows, -1))
        self.assertIsNone(gps_at(rows, 11))

    def test_bad_coordinates_and_duplicate_times_rejected(self):
        for data in (
            b"video_time_seconds,latitude,longitude\n0,nan,28\n",
            b"video_time_seconds,latitude,longitude\n0,40,28\n0,41,29\n",
        ):
            with self.assertRaises(ValueError):
                parse_gps(io.BytesIO(data))


class UploadTests(unittest.TestCase):
    def parse(self, body, limit=1024):
        return parse_form(
            io.BytesIO(body),
            {
                "Content-Type": "multipart/form-data; boundary=x",
                "Content-Length": str(len(body)),
            },
            limit,
        )

    def test_binary_upload(self):
        form = self.parse(
            b'--x\r\nContent-Disposition: form-data; name="video"; filename="x.mp4"\r\n\r\n\x00\xff\r\n--x--\r\n'
        )
        try:
            self.assertEqual(form["video"].file.read(), b"\x00\xff")
        finally:
            form.close()

    def test_reject_before_reading_oversized_body(self):
        stream = io.BytesIO(b"not read")
        with self.assertRaises(UploadError) as cm:
            parse_form(stream, {"Content-Length": "10000"}, 10)
        self.assertEqual(cm.exception.status, 413)
        self.assertEqual(stream.tell(), 0)

    def test_truncated_multipart_and_duplicate_fields(self):
        for body in (
            b'--x\r\nContent-Disposition: form-data; name="video"\r\n\r\ndata',
            b'--x\r\nContent-Disposition: form-data; name="video"\r\n\r\na\r\n--x\r\nContent-Disposition: form-data; name="video"\r\n\r\nb\r\n--x--\r\n',
        ):
            with self.assertRaises(UploadError):
                self.parse(body)


class DatasetTests(unittest.TestCase):
    def test_rdd_converter_keeps_true_negative_scenes(self):
        from src.prepare_rdd import convert

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            (source / "annotations").mkdir(parents=True)
            (source / "images").mkdir()
            for name, objects in [
                ("clean", ""),
                ("unknown", "<object><name>unsupported</name></object>"),
            ]:
                cv2.imwrite(
                    str(source / "images" / f"{name}.jpg"),
                    np.zeros((32, 32, 3), np.uint8),
                )
                (source / "annotations" / f"{name}.xml").write_text(
                    f"<annotation><filename>{name}.jpg</filename><size><width>32</width><height>32</height></size>{objects}</annotation>"
                )
            self.assertEqual(convert(source, root / "ready", 0), (1, 0))
            self.assertEqual(
                (root / "ready/labels/train/clean.txt").read_text().strip(), ""
            )
            self.assertFalse((root / "ready/images/train/unknown.jpg").exists())

    def test_threshold_selection_forbids_test_data_and_counts_missed_truth(self):
        from src.evaluation_protocol import tune_threshold

        with self.assertRaises(ValueError):
            tune_threshold([0.8], [1], split="test")
        best, _ = tune_threshold([0.8, 0.3], [1, 0], missed_positives=1)
        self.assertEqual(best["recall"], 0.5)

    def test_promotion_requires_independent_provenance(self):
        from src.evaluation_protocol import promotion_decision

        protocol = {
            "split": "val",
            "imgsz": 320,
            "task": "detect",
            "data_fingerprint": "same",
            "purpose": "development_validation_not_field_test",
        }
        result = promotion_decision(protocol, protocol, "mAP", "same")
        self.assertFalse(result["promote"])

    def test_duplicate_polygons_flagged(self):
        text = "0 0 0 1 0 1 1\n0 0 0 1 0 1 1"
        findings, _ = validate_label(text, 1, "segment")
        self.assertEqual(findings[0]["code"], "duplicate_annotation")

    def test_shared_mapillary_color_emits_one_polygon(self):
        from src.prepare_mapillary_seg import semantic_polygons

        mask = np.zeros((40, 40, 3), np.uint8)
        mask[5:30, 5:30] = 255
        rows = semantic_polygons(
            mask, {35: 2, 36: 2}, {35: (255, 255, 255), 36: (255, 255, 255)}
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(validate_label(rows[0], 6, "segment")[0], [])

    def test_bad_ids_bounds_nan_polygon_and_full_frame(self):
        for label in (
            "3 .5 .5 .2 .2",
            "0 nan .5 .2 .2",
            "0 .9 .5 .5 .5",
            "0 .5 .5 0 .5",
        ):
            self.assertEqual(validate_label(label, 3)[0][0]["severity"], "error")
        self.assertEqual(
            validate_label("0 .5 .5 1 1", 3)[0][0]["code"], "full_image_bbox"
        )
        self.assertTrue(validate_label("0 0 0 1 1 0 0", 1, "segment")[0])

    def test_decoded_duplicate_detection_across_splits(self):
        import yaml

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ("train", "val"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
                cv2.imwrite(
                    str(root / "images" / split / "x.png"),
                    np.zeros((20, 20, 3), np.uint8),
                )
                (root / "labels" / split / "x.txt").write_text("")
            cfg = root / "data.yaml"
            cfg.write_text(
                yaml.safe_dump(
                    {
                        "path": str(root),
                        "train": "images/train",
                        "val": "images/val",
                        "names": ["x"],
                    }
                )
            )
            result = audit_dataset(cfg)
            self.assertEqual(len(result["exact_cross_split_duplicates"]), 1)
            self.assertFalse(result["splits"]["test"]["configured"])


class VideoTests(unittest.TestCase):
    def test_error_after_validation_still_removes_output(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "input.avi"
            out = root / "output.webm"
            (root / "data").mkdir()
            (root / "data/reference_limits.json").write_text("{")
            writer = cv2.VideoWriter(
                str(raw), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48)
            )
            for _ in range(2):
                writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            privacy = SimpleNamespace(
                process=lambda frame: (frame, []), ids={"face": set(), "plate": set()}
            )
            prediction = {"road_damage": [], "ground_litter": [], "scene_segments": []}
            with patch("src.video_pipeline.ROOT", root), patch(
                "src.video_pipeline.PrivacyProcessor", return_value=privacy
            ), patch("src.video_pipeline.analyze_frame", return_value=prediction):
                with self.assertRaises(json.JSONDecodeError):
                    process_video(raw, out)
            self.assertFalse(raw.exists())
            self.assertFalse(out.exists())

    def test_decoder_validates_all_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.webm"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"VP80"), 5, (64, 48)
            )
            for i in range(3):
                writer.write(np.full((48, 64, 3), i * 70, np.uint8))
            writer.release()
            self.assertEqual(validate_video(path, 3, (64, 48), 5)["frames"], 3)
            with self.assertRaises(ValueError):
                validate_video(path, 4, (64, 48), 5)

    def test_missing_model_cleanup_before_loop(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder) / "input.avi"
            out = Path(folder) / "output.webm"
            writer = cv2.VideoWriter(
                str(raw), cv2.VideoWriter_fourcc(*"MJPG"), 5, (64, 48)
            )
            writer.write(np.zeros((48, 64, 3), np.uint8))
            writer.release()
            with patch(
                "src.video_pipeline.PrivacyProcessor",
                side_effect=RuntimeError("missing model"),
            ):
                with self.assertRaises(RuntimeError):
                    process_video(raw, out)
            self.assertFalse(raw.exists())
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
