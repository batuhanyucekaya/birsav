"""Local video processing, cleanup and review-only outputs."""

import json
import math
from pathlib import Path
import time
import cv2

from src.gps import gps_at
from src.inference import MODEL_LOCK, analyze_frame, annotate
from src.model_registry import version
from src.privacy import PrivacyProcessor
from src.tracking import BoxTracker, EventAccumulator
from src.video_validation import validate_video
from src.video_timing import CfrResampler

ROOT = Path(__file__).resolve().parent.parent


def nearby_limit(gps):
    if not gps:
        return None
    path = ROOT / "data" / "reference_limits.json"
    locations = (
        json.loads(path.read_text(encoding="utf-8")).get("locations", [])
        if path.is_file()
        else []
    )
    found = []
    for item in locations:
        try:
            dy = (float(item["latitude"]) - gps["latitude"]) * 111000
            dx = (
                (float(item["longitude"]) - gps["longitude"])
                * 111000
                * math.cos(math.radians(gps["latitude"]))
            )
            distance = math.hypot(dx, dy)
            if distance <= float(item.get("radius_m", 25)):
                found.append({**item, "distance_m": distance})
        except (KeyError, TypeError, ValueError):
            continue
    return min(found, key=lambda x: x["distance_m"]) if found else None


class Cancelled(Exception):
    pass


def process_video(
    raw_path, output_path, gps_rows=(), cancel=None, progress=None, sample_fps=3
):
    start = time.perf_counter()
    capture = writer = None
    succeeded = False
    try:
        if not 0 < sample_fps <= 30:
            raise ValueError("Örnekleme hızı 0..30 aralığında olmalı")
        capture = cv2.VideoCapture(str(raw_path))
        if not capture.isOpened():
            raise ValueError("Video okunamadı")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or not 0 < fps <= 240:
            raise ValueError("Video FPS bilgisi geçersiz; zaman/GPS uydurulmayacak")
        width, height = int(capture.get(3)), int(capture.get(4))
        if not width or not height or width * height > 3840 * 2160:
            raise ValueError("Video çözünürlüğü geçersiz veya yerel sınırı aşıyor")
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("WebM kodlayıcı açılamadı")

        def check_cancel():
            if cancel is not None and cancel.is_set():
                raise Cancelled()

        resampler = CfrResampler(writer, fps, check_cancel)
        privacy = PrivacyProcessor(ROOT)
        trackers = {
            k: BoxTracker()
            for k in (
                "road_damage",
                "litter_candidate",
                "scene",
                "lane_marking",
                "container",
            )
        }
        versions = {
            "road_damage": version("road"),
            "litter_candidate": version("litter"),
            "scene": version("scene"),
            "lane_marking": version("lane"),
            "container": version("container"),
        }
        accumulator = EventAccumulator(versions)
        frame_idx, next_sample = 0, 0.0
        last_seconds = -1.0
        first_pts = None
        counts = {
            "privacy_frames": 0,
            "faces_blurred": 0,
            "plates_blurred": 0,
            "road_damage": 0,
            "ground_litter": 0,
            "scene": 0,
            "lane_markings": 0,
            "containers": 0,
        }
        result = {}
        inference_seconds = 0.0
        while True:
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            ok, frame = capture.read()
            if not ok:
                break
            pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            # Reject non-monotonic timestamps instead of silently misaligning GPS.
            if not math.isfinite(pts) or pts <= last_seconds:
                raise ValueError(
                    "Video zaman damgaları artmıyor; PTS incelemesi gerekli"
                )
            if first_pts is None:
                first_pts = pts
            seconds = pts - first_pts
            last_seconds = pts
            gps = gps_at(gps_rows, seconds)
            with MODEL_LOCK:
                redacted, privacy_tracks = privacy.process(frame)
                if seconds + 1e-6 >= next_sample:
                    tick = time.perf_counter()
                    # Analyze the original in memory; redaction must not erase issue evidence.
                    result = analyze_frame(
                        frame, include_lane=True, road_conf=0.25, litter_conf=0.35
                    )
                    inference_seconds += time.perf_counter() - tick
                    next_sample = (math.floor(seconds * sample_fps) + 1) / sample_fps
                    for family, key in (
                        ("road_damage", "road_damage"),
                        ("litter_candidate", "ground_litter"),
                        ("scene", "scene_segments"),
                        ("lane_marking", "lane_markings"),
                        ("container", "containers"),
                    ):
                        tracks = trackers[family].update(result.get(key, []))
                        result[key] = [x for x in tracks if x["missing"] == 0]
                        for track in tracks:
                            event = accumulator.observe(family, track, seconds, gps)
                            if event and family == "litter_candidate":
                                event["context"] = track.get("context")
                            if (
                                event
                                and family == "scene"
                                and track.get("segmentation_polygons")
                            ):
                                event.update(
                                    segmentation_polygons=track[
                                        "segmentation_polygons"
                                    ],
                                    merged_detection_count=track.get(
                                        "merged_detection_count", 1
                                    ),
                                    occlusion_bridge_inferred=track.get(
                                        "occlusion_bridge_inferred", False
                                    ),
                                )
                            if event and family == "container":
                                event.update(
                                    state_confidence=track.get("state_confidence"),
                                    experimental=track.get("experimental", False),
                                    localization_validated=track.get(
                                        "localization_validated", False
                                    ),
                                )
                            if (
                                event
                                and family == "scene"
                                and track["class_name"] in {"sign", "billboard"}
                            ):
                                x1, y1, x2, y2 = track["box"]
                                event.update(
                                    width_px=x2 - x1,
                                    height_px=y2 - y1,
                                    measurement_unit="image_pixels",
                                )
                                limit = nearby_limit(gps)
                                if (
                                    limit
                                    and "max_width_px" in limit
                                    and "max_height_px" in limit
                                ):
                                    event.update(
                                        limit_status=(
                                            "pixel_reference_exceeded"
                                            if x2 - x1
                                            > float(limit["max_width_px"]) * 1.05
                                            or y2 - y1
                                            > float(limit["max_height_px"]) * 1.05
                                            else "within_pixel_reference"
                                        ),
                                        limit_name=limit.get("name"),
                                        distance_m=limit["distance_m"],
                                        tolerance=0.05,
                                    )
            # Draw then re-apply opaque masks so annotations never replace privacy pixels.
            annotated = annotate(redacted, result)
            from src.privacy import mask_regions

            annotated = mask_regions(annotated, privacy_tracks)
            resampler.push(annotated, seconds)
            counts["privacy_frames"] += bool(privacy_tracks)
            frame_idx += 1
            if progress:
                progress(frame_idx, expected)
        if expected > 0 and frame_idx != expected:
            raise ValueError(f"Giriş videosu eksik çözüldü: {frame_idx}/{expected}")
        timing_conversion = resampler.finish()
        capture.release()
        capture = None
        writer.release()
        writer = None
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        validation = validate_video(
            output_path, resampler.written, (width, height), fps
        )
        events = list(accumulator.events.values())
        counts.update(
            faces_blurred=len(privacy.ids["face"]),
            plates_blurred=len(privacy.ids["plate"]),
            road_damage=sum(e["event_type"] == "road_damage" for e in events),
            ground_litter=sum(e["event_type"] == "litter_candidate" for e in events),
            scene=sum(e["event_type"] == "scene" for e in events),
            lane_markings=sum(e["event_type"] == "lane_marking" for e in events),
            containers=sum(e["event_type"] == "container" for e in events),
        )
        response = {
            "frames": frame_idx,
            "output_frames": resampler.written,
            "timing_conversion": timing_conversion,
            "fps": fps,
            "sample_fps": sample_fps,
            "counts": counts,
            "events": events,
            "gps_enabled": bool(gps_rows),
            "reference_limits_loaded": bool(
                json.loads(
                    (ROOT / "data/reference_limits.json").read_text(encoding="utf-8")
                ).get("locations", [])
            ),
            "privacy_status": "review_required",
            "review_required": True,
            "output_validation": validation,
            "privacy_note": "Tespit edilen yüz/plakalar opak maskelendi. Dedektör başarımı doğrulanmadı; yerel gizlilik incelemesi gerekli. Ses aktarılmaz.",
            "count_note": "Sayılar takip adaylarıdır; aynı fiziksel olayın yeniden ID alması hâlâ mümkün.",
            "timing": {
                "total_seconds": time.perf_counter() - start,
                "analysis_seconds": inference_seconds,
            },
            "timestamp_source": "opencv_pos_msec",
            "gps_alignment": (
                "first_absolute_gps_row_is_video_zero" if gps_rows else None
            ),
        }
        succeeded = True
        return response
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        Path(raw_path).unlink(missing_ok=True)
        if not succeeded:
            Path(output_path).unlink(missing_ok=True)
