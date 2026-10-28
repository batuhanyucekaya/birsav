"""Konumlu video okuyucu: video karelerini zaman/GPS ile eşleştirir."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from src.privacy import PrivacyProcessor

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("opencv-python gerekli: pip install -r requirements.txt") from exc


@dataclass(frozen=True)
class GpsPoint:
    timestamp: datetime
    latitude: float
    longitude: float


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("GPS/video başlangıç zamanında saat dilimi gerekli")
    return parsed


def load_gps(path: Path) -> list[GpsPoint]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        required = {"timestamp", "latitude", "longitude"}
        if not rows.fieldnames or not required.issubset(rows.fieldnames):
            raise ValueError(f"GPS CSV şu sütunları içermeli: {sorted(required)}")
        points = [
            GpsPoint(
                parse_timestamp(r["timestamp"]),
                float(r["latitude"]),
                float(r["longitude"]),
            )
            for r in rows
        ]
    if not points:
        raise ValueError("GPS CSV boş")
    points = sorted(points, key=lambda p: p.timestamp)
    if any(
        not math.isfinite(p.latitude)
        or not math.isfinite(p.longitude)
        or not -90 <= p.latitude <= 90
        or not -180 <= p.longitude <= 180
        for p in points
    ):
        raise ValueError("Geçersiz GPS koordinatı")
    if any(b.timestamp <= a.timestamp for a, b in zip(points, points[1:])):
        raise ValueError("GPS zamanları tekrarlanamaz")
    return points


def interpolate(
    points: list[GpsPoint], at: datetime
) -> tuple[float | None, float | None]:
    if at < points[0].timestamp or at > points[-1].timestamp:
        return None, None
    if at == points[0].timestamp:
        return points[0].latitude, points[0].longitude
    if at == points[-1].timestamp:
        return points[-1].latitude, points[-1].longitude
    for left, right in zip(points, points[1:]):
        if left.timestamp <= at <= right.timestamp:
            total = (right.timestamp - left.timestamp).total_seconds()
            ratio = 0.0 if total <= 0 else (at - left.timestamp).total_seconds() / total
            return (
                left.latitude + ratio * (right.latitude - left.latitude),
                left.longitude + ratio * (right.longitude - left.longitude),
            )
    return None, None


def process(
    video_path: Path,
    gps_path: Path,
    output_path: Path,
    sample_every: int,
    frames_dir: Path,
    video_start: datetime | None = None,
) -> int:
    if sample_every < 1:
        raise ValueError("--sample-every en az 1 olmalı")
    if output_path.exists():
        raise ValueError("Çıktı zaten var; yeni bir çıktı adı seçin")
    points = load_gps(gps_path)
    start = video_start or points[0].timestamp
    capture = cv2.VideoCapture(str(video_path))
    created = []
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    if partial.exists():
        capture.release()
        raise ValueError("Yarım çıktı dosyası mevcut; yeni bir çıktı adı seçin")
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Video açılamadı: {video_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS bilgisi geçersiz; zaman uydurulmayacak")
        privacy = PrivacyProcessor(Path(__file__).resolve().parent.parent)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Route-scoped names prevent silently overwriting another video's frames.
        frames_dir = frames_dir / video_path.stem
        if frames_dir.exists() and any(frames_dir.iterdir()):
            raise ValueError("Rota kare klasörü dolu; yeni --frames-dir seçin")
        frames_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        first_pts = previous_pts = None
        with partial.open("w", encoding="utf-8") as output:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                if not math.isfinite(pts) or (
                    previous_pts is not None and pts <= previous_pts
                ):
                    raise ValueError("Video zaman damgaları geçersiz")
                if first_pts is None:
                    first_pts = pts
                previous_pts = pts
                redacted, _ = privacy.process(frame)
                if count % sample_every == 0:
                    seconds = pts - first_pts
                    timestamp = start + timedelta(seconds=seconds)
                    latitude, longitude = interpolate(points, timestamp)
                    image_path = frames_dir / f"frame_{count:08d}.jpg"
                    if not cv2.imwrite(str(image_path), redacted):
                        raise RuntimeError(f"Kare yazılamadı: {image_path}")
                    created.append(image_path)
                    record = {
                        "frame_index": count,
                        "video_time_seconds": round(seconds, 3),
                        "timestamp": timestamp.isoformat(),
                        "latitude": (
                            round(latitude, 7) if latitude is not None else None
                        ),
                        "longitude": (
                            round(longitude, 7) if longitude is not None else None
                        ),
                        "location_source": (
                            "camera_gps_estimate" if latitude is not None else "unknown"
                        ),
                        "privacy_status": "review_required",
                        "image_path": image_path.as_posix(),
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        if count == 0:
            raise ValueError("Video kare içermiyor")
        partial.replace(output_path)
        return count
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
        raise
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--frames-dir", type=Path, default=Path("data/processed/frames")
    )
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--video-start", type=parse_timestamp)
    args = parser.parse_args()
    total = process(
        args.video,
        args.gps,
        args.output,
        args.sample_every,
        args.frames_dir,
        args.video_start,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "frames_read": total,
                "records": sum(1 for _ in args.output.open(encoding="utf-8")),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
