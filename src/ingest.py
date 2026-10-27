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
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def load_gps(path: Path) -> list[GpsPoint]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        required = {"timestamp", "latitude", "longitude"}
        if not rows.fieldnames or not required.issubset(rows.fieldnames):
            raise ValueError(f"GPS CSV şu sütunları içermeli: {sorted(required)}")
        points = [GpsPoint(parse_timestamp(r["timestamp"]), float(r["latitude"]), float(r["longitude"])) for r in rows]
    if not points:
        raise ValueError("GPS CSV boş")
    return sorted(points, key=lambda p: p.timestamp)


def interpolate(points: list[GpsPoint], at: datetime) -> tuple[float, float]:
    if at <= points[0].timestamp:
        return points[0].latitude, points[0].longitude
    if at >= points[-1].timestamp:
        return points[-1].latitude, points[-1].longitude
    for left, right in zip(points, points[1:]):
        if left.timestamp <= at <= right.timestamp:
            total = (right.timestamp - left.timestamp).total_seconds()
            ratio = 0.0 if total <= 0 else (at - left.timestamp).total_seconds() / total
            return (left.latitude + ratio * (right.latitude - left.latitude),
                    left.longitude + ratio * (right.longitude - left.longitude))
    return points[-1].latitude, points[-1].longitude


def process(video_path: Path, gps_path: Path, output_path: Path, sample_every: int, frames_dir: Path, video_start: datetime | None = None) -> int:
    if sample_every < 1:
        raise ValueError("--sample-every en az 1 olmalı")
    points = load_gps(gps_path)
    start = video_start or points[0].timestamp
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video açılamadı: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if count % sample_every == 0:
                seconds = count / fps
                timestamp = start + timedelta(seconds=seconds)
                latitude, longitude = interpolate(points, timestamp)
                image_name = f"frame_{count:08d}.jpg"
                image_path = frames_dir / image_name
                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError(f"Kare yazılamadı: {image_path}")
                record = {
                    "frame_index": count,
                    "video_time_seconds": round(seconds, 3),
                    "timestamp": timestamp.isoformat(),
                    "latitude": round(latitude, 7),
                    "longitude": round(longitude, 7),
                    "image_path": str(image_path.as_posix()),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                continue
            count += 1
    capture.release()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, default=Path("data/processed/frames"))
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--video-start", type=parse_timestamp)
    args = parser.parse_args()
    total = process(args.video, args.gps, args.output, args.sample_every, args.frames_dir, args.video_start)
    print(json.dumps({"status": "ok", "frames_read": total, "records": sum(1 for _ in args.output.open(encoding="utf-8"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()

