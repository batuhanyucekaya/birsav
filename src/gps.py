"""Strict, shared GPS parsing; do not fabricate location outside recorded coverage."""

import csv
from datetime import datetime, timezone
import io
import math


def parse_gps(stream):
    text = stream.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    fields = set(reader.fieldnames or [])
    if not {"latitude", "longitude"}.issubset(fields) or not {
        "timestamp",
        "video_time_seconds",
    }.intersection(fields):
        raise ValueError("GPS CSV sütunları eksik")
    result = []
    relative = "video_time_seconds" in fields
    for row in reader:
        lat, lon = float(row["latitude"]), float(row["longitude"])
        if relative:
            sec = float(row["video_time_seconds"])
        else:
            dt = datetime.fromisoformat(row["timestamp"].strip().replace("Z", "+00:00"))
            if dt.tzinfo is None:
                raise ValueError("Mutlak GPS zamanında saat dilimi gerekli")
            sec = dt.timestamp()
        if (
            not all(math.isfinite(v) for v in (sec, lat, lon))
            or not -90 <= lat <= 90
            or not -180 <= lon <= 180
        ):
            raise ValueError("Geçersiz GPS koordinatı/zamanı")
        result.append({"seconds": sec, "latitude": lat, "longitude": lon})
    if not result:
        raise ValueError("GPS CSV boş")
    result.sort(key=lambda x: x["seconds"])
    if not relative:
        start = result[0]["seconds"]
        for row in result:
            row["seconds"] -= start
    if any(b["seconds"] <= a["seconds"] for a, b in zip(result, result[1:])):
        raise ValueError("GPS zamanları tekrarlanamaz")
    return result


def gps_at(rows, seconds):
    if not rows or seconds < rows[0]["seconds"] or seconds > rows[-1]["seconds"]:
        return None
    if len(rows) == 1:
        return {k: rows[0][k] for k in ("latitude", "longitude")}
    for a, b in zip(rows, rows[1:]):
        if a["seconds"] <= seconds <= b["seconds"]:
            r = (seconds - a["seconds"]) / (b["seconds"] - a["seconds"])
            return {k: a[k] + r * (b[k] - a[k]) for k in ("latitude", "longitude")}
    return None
