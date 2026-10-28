"""Small deterministic tracker. IDs never repeat within a video; no re-ID claim."""

from __future__ import annotations


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection / max(
        1e-9,
        max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        + max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        - intersection,
    )


class BoxTracker:
    def __init__(self, match_iou=0.25, alpha=0.35, max_missing=3):
        self.match_iou, self.alpha, self.max_missing = match_iou, alpha, max_missing
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        pairs = []
        for oi, old in enumerate(self.tracks):
            for ni, new in enumerate(detections):
                if (old.get("kind"), old.get("class_name")) != (
                    new.get("kind"),
                    new.get("class_name"),
                ):
                    continue
                score = iou(old["box"], new["box"])
                if score >= self.match_iou:
                    pairs.append((score, oi, ni))
        matches, used_new = {}, set()
        for score, oi, ni in sorted(pairs, reverse=True):
            if oi not in matches and ni not in used_new:
                matches[oi] = ni
                used_new.add(ni)
        updated = []
        for oi, old in enumerate(self.tracks):
            if oi in matches:
                new = detections[matches[oi]]
                box = [
                    (1 - self.alpha) * old["box"][j] + self.alpha * new["box"][j]
                    for j in range(4)
                ]
                updated.append(
                    {
                        **new,
                        "box": box,
                        "observed_box": list(new["box"]),
                        "track_id": old["track_id"],
                        "hits": old["hits"] + 1,
                        "missing": 0,
                    }
                )
            elif old["missing"] < self.max_missing:
                updated.append({**old, "missing": old["missing"] + 1})
        for ni, new in enumerate(detections):
            if ni not in used_new:
                updated.append(
                    {
                        **new,
                        "observed_box": list(new["box"]),
                        "track_id": self.next_id,
                        "hits": 1,
                        "missing": 0,
                    }
                )
                self.next_id += 1
        self.tracks = updated
        return updated


class EventAccumulator:
    """Merge only continuing track observations, never GPS-only nearby objects."""

    def __init__(self, model_versions=None):
        self.events = {}
        self.model_versions = model_versions or {}

    def observe(self, family, track, seconds, gps=None):
        if track.get("missing", 0):
            return None
        key = (family, track["track_id"])
        if key not in self.events:
            self.events[key] = {
                "event_id": f"{family}-{track['track_id']}",
                "event_type": family,
                "event_status": "candidate",
                "class_name": track["class_name"],
                "confidence": track["confidence"],
                "model_version": self.model_versions.get(family),
                "start_time": seconds,
                "end_time": seconds,
                "video_time_seconds": seconds,
                "latitude": None,
                "longitude": None,
                "location_source": "unknown",
                "tracking_id": track["track_id"],
                "review_required": True,
                "reviewed_by": None,
                "review_result": None,
                "observations": 0,
            }
        event = self.events[key]
        event.update(
            end_time=seconds,
            confidence=max(event["confidence"], track["confidence"]),
            bounding_box=list(track.get("observed_box", track["box"])),
            observations=event["observations"] + 1,
        )
        if gps and event["latitude"] is None:
            event.update(
                latitude=gps["latitude"],
                longitude=gps["longitude"],
                location_source="camera_gps_estimate",
            )
        return event
