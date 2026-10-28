"""Local privacy masking. Detection quality must be measured separately."""

from pathlib import Path
import math
import cv2

from src.tracking import BoxTracker


def mask_regions(frame, items, mode="opaque"):
    output = frame.copy()
    height, width = frame.shape[:2]
    for item in items:
        # Include the current detection as well as smoothed/stale regions.
        for box in (item["box"], item.get("observed_box", item["box"])):
            if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
                output[:] = 0
                continue
            x1, y1, x2, y2 = box
            px, py = max(4, (x2 - x1) * 0.2), max(4, (y2 - y1) * 0.2)
            x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
            x2, y2 = min(width, math.ceil(x2 + px)), min(height, math.ceil(y2 + py))
            roi = output[y1:y2, x1:x2]
            if not roi.size:
                continue
            if mode == "opaque":
                roi[:] = 0
            elif mode == "pixelate":
                small = cv2.resize(
                    roi, (max(1, roi.shape[1] // 20), max(1, roi.shape[0] // 20))
                )
                roi[:] = cv2.resize(
                    small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            elif mode == "blur":
                roi[:] = cv2.GaussianBlur(roi, (0, 0), 35)
            else:
                raise ValueError("Unknown privacy mode")
    return output


class PrivacyProcessor:
    def __init__(self, root, mode="opaque"):
        self.mode = mode
        folder = Path(root) / "models" / "privacy"
        face = folder / "face_detection_yunet_2023mar.onnx"
        self.face = (
            cv2.FaceDetectorYN.create(str(face), "", (320, 320), 0.5, 0.3, 5000)
            if face.is_file()
            else None
        )
        self.plate = cv2.CascadeClassifier(
            str(folder / "haarcascade_russian_plate_number.xml")
        )
        if self.face is None or self.plate.empty():
            raise RuntimeError("Gizlilik modelleri eksik; video sunulmayacak")
        self.tracker = BoxTracker(0.2, alpha=1.0, max_missing=10)
        self.ids = {"face": set(), "plate": set()}

    def process(self, frame):
        h, w = frame.shape[:2]
        # Scan every frame, including faces entering after the first six frames.
        self.face.setInputSize((w, h))
        _, rows = self.face.detect(frame)
        detections = []
        for row in [] if rows is None else rows:
            x, y, bw, bh = map(float, row[:4])
            if float(row[14]) >= 0.5 and bw > 0 and bh > 0:
                detections.append(
                    {
                        "kind": "face",
                        "class_name": "face",
                        "box": [x, y, x + bw, y + bh],
                    }
                )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for x, y, bw, bh in self.plate.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 8)
        ):
            detections.append(
                {
                    "kind": "plate",
                    "class_name": "plate",
                    "box": [int(x), int(y), int(x + bw), int(y + bh)],
                }
            )
        tracks = self.tracker.update(detections)
        for item in tracks:
            self.ids[item["kind"]].add(item["track_id"])
        return mask_regions(frame, tracks, self.mode), tracks
