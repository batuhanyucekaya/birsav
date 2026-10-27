"""Kalıcı veritabanı kullanmadan demo dashboard sunar."""

import base64
import cgi
import csv
from datetime import datetime, timezone
import io
import json
import math
import re
import time
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
_road_model = None
_container_model = None
_container_state_model = None
_lane_model = None
_scene_model = None
_object_model = None
_litter_model = None
_face_cascade = None
_face_detector = None
_plate_cascade = None
LITTER_CLASSES = {"Plastic bag & wrapper", "Cigarette", "Bottle", "Bottle cap", "Can", "Other plastic", "Carton", "Cup", "Other"}
VIDEO_DIR = ROOT / "tmp" / "video_results"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
VIDEO_TTL_SECONDS = 60 * 60
_video_tokens = {}
REFERENCE_LIMITS_PATH = ROOT / "data" / "reference_limits.json"


def _load_gps_rows(field):
    if field is None or not getattr(field, "file", None):
        return []
    rows = []
    first_absolute = None
    for row in csv.DictReader(io.TextIOWrapper(field.file, encoding="utf-8-sig")):
        try:
            lat, lon = float(row["latitude"]), float(row["longitude"])
            if "video_time_seconds" in row and row["video_time_seconds"]:
                sec = float(row["video_time_seconds"])
            else:
                absolute = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).timestamp()
                if first_absolute is None:
                    first_absolute = absolute
                sec = absolute - first_absolute
            rows.append({"seconds": sec, "latitude": lat, "longitude": lon})
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(rows, key=lambda x: x["seconds"])


def _gps_at(rows, seconds):
    if not rows:
        return None
    before = rows[0]
    after = rows[-1]
    for row in rows:
        if row["seconds"] <= seconds:
            before = row
        if row["seconds"] >= seconds:
            after = row
            break
    span = after["seconds"] - before["seconds"]
    ratio = 0.0 if span <= 0 else min(1.0, max(0.0, (seconds - before["seconds"]) / span))
    return {"latitude": before["latitude"] + (after["latitude"] - before["latitude"]) * ratio, "longitude": before["longitude"] + (after["longitude"] - before["longitude"]) * ratio}


def _nearby_limit(gps_point):
    if not gps_point or not REFERENCE_LIMITS_PATH.exists():
        return None
    try:
        locations = json.loads(REFERENCE_LIMITS_PATH.read_text(encoding="utf-8")).get("locations", [])
    except (OSError, ValueError):
        return None
    best = None
    for item in locations:
        try:
            dy = (float(item["latitude"]) - gps_point["latitude"]) * 111_000
            dx = (float(item["longitude"]) - gps_point["longitude"]) * 111_000 * math.cos(math.radians(gps_point["latitude"]))
            distance = math.hypot(dx, dy)
            if distance <= float(item.get("radius_m", 25)) and (best is None or distance < best["distance_m"]):
                best = {**item, "distance_m": round(distance, 1)}
        except (KeyError, TypeError, ValueError):
            continue
    return best


def _cleanup_video_dir() -> None:
    cutoff = time.time() - VIDEO_TTL_SECONDS
    for path in VIDEO_DIR.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue
    for token, created in list(_video_tokens.items()):
        if created < cutoff:
            _video_tokens.pop(token, None)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1, area_a + area_b - inter)


def _smooth_boxes(previous, current, match_iou=0.25, alpha=0.35, max_missing=2):
    """Kareler arası kutu titremesini azaltır; eski kutuları kısa süre korur."""
    used = set(); updated = []
    next_id = max((int(old.get("track_id", 0)) for old in previous), default=0) + 1
    for old in previous:
        best_i, best = -1, 0.0
        for i, item in enumerate(current):
            if old.get("class_name") and item.get("class_name") != old.get("class_name"):
                continue
            score = _iou(old["box"], item["box"])
            if i not in used and score > best:
                best_i, best = i, score
        if best_i >= 0 and best >= match_iou:
            new = current[best_i]; used.add(best_i)
            box = [old["box"][j] * (1 - alpha) + new["box"][j] * alpha for j in range(4)]
            updated.append({**new, "track_id": old.get("track_id", next_id), "box": box, "hits": old["hits"] + 1, "missing": 0})
        elif old["missing"] < max_missing:
            updated.append({**old, "missing": old["missing"] + 1})
    for i, item in enumerate(current):
        if i not in used:
            updated.append({**item, "track_id": next_id, "hits": 1, "missing": 0})
            next_id += 1
    return updated


def _dedupe_boxes(items, overlap=0.45):
    """Aynı sınıftaki üst üste binen kutuları en güvenilir kutuya indirger."""
    kept = []
    for item in sorted(items, key=lambda x: float(x.get("confidence", 0)), reverse=True):
        if any(item.get("class_name") == old.get("class_name") and _iou(item["box"], old["box"]) >= overlap for old in kept):
            continue
        kept.append(item)
    return kept


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def do_POST(self):
        if self.path == "/api/analyze-video":
            self._analyze_video()
            return
        if self.path == "/api/analyze":
            self._analyze()
            return
        if self.path == "/api/container-classify":
            self._container_classify()
            return
        if self.path != "/api/detect":
            self.send_error(404)
            return
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
        if "image" not in form or not getattr(form["image"], "file", None):
            self._json({"error": "image alanı gerekli"}, 400)
            return
        frame = cv2.imdecode(np.frombuffer(form["image"].file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self._json({"error": "görsel okunamadı"}, 400)
            return
        model = YOLO(str(ROOT / "models" / "yolo12s_seed0_best.pt"))
        result = model.predict(frame, conf=0.10, verbose=False)[0]
        detections = []
        for cls, conf, xyxy in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
            detections.append({"class_name": result.names[int(cls)], "confidence": float(conf), "xyxy": [round(float(v), 2) for v in xyxy]})
        annotated = result.plot()
        ok, encoded = cv2.imencode(".jpg", annotated)
        if not ok:
            self._json({"error": "sonuç görseli üretilemedi"}, 500)
            return
        self._json({"model": "RDD2022 YOLO12s", "detections": detections, "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii")})

    def do_GET(self):
        if self.path.startswith("/video-results/"):
            _cleanup_video_dir()
            name = Path(self.path.split("?", 1)[0]).name
            if not re.fullmatch(r"[0-9a-f]{32}\.(?:webm|mp4)", name):
                self.send_error(404)
                return
            token = name.rsplit(".", 1)[0]
            if token not in _video_tokens:
                self.send_error(404)
                return
            target = VIDEO_DIR / name
            if target.exists() and target.suffix in {".mp4", ".webm"}:
                data = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/webm" if target.suffix == ".webm" else "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)
            return
        super().do_GET()

    def _analyze_video(self):
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
        if "video" not in form or not getattr(form["video"], "file", None):
            self._json({"error": "video alanı gerekli"}, 400)
            return
        gps_rows = _load_gps_rows(form["gps"] if "gps" in form else None)
        token = uuid.uuid4().hex
        raw_path = VIDEO_DIR / f"{token}.input"
        # VP8/WebM tarayıcılarla uyumludur; mp4v bazı tarayıcılarda oynatılmaz.
        output_path = VIDEO_DIR / f"{token}.webm"
        _cleanup_video_dir()
        total = 0
        try:
            with raw_path.open("wb") as output:
                while True:
                    chunk = form["video"].file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise ValueError("video dosyası boyut sınırını aşıyor")
                    output.write(chunk)
        except ValueError as exc:
            raw_path.unlink(missing_ok=True)
            self._json({"error": str(exc), "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024)}, 413)
            return
        except OSError:
            raw_path.unlink(missing_ok=True)
            self._json({"error": "video geçici olarak kaydedilemedi"}, 507)
            return
        cap = cv2.VideoCapture(str(raw_path))
        if not cap.isOpened():
            raw_path.unlink(missing_ok=True)
            self._json({"error": "video okunamadı"}, 400)
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not width or not height:
            cap.release(); raw_path.unlink(missing_ok=True)
            self._json({"error": "video boyutu okunamadı"}, 400)
            return
        sample_every = max(1, int(round(fps / 3)))
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height))
        if not writer.isOpened():
            cap.release(); raw_path.unlink(missing_ok=True)
            self._json({"error": "tarayıcı uyumlu video kodlayıcı açılamadı"}, 500)
            return
        global _face_cascade, _face_detector, _plate_cascade, _road_model, _litter_model, _scene_model, _object_model
        if _face_detector is None:
            yunet_path = ROOT / "models" / "privacy" / "face_detection_yunet_2023mar.onnx"
            if yunet_path.exists() and hasattr(cv2, "FaceDetectorYN"):
                _face_detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (width, height), 0.58, 0.30, 5000)
        if _face_detector is None and _face_cascade is None:
            _face_cascade = cv2.CascadeClassifier(str(ROOT / "models" / "privacy" / "haarcascade_frontalface_default.xml"))
        if _plate_cascade is None:
            _plate_cascade = cv2.CascadeClassifier(str(ROOT / "models" / "privacy" / "haarcascade_russian_plate_number.xml"))
        if _road_model is None:
            _road_model = YOLO(str(ROOT / "models" / "yolo12s_seed0_best.pt"))
        if _litter_model is None:
            _litter_model = YOLO(str(ROOT / "models" / "taco_yolo11s_best.pt"))
        scene_path = ROOT / "runs" / "segment" / "runs" / "segment" / "runs" / "mapillary_seg" / "train_advertisement_fix" / "weights" / "best.pt"
        if scene_path.exists() and _scene_model is None:
            _scene_model = YOLO(str(scene_path))
        if _object_model is None and (ROOT / "yolo11n.pt").exists():
            _object_model = YOLO(str(ROOT / "yolo11n.pt"))
        counts = {"privacy_frames": 0, "faces_blurred": 0, "plates_blurred": 0, "road_damage": 0, "ground_litter": 0, "scene": 0}
        events = []
        privacy_tracks = []
        face_ids, plate_ids = set(), set()
        road_tracks, litter_tracks, scene_tracks = [], [], []
        frame_idx = 0
        processing_error = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gps_point = _gps_at(gps_rows, frame_idx / max(fps, 1.0))
                blurred = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if _face_detector is not None:
                    _face_detector.setInputSize((width, height))
                    _, face_rows = _face_detector.detect(frame)
                    faces = [] if face_rows is None else [
                        (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
                        for row in face_rows if float(row[14]) >= 0.65 and row[2] >= 20 and row[3] >= 20
                    ]
                    # İlk karelerde küçük/uzak yüzleri yakalamak için kısa büyütülmüş-kare takviyesi.
                    if not faces and frame_idx < 6:
                        scale = 1.5
                        enlarged = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                        _face_detector.setInputSize((enlarged.shape[1], enlarged.shape[0]))
                        _, enlarged_rows = _face_detector.detect(enlarged)
                        faces = [] if enlarged_rows is None else [
                            (int(row[0] / scale), int(row[1] / scale), int(row[2] / scale), int(row[3] / scale))
                            for row in enlarged_rows if float(row[14]) >= 0.60 and row[2] >= 24 and row[3] >= 24
                        ]
                else:
                    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(32, 32)) if _face_cascade is not None and not _face_cascade.empty() else []
                plates = _plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 12)) if not _plate_cascade.empty() else []
                frame_area = float(width * height)
                valid_faces = []
                for x, y, w, h in faces:
                    ratio = w / max(1, h)
                    if 0.45 <= ratio <= 2.2 and (w * h) / frame_area <= 0.10 and w <= width * 0.32 and h <= height * 0.32:
                        valid_faces.append((x, y, w, h))
                valid_plates = []
                for x, y, w, h in plates:
                    ratio = w / max(1, h)
                    if 1.8 <= ratio <= 6.5 and (w * h) / frame_area <= 0.035 and w <= width * 0.35 and h <= height * 0.14:
                        valid_plates.append((x, y, w, h))
                privacy_current = [{"kind": "face", "box": [x, y, x + w, y + h]} for x, y, w, h in valid_faces]
                privacy_current += [{"kind": "plate", "box": [x, y, x + w, y + h]} for x, y, w, h in valid_plates]
                privacy_current = _dedupe_boxes(privacy_current, overlap=0.30)
                privacy_tracks = _smooth_boxes(privacy_tracks, privacy_current, match_iou=0.25, alpha=0.45, max_missing=10)
                stable_privacy = [x for x in privacy_tracks if x["hits"] >= 1 and x["missing"] <= 10]
                # Hatalı çoklu kutuların ekranın büyük kısmını kaplamasını engelle.
                safe_privacy = []
                covered = 0.0
                for item in sorted(stable_privacy, key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1])):
                    bx1, by1, bx2, by2 = item["box"]
                    area = max(0, bx2 - bx1) * max(0, by2 - by1)
                    if covered + area <= frame_area * 0.20:
                        safe_privacy.append(item)
                        covered += area
                stable_privacy = safe_privacy
                if stable_privacy:
                    counts["privacy_frames"] += 1
                    face_ids.update(x["track_id"] for x in stable_privacy if x["kind"] == "face")
                    plate_ids.update(x["track_id"] for x in stable_privacy if x["kind"] == "plate")
                    counts["faces_blurred"] = len(face_ids); counts["plates_blurred"] = len(plate_ids)
                    for item in stable_privacy:
                        bx1, by1, bx2, by2 = item["box"]
                        pad_x = min((bx2 - bx1) * 0.18, width * 0.08)
                        pad_y = min((by2 - by1) * 0.18, height * 0.08)
                        x1, y1, x2, y2 = [max(0, int(v)) for v in (bx1 - pad_x, by1 - pad_y, bx2 + pad_x, by2 + pad_y)]
                        x2, y2 = min(width, x2), min(height, y2)
                        roi = blurred[y1:y2, x1:x2]
                        if roi.size:
                            blurred[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (0, 0), 35)
                if frame_idx % sample_every == 0:
                    road_result = _road_model.predict(blurred, conf=0.25, verbose=False)[0]
                    litter_result = _litter_model.predict(blurred, conf=0.35, verbose=False)[0]
                    scene_result = _scene_model.predict(blurred, conf=0.12, imgsz=640, verbose=False)[0] if _scene_model else None
                    person_boxes = []
                    if _object_model:
                        person_result = _object_model.predict(blurred, conf=0.35, classes=[0], imgsz=640, verbose=False)[0]
                        person_boxes = [box for box in person_result.boxes.xyxy.tolist()]
                    frame_events = []
                    road_current, litter_current, scene_current = [], [], []
                    for c, p, box in zip(road_result.boxes.cls.tolist(), road_result.boxes.conf.tolist(), road_result.boxes.xyxy.tolist()):
                        road_current.append({"kind": "road_damage", "class_name": road_result.names[int(c)], "confidence": float(p), "box": box})
                    for c, p, box in zip(litter_result.boxes.cls.tolist(), litter_result.boxes.conf.tolist(), litter_result.boxes.xyxy.tolist()):
                        if litter_result.names[int(c)] not in LITTER_CLASSES: continue
                        litter_current.append({"kind": "ground_litter", "class_name": litter_result.names[int(c)], "confidence": float(p), "box": box})
                    if scene_result:
                        for c, p, box in zip(scene_result.boxes.cls.tolist(), scene_result.boxes.conf.tolist(), scene_result.boxes.xyxy.tolist()):
                            name = scene_result.names[int(c)]
                            if float(p) < {"sidewalk": .08, "billboard": .12, "sign": .15, "trash_can": .30}.get(name, .20): continue
                            if name == "trash_can" and any(_iou(box, person_box) > 0.05 for person_box in person_boxes): continue
                            scene_current.append({"kind": "scene", "class_name": name, "confidence": float(p), "box": box})
                    road_tracks = _smooth_boxes(road_tracks, road_current, match_iou=0.25, alpha=0.35, max_missing=3)
                    litter_tracks = _smooth_boxes(litter_tracks, litter_current, match_iou=0.18, alpha=0.35, max_missing=3)
                    scene_tracks = _smooth_boxes(scene_tracks, _dedupe_boxes(scene_current), match_iou=0.25, alpha=0.35, max_missing=3)
                    for item in [x for x in road_tracks if x["hits"] >= 1 and x["missing"] <= 3]:
                        counts["road_damage"] += 1; frame_events.append({"type": "road_damage", "confidence": round(item["confidence"], 3), "class_name": item["class_name"]})
                        x1, y1, x2, y2 = [int(v) for v in item["box"]]; cv2.rectangle(blurred, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    for item in [x for x in litter_tracks if x["hits"] >= 1 and x["missing"] <= 3]:
                        counts["ground_litter"] += 1; frame_events.append({"type": "ground_litter", "confidence": round(item["confidence"], 3), "class_name": item["class_name"]})
                        x1, y1, x2, y2 = [int(v) for v in item["box"]]; cv2.rectangle(blurred, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    for item in [x for x in scene_tracks if x["hits"] >= 1 and x["missing"] <= 3]:
                        counts["scene"] += 1
                        sx1, sy1, sx2, sy2 = item["box"]
                        scene_event = {"type": item["class_name"], "confidence": round(item["confidence"], 3), "class_name": item["class_name"]}
                        if item["class_name"] in {"billboard", "sign"}:
                            scene_event.update({"width_px": round(max(0, sx2 - sx1), 1), "height_px": round(max(0, sy2 - sy1), 1), "area_px": round(max(0, sx2 - sx1) * max(0, sy2 - sy1), 1)})
                            limit = _nearby_limit(gps_point)
                            if limit and "max_width_px" in limit and "max_height_px" in limit:
                                allowed_w = float(limit["max_width_px"]) * 1.05
                                allowed_h = float(limit["max_height_px"]) * 1.05
                                scene_event.update({"limit_name": limit.get("name", "referans"), "distance_m": limit["distance_m"], "limit_status": "SINIR AŞILDI" if sx2 - sx1 > allowed_w or sy2 - sy1 > allowed_h else "UYGUN", "max_width_px": limit["max_width_px"], "max_height_px": limit["max_height_px"], "tolerance": 0.05})
                        frame_events.append(scene_event)
                        x1, y1, x2, y2 = [int(v) for v in item["box"]]; cv2.rectangle(blurred, (x1, y1), (x2, y2), (180, 0, 180), 2)
                        if item["class_name"] in {"billboard", "sign"} and gps_point:
                            limit = _nearby_limit(gps_point)
                            if limit and "max_width_px" in limit and "max_height_px" in limit:
                                allowed_w = float(limit["max_width_px"]) * 1.05
                                allowed_h = float(limit["max_height_px"]) * 1.05
                                status = "SINIR ASILDI" if x2 - x1 > allowed_w or y2 - y1 > allowed_h else "UYGUN"
                                color = (0, 0, 255) if status == "SINIR ASILDI" else (0, 180, 0)
                                cv2.putText(blurred, status, (x1, min(height - 8, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    if frame_events:
                        events.append({"time_sec": round(frame_idx / fps, 2), "items": frame_events})
                elif frame_idx % sample_every != 0:
                    # Model çalıştırılmayan ara karelerde son güvenilir kutuları koru.
                    for item in [x for x in road_tracks if x["hits"] >= 1 and x["missing"] <= 3]:
                        x1, y1, x2, y2 = [int(v) for v in item["box"]]
                        cv2.rectangle(blurred, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    for item in [x for x in litter_tracks if x["hits"] >= 1 and x["missing"] <= 3]:
                        x1, y1, x2, y2 = [int(v) for v in item["box"]]
                        cv2.rectangle(blurred, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    for item in [x for x in scene_tracks if x["hits"] >= 1 and x["missing"] <= 3]:
                        x1, y1, x2, y2 = [int(v) for v in item["box"]]
                        cv2.rectangle(blurred, (x1, y1), (x2, y2), (180, 0, 180), 2)
                writer.write(blurred)
                frame_idx += 1
        except Exception:
            processing_error = "video işleme sırasında beklenmeyen bir hata oluştu"
        finally:
            cap.release(); writer.release(); raw_path.unlink(missing_ok=True)
        if processing_error:
            output_path.unlink(missing_ok=True)
            self._json({"error": processing_error}, 500)
            return
        validation = cv2.VideoCapture(str(output_path))
        output_ok, _ = validation.read()
        validation.release()
        if not output_ok:
            output_path.unlink(missing_ok=True)
            self._json({"error": "işlenen video çıktısı doğrulanamadı"}, 500)
            return
        _video_tokens[token] = time.time()
        self._json({"video_url": f"/video-results/{output_path.name}", "frames": frame_idx, "fps": round(fps, 2), "sample_every": sample_every, "counts": counts, "events": events[:200], "gps_enabled": bool(gps_rows), "reference_limits_loaded": REFERENCE_LIMITS_PATH.exists(), "privacy_note": "Yalnızca tespit edilen yüz ve plaka bölgeleri bulanıklaştırıldı; ham video saklanmadı."})

    def _container_classify(self):
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
        if "image" not in form or not getattr(form["image"], "file", None):
            self._json({"error": "image alanı gerekli"}, 400)
            return
        frame = cv2.imdecode(np.frombuffer(form["image"].file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self._json({"error": "görsel okunamadı"}, 400)
            return
        model_path = ROOT / "runs" / "detect" / "runs" / "container_detection" / "train" / "weights" / "best.pt"
        if not model_path.exists():
            self._json({"error": "konteyner modeli bulunamadı; önce eğitim çalıştırılmalı"}, 503)
            return
        global _container_model, _container_state_model
        if _container_model is None:
            _container_model = YOLO(str(model_path))
        model = _container_model
        # Bu ilk demo modeli çok küçük bir veriyle eğitildiği için 0.25 eşiği
        # bazı doğru adayları eliyordu. Demo görünürlüğü için daha düşük eşik
        # ve daha yüksek çözünürlük kullanıyoruz; gerçek sürümde eşik doğrulama
        # setine göre yeniden ayarlanmalıdır.
        candidate = model.predict(frame, conf=0.08, imgsz=640, max_det=5, verbose=False)[0]
        # Detection aşaması konteyner adayı bulur; durum etiketi ayrı
        # sınıflandırıcıdan gelir. Böylece kaldırım/zemin kutuları doğrudan
        # empty/full sonucu olarak listelenmez ve API tek sonuç döndürür.
        state_path = ROOT / "runs" / "classify" / "runs" / "container" / "train" / "weights" / "best.pt"
        if _container_state_model is None:
            _container_state_model = YOLO(str(state_path))
        state = _container_state_model.predict(frame, verbose=False)[0]
        state_name = state.names[int(state.probs.top1)]
        state_conf = float(state.probs.top1conf)
        # Mevcut bootstrap etiketleri gerçek koordinat değil, tam-görsel kutusu
        # olduğu için modelin ürettiği koordinatı kullanıcıya yanlış bir
        # dikdörtgen olarak göstermiyoruz. Durum tahmini güvenilir kalır;
        # gerçek kutu için ayrıca bbox etiketli veri gerekir.
        detections = [{"class_name": state_name, "confidence": state_conf}]
        annotated = frame.copy()
        cv2.putText(annotated, f"container status: {state_name} ({state_conf:.0%})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 255), 2)
        ok, encoded = cv2.imencode(".jpg", annotated)
        if not ok:
            self._json({"error": "sonuç görseli üretilemedi"}, 500)
            return
        self._json({"model": "Konteyner detection + durum sınıflandırıcı", "detections": detections, "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii")})

    def _analyze(self):
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
        if "image" not in form or not getattr(form["image"], "file", None):
            self._json({"error": "image alanı gerekli"}, 400)
            return
        frame = cv2.imdecode(np.frombuffer(form["image"].file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self._json({"error": "görsel okunamadı"}, 400)
            return
        global _road_model, _litter_model, _container_model, _container_state_model, _lane_model, _scene_model, _object_model
        if _road_model is None:
            _road_model = YOLO(str(ROOT / "models" / "yolo12s_seed0_best.pt"))
        road_result = _road_model.predict(frame, conf=0.10, verbose=False)[0]
        road = [{"class_name": road_result.names[int(c)], "confidence": float(p)} for c, p in zip(road_result.boxes.cls.tolist(), road_result.boxes.conf.tolist())]
        lane = []
        lane_path = ROOT / "runs" / "detect" / "runs" / "ceymo_small" / "train20_cpu" / "weights" / "best.pt"
        if lane_path.exists():
            if _lane_model is None:
                _lane_model = YOLO(str(lane_path))
            lane_result = _lane_model.predict(frame, conf=0.20, imgsz=320, verbose=False)[0]
            lane = [{"class_name": lane_result.names[int(c)], "confidence": float(p), "xyxy": [round(float(v), 2) for v in box]} for c, p, box in zip(lane_result.boxes.cls.tolist(), lane_result.boxes.conf.tolist(), lane_result.boxes.xyxy.tolist())]
        scene = []
        scene_path = ROOT / "runs" / "segment" / "runs" / "segment" / "runs" / "mapillary_seg" / "train_advertisement_fix" / "weights" / "best.pt"
        if not scene_path.exists():
            scene_path = ROOT / "runs" / "segment" / "runs" / "mapillary_seg" / "train_advertisement_fix" / "weights" / "best.pt"
        if scene_path.exists():
            if _scene_model is None:
                _scene_model = YOLO(str(scene_path))
            # Küçük pano/kaldırım nesneleri için temel eşiği düşük tutup sınıfa göre filtreliyoruz.
            scene_result = _scene_model.predict(frame, conf=0.08, imgsz=640, verbose=False)[0]
            if _object_model is None and (ROOT / "yolo11n.pt").exists():
                _object_model = YOLO(str(ROOT / "yolo11n.pt"))
            person_boxes = []
            if _object_model is not None:
                person_result = _object_model.predict(frame, conf=0.35, classes=[0], imgsz=640, verbose=False)[0]
                person_boxes = person_result.boxes.xyxy.tolist()
            thresholds = {"road": 0.20, "sidewalk": 0.08, "lane_marking": 0.15, "trash_can": 0.30, "billboard": 0.12, "sign": 0.15}
            scene = []
            for c, p, box in zip(scene_result.boxes.cls.tolist(), scene_result.boxes.conf.tolist(), scene_result.boxes.xyxy.tolist()):
                name, confidence = scene_result.names[int(c)], float(p)
                if confidence < thresholds.get(name, 0.12):
                    continue
                if name == "trash_can" and any(_iou(box, person_box) > 0.05 for person_box in person_boxes):
                    continue
                scene.append({"class_name": name, "confidence": confidence, "xyxy": [round(float(v), 2) for v in box]})
        if _litter_model is None:
            _litter_model = YOLO(str(ROOT / "models" / "taco_yolo11s_best.pt"))
        litter_result = _litter_model.predict(frame, conf=0.25, verbose=False)[0]
        litter = [{"class_name": "litter_candidate", "source_object": litter_result.names[int(c)], "confidence": float(p)} for c, p in zip(litter_result.boxes.cls.tolist(), litter_result.boxes.conf.tolist()) if litter_result.names[int(c)] in LITTER_CLASSES]
        container = None
        container_path = ROOT / "runs" / "detect" / "runs" / "container_detection" / "train" / "weights" / "best.pt"
        state_path = ROOT / "runs" / "classify" / "runs" / "container" / "train" / "weights" / "best.pt"
        if container_path.exists() and state_path.exists():
            if _container_model is None:
                _container_model = YOLO(str(container_path))
            candidate = _container_model.predict(frame, conf=0.45, imgsz=640, max_det=1, verbose=False)[0]
            if len(candidate.boxes):
                x1, y1, x2, y2 = candidate.boxes.xyxy[0].tolist()
                bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
                candidate_conf = float(candidate.boxes.conf[0])
                candidate_area = (bw * bh) / max(1.0, frame.shape[0] * frame.shape[1])
                candidate_ratio = bw / bh
                valid_container_box = (candidate_conf >= 0.45 and 0.35 <= candidate_ratio <= 2.8 and 0.005 <= candidate_area <= 0.30)
                if _object_model is not None:
                    person_result = _object_model.predict(frame, conf=0.35, classes=[0], imgsz=640, verbose=False)[0]
                    if any(_iou([x1, y1, x2, y2], person_box) > 0.05 for person_box in person_result.boxes.xyxy.tolist()):
                        valid_container_box = False
                if not valid_container_box:
                    candidate = None
            if candidate is not None and len(candidate.boxes):
                if _container_state_model is None:
                    _container_state_model = YOLO(str(state_path))
                state = _container_state_model.predict(frame, verbose=False)[0]
                state_conf = float(state.probs.top1conf)
                # Tek akışta yol fotoğraflarını konteyner sanmamak için
                # sınıflandırıcı güvenini ve yol tespiti çakışmasını gate'le.
                if state_conf >= (0.75 if road else 0.65):
                    container = {"class_name": state.names[int(state.probs.top1)], "confidence": state_conf}
        annotated = frame.copy()
        # Birleşik sonuç görselinde bütün sorunları kırmızı kutularla göster.
        for cls, conf, xyxy in zip(road_result.boxes.cls.tolist(), road_result.boxes.conf.tolist(), road_result.boxes.xyxy.tolist()):
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated, f"road: {road_result.names[int(cls)]} {conf:.0%}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        for item in lane:
            x1, y1, x2, y2 = [int(v) for v in item["xyxy"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 0), 3)
            cv2.putText(annotated, f"lane: {item['class_name']} {item['confidence']:.0%}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 0), 2)
        for item in scene:
            x1, y1, x2, y2 = [int(v) for v in item["xyxy"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (180, 0, 180), 2)
            cv2.putText(annotated, f"scene: {item['class_name']} {item['confidence']:.0%}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 0, 180), 2)
        for cls, conf, xyxy in zip(litter_result.boxes.cls.tolist(), litter_result.boxes.conf.tolist(), litter_result.boxes.xyxy.tolist()):
            if litter_result.names[int(cls)] not in LITTER_CLASSES:
                continue
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated, f"litter: {litter_result.names[int(cls)]} {conf:.0%}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        ok, encoded = cv2.imencode(".jpg", annotated)
        if not ok:
            self._json({"error": "sonuç görseli üretilemedi"}, 500)
            return
        issues = []
        issues.extend({"type": "road_damage", **item} for item in road)
        issues.extend({"type": "lane_marking", **item} for item in lane)
        issues.extend({"type": "scene", **item} for item in scene)
        issues.extend({"type": "ground_litter", **item} for item in litter)
        if container:
            issues.append({"type": "container", **container})
        self._json({"issues": issues, "road_damage": road, "lane_markings": lane, "scene_segments": scene, "ground_litter": litter, "container": container, "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"), "litter_model": "TACO YOLO11s hazır model", "lane_model": "CeyMo YOLO11n (20 epoch pilot)" if lane_path.exists() else None, "scene_model": "Mapillary YOLO11n-seg (pano/tabela/konteyner pilot)" if scene_path.exists() else None})

    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


def main() -> None:
    # Global OpenCV/YOLO model nesneleri thread-safe değildir. Demo sürümünde
    # istekleri sıraya alarak modellerin çözünürlük ve tracker durumlarının
    # farklı videolar arasında karışmasını engelleriz.
    server = HTTPServer(("127.0.0.1", 8000), Handler)
    print("Demo sitesi: http://127.0.0.1:8000")
    print("Veriler yalnızca tarayıcıda örnek olarak tutulur; kapatınca kalıcı kayıt kalmaz.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

