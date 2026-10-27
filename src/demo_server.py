"""Kalıcı veritabanı kullanmadan demo dashboard sunar."""

import base64
import cgi
import io
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
_road_model = None
_container_model = None
_litter_model = None
LITTER_CLASSES = {"bottle", "cup", "banana", "apple", "backpack", "handbag", "bowl", "fork", "knife", "spoon", "wine glass", "book", "umbrella", "sports ball"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def do_POST(self):
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
        global _container_model
        if _container_model is None:
            _container_model = YOLO(str(model_path))
        model = _container_model
        # Bu ilk demo modeli çok küçük bir veriyle eğitildiği için 0.25 eşiği
        # bazı doğru adayları eliyordu. Demo görünürlüğü için daha düşük eşik
        # ve daha yüksek çözünürlük kullanıyoruz; gerçek sürümde eşik doğrulama
        # setine göre yeniden ayarlanmalıdır.
        result = model.predict(frame, conf=0.08, imgsz=640, max_det=5, verbose=False)[0]
        raw = sorted(zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()), key=lambda x: x[1], reverse=True)
        kept = []
        for c, p, box in raw:
            # Küçük eğitim setinde aynı konteyner için farklı sınıflardan
            # üst üste kutular gelebiliyor; örtüşen düşük skorlu kutuyu ele.
            x1, y1, x2, y2 = box
            area = max(1.0, (x2 - x1) * (y2 - y1))
            duplicate = False
            for _, _, other in kept:
                ox1, oy1, ox2, oy2 = other
                ix1, iy1, ix2, iy2 = max(x1, ox1), max(y1, oy1), min(x2, ox2), min(y2, oy2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                other_area = max(1.0, (ox2 - ox1) * (oy2 - oy1))
                if inter / min(area, other_area) > 0.6:
                    duplicate = True
                    break
            if not duplicate:
                kept.append((c, p, box))
        detections = [{"class_name": result.names[int(c)], "confidence": float(p), "xyxy": [round(float(v), 2) for v in box]} for c, p, box in kept]
        annotated = result.plot()
        ok, encoded = cv2.imencode(".jpg", annotated)
        if not ok:
            self._json({"error": "sonuç görseli üretilemedi"}, 500)
            return
        self._json({"model": "Konteyner YOLO11n detection", "detections": detections, "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii")})

    def _analyze(self):
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type", "")})
        if "image" not in form or not getattr(form["image"], "file", None):
            self._json({"error": "image alanı gerekli"}, 400)
            return
        frame = cv2.imdecode(np.frombuffer(form["image"].file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self._json({"error": "görsel okunamadı"}, 400)
            return
        global _road_model, _litter_model
        if _road_model is None:
            _road_model = YOLO(str(ROOT / "models" / "yolo12s_seed0_best.pt"))
        road_result = _road_model.predict(frame, conf=0.10, verbose=False)[0]
        road = [{"class_name": road_result.names[int(c)], "confidence": float(p)} for c, p in zip(road_result.boxes.cls.tolist(), road_result.boxes.conf.tolist())]
        if _litter_model is None:
            _litter_model = YOLO("yolo11n.pt")
        litter_result = _litter_model.predict(frame, conf=0.25, verbose=False)[0]
        litter = [{"class_name": "litter_candidate", "source_object": litter_result.names[int(c)], "confidence": float(p)} for c, p in zip(litter_result.boxes.cls.tolist(), litter_result.boxes.conf.tolist()) if litter_result.names[int(c)] in LITTER_CLASSES]
        self._json({"road_damage": road, "ground_litter": litter, "litter_model": "YOLO11n COCO baseline"})

    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
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

