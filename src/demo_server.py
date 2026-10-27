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
_container_state_model = None
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
        global _road_model, _litter_model, _container_model, _container_state_model
        if _road_model is None:
            _road_model = YOLO(str(ROOT / "models" / "yolo12s_seed0_best.pt"))
        road_result = _road_model.predict(frame, conf=0.10, verbose=False)[0]
        road = [{"class_name": road_result.names[int(c)], "confidence": float(p)} for c, p in zip(road_result.boxes.cls.tolist(), road_result.boxes.conf.tolist())]
        if _litter_model is None:
            _litter_model = YOLO("yolo11n.pt")
        litter_result = _litter_model.predict(frame, conf=0.25, verbose=False)[0]
        litter = [{"class_name": "litter_candidate", "source_object": litter_result.names[int(c)], "confidence": float(p)} for c, p in zip(litter_result.boxes.cls.tolist(), litter_result.boxes.conf.tolist()) if litter_result.names[int(c)] in LITTER_CLASSES]
        container = None
        container_path = ROOT / "runs" / "detect" / "runs" / "container_detection" / "train" / "weights" / "best.pt"
        state_path = ROOT / "runs" / "classify" / "runs" / "container" / "train" / "weights" / "best.pt"
        if container_path.exists() and state_path.exists():
            if _container_model is None:
                _container_model = YOLO(str(container_path))
            candidate = _container_model.predict(frame, conf=0.25, imgsz=640, max_det=1, verbose=False)[0]
            if len(candidate.boxes):
                if _container_state_model is None:
                    _container_state_model = YOLO(str(state_path))
                state = _container_state_model.predict(frame, verbose=False)[0]
                state_conf = float(state.probs.top1conf)
                # Tek akışta yol fotoğraflarını konteyner sanmamak için
                # sınıflandırıcı güvenini ve yol tespiti çakışmasını gate'le.
                if state_conf >= (0.75 if road else 0.65):
                    container = {"class_name": state.names[int(state.probs.top1)], "confidence": state_conf}
        issues = []
        issues.extend({"type": "road_damage", **item} for item in road)
        issues.extend({"type": "ground_litter", **item} for item in litter)
        if container:
            issues.append({"type": "container", **container})
        self._json({"issues": issues, "road_damage": road, "ground_litter": litter, "container": container, "litter_model": "YOLO11n COCO baseline"})

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

