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


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def do_POST(self):
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

