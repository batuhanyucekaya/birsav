"""Local demo HTTP API with bounded uploads and background video processing."""

from __future__ import annotations
from http.cookies import SimpleCookie
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import socket
import threading
import time
from urllib.parse import urlsplit
import uuid
import cv2
import numpy as np
from src.gps import parse_gps
from src.http_upload import parse_form, UploadError
from src.inference import (
    MODEL_LOCK,
    get_model,
    analyze_frame,
    annotate,
    boxes,
    container_predictions,
    jpeg_payload,
)
from src.jobs import JobManager
from src.model_registry import manifest
from src.privacy import PrivacyProcessor, mask_regions

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "tmp" / "video_results"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
_sessions = {}
_session_lock = threading.Lock()
_upload_slots = threading.BoundedSemaphore(2)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.owner = None
        self.new_cookie = None
        super().__init__(*args, directory=str(ROOT / "web"), **kwargs)

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def _session(self, create=False):
        host = urlsplit("//" + self.headers.get("Host", "")).hostname
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise UploadError("Yalnızca yerel Host desteklenir", 403)
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            raise UploadError("Siteler arası istek reddedildi", 403)
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            raise UploadError("Origin uyuşmuyor", 403)
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get("Cookie", ""))
        except Exception:
            pass
        token = cookie["kent_session"].value if "kent_session" in cookie else None
        with _session_lock:
            now = time.time()
            for sid, seen in list(_sessions.items()):
                if now - seen > 3600:
                    del _sessions[sid]
            if token not in _sessions:
                if not create:
                    raise UploadError(
                        "Yerel oturum gerekli; önce ana sayfayı açın", 401
                    )
                token = uuid.uuid4().hex
                self.new_cookie = f"kent_session={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=3600"
            _sessions[token] = now
        self.owner = token

    def end_headers(self):
        if self.new_cookie:
            self.send_header("Set-Cookie", self.new_cookie)
            self.new_cookie = None
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        form = None
        slot = False
        try:
            self._session()
            action_match = re.fullmatch(
                r"/api/jobs/([0-9a-f]{32})/(cancel|review)", self.path
            )
            if action_match:
                token, action = action_match.groups()
                ok = (
                    self.server.jobs.cancel(token, self.owner)
                    if action == "cancel"
                    else self.server.jobs.approve(token, self.owner)
                )
                self._json({"ok": ok}, 200 if ok else 404)
                return
            if self.path not in {
                "/api/analyze-video",
                "/api/analyze",
                "/api/detect",
                "/api/container-classify",
            }:
                self._json({"error": "Endpoint bulunamadı"}, 404)
                return
            slot = _upload_slots.acquire(blocking=False)
            if not slot:
                raise UploadError("Eşzamanlı yükleme sınırına ulaşıldı", 429)
            limit = (
                MAX_UPLOAD_BYTES
                if self.path == "/api/analyze-video"
                else MAX_IMAGE_BYTES
            )
            form = parse_form(self.rfile, self.headers, limit)
            if self.path == "/api/analyze-video":
                if "video" not in form or form["video"].size == 0:
                    raise UploadError("video alanı gerekli")
                gps = parse_gps(form["gps"].file) if "gps" in form else []
                self._json(
                    self.server.jobs.submit(self.owner, form["video"].file, gps), 202
                )
                return
            if "image" not in form or form["image"].size == 0:
                raise UploadError("image alanı gerekli")
            frame = cv2.imdecode(
                np.frombuffer(form["image"].file.read(), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if frame is None or frame.shape[0] * frame.shape[1] > 3840 * 2160:
                raise UploadError("Görsel okunamadı veya çözünürlük sınırını aşıyor")
            with MODEL_LOCK:
                privacy = PrivacyProcessor(ROOT)
                redacted, tracks = privacy.process(frame)
                if self.path == "/api/container-classify":
                    detections, note = container_predictions(frame)
                    result = {
                        "detections": detections,
                        "containers": detections,
                        "model": "Konteyner adayları",
                        "note": note,
                        "review_required": True,
                    }
                elif self.path == "/api/detect":
                    detections = boxes(
                        get_model("road").predict(frame, conf=0.10, verbose=False)[0],
                        "road_damage",
                    )
                    result = {
                        "detections": detections,
                        "road_damage": detections,
                        "model": "RDD2022 YOLO12s",
                        "review_required": True,
                    }
                else:
                    result = analyze_frame(frame)
                result["image_base64"] = jpeg_payload(
                    mask_regions(annotate(redacted, result), tracks)
                )
                result["privacy_status"] = "review_required"
                result["confidence_note"] = "Model skoru doğruluk yüzdesi değildir."
            self._json(result)
        except UploadError as exc:
            self.close_connection = True
            self._json({"error": str(exc)}, exc.status)
        except (ValueError, UnicodeError) as exc:
            self.close_connection = True
            self._json({"error": str(exc)}, 400)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            self.close_connection = True
        except Exception as exc:
            self.log_error("Analysis failed: %s", type(exc).__name__)
            self.close_connection = True
            self._json(
                {
                    "error": "Analiz tamamlanamadı; model ve sunucu kayıtlarını kontrol edin."
                },
                503,
            )
        finally:
            if form is not None:
                form.close()
            if slot:
                _upload_slots.release()

    def do_GET(self):
        try:
            self._session(create=not self.path.startswith(("/api/", "/video-results/")))
            if self.path == "/api/models":
                self._json(manifest())
                return
            match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})", self.path)
            if match:
                result = self.server.jobs.snapshot(match[1], self.owner)
                self._json(result or {"error": "İş bulunamadı"}, 200 if result else 404)
                return
            if self.path.startswith("/video-results/"):
                match = re.fullmatch(r"/video-results/([0-9a-f]{32})\.webm", self.path)
                path = self.server.jobs.output(match[1], self.owner) if match else None
                if path is None:
                    self._json({"error": "Video bulunamadı"}, 404)
                    return
                self._serve_video(path)
                return
            if self.path.startswith("/api/"):
                self._json({"error": "Endpoint bulunamadı"}, 404)
                return
            super().do_GET()
        except UploadError as exc:
            self._json({"error": str(exc)}, exc.status)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            self.close_connection = True

    def do_HEAD(self):
        self._json({"error": "HEAD desteklenmiyor"}, 405)

    def _serve_video(self, path):
        size = path.stat().st_size
        start, end = 0, size - 1
        request_range = self.headers.get("Range")
        if request_range:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", request_range)
            if not match or not any(match.groups()):
                self._json({"error": "Geçersiz Range"}, 416)
                return
            a, b = match.groups()
            if a:
                start, end = int(a), min(int(b), size - 1) if b else size - 1
            else:
                start = max(0, size - int(b))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        with path.open("rb") as stream:
            self.send_response(206 if request_range else 200)
            self.send_header("Content-Type", "video/webm")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if request_range:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address=("127.0.0.1", 8000), folder=VIDEO_DIR):
        super().__init__(address, Handler)
        self.jobs = JobManager(folder)

    def server_close(self):
        self.jobs.close()
        super().server_close()


def main():
    server = DemoServer()
    print(
        "Demo sitesi: http://127.0.0.1:8000 — yerel oturum; çıktı saklama süresi: 1 saat"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
