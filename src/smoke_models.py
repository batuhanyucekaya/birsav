"""Real weights + synthetic short video/API smoke; NOT an accuracy benchmark."""

import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import cv2
import numpy as np
import torch
from src.demo_server import DemoServer


def multipart(parts):
    boundary = "SmokeBoundaryBirsav"
    body = bytearray()
    for name, filename, content in parts:
        body.extend(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n\r\n'.encode()
        )
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def main():
    torch.set_num_threads(2)
    report = {
        "purpose": "real_weights_integration_on_synthetic_input_NOT_accuracy",
        "checks": {},
    }
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        video = folder / "input.avi"
        writer = cv2.VideoWriter(
            str(video), cv2.VideoWriter_fourcc(*"MJPG"), 5, (320, 240)
        )
        for i in range(5):
            writer.write(np.full((240, 320, 3), 50 + i * 20, np.uint8))
        writer.release()
        server = DemoServer(("127.0.0.1", 0), folder / "jobs")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        cookie = None

        def request(method, path, body=None, headers=None):
            c = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=120)
            h = {"Cookie": cookie} if cookie else {}
            h.update(headers or {})
            c.request(method, path, body, headers=h)
            r = c.getresponse()
            data = r.read()
            c.close()
            return r, data

        try:
            r, _ = request("GET", "/")
            cookie = r.getheader("Set-Cookie").split(";")[0]
            frame = cv2.imread("data/demo/road_damaged.jpg")
            if frame is None:
                frame = np.zeros((240, 320, 3), np.uint8)
            ok, encoded = cv2.imencode(".jpg", frame)
            for endpoint in ("/api/detect", "/api/analyze", "/api/container-classify"):
                body, headers = multipart([("image", "frame.jpg", encoded.tobytes())])
                start = time.perf_counter()
                r, data = request("POST", endpoint, body, headers)
                result = json.loads(data)
                if r.status != 200:
                    raise RuntimeError((endpoint, r.status, result))
                report["checks"][endpoint] = {
                    "http_status": r.status,
                    "elapsed_seconds": time.perf_counter() - start,
                    "image_returned": bool(result.get("image_base64")),
                    "issue_count": len(
                        result.get("issues", result.get("detections", []))
                    ),
                    "privacy_status": result.get("privacy_status"),
                }
            body, headers = multipart(
                [
                    ("video", "input.avi", video.read_bytes()),
                    (
                        "gps",
                        "gps.csv",
                        b"video_time_seconds,latitude,longitude\n0,41,29\n2,41.001,29.001\n",
                    ),
                ]
            )
            r, data = request("POST", "/api/analyze-video", body, headers)
            assert r.status == 202, (r.status, data)
            job = json.loads(data)
            start = time.perf_counter()
            r, _ = request("GET", "/")
            assert r.status == 200
            report["checks"]["static_during_video"] = True
            while job["state"] in ("queued", "processing"):
                if time.perf_counter() - start > 120:
                    raise TimeoutError("job")
                time.sleep(0.1)
                r, data = request("GET", f"/api/jobs/{job['job_id']}")
                job = json.loads(data)
            assert job["state"] == "review_required", job
            result = job["result"]
            r, output = request("GET", result["preview_url"])
            assert r.status == 200
            report["checks"]["video"] = {
                "state": job["state"],
                "result": result,
                "output_bytes": len(output),
                "raw_files_remaining": len(list((folder / "jobs").glob("*.input"))),
            }
            r, _ = request("POST", f"/api/jobs/{job['job_id']}/review", b"")
            assert r.status == 200
            r, _ = request("GET", result["preview_url"], headers={"Cookie": ""})
            assert r.status == 401
            body, headers = multipart([("video", "bad.mp4", b"broken video")])
            r, data = request("POST", "/api/analyze-video", body, headers)
            bad = json.loads(data)
            while bad["state"] in ("queued", "processing"):
                time.sleep(0.1)
                r, data = request("GET", f"/api/jobs/{bad['job_id']}")
                bad = json.loads(data)
            assert bad["state"] == "failed", bad
            report["checks"]["broken_video"] = {
                "state": bad["state"],
                "raw_files_remaining": len(list((folder / "jobs").glob("*.input"))),
            }
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
    Path("reports/real_model_smoke.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Real model integration smoke passed")


if __name__ == "__main__":
    main()
