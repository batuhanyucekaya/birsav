"""HTTP integration tests; fake inference is explicitly separated from model tests."""

import http.client
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from src.demo_server import DemoServer, MAX_UPLOAD_BYTES


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.server = DemoServer(("127.0.0.1", 0), self.tmp.name)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]
        response, body = self.request("GET", "/")
        self.cookie = response.getheader("Set-Cookie").split(";")[0]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        h = {"Cookie": self.cookie} if getattr(self, "cookie", None) else {}
        h.update(headers or {})
        connection.request(method, path, body=body, headers=h)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response, data

    def test_session_and_host_restrictions(self):
        r, _ = self.request("GET", "/api/models", headers={"Cookie": ""})
        self.assertEqual(r.status, 401)
        r, _ = self.request("GET", "/", headers={"Host": "evil.example"})
        self.assertEqual(r.status, 403)
        r, _ = self.request(
            "POST", "/api/analyze", headers={"Origin": "http://evil.example"}
        )
        self.assertEqual(r.status, 403)

    def test_limit_checked_before_body(self):
        r, _ = self.request(
            "POST",
            "/api/analyze-video",
            body=b"",
            headers={"Content-Length": str(MAX_UPLOAD_BYTES + 1)},
        )
        self.assertEqual(r.status, 413)

    def test_invalid_image_is_controlled_error(self):
        body = b'--x\r\nContent-Disposition: form-data; name="image"; filename="a.jpg"\r\n\r\nnot-image\r\n--x--\r\n'
        r, _ = self.request(
            "POST",
            "/api/analyze",
            body,
            {"Content-Type": "multipart/form-data; boundary=x"},
        )
        self.assertEqual(r.status, 400)

    def test_other_session_cannot_access_job_and_video(self):
        token = "a" * 32
        path = Path(self.tmp.name) / f"{token}.webm"
        path.write_bytes(b"1234567890")
        owner = self.cookie.split("=")[1]
        self.server.jobs.jobs[token] = {
            "job_id": token,
            "owner": owner,
            "state": "review_required",
            "created": time.time(),
            "raw": Path(self.tmp.name) / f"{token}.input",
            "output": path,
            "cancel": threading.Event(),
            "frames_done": 1,
            "frames_total": 1,
            "result": {},
            "error": None,
        }
        r, data = self.request(
            "GET", f"/video-results/{token}.webm", headers={"Range": "bytes=2-5"}
        )
        self.assertEqual(r.status, 206)
        self.assertEqual(data, b"3456")
        r, _ = self.request("GET", f"/api/jobs/{token}", headers={"Cookie": ""})
        self.assertEqual(r.status, 401)
        r, _ = self.request("GET", "/", headers={"Cookie": ""})
        other = r.getheader("Set-Cookie").split(";")[0]
        for url in (f"/api/jobs/{token}", f"/video-results/{token}.webm"):
            r, _ = self.request("GET", url, headers={"Cookie": other})
            self.assertEqual(r.status, 404)

    def test_queue_cancel_review_and_cleanup(self):
        gate = threading.Event()

        def fake_process(raw, output, gps, cancel, progress):
            gate.wait(3)
            output.write_bytes(b"fake video")
            return {"review_required": True}

        owner = self.cookie.split("=")[1]
        with patch("src.jobs.process_video", side_effect=fake_process):
            first = self.server.jobs.submit(owner, io.BytesIO(b"fake"), [])
            second = self.server.jobs.submit(owner, io.BytesIO(b"fake"), [])
            self.assertTrue(self.server.jobs.cancel(second["job_id"], owner))
            r, _ = self.request("GET", "/")
            self.assertEqual(r.status, 200)
            gate.set()
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline:
                status = self.server.jobs.snapshot(first["job_id"], owner)
                if status["state"] == "review_required":
                    break
                time.sleep(0.01)
            self.assertEqual(status["state"], "review_required")
            self.assertFalse(list(Path(self.tmp.name).glob("*.input")))
            self.assertTrue(self.server.jobs.approve(first["job_id"], owner))
            self.assertEqual(
                self.server.jobs.snapshot(first["job_id"], owner)["state"], "completed"
            )
            self.server.jobs.jobs[first["job_id"]]["created"] = 0
            self.server.jobs.cleanup()
            self.assertFalse(list(Path(self.tmp.name).glob("*.webm")))


if __name__ == "__main__":
    unittest.main()
