"""Bounded single-worker queue. Ephemeral jobs/files belong to a local session."""

from collections import deque
from pathlib import Path
import threading
import time
import uuid
import shutil
from src.video_pipeline import process_video, Cancelled


class JobManager:
    def __init__(self, folder, ttl=3600, max_jobs=3):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.ttl, self.max_jobs = ttl, max_jobs
        self.jobs = {}
        self.pending = deque()
        self.stopped = False
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.worker = threading.Thread(
            target=self._run, name="video-worker", daemon=True
        )
        self.worker.start()

    def submit(self, owner, source, gps_rows):
        with self.condition:
            self.cleanup()
            if (
                sum(j["state"] in {"queued", "processing"} for j in self.jobs.values())
                >= self.max_jobs
            ):
                raise ValueError("Video kuyruğu dolu")
            token = uuid.uuid4().hex
            raw = self.folder / f"{token}.input"
            source.seek(0, 2)
            size = source.tell()
            source.seek(0)
            if shutil.disk_usage(self.folder).free < size * 2 + 128 * 1024 * 1024:
                raise ValueError("Video ve çıktı için yeterli boş disk alanı yok")
            try:
                with raw.open("wb") as stream:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        stream.write(chunk)
            except Exception:
                raw.unlink(missing_ok=True)
                raise
            self.jobs[token] = {
                "job_id": token,
                "owner": owner,
                "state": "queued",
                "created": time.time(),
                "cancel": threading.Event(),
                "raw": raw,
                "output": self.folder / f"{token}.webm",
                "gps": gps_rows,
                "frames_done": 0,
                "frames_total": None,
                "result": None,
                "error": None,
            }
            self.pending.append(token)
            self.condition.notify()
            return self.snapshot(token, owner)

    def snapshot(self, token, owner):
        with self.lock:
            job = self.jobs.get(token)
            if not job or job["owner"] != owner:
                return None
            return {
                k: job[k]
                for k in (
                    "job_id",
                    "state",
                    "frames_done",
                    "frames_total",
                    "result",
                    "error",
                )
            }

    def cancel(self, token, owner):
        with self.condition:
            job = self.jobs.get(token)
            if not job or job["owner"] != owner:
                return False
            job["cancel"].set()
            if job["state"] != "processing":
                job.update(state="cancelled", result=None)
                job["raw"].unlink(missing_ok=True)
                job["output"].unlink(missing_ok=True)
            self.condition.notify()
            return True

    def approve(self, token, owner):
        with self.lock:
            job = self.jobs.get(token)
            if not job or job["owner"] != owner or job["state"] != "review_required":
                return False
            job["state"] = "completed"
            job["result"].update(
                privacy_status="user_reviewed",
                review_required=False,
                video_url=f"/video-results/{token}.webm",
            )
            return True

    def output(self, token, owner):
        with self.lock:
            self.cleanup()
            job = self.jobs.get(token)
            if (
                not job
                or job["owner"] != owner
                or job["state"] not in {"review_required", "completed"}
            ):
                return None
            return job["output"] if job["output"].is_file() else None

    def cleanup(self):
        with self.lock:
            cutoff = time.time() - self.ttl
            for token, job in list(self.jobs.items()):
                if job["created"] < cutoff:
                    if job["state"] == "processing":
                        job["cancel"].set()
                    else:
                        job["raw"].unlink(missing_ok=True)
                        job["output"].unlink(missing_ok=True)
                        del self.jobs[token]
            active = {j["raw"] for j in self.jobs.values()} | {
                j["output"] for j in self.jobs.values()
            }
            for path in self.folder.iterdir():
                if (
                    path.is_file()
                    and path not in active
                    and path.stat().st_mtime < cutoff
                ):
                    path.unlink(missing_ok=True)

    def _run(self):
        while True:
            with self.condition:
                self.cleanup()
                if self.stopped:
                    return
                if not self.pending:
                    self.condition.wait(timeout=30)
                    continue
                token = self.pending.popleft()
                job = self.jobs.get(token)
                if not job or job["state"] != "queued":
                    continue
                job["state"] = "processing"

            def progress(done, total):
                with self.lock:
                    job.update(frames_done=done, frames_total=total)

            try:
                result = process_video(
                    job["raw"], job["output"], job["gps"], job["cancel"], progress
                )
                with self.lock:
                    if job["cancel"].is_set():
                        raise Cancelled()
                    result["preview_url"] = f"/video-results/{token}.webm"
                    job.update(state="review_required", result=result)
            except Cancelled:
                with self.lock:
                    job.update(state="cancelled", result=None)
            except Exception as exc:
                with self.lock:
                    job.update(
                        state="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        result=None,
                    )
            finally:
                job["raw"].unlink(missing_ok=True)
                if job["state"] in {"failed", "cancelled"}:
                    job["output"].unlink(missing_ok=True)

    def close(self):
        with self.condition:
            self.stopped = True
            for job in self.jobs.values():
                job["cancel"].set()
                if job["state"] == "queued":
                    job["raw"].unlink(missing_ok=True)
                    job["state"] = "cancelled"
            self.condition.notify_all()
        self.worker.join(timeout=10)
