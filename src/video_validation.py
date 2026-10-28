"""Full decode check; optional ffprobe verifies codec and absence of audio."""

import json
import math
import shutil
import subprocess
import cv2


def validate_video(path, expected_frames, expected_size, expected_fps):
    capture = cv2.VideoCapture(str(path))
    count = 0
    try:
        if not capture.isOpened():
            raise ValueError("Çıktı videosu açılamadı")
        fps = capture.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != tuple(expected_size):
                raise ValueError("Çıktı çözünürlüğü değişti")
            count += 1
    finally:
        capture.release()
    if count != expected_frames or count == 0:
        raise ValueError(f"Eksik çıktı kareleri: {count}/{expected_frames}")
    if not math.isfinite(fps) or abs(fps - expected_fps) > max(
        0.1, expected_fps * 0.01
    ):
        raise ValueError("Çıktı FPS değeri uyuşmuyor")
    result = {
        "fully_decoded": True,
        "frames": count,
        "fps": fps,
        "duration_seconds": count / fps,
        "ffprobe": None,
    }
    executable = shutil.which("ffprobe")
    if executable:
        completed = subprocess.run(
            [
                executable,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        probe = json.loads(completed.stdout)
        if any(s.get("codec_type") == "audio" for s in probe["streams"]):
            raise ValueError("Çıktıda ses akışı var")
        result["ffprobe"] = probe
    return result
