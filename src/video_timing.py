"""Resample already-redacted frames onto a constant-rate timeline using input PTS."""

from collections import deque
import math
import statistics


class CfrResampler:
    def __init__(self, writer, fps, cancel=None):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid output FPS")
        self.writer, self.fps, self.cancel = writer, fps, cancel
        self.previous = None
        self.previous_time = None
        self.intervals = deque(maxlen=120)
        self.written = self.duplicated = self.dropped = 0
        self.minimum_delta = float("inf")
        self.maximum_delta = 0.0

    def _emit_until(self, end):
        emitted = 0
        while self.written / self.fps < end - 1e-8:
            if self.cancel:
                self.cancel()
            self.writer.write(self.previous)
            self.written += 1
            emitted += 1
        if emitted == 0:
            self.dropped += 1
        else:
            self.duplicated += max(0, emitted - 1)

    def push(self, frame, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid frame timestamp")
        if self.previous is not None:
            delta = seconds - self.previous_time
            if delta <= 0:
                raise ValueError("Non-monotonic frame timestamps")
            self.intervals.append(delta)
            self.minimum_delta = min(self.minimum_delta, delta)
            self.maximum_delta = max(self.maximum_delta, delta)
            self._emit_until(seconds)
        elif abs(seconds) > 1e-8:
            raise ValueError("Timeline must start at zero")
        self.previous, self.previous_time = frame, seconds

    def finish(self):
        if self.previous is None:
            raise ValueError("Video has no frames")
        # OpenCV does not expose the last packet duration; record its estimate explicitly.
        tail = statistics.median(self.intervals) if self.intervals else 1 / self.fps
        end = self.previous_time + tail
        self._emit_until(end)
        variable = bool(
            self.intervals
            and self.maximum_delta - self.minimum_delta > max(0.005, 0.1 / self.fps)
        )
        return {
            "mode": "pts_resampled_to_cfr",
            "variable_intervals_detected": variable,
            "output_frames": self.written,
            "duplicated_frames": self.duplicated,
            "dropped_frames": self.dropped,
            "estimated_source_duration_seconds": end,
            "output_duration_seconds": self.written / self.fps,
            "duration_rounding_error_seconds": self.written / self.fps - end,
            "last_frame_duration_source": (
                "median_of_last_positive_pts_intervals"
                if self.intervals
                else "nominal_fps"
            ),
        }
