"""
Đọc RTSP/HTTP bằng FFmpeg subprocess → BGR24 (ổn định hơn OpenCV khi chạy lâu).

Cùng hợp đồng với StableCameraReader: get_frame, callback, reconnect.
"""

from __future__ import annotations

import logging
import os
import random
import subprocess
import threading
import time
from typing import Any, Callable, Optional

import cv2
import numpy as np

from identity_vm_app import settings as s
from packages.camera_stream.config import StreamConnectionConfig
from packages.camera_stream.ffmpeg_common import (
    cached_ffprobe_stream_size,
    even_dims,
    ffmpeg_bin,
    ffmpeg_cli_available,
    popen_kwargs,
    rtsp_url,
    scale_dims_keep_aspect,
    terminate_process,
)
from packages.camera_stream.opencv_reader import FrameDecodedCallback

logger = logging.getLogger("camera_stream.ffmpeg_reader")


class FfmpegCameraReader:
    """Thread đọc FFmpeg rawvideo; frame mới nhất + callback sample."""

    def __init__(
        self,
        camera_id: str,
        source: Any,
        *,
        config: Optional[StreamConnectionConfig] = None,
    ) -> None:
        self.camera_id = str(camera_id)
        url = rtsp_url(source)
        if not url:
            raise ValueError(f"[{camera_id}] FfmpegCameraReader cần URL rtsp/http")
        self.source = url
        self.config = config or StreamConnectionConfig()

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_decoded_cb: Optional[FrameDecodedCallback] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None

        self._last_error: Optional[str] = None
        self.frame_count = 0
        self.connect_success_count = 0
        self._fps_ts = time.monotonic()
        self._fps_cnt = 0
        self.fps_actual = 0.0
        self._last_good_mono = time.monotonic()
        self._session_opened_mono = time.monotonic()
        self._stream_connected = False

    @property
    def is_connected(self) -> bool:
        return bool(self._stream_connected)

    @property
    def is_running(self) -> bool:
        return self._running

    def last_error(self) -> Optional[str]:
        return self._last_error

    def set_frame_decoded_callback(self, cb: Optional[FrameDecodedCallback]) -> None:
        with self._lock:
            self._frame_decoded_cb = cb

    def start(self) -> None:
        if self._running:
            return
        if not ffmpeg_cli_available():
            self._last_error = "Không tìm thấy ffmpeg (IVM_FFMPEG_BIN)"
            logger.error("[%s] %s", self.camera_id, self._last_error)
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name=f"cam-ffmpeg-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_proc()
        if self._thread is not None:
            self._thread.join(timeout=12.0)
            self._thread = None
        with self._lock:
            self._frame = None
        self._stream_connected = False

    def _stop_proc(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is not None:
            terminate_process(proc)
        self._stream_connected = False

    def _run(self) -> None:
        cfg = self.config
        url = self.source
        open_failures = 0
        max_h = int(getattr(s, "IVM_CAMERA_FFMPEG_READ_MAX_HEIGHT", 0) or 0)
        timeout_us = str(int(os.getenv("IVM_FFMPEG_STIMEOUT_US", "8000000")))

        while self._running:
            wh = cached_ffprobe_stream_size(url)
            if not wh:
                open_failures += 1
                self._last_error = f"ffprobe không đọc được stream (lần {open_failures})"
                exp = min(
                    cfg.open_backoff_cap_s,
                    cfg.open_backoff_base_s * (1.45 ** min(open_failures, 12)),
                )
                time.sleep(min(cfg.open_backoff_cap_s, exp + random.uniform(0, 0.75)))
                continue

            sw, sh = wh
            w, h = scale_dims_keep_aspect(sw, sh, max_h)
            frame_bytes = w * h * 3

            cmd = [
                ffmpeg_bin(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-rtsp_transport",
                "tcp",
                "-timeout",
                timeout_us,
                "-fflags",
                "+discardcorrupt+genpts",
                "-i",
                url,
                "-an",
                "-map",
                "0:v:0",
                "-vf",
                f"scale={w}:{h}:flags=fast_bilinear,format=bgr24",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ]

            proc: Optional[subprocess.Popen] = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    **popen_kwargs(bufsize=frame_bytes * 2),
                )
                with self._lock:
                    self._proc = proc

                if proc.stderr:

                    def _drain_stderr() -> None:
                        try:
                            if proc and proc.stderr:
                                proc.stderr.read()
                        except Exception:
                            pass

                    threading.Thread(target=_drain_stderr, daemon=True).start()

                self._last_error = None
                open_failures = 0
                self.connect_success_count += 1
                self._session_opened_mono = time.monotonic()
                self._stream_connected = True
                logger.info(
                    "[%s] FFmpeg RTSP kết nối OK (lần %d, %dx%d)",
                    self.camera_id,
                    self.connect_success_count,
                    w,
                    h,
                )

                for _ in range(max(0, int(cfg.discard_frames_on_open))):
                    if not self._running or proc.stdout is None:
                        break
                    proc.stdout.read(frame_bytes)

                assert proc.stdout is not None
                read_streak = 0

                while self._running and proc.poll() is None:
                    now_mono = time.monotonic()
                    proactive_s = float(cfg.rtsp_proactive_reconnect_s)
                    if proactive_s > 0 and (now_mono - self._session_opened_mono) >= proactive_s:
                        logger.info(
                            "[%s] FFmpeg proactive reconnect sau %.0fs",
                            self.camera_id,
                            now_mono - self._session_opened_mono,
                        )
                        break

                    buf = b""
                    while len(buf) < frame_bytes and self._running:
                        chunk = proc.stdout.read(frame_bytes - len(buf))
                        if not chunk:
                            break
                        buf += chunk

                    if len(buf) != frame_bytes:
                        read_streak += 1
                        stale_s = float(cfg.rtsp_stale_reconnect_s)
                        idle_s = time.monotonic() - self._last_good_mono
                        if stale_s > 0 and idle_s >= stale_s:
                            logger.warning(
                                "[%s] FFmpeg không có frame %.0fs — reconnect",
                                self.camera_id,
                                idle_s,
                            )
                            self._last_error = f"FFmpeg stale {idle_s:.0f}s"
                            break
                        if read_streak >= max(1, int(cfg.read_fails_before_reconnect)):
                            self._last_error = "Thiếu byte rawvideo — reconnect"
                            break
                        time.sleep(cfg.read_error_sleep_s)
                        continue

                    read_streak = 0
                    self._last_good_mono = time.monotonic()
                    frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
                    self.frame_count += 1
                    self._update_fps()
                    fc = int(self.frame_count)
                    fps_hint = max(1.0, float(self.fps_actual) or 0.0)
                    frame_copy = frame.copy()
                    with self._lock:
                        self._frame = frame
                        cb = self._frame_decoded_cb
                    if cb is not None:
                        try:
                            cb(fc, frame_copy, fps_hint)
                        except Exception:
                            pass

            except Exception as ex:
                self._last_error = str(ex)
                open_failures += 1
                logger.warning("[%s] FFmpeg reader: %s", self.camera_id, ex)
            finally:
                self._stop_proc()

            if not self._running:
                break
            time.sleep(cfg.reconnect_delay_s + random.uniform(0, 1.0))

    def _update_fps(self) -> None:
        self._fps_cnt += 1
        now = time.monotonic()
        if now - self._fps_ts >= 1.0:
            self.fps_actual = float(self._fps_cnt)
            self._fps_cnt = 0
            self._fps_ts = now

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def get_jpeg(self, quality: int = 78) -> Optional[bytes]:
        frame = self.get_frame()
        if frame is None:
            return None
        q = max(40, min(95, int(quality)))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return buf.tobytes() if ok else None
