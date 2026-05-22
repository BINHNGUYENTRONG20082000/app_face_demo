"""
Đọc camera / RTSP ổn định: một thread, buffer frame mới nhất, reconnect có backoff.

Không import nhận diện / API — chỉ OpenCV + NumPy.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger("camera_stream.opencv_reader")

try:
    from identity_vm_app.preview.ffmpeg_env import apply_ffmpeg_capture_env

    apply_ffmpeg_capture_env()
except ImportError:
    _FALLBACK_FFMPEG = (
        "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|reorder_queue_size;0|stimeout;8000000"
    )
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _FALLBACK_FFMPEG)

import cv2
import numpy as np

from packages.camera_stream.config import StreamConnectionConfig

FrameDecodedCallback = Callable[[int, np.ndarray, float], None]


class StableCameraReader:
    """
    Thread nền đọc liên tục; luồng khác lấy `get_frame()` hoặc `get_jpeg()`.

    Hợp đồng: nhận diện là tầng trên — chỉ đọc khi `get_frame()` không None.
    """

    def __init__(
        self,
        camera_id: str,
        source: Any,
        *,
        config: Optional[StreamConnectionConfig] = None,
    ) -> None:
        self.camera_id = str(camera_id)
        self.source = source
        self.config = config or StreamConnectionConfig()

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_decoded_cb: Optional[FrameDecodedCallback] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._last_error: Optional[str] = None
        self.frame_count = 0
        self.connect_success_count = 0
        self._fps_ts = time.monotonic()
        self._fps_cnt = 0
        self.fps_actual = 0.0
        self._last_good_mono = time.monotonic()
        self._session_opened_mono = time.monotonic()

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and bool(self._cap.isOpened())

    @property
    def is_running(self) -> bool:
        return self._running

    def last_error(self) -> Optional[str]:
        return self._last_error

    def set_frame_decoded_callback(
        self, cb: Optional[FrameDecodedCallback]
    ) -> None:
        """Gọi trong thread đọc ngay khi có frame mới (trước khi worker infer)."""
        with self._lock:
            self._frame_decoded_cb = cb

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name=f"cam-stream-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=12.0)
            self._thread = None
        self._release_cap()
        with self._lock:
            self._frame = None

    def _release_cap(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _open_cap(self) -> bool:
        self._release_cap()
        cfg = self.config
        try:
            src = self.source
            if isinstance(src, int):
                cap = cv2.VideoCapture(src)
            elif isinstance(src, str) and src.strip().isdigit():
                cap = cv2.VideoCapture(int(src.strip()))
            else:
                cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG)
            if cap is not None:
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(cfg.cap_buffer_size))
                except Exception:
                    pass
            if cap is None or not cap.isOpened():
                self._last_error = "Không mở được nguồn"
                return False
            self._cap = cap
            for _ in range(max(0, int(cfg.discard_frames_on_open))):
                self._cap.grab()
            self._last_error = None
            self.connect_success_count += 1
            self._session_opened_mono = time.monotonic()
            logger.info(
                "[%s] RTSP kết nối lại OK (lần %d)",
                self.camera_id,
                self.connect_success_count,
            )
            return True
        except Exception as e:
            self._last_error = str(e)
            return False

    def _run(self) -> None:
        cfg = self.config
        open_failures = 0
        read_streak = 0

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if not self._open_cap():
                    open_failures += 1
                    exp = min(
                        cfg.open_backoff_cap_s,
                        cfg.open_backoff_base_s * (1.45 ** min(open_failures, 12)),
                    )
                    delay = min(cfg.open_backoff_cap_s, exp + random.uniform(0, 0.75))
                    self._last_error = f"Chờ mở lại nguồn (thử {open_failures})"
                    time.sleep(delay)
                    continue
                open_failures = 0
                read_streak = 0

            assert self._cap is not None
            ok, frame = self._cap.read()
            good = bool(ok and frame is not None and getattr(frame, "size", 0) > 0)
            if good:
                read_streak = 0
                now_mono = time.monotonic()
                proactive_s = float(cfg.rtsp_proactive_reconnect_s)
                if (
                    proactive_s > 0
                    and (now_mono - self._session_opened_mono) >= proactive_s
                ):
                    logger.info(
                        "[%s] RTSP proactive reconnect sau %.0fs phiên (tránh mất kết nối chạy lâu)",
                        self.camera_id,
                        now_mono - self._session_opened_mono,
                    )
                    self._release_cap()
                    time.sleep(cfg.reconnect_delay_s + random.uniform(0, 0.5))
                    continue
                self._last_good_mono = now_mono
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
                continue

            stale_s = float(cfg.rtsp_stale_reconnect_s)
            idle_s = time.monotonic() - self._last_good_mono
            if stale_s > 0 and idle_s >= stale_s:
                logger.warning(
                    "[%s] RTSP không có frame %.0fs — reconnect (VLC vẫn xem được = kết nối cũ chết)",
                    self.camera_id,
                    idle_s,
                )
                self._last_error = f"RTSP stale {idle_s:.0f}s — reconnect"
                self._release_cap()
                time.sleep(cfg.reconnect_delay_s + random.uniform(0, 1.0))
                read_streak = 0
                continue

            read_streak += 1
            if read_streak < max(1, int(cfg.read_fails_before_reconnect)):
                time.sleep(cfg.read_error_sleep_s)
                continue

            read_streak = 0
            logger.warning("[%s] Mất frame liên tiếp — reconnect RTSP", self.camera_id)
            self._last_error = "Mất frame — reconnect"
            self._release_cap()
            time.sleep(cfg.reconnect_delay_s + random.uniform(0, 1.0))

        self._release_cap()

    def _update_fps(self) -> None:
        self._fps_cnt += 1
        now = time.monotonic()
        if now - self._fps_ts >= 1.0:
            self.fps_actual = float(self._fps_cnt)
            self._fps_cnt = 0
            self._fps_ts = now

    def get_frame(self) -> Optional[np.ndarray]:
        """BGR frame gần nhất (bản copy an toàn cho infer)."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def get_jpeg(self, quality: int = 78) -> Optional[bytes]:
        """JPEG từ frame gần nhất — dùng cho preview/MJPEG, không infer."""
        frame = self.get_frame()
        if frame is None:
            return None
        q = max(40, min(95, int(quality)))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return buf.tobytes() if ok else None
