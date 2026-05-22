"""Mỗi camera một thread đọc RTSP, cập nhật JPEG mới nhất — không gọi InsightFace / không chặn UI."""

from __future__ import annotations

import os
import random
import shutil
import threading
import time
from typing import Any, Dict, List, Optional

from identity_vm_app import settings as s
from identity_vm_app.preview.ffmpeg_env import apply_ffmpeg_capture_env

apply_ffmpeg_capture_env()

import cv2

from identity_vm_app.preview.ffmpeg_rtsp_bgr import preview_should_use_ffmpeg, run_ffmpeg_preview_loop


def _ffmpeg_cli_available() -> bool:
    fb = s.IVM_FFMPEG_BIN
    return bool(shutil.which(fb) or (os.path.isfile(fb) if fb else False))


class CameraPreviewStream:
    def __init__(
        self,
        camera_id: str,
        source: Any,
        *,
        jpeg_quality: int = 78,
        capture_fps: float = 20.0,
        reconnect_delay_s: float = 2.5,
        read_fails_before_reconnect: int = 4,
        open_backoff_cap_s: float = 60.0,
        cap_prop_buffersize: int = 2,
        use_ffmpeg: Optional[bool] = None,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.jpeg_quality = int(jpeg_quality)
        self._period = 1.0 / max(5.0, min(60.0, float(capture_fps)))
        self._reconnect_delay = max(0.1, float(reconnect_delay_s))
        self._read_fails_reconnect = max(1, int(read_fails_before_reconnect))
        self._open_backoff_cap = max(5.0, float(open_backoff_cap_s))
        self._cap_buffer = max(1, min(16, int(cap_prop_buffersize)))
        if use_ffmpeg is None:
            self._use_ffmpeg = bool(preview_should_use_ffmpeg(source) and _ffmpeg_cli_available())
        else:
            self._use_ffmpeg = bool(use_ffmpeg and preview_should_use_ffmpeg(source) and _ffmpeg_cli_available())

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._jpeg_fb: List[bytes] = [b""]
        self._err_msg: List[Optional[str]] = [None]
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_ev = threading.Event()

    def start(self) -> None:
        if self._running:
            return
        self._stop_ev.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, name=f"ivm-preview-{self.camera_id}", daemon=True)
        self._thread.start()

    def _release_cap(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = None

    def _open_cap(self) -> bool:
        self._release_cap()
        try:
            if isinstance(self.source, int):
                cap = cv2.VideoCapture(self.source)
            elif isinstance(self.source, str) and self.source.strip().isdigit():
                cap = cv2.VideoCapture(int(self.source.strip()))
            else:
                cap = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG)
            if cap is not None:
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, self._cap_buffer)
                except Exception:
                    pass
            if cap is None or not cap.isOpened():
                self._err_msg[0] = "Không mở được nguồn video"
                return False
            self._cap = cap
            for _ in range(8):
                self._cap.grab()
            self._err_msg[0] = None
            return True
        except Exception as e:
            self._err_msg[0] = str(e)
            return False

    def _run(self) -> None:
        if self._use_ffmpeg:
            run_ffmpeg_preview_loop(
                str(self.source),
                jpeg_quality=self.jpeg_quality,
                capture_period_s=self._period,
                reconnect_delay_s=self._reconnect_delay,
                open_backoff_cap_s=self._open_backoff_cap,
                max_height=int(s.IVM_PREVIEW_FFMPEG_MAX_HEIGHT),
                stop_event=self._stop_ev,
                lock=self._lock,
                last_jpeg_out=self._jpeg_fb,
                error_out=self._err_msg,
                camera_id=self.camera_id,
            )
        else:
            self._run_opencv()
        self._running = False

    def _run_opencv(self) -> None:
        open_failures = 0
        read_streak = 0

        while self._running and not self._stop_ev.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._open_cap():
                    open_failures += 1
                    exp = min(
                        self._open_backoff_cap,
                        self._reconnect_delay * (1.45 ** min(open_failures, 12)),
                    )
                    delay = min(self._open_backoff_cap, exp + random.uniform(0, 0.75))
                    self._err_msg[0] = f"Chờ kết nối lại (thử {open_failures})"
                    time.sleep(delay)
                    continue
                open_failures = 0
                read_streak = 0

            assert self._cap is not None
            ok, frame = self._cap.read()
            good = bool(ok and frame is not None and getattr(frame, "size", 0) > 0)
            if good:
                read_streak = 0
                enc_ok, buf = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if enc_ok:
                    with self._lock:
                        self._jpeg_fb[0] = buf.tobytes()
                time.sleep(self._period)
                continue

            read_streak += 1
            if read_streak < self._read_fails_reconnect:
                time.sleep(0.05)
                continue

            read_streak = 0
            self._err_msg[0] = "Mất frame — đang kết nối lại"
            self._release_cap()
            time.sleep(self._reconnect_delay + random.uniform(0, 1.0))

        self._release_cap()

    def stop(self) -> None:
        self._running = False
        self._stop_ev.set()
        if self._thread is not None:
            self._thread.join(timeout=12.0)
            self._thread = None

    def get_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg_fb[0]

    def error_message(self) -> Optional[str]:
        return self._err_msg[0]


class PreviewHub:
    def __init__(self) -> None:
        self._streams: Dict[str, CameraPreviewStream] = {}
        self._lock = threading.Lock()
        self._armed = False

    def is_armed(self) -> bool:
        with self._lock:
            return bool(self._armed)

    def get_running(self, camera_id: str) -> Optional[CameraPreviewStream]:
        with self._lock:
            st = self._streams.get(camera_id)
            if st is not None and st._running:
                return st
            return None

    def ensure(
        self,
        camera_id: str,
        source: Any,
        *,
        jpeg_quality: int,
        capture_fps: float,
        reconnect_delay_s: float = 2.5,
        read_fails_before_reconnect: int = 4,
        open_backoff_cap_s: float = 60.0,
        cap_prop_buffersize: int = 2,
        use_ffmpeg: Optional[bool] = None,
    ) -> CameraPreviewStream:
        with self._lock:
            old = self._streams.get(camera_id)
            if old is not None and old._running:
                return old
            if old is not None:
                old.stop()
                self._streams.pop(camera_id, None)
            st = CameraPreviewStream(
                camera_id,
                source,
                jpeg_quality=jpeg_quality,
                capture_fps=capture_fps,
                reconnect_delay_s=reconnect_delay_s,
                read_fails_before_reconnect=read_fails_before_reconnect,
                open_backoff_cap_s=open_backoff_cap_s,
                cap_prop_buffersize=cap_prop_buffersize,
                use_ffmpeg=use_ffmpeg,
            )
            st.start()
            self._streams[camera_id] = st
            return st

    def stop(self, camera_id: str) -> None:
        with self._lock:
            s = self._streams.pop(camera_id, None)
        if s is not None:
            s.stop()

    def stop_all(self) -> None:
        with self._lock:
            self._armed = False
            items = list(self._streams.values())
            self._streams.clear()
        for s in items:
            s.stop()

    def warm_all(
        self,
        sources: Dict[str, Any],
        *,
        jpeg_quality: int,
        capture_fps: float,
        reconnect_delay_s: float = 2.5,
        read_fails_before_reconnect: int = 4,
        open_backoff_cap_s: float = 60.0,
        cap_prop_buffersize: int = 2,
        stagger_s: float = 0.35,
    ) -> int:
        """Bật reader cho mọi camera — stagger để không dồn ffprobe/FFmpeg cùng lúc."""
        with self._lock:
            self._armed = True
        items = list(sources.items())
        for i, (camera_id, source) in enumerate(items):
            self.ensure(
                str(camera_id),
                source,
                jpeg_quality=jpeg_quality,
                capture_fps=capture_fps,
                reconnect_delay_s=reconnect_delay_s,
                read_fails_before_reconnect=read_fails_before_reconnect,
                open_backoff_cap_s=open_backoff_cap_s,
                cap_prop_buffersize=cap_prop_buffersize,
            )
            if i + 1 < len(items) and stagger_s > 0:
                time.sleep(stagger_s)
        return len(items)


_hub: Optional[PreviewHub] = None
_hub_lock = threading.Lock()


def get_preview_hub() -> PreviewHub:
    global _hub
    with _hub_lock:
        if _hub is None:
            _hub = PreviewHub()
        return _hub


def shutdown_preview_hub() -> None:
    global _hub
    with _hub_lock:
        if _hub is not None:
            _hub.stop_all()
            _hub = None
