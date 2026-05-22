"""MJPEG qua `StableCameraReader` — tách khỏi `mjpeg_hub` để test module stream ổn định."""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from packages.camera_stream import StableCameraReader, StreamConnectionConfig

from identity_vm_app import settings as s


class NativePreviewHub:
    """Một reader OpenCV/`StableCameraReader` mỗi camera — dùng cho tab thử nghiệm UI."""

    def __init__(self) -> None:
        self._readers: Dict[str, StableCameraReader] = {}
        self._lock = threading.Lock()

    def _stream_config(self) -> StreamConnectionConfig:
        return StreamConnectionConfig(
            read_fails_before_reconnect=int(s.IVM_PREVIEW_READ_FAILS_BEFORE_RECONNECT),
            reconnect_delay_s=float(s.IVM_PREVIEW_RECONNECT_DELAY_S),
            open_backoff_base_s=float(s.IVM_PREVIEW_RECONNECT_DELAY_S),
            open_backoff_cap_s=float(s.IVM_PREVIEW_OPEN_BACKOFF_CAP_S),
            cap_buffer_size=int(s.IVM_CAP_PROP_BUFFERSIZE),
        )

    def ensure(self, camera_id: str, source: Any) -> StableCameraReader:
        with self._lock:
            old = self._readers.get(camera_id)
            if old is not None and old.is_running:
                return old
            if old is not None:
                old.stop()
                self._readers.pop(camera_id, None)
            r = StableCameraReader(camera_id, source, config=self._stream_config())
            r.start()
            self._readers[camera_id] = r
            return r

    def get(self, camera_id: str) -> Optional[StableCameraReader]:
        with self._lock:
            return self._readers.get(camera_id)

    def stop(self, camera_id: str) -> None:
        with self._lock:
            reader = self._readers.pop(camera_id, None)
        if reader is not None:
            reader.stop()

    def stop_all(self) -> None:
        with self._lock:
            items = list(self._readers.values())
            self._readers.clear()
        for r in items:
            r.stop()


_hub: Optional[NativePreviewHub] = None
_hub_lock = threading.Lock()


def get_native_preview_hub() -> NativePreviewHub:
    global _hub
    with _hub_lock:
        if _hub is None:
            _hub = NativePreviewHub()
        return _hub


def shutdown_native_preview_hub() -> None:
    global _hub
    with _hub_lock:
        if _hub is not None:
            _hub.stop_all()
            _hub = None
