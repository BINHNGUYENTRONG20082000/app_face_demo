from __future__ import annotations

import threading
from typing import Dict, List, Optional

from identity_vm_app.recorder.rolling_ffmpeg import RollingFfmpegRecorder


class RecorderRegistry:
    def __init__(self):
        self._lock = threading.RLock()
        self._by_camera: Dict[str, RollingFfmpegRecorder] = {}

    def get(self, camera_id: str) -> Optional[RollingFfmpegRecorder]:
        with self._lock:
            return self._by_camera.get(camera_id)

    def start(self, camera_id: str, recorder: RollingFfmpegRecorder) -> None:
        with self._lock:
            old = self._by_camera.get(camera_id)
            if old and old is not recorder:
                old.stop()
            self._by_camera[camera_id] = recorder
        recorder.start()

    def stop(self, camera_id: str) -> None:
        with self._lock:
            r = self._by_camera.pop(camera_id, None)
        if r:
            r.stop()

    def stop_all(self) -> None:
        with self._lock:
            items: List[RollingFfmpegRecorder] = list(self._by_camera.values())
            self._by_camera.clear()
        for r in items:
            r.stop()
