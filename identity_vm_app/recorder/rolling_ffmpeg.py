from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

from identity_vm_app import settings as s

SegmentHook = Callable[[Path, float], int]


class RollingFfmpegRecorder:
    """
    Ghi RTSP → file segment rolling bằng ffmpeg.
    `segment_hook(path, started_ts) -> segment_db_id` gọi khi phát hiện file segment mới.
    """

    def __init__(
        self,
        camera_id: str,
        source_url: str,
        *,
        archive_root: Optional[Path] = None,
        segment_seconds: Optional[int] = None,
        ffmpeg_bin: Optional[str] = None,
        segment_hook: Optional[SegmentHook] = None,
    ):
        self.camera_id = camera_id
        self.source_url = source_url
        self.archive_root = Path(archive_root or s.IVM_ARCHIVE_ROOT) / camera_id
        self.segment_seconds = int(segment_seconds or s.IVM_SEGMENT_SECONDS)
        self.ffmpeg_bin = ffmpeg_bin or s.IVM_FFMPEG_BIN
        self._segment_hook = segment_hook
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()
        self._current_path: Optional[Path] = None
        self._current_segment_id: Optional[int] = None
        self._segment_started_utc: float = 0.0

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def current_archive_ref(self) -> Tuple[Optional[int], Optional[str], float, float]:
        with self._lock:
            sid = self._current_segment_id
            p = str(self._current_path) if self._current_path else None
            t0 = self._segment_started_utc
            return sid, p, t0, time.time()

    def start(self) -> None:
        self.archive_root.mkdir(parents=True, exist_ok=True)
        pattern = str(self.archive_root / f"{self.camera_id}_%Y%m%d_%H%M%S.mkv")
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-rtsp_transport",
            "tcp",
            "-i",
            self.source_url,
            "-an",
            "-c:v",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(self.segment_seconds),
            "-reset_timestamps",
            "1",
            "-strftime",
            "1",
            pattern,
        ]
        with self._lock:
            if self.is_running():
                return
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._current_path = None
            self._current_segment_id = None
            self._segment_started_utc = time.time()
        threading.Thread(target=self._watch_segments, daemon=True).start()

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=timeout)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def _watch_segments(self) -> None:
        last_path: Optional[Path] = None
        while self.is_running():
            try:
                files = sorted(
                    self.archive_root.glob(f"{self.camera_id}_*.mkv"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                newest = files[0] if files else None
                if newest and newest != last_path:
                    last_path = newest
                    started = float(newest.stat().st_mtime)
                    sid: Optional[int] = None
                    if self._segment_hook:
                        try:
                            sid = int(self._segment_hook(newest, started))
                        except Exception:
                            sid = None
                    with self._lock:
                        self._current_path = newest
                        self._segment_started_utc = started
                        self._current_segment_id = sid
            except Exception:
                pass
            time.sleep(1.0)
