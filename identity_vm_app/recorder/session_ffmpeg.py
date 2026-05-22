"""Ghi full luồng camera trong phiên nhận diện live — tách khỏi infer."""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from identity_vm_app import settings as s
from identity_vm_app.services.video_analyze_split import probe_duration_s

logger = logging.getLogger("session_ffmpeg")


def _resolve_ffprobe(ffmpeg_bin: str) -> str:
    fb = Path(ffmpeg_bin)
    if fb.is_file():
        sibling = fb.with_name("ffprobe.exe" if fb.suffix.lower() == ".exe" else "ffprobe")
        if sibling.is_file():
            return str(sibling)
    for name in ("ffprobe.exe", "ffprobe"):
        w = shutil.which(name)
        if w:
            return w
    return "ffprobe"


def _is_stream_url(source: Any) -> bool:
    if not isinstance(source, str):
        return False
    u = source.strip().lower()
    return u.startswith("rtsp://") or u.startswith("http://") or u.startswith("https://")


class SessionFfmpegRecorder:
    """Ghi RTSP/HTTP → MKV (copy) song song OpenCV reader; remux MP4 khi đóng."""

    def __init__(
        self,
        camera_id: str,
        source_url: str,
        raw_mkv: Path,
        *,
        final_mp4: Path,
        ffmpeg_bin: Optional[str] = None,
    ) -> None:
        self.camera_id = camera_id
        self.source_url = str(source_url)
        self.raw_mkv = Path(raw_mkv)
        self.final_mp4 = Path(final_mp4)
        self.ffmpeg_bin = ffmpeg_bin or s.IVM_FFMPEG_BIN
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()
        self.started_utc = time.time()

    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        self.raw_mkv.parent.mkdir(parents=True, exist_ok=True)
        if self.raw_mkv.is_file():
            try:
                self.raw_mkv.unlink()
            except OSError:
                pass
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
            "matroska",
            str(self.raw_mkv),
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
        self.started_utc = time.time()
        logger.info("[%s] Ghi full stream ffmpeg → %s", self.camera_id, self.raw_mkv)

    def stop(self, timeout: float = 8.0) -> Dict[str, Any]:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=timeout)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        ended = time.time()
        return self._finalize(ended)

    def close(self) -> Dict[str, Any]:
        return self.stop()

    def _finalize(self, ended_utc: float) -> Dict[str, Any]:
        browser_path: Optional[str] = None
        exists = self.raw_mkv.is_file() and self.raw_mkv.stat().st_size > 0
        out_path = self.final_mp4
        duration_s = max(0.0, ended_utc - self.started_utc)

        if exists:
            try:
                self._remux_to_mp4(self.raw_mkv, out_path)
                if out_path.is_file() and out_path.stat().st_size > 0:
                    probed = probe_duration_s(out_path)
                    if probed > 0:
                        duration_s = probed
                    try:
                        from identity_vm_app.services.visual_mp4 import remux_visual_mp4_for_browser

                        web = remux_visual_mp4_for_browser(out_path)
                        browser_path = str(web)
                    except Exception as ex:
                        logger.warning("[%s] Remux web MP4: %s", self.camera_id, ex)
            except Exception as ex:
                logger.warning("[%s] Remux session MP4: %s", self.camera_id, ex)
                out_path = self.raw_mkv

        return {
            "mode": "ffmpeg",
            "started_utc": self.started_utc,
            "ended_utc": ended_utc,
            "duration_s": duration_s,
            "path": str(out_path) if out_path.is_file() else str(self.raw_mkv),
            "raw_path": str(self.raw_mkv) if exists else None,
            "browser_path": browser_path,
            "exists": exists and (out_path.is_file() or self.raw_mkv.is_file()),
        }

    def _remux_to_mp4(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp.mp4")
        cmd = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-c:v",
            "copy",
            "-movflags",
            "+faststart",
            "-an",
            str(tmp),
        ]
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
            err = (proc.stderr or b"").decode("utf-8", errors="replace")[-300:]
            raise RuntimeError(err or f"ffmpeg remux failed ({proc.returncode})")
        if dst.is_file():
            dst.unlink(missing_ok=True)
        tmp.replace(dst)


class SessionFrameRecorder:
    """Ghi mọi khung reader đọc được (webcam / file) — full phiên, không chờ infer."""

    def __init__(
        self,
        camera_id: str,
        out_path: Path,
        *,
        fps: float,
    ) -> None:
        self.camera_id = camera_id
        self.out_path = Path(out_path)
        self._fps = max(1.0, min(60.0, float(fps)))
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[tuple[int, int]] = None
        self._frame_count = 0
        self._lock = threading.Lock()
        self.started_utc = time.time()

    def write_bgr(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        with self._lock:
            h, w = frame.shape[:2]
            if self._writer is None:
                self.out_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(str(self.out_path), fourcc, self._fps, (w, h))
                if not self._writer.isOpened():
                    logger.warning("[%s] Không mở VideoWriter %s", self.camera_id, self.out_path)
                    self._writer = None
                    return
                self._size = (w, h)
            fr = frame
            if self._size and (w, h) != self._size:
                fr = cv2.resize(frame, self._size)
            self._writer.write(fr)
            self._frame_count += 1

    def close(self) -> Dict[str, Any]:
        with self._lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
        ended = time.time()
        exists = self.out_path.is_file() and self.out_path.stat().st_size > 0
        duration_s = self._frame_count / self._fps if self._frame_count > 0 else max(
            0.0, ended - self.started_utc
        )
        browser_path: Optional[str] = None
        if exists:
            probed = probe_duration_s(self.out_path)
            if probed > 0:
                duration_s = probed
            try:
                from identity_vm_app.services.visual_mp4 import remux_visual_mp4_for_browser

                web = remux_visual_mp4_for_browser(self.out_path)
                browser_path = str(web)
            except Exception as ex:
                logger.warning("[%s] Remux web MP4: %s", self.camera_id, ex)
        return {
            "mode": "frame",
            "started_utc": self.started_utc,
            "ended_utc": ended,
            "duration_s": duration_s,
            "frame_count": self._frame_count,
            "path": str(self.out_path),
            "browser_path": browser_path,
            "exists": exists,
        }


def start_session_stream_recorder(
    camera_id: str,
    source: Any,
    *,
    job_id: str,
    out_mp4: Path,
    stream_fps: float,
) -> tuple[Optional[Any], str]:
    """Khởi tạo recorder full stream. Trả (recorder, mode)."""
    if not s.IVM_CAMERA_SESSION_STREAM_RECORD:
        return None, "none"
    cam = str(camera_id)
    if _is_stream_url(source):
        raw = out_mp4.with_name("session_raw.mkv")
        rec = SessionFfmpegRecorder(cam, str(source), raw, final_mp4=out_mp4)
        rec.start()
        return rec, "ffmpeg"
    rec = SessionFrameRecorder(cam, out_mp4, fps=stream_fps)
    return rec, "frame"
