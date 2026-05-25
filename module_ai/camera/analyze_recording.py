"""
Ghi video khi bật nhận diện:
- Archive RTSP (ffmpeg segment) — xem lại + export-cut theo offset sự kiện.
- Video overlay (mp4) — bbox + tên trên khung để visualize kết quả phân tích.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from module_ai.config import settings as s

logger = logging.getLogger("camera_recognition.recording")

_lock = threading.RLock()
_auto_archive_cameras: set[str] = set()
_visual_sessions: Dict[str, "_VisualSessionRecorder"] = {}


def _is_stream_url(source: Any) -> bool:
    if not isinstance(source, str):
        return False
    u = source.strip().lower()
    return u.startswith("rtsp://") or u.startswith("http://") or u.startswith("https://")


def _visual_root() -> Path:
    return Path(
        os.getenv("IVM_ANALYZE_VISUAL_DIR", str(s.IVM_DATA_DIR / "analyze_visual"))
    ).resolve()


class _VisualSessionRecorder:
    def __init__(
        self,
        camera_id: str,
        session_id: str,
        out_path: Path,
        *,
        output_fps: Optional[float] = None,
    ) -> None:
        self.camera_id = camera_id
        self.session_id = session_id
        self.out_path = out_path
        self.meta_path = out_path.with_suffix(".json")
        self.started_utc = time.time()
        self._output_fps = float(output_fps) if output_fps and output_fps > 0 else 0.0
        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[tuple[int, int]] = None
        self._frame_count = 0
        self._wlock = threading.Lock()

    def _writer_fps(self) -> float:
        if self._output_fps > 0:
            return min(60.0, self._output_fps)
        env_fps = float(s.IVM_ANALYZE_VISUAL_FPS)
        return env_fps if env_fps > 0 else 10.0

    def write_bgr(self, frame: np.ndarray) -> None:
        if frame is None or frame.size == 0:
            return
        with self._wlock:
            h, w = frame.shape[:2]
            if self._writer is None:
                self._size = (w, h)
                fps = self._writer_fps()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.out_path.parent.mkdir(parents=True, exist_ok=True)
                self._writer = cv2.VideoWriter(str(self.out_path), fourcc, fps, (w, h))
                if not self._writer.isOpened():
                    logger.warning("[%s] Không mở VideoWriter %s", self.camera_id, self.out_path)
                    self._writer = None
                    return
            if self._writer is not None:
                fr = frame
                if (w, h) != self._size and self._size:
                    fr = cv2.resize(frame, self._size)
                self._writer.write(fr)
                self._frame_count += 1

    def close(self) -> Dict[str, Any]:
        with self._wlock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
        ended = time.time()
        exists = self.out_path.is_file() and self.out_path.stat().st_size > 0
        browser_path: Optional[str] = None
        if exists:
            try:
                from identity_vm_app.services.visual_mp4 import remux_visual_mp4_for_browser

                web = remux_visual_mp4_for_browser(self.out_path)
                browser_path = str(web)
            except Exception as ex:
                logger.warning("[%s] Remux H.264 thất bại: %s", self.camera_id, ex)
        info = {
            "camera_id": self.camera_id,
            "session_id": self.session_id,
            "started_utc": self.started_utc,
            "ended_utc": ended,
            "frame_count": self._frame_count,
            "path": str(self.out_path),
            "browser_path": browser_path,
            "exists": exists,
        }
        try:
            self.meta_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return info


def _start_archive_recording(camera_id: str, source_url: str) -> bool:
    from identity_vm_app.recorder.rolling_ffmpeg import RollingFfmpegRecorder
    from identity_vm_app.state import state

    recorders = state.recorders
    store = state.store
    if recorders is None or store is None:
        logger.warning("[%s] Chưa có recorder registry / store — bỏ qua archive", camera_id)
        return False

    existing = recorders.get(camera_id)
    if existing is not None and existing.is_running():
        if camera_id not in _auto_archive_cameras:
            logger.info("[%s] Archive đã chạy (manual) — giữ nguyên", camera_id)
        return True

    prev_holder: Dict[str, Optional[int]] = {"id": None}

    def hook(path: Path, started: float) -> int:
        prev = prev_holder["id"]
        if prev is not None:
            store.finalize_segment(prev, started)
        sid = store.insert_segment(camera_id, str(path), started, None)
        prev_holder["id"] = sid
        return sid

    rec = RollingFfmpegRecorder(camera_id, source_url, segment_hook=hook)
    recorders.start(camera_id, rec)
    _auto_archive_cameras.add(camera_id)
    logger.info("[%s] Bắt đầu ghi archive RTSP", camera_id)
    return True


def _stop_archive_recording(camera_id: str) -> None:
    if camera_id not in _auto_archive_cameras:
        return
    from identity_vm_app.state import state

    if state.recorders is None:
        return
    state.recorders.stop(camera_id)
    _auto_archive_cameras.discard(camera_id)
    logger.info("[%s] Dừng ghi archive (tắt nhận diện)", camera_id)


def start_visual_recorder(
    camera_id: str,
    job_id: str,
    out_path: Path,
    *,
    output_fps: float,
) -> Optional[_VisualSessionRecorder]:
    """Ghi session.mp4 overlay — 1 khung / lần infer, FPS = analyze_fps_eff."""
    if not s.IVM_ANALYZE_RECORD_VISUAL:
        return None
    sess = _VisualSessionRecorder(
        camera_id, job_id, out_path, output_fps=float(output_fps)
    )
    with _lock:
        old = _visual_sessions.get(camera_id)
        if old is not None:
            old.close()
        _visual_sessions[camera_id] = sess
    logger.info("[%s] Ghi video phiên → %s (fps=%.2f)", camera_id, out_path, output_fps)
    return sess


def _start_visual_session(camera_id: str) -> Optional[_VisualSessionRecorder]:
    if not s.IVM_ANALYZE_RECORD_VISUAL:
        return None
    sid = str(int(time.time()))
    out_dir = _visual_root() / camera_id
    out_path = out_dir / f"session_{sid}.mp4"
    sess = _VisualSessionRecorder(camera_id, sid, out_path)
    with _lock:
        old = _visual_sessions.get(camera_id)
        if old is not None:
            old.close()
        _visual_sessions[camera_id] = sess
    logger.info("[%s] Ghi video overlay → %s", camera_id, out_path)
    return sess


def _stop_visual_session(camera_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        sess = _visual_sessions.pop(camera_id, None)
    if sess is None:
        return None
    info = sess.close()
    from module_ai.camera.activity_log import record

    record(
        camera_id,
        "visual_session_end",
        f"Video phân tích: {info.get('frame_count', 0)} khung → {info.get('path', '')}",
        extra=info,
    )
    return info


def record_visual_frame(camera_id: str, frame_bgr: np.ndarray) -> None:
    """Gọi từ worker sau mỗi lần infer (khung đã vẽ bbox)."""
    if not s.IVM_ANALYZE_RECORD_VISUAL:
        return
    with _lock:
        sess = _visual_sessions.get(camera_id)
    if sess is not None:
        sess.write_bgr(frame_bgr)


def sync_analyze_recording(camera_id: str, enabled: bool) -> Dict[str, Any]:
    """
    Bật nhận diện → start archive (+ video overlay).
    Tắt → stop overlay + stop archive nếu do analyze tự bật.
    """
    from camera_channel_config import load_camera_channel_specs

    out: Dict[str, Any] = {
        "camera_id": camera_id,
        "enabled": enabled,
        "archive_started": False,
        "visual_session": None,
    }
    specs = {str(c["id"]): c for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    spec = specs.get(camera_id)
    if spec is None:
        return out

    if enabled:
        src = spec.get("source")
        if s.IVM_ANALYZE_AUTO_ARCHIVE and _is_stream_url(src):
            out["archive_started"] = _start_archive_recording(camera_id, str(src))
        try:
            from identity_vm_app.services.camera_live_session import get_active_session

            live = get_active_session(camera_id)
        except Exception:
            live = None
        if live is None:
            sess = _start_visual_session(camera_id)
            if sess is not None:
                out["visual_session"] = {
                    "session_id": sess.session_id,
                    "path": str(sess.out_path),
                }
        elif live.overlay_recorder is not None:
            out["visual_session"] = {
                "session_id": live.job_id,
                "path": str(live.overlay_recorder.out_path),
                "job_id": live.job_id,
                "kind": "overlay",
            }
        elif live.stream_recorder is not None:
            out["visual_session"] = {
                "session_id": live.job_id,
                "path": str(live.video_path),
                "job_id": live.job_id,
                "record_mode": live.record_mode,
                "kind": "stream",
            }
    else:
        try:
            from identity_vm_app.services.camera_live_session import get_active_session

            live = get_active_session(camera_id)
        except Exception:
            live = None
        if live is None:
            out["visual_session"] = _stop_visual_session(camera_id)
        if s.IVM_ANALYZE_AUTO_ARCHIVE:
            _stop_archive_recording(camera_id)

    return out


def get_visual_session(camera_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        sess = _visual_sessions.get(camera_id)
    if sess is None:
        return None
    return {
        "session_id": sess.session_id,
        "path": str(sess.out_path),
        "started_utc": sess.started_utc,
        "frame_count": sess._frame_count,
        "recording": True,
    }


def list_visual_sessions(camera_id: str, *, limit: int = 20) -> List[Dict[str, Any]]:
    cam_dir = _visual_root() / camera_id
    if not cam_dir.is_dir():
        return []
    items: List[Dict[str, Any]] = []
    for mp4 in sorted(cam_dir.glob("session_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        if mp4.stem.endswith("_web"):
            continue
        if mp4.stat().st_size <= 0:
            continue
        sid = mp4.stem.replace("session_", "", 1)
        meta_path = mp4.with_suffix(".json")
        meta: Dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        items.append(
            {
                "session_id": sid,
                "path": str(mp4),
                "size_bytes": mp4.stat().st_size,
                "mtime_utc": mp4.stat().st_mtime,
                "frame_count": meta.get("frame_count"),
                "started_utc": meta.get("started_utc"),
                "ended_utc": meta.get("ended_utc"),
            }
        )
        if len(items) >= limit:
            break
    live = get_visual_session(camera_id)
    if live and not any(x["session_id"] == live["session_id"] for x in items):
        items.insert(0, {**live, "size_bytes": None, "recording": True})
    return items
