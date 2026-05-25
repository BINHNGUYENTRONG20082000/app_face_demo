"""Quản lý phiên nhận diện camera live — job DB + báo cáo + session.mp4 (full stream)."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from identity_vm_app import settings as s
from identity_vm_app.services import camera_session_media as cmedia
from identity_vm_app.services.camera_report_writer import build_person_report_rows_from_assignments
from identity_vm_app.services.video_analyze_fps import default_display_name, sample_fps_label
from identity_vm_app.services.video_analyze_tracking import VideoClipCounter
from identity_vm_app.store.video_analyze_store import get_video_analyze_store

if TYPE_CHECKING:
    from module_ai.camera.analyze_recording import _VisualSessionRecorder

logger = logging.getLogger("camera_live_session")

_lock = threading.RLock()
_toggle_locks: Dict[str, threading.Lock] = {}
_sessions: Dict[str, "LiveSession"] = {}


def _toggle_lock(camera_id: str) -> threading.Lock:
    with _lock:
        if camera_id not in _toggle_locks:
            _toggle_locks[camera_id] = threading.Lock()
        return _toggle_locks[camera_id]


@dataclass
class LiveSession:
    camera_id: str
    job_id: str
    session_start_utc: float
    sample_fps: float
    analyze_fps_eff: float
    distance_threshold: float
    save_crops: bool
    video_path: str
    start_frame_count: int = 0
    sample_index: int = 0
    frame_index: int = 0
    clip_counter: VideoClipCounter = field(default_factory=lambda: VideoClipCounter(max_miss=10))
    report_buffer: List[Dict[str, Any]] = field(default_factory=list)
    stream_recorder: Optional[Any] = None
    overlay_recorder: Optional[Any] = None
    record_mode: str = "none"
    thumb_written: bool = False
    width: int = 0
    height: int = 0
    stream_fps: float = 10.0

    def t_analyze_s(self) -> float:
        if self.analyze_fps_eff <= 0:
            return float(self.sample_index)
        return float(self.sample_index) / float(self.analyze_fps_eff)

    def wall_t_s(self) -> float:
        return max(0.0, time.time() - self.session_start_utc)

    def t_analyze_s(self) -> float:
        """Thời điểm video theo sample (giống offline: sample_index / analyze_fps)."""
        if self.analyze_fps_eff <= 0:
            return float(self.sample_index)
        return float(self.sample_index) / float(self.analyze_fps_eff)


def get_active_session(camera_id: str) -> Optional[LiveSession]:
    with _lock:
        return _sessions.get(str(camera_id))


def resolve_camera_source(camera_id: str) -> Optional[Any]:
    try:
        from camera_channel_config import load_camera_channel_specs

        for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG):
            if str(c.get("id")) == str(camera_id):
                return c.get("source")
    except Exception:
        pass
    return None


def start_live_session(
    camera_id: str,
    *,
    sample_fps: float = 5.0,
    display_name: Optional[str] = None,
    distance_threshold: Optional[float] = None,
    save_crops: Optional[bool] = None,
    stream_fps: float = 10.0,
    start_frame_count: int = 0,
    source: Optional[Any] = None,
) -> LiveSession:
    cam = str(camera_id)
    with _toggle_lock(cam):
        existing = get_active_session(cam)
        if existing is not None:
            return existing

        sf = float(sample_fps)
        stream = max(1.0, float(stream_fps or 10.0))
        analyze_fps_eff = sf if sf > 0 else stream
        dist = float(distance_threshold if distance_threshold is not None else s.IVM_DISTANCE_THRESHOLD)
        crops = bool(save_crops if save_crops is not None else s.IVM_VIDEO_ANALYZE_SAVE_CROPS)
        dn = (display_name or "").strip() or default_display_name(f"live-{cam}", sf)

        job_id = f"live-{cam}-{uuid.uuid4().hex[:12]}"
        mp4 = cmedia.session_mp4_path(cam, job_id)
        video_path = str(mp4.resolve())
        now = time.time()

        src = source if source is not None else resolve_camera_source(cam)
        record_mode = "none"
        stream_rec: Optional[Any] = None
        if s.IVM_CAMERA_SESSION_STREAM_RECORD:
            from identity_vm_app.recorder.session_ffmpeg import start_session_stream_recorder

            stream_rec, record_mode = start_session_stream_recorder(
                cam,
                src,
                job_id=job_id,
                out_mp4=mp4,
                stream_fps=stream,
            )

        overlay_rec: Optional[_VisualSessionRecorder] = None
        if s.IVM_CAMERA_SESSION_OVERLAY_LIVE and s.IVM_ANALYZE_RECORD_VISUAL:
            from module_ai.camera.analyze_recording import start_visual_recorder

            overlay_path = cmedia.session_overlay_mp4_path(cam, job_id)
            overlay_rec = start_visual_recorder(
                cam, job_id, overlay_path, output_fps=analyze_fps_eff
            )

        feature = {
            "pipeline": "camera_live",
            "camera_id": cam,
            "distance_threshold": dist,
            "save_crops": crops,
            "sample_fps_label": sample_fps_label(sf),
            "record_mode": record_mode,
            "timeline": "video_sample",
        }
        store = get_video_analyze_store()
        store.insert_camera_live_job(
            job_id,
            camera_id=cam,
            video_path=video_path,
            display_name=dn,
            sample_fps=sf,
            feature_analyze=feature,
            session_start_utc=now,
        )

        sess = LiveSession(
            camera_id=cam,
            job_id=job_id,
            session_start_utc=now,
            sample_fps=sf,
            analyze_fps_eff=analyze_fps_eff,
            distance_threshold=dist,
            save_crops=crops,
            video_path=video_path,
            start_frame_count=int(start_frame_count),
            stream_recorder=stream_rec,
            overlay_recorder=overlay_rec,
            record_mode=record_mode,
            stream_fps=stream,
        )
        with _lock:
            _sessions[cam] = sess
        logger.info(
            "[%s] Phiên live %s sample_fps=%s analyze_fps=%.2f record=%s",
            cam,
            job_id,
            sf,
            analyze_fps_eff,
            record_mode,
        )
        return sess


def record_stream_frame(
    camera_id: str,
    frame_bgr: np.ndarray,
    *,
    stream_fps: Optional[float] = None,
) -> None:
    """Ghi khung reader (chỉ SessionFrameRecorder — ffmpeg ghi song song)."""
    sess = get_active_session(camera_id)
    if sess is None or sess.stream_recorder is None:
        return
    if stream_fps and stream_fps > 0:
        sess.stream_fps = float(stream_fps)
    rec = sess.stream_recorder
    if hasattr(rec, "write_bgr") and not hasattr(rec, "raw_mkv"):
        rec.write_bgr(frame_bgr)


def _flush_reports(sess: LiveSession) -> None:
    if not sess.report_buffer:
        return
    n = get_video_analyze_store().insert_person_reports_batch(sess.report_buffer)
    sess.report_buffer.clear()
    if n:
        get_video_analyze_store().update_job_progress(sess.job_id, sess.sample_index)


def on_frame(
    camera_id: str,
    frame_bgr: np.ndarray,
    *,
    frame_count: int,
    assignments: List[Dict[str, Any]],
    faces_with_matches: List[Dict[str, Any]],
    visual_bgr: Optional[np.ndarray] = None,
    t_analyze_s: Optional[float] = None,
    wall_t_s: Optional[float] = None,
    sample_index: Optional[int] = None,
) -> int:
    """Ghi báo cáo DB; overlay tùy chọn. Trả số row đã buffer."""
    sess = get_active_session(camera_id)
    if sess is None or frame_bgr is None or frame_bgr.size == 0:
        return 0

    h, w = frame_bgr.shape[:2]
    if w > 0 and h > 0:
        sess.width, sess.height = int(w), int(h)

    if t_analyze_s is not None:
        t_s = float(t_analyze_s)
    elif wall_t_s is not None:
        t_s = float(wall_t_s)
    else:
        t_s = sess.t_analyze_s()
    si = int(sample_index) if sample_index is not None else int(sess.sample_index)
    had_detection = bool(assignments)
    vclip = sess.clip_counter.on_frame(had_detection)

    persist_emb = bool(s.IVM_VIDEO_ANALYZE_PERSIST_EMBEDDINGS)
    rows = build_person_report_rows_from_assignments(
        sess.camera_id,
        sess.job_id,
        frame_bgr,
        t_s=t_s,
        frame_index=int(frame_count),
        sample_index=si,
        video_clip=vclip,
        assignments=assignments,
        faces_with_matches=faces_with_matches,
        persist_embeddings=persist_emb,
        save_crops=sess.save_crops,
    )
    if rows:
        sess.report_buffer.extend(rows)
        batch_sz = max(1, int(s.IVM_VIDEO_ANALYZE_DB_BATCH))
        if len(sess.report_buffer) >= batch_sz:
            _flush_reports(sess)

    if visual_bgr is not None and sess.overlay_recorder is not None:
        sess.overlay_recorder.write_bgr(visual_bgr)

    if not sess.thumb_written and frame_bgr is not None and frame_bgr.size > 0:
        thumb = cmedia.write_session_thumb(sess.camera_id, sess.job_id, frame_bgr)
        get_video_analyze_store().update_job_thumb(sess.job_id, thumb)
        sess.thumb_written = True

    sess.sample_index = si + 1
    sess.frame_index = int(frame_count)
    return len(rows)


def stop_live_session(camera_id: str) -> Optional[Dict[str, Any]]:
    cam = str(camera_id)
    with _toggle_lock(cam):
        with _lock:
            sess = _sessions.pop(cam, None)
        if sess is None:
            return None

        _flush_reports(sess)

        stream_info: Dict[str, Any] = {}
        if sess.stream_recorder is not None:
            try:
                stream_info = sess.stream_recorder.close()
            except Exception as ex:
                logger.warning("[%s] stream_recorder.close: %s", cam, ex)

        overlay_info: Dict[str, Any] = {}
        if sess.overlay_recorder is not None:
            overlay_info = sess.overlay_recorder.close()
            from module_ai.camera.analyze_recording import (
                _lock as vis_lock,
                _visual_sessions,
            )

            with vis_lock:
                if _visual_sessions.get(cam) is sess.overlay_recorder:
                    _visual_sessions.pop(cam, None)

        ended = time.time()
        duration_s = max(0.0, ended - sess.session_start_utc)
        if stream_info.get("duration_s"):
            duration_s = max(duration_s, float(stream_info["duration_s"]))

        video_path = str(sess.video_path)
        if stream_info.get("browser_path"):
            video_path = str(stream_info["browser_path"])
        elif stream_info.get("path"):
            video_path = str(stream_info["path"])

        thumb_path = str(cmedia.session_thumb_path(cam, sess.job_id))
        if not Path(thumb_path).is_file():
            thumb_path = None

        get_video_analyze_store().finalize_camera_live_job(
            sess.job_id,
            video_path=video_path,
            thumb_path=thumb_path,
            duration_s=duration_s,
            analyze_fps=sess.analyze_fps_eff,
            total_sample_frames=sess.sample_index,
            session_end_utc=ended,
            width=sess.width,
            height=sess.height,
        )

        from module_ai.camera.activity_log import record

        record(
            cam,
            "live_session_end",
            f"Phiên {sess.job_id}: {sess.sample_index} mẫu → {video_path}",
            extra={
                "job_id": sess.job_id,
                "sample_index": sess.sample_index,
                "duration_s": round(duration_s, 2),
                "video_path": video_path,
                "record_mode": sess.record_mode,
            },
        )
        out = {
            "camera_id": cam,
            "job_id": sess.job_id,
            "session_start_utc": sess.session_start_utc,
            "session_end_utc": ended,
            "sample_index": sess.sample_index,
            "video_path": video_path,
            "record_mode": sess.record_mode,
            "stream_recording": stream_info,
            "overlay_recording": overlay_info,
        }
        logger.info("[%s] Kết thúc phiên live %s (%.1fs video)", cam, sess.job_id, duration_s)
        return out
