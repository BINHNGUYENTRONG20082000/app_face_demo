"""Trạng thái bật/tắt nhận diện theo camera (RAM, thread-safe) — worker đọc tại chỗ."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from identity_vm_app import settings as s

logger = logging.getLogger("camera_recognition.analyze")

_lock = threading.Lock()
_stop_lock = threading.Lock()
# Thiếu key = mặc định tắt (False)
_enabled: Dict[str, bool] = {}
# Tắt nhận diện nhưng worker vẫn drain queue trước khi đóng phiên live
_stopping: Dict[str, bool] = {}


def get_analyze_enabled(camera_id: str) -> bool:
    with _lock:
        return _enabled.get(camera_id, False)


def is_analyze_stopping(camera_id: str) -> bool:
    with _stop_lock:
        return bool(_stopping.get(str(camera_id), False))


def finish_analyze_stop(camera_id: str) -> Optional[Dict[str, Any]]:
    """Đóng phiên live sau khi hàng đợi infer đã rỗng. Idempotent."""
    cam = str(camera_id)
    with _stop_lock:
        if not _stopping.get(cam, False):
            return None
        _stopping[cam] = False

    session_info: Optional[Dict[str, Any]] = None
    try:
        from identity_vm_app.services.camera_live_session import stop_live_session

        session_info = stop_live_session(cam)
    except Exception as ex:
        logger.warning("[%s] stop_live_session: %s", cam, ex)
    try:
        from identity_vm_app.engine.gpu_cleanup import finalize_camera_recognition_session

        finalize_camera_recognition_session(cam)
    except Exception as ex:
        logger.warning("[%s] finalize_camera_recognition_session: %s", cam, ex)

    try:
        from identity_vm_app.camera_recognition.activity_log import record

        record(
            cam,
            "session_closed",
            "Phiên live đóng sau drain hàng đợi infer",
            extra={"session": session_info},
        )
    except Exception:
        pass
    logger.info("[%s] Phiên live đóng (drain xong)", cam)
    return session_info


def set_analyze_enabled(
    camera_id: str,
    enabled: bool,
    *,
    sample_fps: Optional[float] = None,
    display_name: Optional[str] = None,
    distance_threshold: Optional[float] = None,
    save_crops: Optional[bool] = None,
    stream_fps: float = 10.0,
    start_frame_count: int = 0,
) -> Dict[str, Any]:
    en = bool(enabled)
    cam = str(camera_id)
    session_info: Optional[Dict[str, Any]] = None
    draining = False
    infer_queue_pending = 0

    with _lock:
        prev = _enabled.get(cam, False)

    if en and not prev:
        with _stop_lock:
            _stopping.pop(cam, None)
        from identity_vm_app.services.camera_live_session import start_live_session
        from identity_vm_app.services.video_analyze_fps import parse_sample_fps

        sf = parse_sample_fps(
            sample_fps if sample_fps is not None else float(s.IVM_CAMERA_DEFAULT_SAMPLE_FPS)
        )
        source: Any = None
        try:
            from identity_vm_app.camera_recognition.hub import get_recognition_hub

            w = get_recognition_hub().get_worker(cam)
            if w is not None:
                source = w.cfg.source
                w.ensure_rtsp_reader()
        except Exception:
            pass
        try:
            from identity_vm_app.camera_recognition.weapon_alerts import reset_weapon_alerts

            reset_weapon_alerts(cam)
        except Exception:
            pass
        sess = start_live_session(
            cam,
            sample_fps=sf,
            display_name=display_name,
            distance_threshold=distance_threshold,
            save_crops=save_crops,
            stream_fps=stream_fps,
            start_frame_count=start_frame_count,
            source=source,
        )
        session_info = {
            "job_id": sess.job_id,
            "sample_fps": sess.sample_fps,
            "analyze_fps_eff": sess.analyze_fps_eff,
            "video_path": sess.video_path,
            "record_mode": sess.record_mode,
        }

    with _lock:
        _enabled[cam] = en

    if prev != en:
        from identity_vm_app.camera_recognition.activity_log import record

        state = "BẬT" if en else "TẮT"
        record(
            cam,
            "analyze_toggle",
            f"Nhận diện {state} (API /ivm/cameras/{cam}/analyze)",
            level="info",
            extra={"enabled": en, "session": session_info},
        )
        logger.info("[%s] Nhận diện %s", cam, state)
        try:
            from identity_vm_app.camera_recognition.analyze_recording import sync_analyze_recording

            sync_analyze_recording(cam, en)
        except Exception as ex:
            logger.warning("[%s] sync_analyze_recording: %s", cam, ex)

        if not en:
            try:
                from identity_vm_app.camera_recognition.weapon_alerts import reset_weapon_alerts

                reset_weapon_alerts(cam)
            except Exception:
                pass
            with _stop_lock:
                _stopping[cam] = True
            try:
                from identity_vm_app.camera_recognition.hub import get_recognition_hub

                w = get_recognition_hub().get_worker(cam)
                if w is not None:
                    drain_on_stop = bool(s.IVM_CAMERA_DRAIN_INFER_ON_STOP)
                    infer_queue_pending = w.infer_queue_size()
                    if not drain_on_stop:
                        dropped = w.clear_infer_queue()
                        infer_queue_pending = w.infer_queue_size()
                        if dropped > 0:
                            logger.info(
                                "[%s] Tắt nhận diện — bỏ %d khung chờ trong queue",
                                cam,
                                dropped,
                            )
                        if w.infer_in_progress:
                            draining = True
                            logger.info(
                                "[%s] Tắt nhận diện — chờ xong khung infer hiện tại",
                                cam,
                            )
                        else:
                            w.dispose_session_models()
                            session_info = finish_analyze_stop(cam)
                    elif infer_queue_pending or w.infer_in_progress:
                        draining = True
                        logger.info(
                            "[%s] Tắt nhận diện — drain %d khung infer + chờ infer hiện tại",
                            cam,
                            infer_queue_pending,
                        )
                    else:
                        w.dispose_session_models()
                        session_info = finish_analyze_stop(cam)
                else:
                    session_info = finish_analyze_stop(cam)
            except Exception as ex:
                logger.warning("[%s] schedule analyze stop: %s", cam, ex)
                session_info = finish_analyze_stop(cam)

    return {
        "camera_id": cam,
        "enabled": en,
        "session": session_info,
        "draining": draining,
        "infer_queue_pending": infer_queue_pending,
    }


def snapshot_states() -> Dict[str, bool]:
    with _lock:
        return dict(_enabled)
