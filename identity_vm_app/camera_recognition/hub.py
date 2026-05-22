"""Quản lý worker nhận diện đa camera — khởi động từ main.py."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Sequence

from identity_vm_app import settings as s
from identity_vm_app.camera_recognition.worker import CameraRecognitionConfig, CameraRecognitionWorker

logger = logging.getLogger("camera_recognition.hub")

_hub: Optional["RecognitionHub"] = None
_hub_lock = threading.Lock()
_status_log_stop = threading.Event()
_status_log_started = False
_status_log_lock = threading.Lock()


class RecognitionHub:
    def __init__(self) -> None:
        self._workers: Dict[str, CameraRecognitionWorker] = {}
        self._lock = threading.Lock()

    def start_cameras(
        self,
        specs: Sequence[Dict[str, Any]],
        *,
        api_base: str,
        interval_s: Optional[float] = None,
        threshold: Optional[float] = None,
        stagger_s: Optional[float] = None,
    ) -> None:
        stagger = float(stagger_s if stagger_s is not None else s.IVM_PIPELINE_START_STAGGER_S)
        with self._lock:
            for i, spec in enumerate(specs):
                cid = str(spec["id"])
                if cid in self._workers:
                    continue
                eff_iv = interval_s if interval_s is not None and float(interval_s) > 0 else None
                cfg = CameraRecognitionConfig(
                    camera_id=cid,
                    source=spec["source"],
                    api_base=api_base,
                    interval_s=eff_iv,
                    distance_threshold=threshold,
                )
                w = CameraRecognitionWorker(cfg)
                self._workers[cid] = w
                w.start()
                if stagger > 0 and i + 1 < len(specs):
                    time.sleep(stagger)
        logger.info("RecognitionHub: %d camera worker(s) started", len(self._workers))
        _ensure_hub_status_logger()

    def stop_all(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for w in workers:
            w.stop()
        for w in workers:
            w.join(timeout=12.0)
        logger.info("RecognitionHub: all workers stopped")

    def get_worker(self, camera_id: str) -> Optional[CameraRecognitionWorker]:
        with self._lock:
            return self._workers.get(camera_id)

    def list_camera_ids(self) -> List[str]:
        with self._lock:
            return list(self._workers.keys())

    def status(self) -> Dict[str, Any]:
        with self._lock:
            items = []
            for cid, w in self._workers.items():
                items.append(
                    {
                        "camera_id": cid,
                        "reader_connected": w.reader.is_connected,
                        "reader_fps": w.reader.fps_actual,
                        "frame_count": w.reader.frame_count,
                        "meta": w.get_meta(),
                        "infer_queue_depth": w.infer_queue_size(),
                    }
                )
        return {"cameras": items, "count": len(items)}


def get_recognition_hub() -> RecognitionHub:
    global _hub
    with _hub_lock:
        if _hub is None:
            _hub = RecognitionHub()
        return _hub


def start_recognition_hub(
    specs: Sequence[Dict[str, Any]],
    *,
    api_base: str,
    interval_s: Optional[float] = None,
    threshold: Optional[float] = None,
) -> RecognitionHub:
    hub = get_recognition_hub()
    hub.start_cameras(
        specs,
        api_base=api_base,
        interval_s=interval_s,
        threshold=threshold,
    )
    return hub


def ensure_recognition_hub_started(
    *,
    api_base: Optional[str] = None,
    config_path: Optional[str] = None,
    interval_s: Optional[float] = None,
    threshold: Optional[float] = None,
) -> RecognitionHub:
    """Khởi động hub một lần (bỏ qua nếu đã có worker)."""
    from camera_channel_config import load_camera_channel_specs

    hub = get_recognition_hub()
    if hub.list_camera_ids():
        return hub
    base = (api_base or f"http://127.0.0.1:{s.IVM_API_PORT}").rstrip("/")
    specs = load_camera_channel_specs(config_path or s.IVM_CAMERA_CONFIG)
    if not specs:
        logger.warning("ensure_recognition_hub_started: không có camera trong config")
        return hub
    logger.info("Đang khởi động %d worker nhận diện (api=%s)", len(specs), base)
    return start_recognition_hub(
        specs,
        api_base=base,
        interval_s=interval_s,
        threshold=threshold,
    )


def _hub_status_log_loop() -> None:
    """In định kỳ trạng thái mọi camera đang analyze — tránh tưởng cam0 'mất' khi cam8 spam log."""
    interval = float(s.IVM_CAMERA_HUB_STATUS_LOG_INTERVAL_S)
    while not _status_log_stop.wait(interval):
        try:
            from identity_vm_app.camera_analyze_control import get_analyze_enabled

            hub = get_recognition_hub()
            parts: List[str] = []
            for cid in sorted(hub.list_camera_ids()):
                if not get_analyze_enabled(cid):
                    continue
                w = hub.get_worker(cid)
                if w is None:
                    parts.append(f"{cid}: no-worker")
                    continue
                meta = w.get_meta() or {}
                fc = int(meta.get("frame_count") or w.reader.frame_count or 0)
                stale = ""
                err = w.reader.last_error()
                if err:
                    stale = f" err={err[:40]}"
                parts.append(
                    f"{cid}: q={w.infer_queue_size()}"
                    f"{' infer' if w.infer_in_progress else ''}"
                    f" fc={fc}{stale}"
                )
            if parts:
                logger.info("📹 Analyze cameras: %s", " | ".join(parts))
        except Exception as ex:
            logger.debug("hub status log: %s", ex)


def _ensure_hub_status_logger() -> None:
    global _status_log_started
    interval = float(s.IVM_CAMERA_HUB_STATUS_LOG_INTERVAL_S)
    if interval <= 0:
        return
    with _status_log_lock:
        if _status_log_started:
            return
        _status_log_started = True
        _status_log_stop.clear()
        threading.Thread(
            target=_hub_status_log_loop,
            name="ivm-hub-status-log",
            daemon=True,
        ).start()
        logger.info("Hub status log mỗi %.0fs (IVM_CAMERA_HUB_STATUS_LOG_INTERVAL_S)", interval)


def shutdown_recognition_hub() -> None:
    global _hub, _status_log_started
    _status_log_stop.set()
    with _status_log_lock:
        _status_log_started = False
    with _hub_lock:
        if _hub is not None:
            _hub.stop_all()
            _hub = None
