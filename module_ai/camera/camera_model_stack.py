"""Bộ model riêng từng camera — VisionMaster CameraAnalyze (lazy load khi BẬT analyze)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional, Tuple

from module_ai.config import settings as s

logger = logging.getLogger("camera_recognition.model_stack")


def _ctx_id_for_camera(camera_id: str) -> int:
    ids = s.ivm_bulk_worker_ctx_ids()
    if not ids:
        return int(s.IVM_CTX_ID)
    key = sum(ord(c) for c in str(camera_id)) % len(ids)
    return int(ids[key])


class CameraModelStack:
    """InsightFace + YOLO person tracker riêng / camera; face_db dùng chung (FAISS)."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = str(camera_id)
        self._lock = threading.Lock()
        self._engine: Any = None
        self._person_tracker: Any = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._loaded

    def ensure_loaded(self) -> Tuple[Any, Optional[Any], Any]:
        """Trả (engine, person_tracker|None, face_db)."""
        from identity_vm_app.state import state

        if state.face_db is None:
            raise RuntimeError("face_db chưa sẵn sàng — khởi động API trước")

        with self._lock:
            if self._loaded:
                return self._engine, self._person_tracker, state.face_db

            dedicated = bool(s.IVM_CAMERA_DEDICATED_MODELS)
            if dedicated:
                from module_ai.engine.insightface_engine import InsightFaceEngine

                ctx = _ctx_id_for_camera(self.camera_id)
                logger.info(
                    "[%s] Đang load InsightFace engine riêng (ctx_id=%s) — camera khác có thể chậm/ im log vài chục giây",
                    self.camera_id,
                    ctx,
                )
                t0 = time.perf_counter()
                self._engine = InsightFaceEngine(ctx_id=ctx)
                logger.info(
                    "[%s] InsightFace engine sẵn sàng (%.1fs)",
                    self.camera_id,
                    time.perf_counter() - t0,
                )
            else:
                from identity_vm_app.lifecycle import ensure_inference_engine

                self._engine = ensure_inference_engine()

            if bool(s.IVM_USE_PERSON_FIRST_PIPELINE):
                try:
                    from module_ai.engine.yolo_person_tracker import (
                        create_person_tracker,
                        vm_tracking_available,
                    )

                    if vm_tracking_available():
                        logger.info(
                            "[%s] Đang load YOLO person tracker riêng…",
                            self.camera_id,
                        )
                        t1 = time.perf_counter()
                        self._person_tracker = create_person_tracker(self.camera_id)
                        logger.info(
                            "[%s] YOLO person tracker sẵn sàng (%.1fs)",
                            self.camera_id,
                            time.perf_counter() - t1,
                        )
                except Exception as ex:
                    logger.warning("[%s] Person tracker không khởi tạo: %s", self.camera_id, ex)

            self._loaded = True
            return self._engine, self._person_tracker, state.face_db

    def dispose(self) -> None:
        with self._lock:
            if not self._loaded:
                return
            if bool(s.IVM_CAMERA_DEDICATED_MODELS) and self._engine is not None:
                try:
                    from module_ai.engine.gpu_cleanup import dispose_insightface_engine

                    dispose_insightface_engine(self._engine)
                except Exception as ex:
                    logger.warning("[%s] dispose InsightFace: %s", self.camera_id, ex)
            self._engine = None

            if self._person_tracker is not None:
                try:
                    self._person_tracker.dispose()
                except Exception:
                    pass
                self._person_tracker = None

            self._loaded = False
            logger.info("[%s] Đã giải phóng model stack camera", self.camera_id)

            try:
                from module_ai.engine.gpu_cleanup import finalize_camera_recognition_session

                finalize_camera_recognition_session(self.camera_id)
            except Exception as ex:
                logger.warning("[%s] finalize session cleanup: %s", self.camera_id, ex)
