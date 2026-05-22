"""Dọn VRAM/RAM khi tiến trình kết thúc — tùy chọn unload InsightFace sau infer ảnh."""

from __future__ import annotations

import gc
import logging
from typing import Any, Optional

logger = logging.getLogger("identity_vm_app.gpu_cleanup")


def _any_camera_analyze_active() -> bool:
    """True nếu có camera đang BẬT nhận diện (dùng chung state.engine khi dedicated=0)."""
    try:
        from camera_channel_config import load_camera_channel_specs

        from identity_vm_app import settings as s
        from identity_vm_app.camera_analyze_control import get_analyze_enabled

        for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG):
            if get_analyze_enabled(str(c["id"])):
                return True
    except Exception:
        pass
    return False


def maybe_gpu_soft_cleanup(*, log_label: str = "") -> None:
    """Theo IVM_GPU_SOFT_CLEANUP_ON_PROCESS_DONE — chỉ khi tiến trình kết thúc."""
    from identity_vm_app import settings as s

    if s.IVM_GPU_SOFT_CLEANUP_ON_PROCESS_DONE:
        gpu_soft_cleanup(log_label=log_label)


def gpu_soft_cleanup(*, log_label: str = "") -> None:
    """GC + CUDA cache; không đóng ONNX session."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()
    if log_label:
        logger.debug("gpu_soft_cleanup: %s", log_label)


def maybe_release_global_engine_after_image_infer(*, log_label: str = "") -> None:
    """Sau infer ảnh qua API: soft cleanup hoặc unload InsightFace toàn cục (VRAM ~ lúc boot)."""
    from identity_vm_app import settings as s

    if not s.IVM_GPU_RELEASE_ENGINE_AFTER_IMAGE_INFER:
        maybe_gpu_soft_cleanup(log_label=log_label)
        return

    if not s.IVM_CAMERA_DEDICATED_MODELS and _any_camera_analyze_active():
        logger.info(
            "skip unload global engine (%s): camera analyze ON + IVM_CAMERA_DEDICATED_MODELS=0",
            log_label or "image_infer",
        )
        maybe_gpu_soft_cleanup(log_label=log_label)
        return

    from identity_vm_app.lifecycle import release_inference_engine

    release_inference_engine()
    if log_label:
        logger.info("Released global InsightFace engine after %s", log_label)


def dispose_insightface_engine(eng: Any) -> None:
    """Gỡ engine tạm (bulk / identify_images multi-worker) hoặc toàn cục."""
    if eng is None:
        return
    try:
        app = getattr(eng, "_app", None)
        try:
            eng._app = None  # type: ignore[attr-defined]
        except Exception:
            pass
        if app is not None:
            models = getattr(app, "models", None)
            if isinstance(models, dict):
                for model in list(models.values()):
                    try:
                        sess = getattr(model, "session", None)
                        if sess is not None:
                            try:
                                sess.close()
                            except Exception:
                                pass
                    except Exception:
                        pass
                try:
                    models.clear()
                except Exception:
                    pass
            try:
                del app
            except Exception:
                pass
    except Exception:
        pass
    gpu_soft_cleanup()


def dispose_ultralytics_yolo(model: Any) -> None:
    if model is None:
        return
    for attr in ("model", "predictor", "trainer"):
        try:
            if getattr(model, attr, None) is not None:
                setattr(model, attr, None)
        except Exception:
            pass


def finalize_camera_recognition_session(camera_id: str) -> None:
    """Tắt nhận diện: gỡ model tạm (vũ khí) rồi dọn VRAM một lần."""
    from identity_vm_app.camera_recognition.weapon import release_weapon_detector

    release_weapon_detector(camera_id)
    maybe_gpu_soft_cleanup(log_label=f"camera_session_done:{camera_id}")
