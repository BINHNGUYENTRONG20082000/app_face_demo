from __future__ import annotations

import gc
import os
import threading
import time

from identity_vm_app import settings as s
from identity_vm_app.preview.mjpeg_hub import shutdown_preview_hub
from identity_vm_app.preview.native_reader_hub import shutdown_native_preview_hub
from module_ai.engine.insightface_engine import InsightFaceEngine
from identity_vm_app.recorder.registry import RecorderRegistry
from identity_vm_app.state import state
from identity_vm_app.store.sqlite_store import IdentityVmStore
from module_ai.persistence.face_database import FaceDatabase

_engine_lock = threading.Lock()


def _load_global_inference_engine() -> InsightFaceEngine:
    print("identity_vm_app: loading InsightFace FaceAnalysis…", flush=True)
    t0 = time.perf_counter()
    eng = InsightFaceEngine()
    eng.log_loaded_models()
    print(
        f"identity_vm_app: InsightFace ready ({time.perf_counter() - t0:.2f}s) "
        f"{eng.get_runtime_info()}",
        flush=True,
    )
    return eng


def ensure_inference_engine() -> InsightFaceEngine:
    """Nạp InsightFace toàn cục nếu chưa có (thread-safe, dùng cho lazy load sau release)."""
    with _engine_lock:
        if state.engine is None:
            state.engine = _load_global_inference_engine()
        return state.engine


def startup(*, load_face_model: bool = True) -> None:
    """Khởi tạo FaceDB + SQLite + recorders; tùy chọn nạp InsightFace (một FaceAnalysis toàn cục)."""
    if load_face_model and state.engine is None:
        if s.IVM_GPU_RELEASE_ENGINE_AFTER_IMAGE_INFER:
            print(
                "identity_vm_app: hoãn nạp InsightFace lúc boot "
                "(IVM_GPU_RELEASE_ENGINE_AFTER_IMAGE_INFER=1 — load khi request ảnh đầu tiên)",
                flush=True,
            )
        else:
            with _engine_lock:
                if state.engine is None:
                    state.engine = _load_global_inference_engine()

    s.IVM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    s.IVM_FACE_DB_DIR.mkdir(parents=True, exist_ok=True)
    s.IVM_ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    s.IVM_EXPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s.IVM_CAMERA_SESSION_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from identity_vm_app.store.video_analyze_store import get_video_analyze_store

        n = get_video_analyze_store().cleanup_stuck_camera_live_jobs()
        if n:
            print(f"identity_vm_app: cleanup {n} phiên camera_live treo → ERROR", flush=True)
    except Exception as ex:
        print(f"identity_vm_app: cleanup camera_live jobs: {ex}", flush=True)

    if state.face_db is None:
        state.face_db = FaceDatabase(str(s.IVM_FACE_DB_DIR), use_faiss=s.IVM_USE_FAISS)
        print(f"identity_vm_app: FaceDatabase @ {s.IVM_FACE_DB_DIR}")
    if state.store is None:
        state.store = IdentityVmStore()
        print(f"identity_vm_app: SQLite @ {s.IVM_SQLITE_PATH}")
    if state.recorders is None:
        state.recorders = RecorderRegistry()

    if s.IVM_AUTO_START_CAMERA_WORKERS and not os.environ.get("IVM_NO_CAMERA_WORKERS"):
        try:
            from module_ai.camera.hub import ensure_recognition_hub_started

            ensure_recognition_hub_started()
            print(
                "identity_vm_app: worker camera đã bật (IVM_AUTO_START_CAMERA_WORKERS=1)",
                flush=True,
            )
        except Exception as ex:
            print(f"identity_vm_app: không khởi động worker camera: {ex}", flush=True)
    else:
        print(
            "identity_vm_app: worker camera không tự bật (mặc định) — bật nhận diện từ UI hoặc "
            "IVM_AUTO_START_CAMERA_WORKERS=1",
            flush=True,
        )


def release_inference_engine() -> None:
    """Gỡ model InsightFace toàn cục + đóng ONNX session + dọn CUDA (giữ FaceDB / SQLite)."""
    with _engine_lock:
        eng = state.engine
        state.engine = None
    if eng is not None:
        from module_ai.engine.gpu_cleanup import dispose_insightface_engine

        dispose_insightface_engine(eng)
    else:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass


def shutdown() -> None:
    shutdown_preview_hub()
    shutdown_native_preview_hub()
    try:
        from module_ai.camera.weapon import release_weapon_detectors

        release_weapon_detectors()
    except Exception:
        pass
    if state.recorders is not None:
        state.recorders.stop_all()
    state.engine = None
    state.face_db = None
    state.store = None
    state.recorders = None
