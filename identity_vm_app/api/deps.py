from fastapi import HTTPException

from identity_vm_app.engine.insightface_engine import InsightFaceEngine
from identity_vm_app.recorder.registry import RecorderRegistry
from identity_vm_app.state import state
from identity_vm_app.store.sqlite_store import IdentityVmStore
from packages.persistence.face_database import FaceDatabase


def get_engine() -> InsightFaceEngine:
    if state.engine is None:
        try:
            from identity_vm_app.lifecycle import ensure_inference_engine

            return ensure_inference_engine()
        except Exception as ex:
            raise HTTPException(
                status_code=503,
                detail=f"InsightFace engine not ready: {ex}",
            ) from ex
    return state.engine


def get_face_db() -> FaceDatabase:
    if state.face_db is None:
        raise HTTPException(status_code=503, detail="Face database not ready")
    return state.face_db


def get_store() -> IdentityVmStore:
    if state.store is None:
        raise HTTPException(status_code=503, detail="Event store not ready")
    return state.store


def get_recorders() -> RecorderRegistry:
    if state.recorders is None:
        raise HTTPException(status_code=503, detail="Recorder registry not ready")
    return state.recorders
