from identity_vm_app.recorder.registry import RecorderRegistry
from identity_vm_app.state import state
from identity_vm_app.store.sqlite_store import IdentityVmStore
from module_ai.api.deps import get_engine, get_face_db  # noqa: F401
from module_ai.engine.insightface_engine import InsightFaceEngine
from module_ai.persistence.face_database import FaceDatabase


def get_store() -> IdentityVmStore:
    if state.store is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Event store not ready")
    return state.store


def get_recorders() -> RecorderRegistry:
    if state.recorders is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail="Recorder registry not ready")
    return state.recorders
