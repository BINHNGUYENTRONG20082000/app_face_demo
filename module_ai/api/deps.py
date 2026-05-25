from fastapi import HTTPException

from module_ai.engine.insightface_engine import InsightFaceEngine
from module_ai.persistence.face_database import FaceDatabase


def get_engine() -> InsightFaceEngine:
    try:
        from identity_vm_app.state import state
    except ImportError as ex:
        raise HTTPException(status_code=503, detail="App state not available") from ex
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
    from identity_vm_app.state import state

    if state.face_db is None:
        raise HTTPException(status_code=503, detail="Face database not ready")
    return state.face_db


def get_store():
    """Host SQLite store — injected from identity_vm_app.state."""
    from identity_vm_app.state import state

    if state.store is None:
        raise HTTPException(status_code=503, detail="Event store not ready")
    return state.store
