from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from module_ai.persistence.face_database import FaceDatabase

    from module_ai.engine.insightface_engine import InsightFaceEngine
    from identity_vm_app.recorder.registry import RecorderRegistry
    from identity_vm_app.store.sqlite_store import IdentityVmStore


class IdentityVmState:
    engine: Optional["InsightFaceEngine"] = None
    face_db: Optional["FaceDatabase"] = None
    store: Optional["IdentityVmStore"] = None
    recorders: Optional["RecorderRegistry"] = None


state = IdentityVmState()
