from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from identity_vm_app import settings as s
from identity_vm_app.api.deps import get_engine, get_face_db, get_store
from module_ai.engine.gpu_cleanup import maybe_release_global_engine_after_image_infer

router = APIRouter(prefix="/ivm", tags=["identity-vm-people"])


def _gallery_dir(face_id: int) -> Path:
    d = Path(s.IVM_DATA_DIR) / "gallery" / str(face_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _decode(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")
    return img


@router.get("/people")
def list_people() -> Dict[str, Any]:
    db = get_face_db()
    faces = db.get_all_faces()
    by_name: Dict[str, List[dict]] = {}
    for f in faces:
        n = str(f.get("name") or "")
        by_name.setdefault(n, []).append(f)
    return {"total_faces": len(faces), "faces": faces, "grouped_by_name": by_name}


@router.delete("/people/{face_id:int}")
def delete_person_face(face_id: int) -> Dict[str, Any]:
    db = get_face_db()
    if str(face_id) not in db.metadata:
        raise HTTPException(status_code=404, detail="face_id not found")
    db.delete_face(face_id)
    return {"ok": True, "deleted_face_id": face_id}


@router.post("/people/{face_id:int}/media")
async def add_person_media(
    face_id: int,
    file: UploadFile = File(...),
    media_type: str = "enrollment_extra",
    also_embed: bool = Query(False),
):
    db = get_face_db()
    meta = db.metadata.get(str(face_id))
    if not meta:
        raise HTTPException(status_code=404, detail="face_id not found")
    display_name = str(meta.get("name") or "")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    img = _decode(raw)
    path = str(_gallery_dir(face_id) / f"extra_{int(time.time() * 1000)}.jpg")
    cv2.imwrite(path, img)

    store = get_store()
    mid = store.insert_person_media(
        face_id=face_id,
        display_name=display_name or None,
        media_type=media_type,
        path=path,
        created_at_utc=time.time(),
    )
    new_fid = None
    try:
        if also_embed:
            faces = get_engine().analyze_bgr(img)
            if faces:
                face = max(faces, key=lambda f: f.det_score)
                new_fid = db.add_face(face.embedding, display_name, image_path=path)
    finally:
        if also_embed:
            maybe_release_global_engine_after_image_infer(
                log_label=f"people_media_embed:{face_id}"
            )
    return {"media_id": mid, "path": path, "also_embed_face_id": new_fid}


@router.get("/people/{face_id:int}/media")
def list_media(face_id: int) -> Dict[str, Any]:
    db = get_face_db()
    if str(face_id) not in db.metadata:
        raise HTTPException(status_code=404, detail="face_id not found")
    return {"face_id": face_id, "items": get_store().list_person_media(face_id)}


@router.delete("/people/media/{media_id}")
def delete_media(media_id: str) -> Dict[str, Any]:
    store = get_store()
    row = store.get_person_media(media_id)
    if not row:
        raise HTTPException(status_code=404, detail="media not found")
    p = Path(str(row["path"]))
    store.delete_person_media(media_id)
    try:
        if p.is_file() and str(p.resolve()).startswith(str(Path(s.IVM_DATA_DIR).resolve())):
            p.unlink(missing_ok=True)
    except Exception:
        pass
    return {"ok": True}


@router.get("/people/media/{media_id}/file")
def get_media_file(media_id: str) -> FileResponse:
    store = get_store()
    row = store.get_person_media(media_id)
    if not row:
        raise HTTPException(status_code=404, detail="media not found")
    p = Path(str(row["path"])).resolve()
    root = Path(s.IVM_DATA_DIR).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid path")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="file missing on disk")
    return FileResponse(p, filename=p.name)
