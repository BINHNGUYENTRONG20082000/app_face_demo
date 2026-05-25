from __future__ import annotations

import base64
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import APIRouter, Body, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from identity_vm_app import settings as s
from identity_vm_app.api.deps import get_recorders
from module_ai.api.deps import get_engine, get_face_db, get_store
from module_ai.jobs.bulk_folder_register import (
    _dispose_worker_engine,
    _release_bulk_infer_resources,
    assert_bulk_root_allowed,
    resolve_bulk_infer_workers,
    run_folder_register,
    write_register_folder_progress,
)
from module_ai.engine.gpu_cleanup import (
    gpu_soft_cleanup,
    maybe_release_global_engine_after_image_infer,
)
from module_ai.engine.insightface_engine import InsightFaceEngine
from module_ai.persistence.face_database import FaceDatabase
from module_ai.utils.text import remove_accents

router = APIRouter(prefix="/ivm", tags=["identity-vm-ai"])

_bulk_folder_lock = threading.Lock()


@router.get("/health")
def health() -> Dict[str, Any]:
    eng = get_engine()
    db = get_face_db()
    return {
        "status": "ok",
        "model_tag": s.IVM_MODEL_TAG,
        "event_debounce_s": s.IVM_EVENT_DEBOUNCE_S,
        "insightface": eng.get_runtime_info(),
        "face_db": db.get_stats(),
        "sqlite": str(s.IVM_SQLITE_PATH),
    }


def _decode_upload(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")
    return img


def _identify_faces_payload(faces_det: List[Any], db: FaceDatabase, thr: float) -> Tuple[List[Dict[str, Any]], float]:
    """Chuyá»ƒn káº¿t quáº£ analyze_bgr â†’ payload JSON; má»™t láº§n search_batch cho toÃ n bá»™ máº·t. Tráº£ vá» (payload, search_ms)."""
    if not faces_det:
        return [], 0.0
    t0 = time.perf_counter()
    embs = np.stack(
        [np.asarray(f.embedding, dtype=np.float32).reshape(-1) for f in faces_det],
        axis=0,
    )
    all_matches = db.search_batch(embs, k=s.IVM_SEARCH_K, distance_threshold=thr)
    search_ms = (time.perf_counter() - t0) * 1000
    out: List[Dict[str, Any]] = []
    for f, matches in zip(faces_det, all_matches):
        out.append(
            {
                "bbox": f.bbox.tolist(),
                "det_score": f.det_score,
                "gender": f.gender,
                "age": f.age,
                "matches": matches,
            }
        )
    return out, search_ms


def _round_ms(v: float) -> float:
    return round(float(v), 3)


def _identify_decode_infer_only(fname: str, raw: bytes, eng: InsightFaceEngine) -> Dict[str, Any]:
    """Decode + analyze_bgr cho má»™t áº£nh (dÃ¹ng trong identify_images tuáº§n tá»± hoáº·c worker song song)."""
    if not raw:
        return {
            "result_entry": {"filename": fname, "error": "empty", "faces": []},
            "timing": {
                "filename": fname,
                "decode_ms": 0.0,
                "detect_ms": 0.0,
                "embedding_ms": 0.0,
                "infer_ms": 0.0,
                "face_count": 0,
                "search_ms": 0.0,
                "error": "empty",
            },
            "embeddings": [],
        }

    t_dec = time.perf_counter()
    try:
        img = _decode_upload(raw)
        decode_ms = (time.perf_counter() - t_dec) * 1000
    except HTTPException as ex:
        decode_ms = (time.perf_counter() - t_dec) * 1000
        detail = ex.detail
        msg = detail if isinstance(detail, str) else str(detail)
        return {
            "result_entry": {"filename": fname, "error": msg, "faces": []},
            "timing": {
                "filename": fname,
                "decode_ms": _round_ms(decode_ms),
                "detect_ms": 0.0,
                "embedding_ms": 0.0,
                "infer_ms": 0.0,
                "face_count": 0,
                "search_ms": 0.0,
                "error": msg,
            },
            "embeddings": [],
        }

    infer_prof: Dict[str, float] = {}
    faces_det = eng.analyze_bgr(img, timing_out=infer_prof)
    detect_ms = float(infer_prof.get("detect_ms", 0.0))
    embedding_ms = float(infer_prof.get("embedding_ms", 0.0))
    infer_ms = detect_ms + embedding_ms
    fc = len(faces_det)
    if not faces_det:
        return {
            "result_entry": {"filename": fname, "faces": []},
            "timing": {
                "filename": fname,
                "decode_ms": _round_ms(decode_ms),
                "detect_ms": _round_ms(detect_ms),
                "embedding_ms": _round_ms(embedding_ms),
                "infer_ms": _round_ms(infer_ms),
                "face_count": 0,
                "search_ms": 0.0,
            },
            "embeddings": [],
        }

    faces_out: List[Dict[str, Any]] = []
    embeddings: List[np.ndarray] = []
    for f in faces_det:
        embeddings.append(np.asarray(f.embedding, dtype=np.float32).reshape(-1))
        faces_out.append(
            {
                "bbox": f.bbox.tolist(),
                "det_score": f.det_score,
                "gender": f.gender,
                "age": f.age,
                "matches": [],
            }
        )
    return {
        "result_entry": {"filename": fname, "faces": faces_out},
        "timing": {
            "filename": fname,
            "decode_ms": _round_ms(decode_ms),
            "detect_ms": _round_ms(detect_ms),
            "embedding_ms": _round_ms(embedding_ms),
            "infer_ms": _round_ms(infer_ms),
            "face_count": fc,
            "search_ms": 0.0,
        },
        "embeddings": embeddings,
    }


def _identify_images_merge_recognition_chunk(
    indexed_payloads: List[Tuple[int, str, bytes]],
    eng: InsightFaceEngine,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[np.ndarray], float]:
    """
    Decode â†’ detect+align â†’ gom crop â†’ get_feat theo lÃ´ (IVM_REC_GET_FEAT_MAX_BATCH) cho má»™t nhÃ³m áº£nh.
    Tráº£ (results, timing_images, embeddings_flat, merged_recognition_ms).
    """
    n = len(indexed_payloads)
    imgs: List[Optional[np.ndarray]] = [None] * n
    decode_ms_arr = [0.0] * n
    err_msg: List[Optional[str]] = [None] * n

    for i, (_idx, fname, raw) in enumerate(indexed_payloads):
        fname = fname or f"image_{i}"
        if not raw:
            err_msg[i] = "empty"
            continue
        t_dec = time.perf_counter()
        try:
            imgs[i] = _decode_upload(raw)
            decode_ms_arr[i] = (time.perf_counter() - t_dec) * 1000
        except HTTPException as ex:
            decode_ms_arr[i] = (time.perf_counter() - t_dec) * 1000
            detail = ex.detail
            err_msg[i] = detail if isinstance(detail, str) else str(detail)

    aligned_per: List[List[np.ndarray]] = [[] for _ in range(n)]
    meta_per: List[List[Tuple[np.ndarray, float]]] = [[] for _ in range(n)]
    detect_ms_arr = [0.0] * n

    for i in range(n):
        if err_msg[i] is not None:
            continue
        assert imgs[i] is not None
        d_ms, al, mt = eng.detect_align_faces(imgs[i])
        detect_ms_arr[i] = d_ms
        aligned_per[i] = al
        meta_per[i] = mt

    all_aligned: List[np.ndarray] = []
    for i in range(n):
        all_aligned.extend(aligned_per[i])

    merge_emb_ms = 0.0
    feats: np.ndarray
    if all_aligned:
        feats, merge_emb_ms = eng.embed_aligned_crops(
            all_aligned, max_batch=int(s.IVM_REC_GET_FEAT_MAX_BATCH)
        )
    else:
        dim = eng.recognition_feature_dim()
        feats = np.empty((0, dim), dtype=np.float32)

    total_faces = int(feats.shape[0])
    emb_share = [0.0] * n
    if total_faces > 0:
        for i in range(n):
            fc_i = len(aligned_per[i])
            emb_share[i] = merge_emb_ms * (fc_i / float(total_faces))

    results: List[Dict[str, Any]] = []
    timing_images: List[Dict[str, Any]] = []
    embs_flat: List[np.ndarray] = []

    off = 0
    for i in range(n):
        _idx, fname, raw = indexed_payloads[i]
        fname = fname or f"image_{i}"
        if err_msg[i] is not None:
            msg = err_msg[i] or "error"
            if msg == "empty":
                results.append({"filename": fname, "error": "empty", "faces": []})
                timing_images.append(
                    {
                        "filename": fname,
                        "decode_ms": 0.0,
                        "detect_ms": 0.0,
                        "embedding_ms": 0.0,
                        "infer_ms": 0.0,
                        "face_count": 0,
                        "search_ms": 0.0,
                        "error": "empty",
                    }
                )
            else:
                results.append({"filename": fname, "error": msg, "faces": []})
                timing_images.append(
                    {
                        "filename": fname,
                        "decode_ms": _round_ms(decode_ms_arr[i]),
                        "detect_ms": 0.0,
                        "embedding_ms": 0.0,
                        "infer_ms": 0.0,
                        "face_count": 0,
                        "search_ms": 0.0,
                        "error": msg,
                    }
                )
            continue

        decode_ms = decode_ms_arr[i]
        detect_ms = float(detect_ms_arr[i])
        fc = len(aligned_per[i])
        if fc == 0:
            results.append({"filename": fname, "faces": []})
            timing_images.append(
                {
                    "filename": fname,
                    "decode_ms": _round_ms(decode_ms),
                    "detect_ms": _round_ms(detect_ms),
                    "embedding_ms": 0.0,
                    "infer_ms": _round_ms(detect_ms),
                    "face_count": 0,
                    "search_ms": 0.0,
                }
            )
            continue

        slice_feats = feats[off : off + fc]
        off += fc
        faces_out: List[Dict[str, Any]] = []
        for j in range(fc):
            emb = np.asarray(slice_feats[j], dtype=np.float32).reshape(-1)
            embs_flat.append(emb)
            bbox, det_score = meta_per[i][j]
            faces_out.append(
                {
                    "bbox": bbox.tolist(),
                    "det_score": det_score,
                    "gender": None,
                    "age": None,
                    "matches": [],
                }
            )
        e_ms = emb_share[i]
        infer_ms = detect_ms + e_ms
        results.append({"filename": fname, "faces": faces_out})
        timing_images.append(
            {
                "filename": fname,
                "decode_ms": _round_ms(decode_ms),
                "detect_ms": _round_ms(detect_ms),
                "embedding_ms": _round_ms(e_ms),
                "infer_ms": _round_ms(infer_ms),
                "face_count": fc,
                "search_ms": 0.0,
            }
        )

    assert off == total_faces
    return results, timing_images, embs_flat, merge_emb_ms


def _identify_images_merge_recognition(
    indexed_payloads: List[Tuple[int, str, bytes]],
    eng: InsightFaceEngine,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[np.ndarray], float]:
    """
    Merge recognition cross-image; chia lÃ´ theo IVM_IDENTIFY_IMAGES_PROCESS_CHUNK Ä‘á»ƒ giáº£m peak VRAM.
    """
    chunk_size = int(s.IVM_IDENTIFY_IMAGES_PROCESS_CHUNK)
    n = len(indexed_payloads)
    if chunk_size <= 0 or chunk_size >= n:
        return _identify_images_merge_recognition_chunk(indexed_payloads, eng)

    all_results: List[Dict[str, Any]] = []
    all_timing: List[Dict[str, Any]] = []
    all_embs: List[np.ndarray] = []
    total_merge_ms = 0.0
    for start in range(0, n, chunk_size):
        sub = indexed_payloads[start : start + chunk_size]
        res, tim, emb, ms = _identify_images_merge_recognition_chunk(sub, eng)
        all_results.extend(res)
        all_timing.extend(tim)
        all_embs.extend(emb)
        total_merge_ms += ms
        gpu_soft_cleanup(log_label=f"identify_images_chunk:{start // chunk_size + 1}")
    return all_results, all_timing, all_embs, total_merge_ms


def _image_path_for_person(clean_name: str, idx: int) -> str:
    images_dir = Path(s.IVM_FACE_DB_DIR) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in clean_name)
    name = f"{safe}_{int(time.time() * 1000)}_{idx}.jpg"
    return str(images_dir / name)


def _face_bbox_xywh(face: Any) -> Dict[str, int]:
    bb = np.asarray(face.bbox, dtype=np.float64).reshape(-1)
    x1, y1, x2, y2 = bb[:4]
    return {
        "x": int(x1),
        "y": int(y1),
        "w": max(0, int(x2 - x1)),
        "h": max(0, int(y2 - y1)),
    }


def _face_display_crop(image: np.ndarray, xywh: Dict[str, int], out_size: int = 160) -> np.ndarray:
    h_img, w_img = image.shape[:2]
    x, y, w, h = xywh["x"], xywh["y"], xywh["w"], xywh["h"]
    pad_x = int(w * 0.15)
    pad_y = int(h * 0.15)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (out_size, out_size))


def _jpeg_b64(img_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_jpeg_b64(b64_text: str) -> Optional[np.ndarray]:
    try:
        raw = base64.b64decode(b64_text)
        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _default_name_from_stem(stem: str, face_idx: int, face_count: int) -> str:
    base = stem.replace("_", " ").replace("-", " ").strip()
    if not base:
        base = "person"
    if face_count <= 1:
        return base
    return f"{base} {face_idx + 1}"


class RegisterCommitFaceItem(BaseModel):
    name: str
    embedding: List[float]
    crop_jpeg_b64: Optional[str] = None
    source_filename: Optional[str] = None


class RegisterCommitBody(BaseModel):
    faces: List[RegisterCommitFaceItem] = Field(..., min_length=1)


def _name_from_upload_filename(filename: str) -> str:
    """TÃªn Ä‘á»‘i tÆ°á»£ng = stem tÃªn file, bá» dáº¥u; gáº¡ch dÆ°á»›i Ä‘á»•i thÃ nh khoáº£ng tráº¯ng."""
    stem = Path(filename or "").stem.strip()
    if not stem:
        raise ValueError("Thiáº¿u tÃªn file Ä‘á»ƒ suy ra tÃªn Ä‘á»‘i tÆ°á»£ng.")
    s = remove_accents(stem)
    s = s.replace("_", " ").strip()
    s = " ".join(s.split())
    if not s:
        raise ValueError("TÃªn Ä‘á»‘i tÆ°á»£ng (tá»« tÃªn file) sau khi chuáº©n hoÃ¡ bá»‹ rá»—ng.")
    return s


def _append_reg_error(
    store,
    errors: List[Dict[str, Any]],
    filename: str,
    err_msg: str,
    raw: Optional[bytes],
) -> None:
    log_id = store.insert_registration_failure(
        original_filename=filename or None,
        error_message=str(err_msg),
        raw_bytes=raw,
    )
    errors.append({"filename": filename, "error": str(err_msg), "log_id": log_id})


@router.post("/register/preview")
async def register_face_preview(
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
):
    """PhÃ¡t hiá»‡n má»i khuÃ´n máº·t; tráº£ crop + embedding Ä‘á»ƒ UI nháº­p tÃªn trÆ°á»›c khi lÆ°u DB."""
    uploads: List[UploadFile] = []
    if files:
        uploads.extend([u for u in files if u is not None])
    if file is not None:
        uploads.append(file)
    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="Cáº§n Ã­t nháº¥t má»™t áº£nh: form `file` hoáº·c `files`.",
        )

    engine = get_engine()
    images_out: List[Dict[str, Any]] = []
    total_faces = 0

    try:
        for idx, up in enumerate(uploads):
            label = up.filename or f"image_{idx + 1}"
            stem = Path(label).stem
            raw = await up.read()
            if not raw:
                images_out.append({
                    "filename": label,
                    "faces_count": 0,
                    "faces": [],
                    "error": "empty file",
                })
                continue
            try:
                img = _decode_upload(raw)
            except HTTPException as e:
                images_out.append({
                    "filename": label,
                    "faces_count": 0,
                    "faces": [],
                    "error": str(e.detail),
                })
                continue

            faces = engine.analyze_bgr(img)
            face_items: List[Dict[str, Any]] = []
            for face_idx, face in enumerate(faces):
                xywh = _face_bbox_xywh(face)
                default_name = _default_name_from_stem(stem, face_idx, len(faces))
                crop_bgr = _face_display_crop(img, xywh)
                emb = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
                face_items.append({
                    "face_index": face_idx,
                    **xywh,
                    "confidence": float(face.det_score),
                    "default_name": default_name,
                    "crop_jpeg_b64": _jpeg_b64(crop_bgr),
                    "embedding": emb.tolist(),
                })

            total_faces += len(face_items)
            images_out.append({
                "filename": label,
                "faces_count": len(face_items),
                "faces": face_items,
                "error": None if face_items else "no face detected",
            })
    finally:
        maybe_release_global_engine_after_image_infer(log_label="register_preview:done")

    if total_faces == 0:
        raise HTTPException(status_code=400, detail="No faces detected in uploaded images")

    return {
        "success": True,
        "images": images_out,
        "total_faces_count": total_faces,
        "model_tag": s.IVM_MODEL_TAG,
    }


@router.post("/register/commit")
def register_face_commit(body: RegisterCommitBody = Body(...)):
    db = get_face_db()
    expected_dim = 512
    if db.embeddings is not None and getattr(db.embeddings, "shape", (0,))[0] > 0:
        expected_dim = int(db.embeddings.shape[1])

    embeddings: List[np.ndarray] = []
    names: List[str] = []
    image_paths: List[Optional[str]] = []
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for idx, item in enumerate(body.faces):
        clean_name = remove_accents((item.name or "").strip())
        if not clean_name:
            errors.append({"index": idx, "detail": "Name is required"})
            continue
        if not item.embedding:
            errors.append({"index": idx, "name": item.name, "detail": "Missing embedding"})
            continue

        emb = np.asarray(item.embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != expected_dim:
            errors.append({
                "index": idx,
                "name": item.name,
                "detail": f"Invalid embedding dimension (expected {expected_dim})",
            })
            continue

        image_path: Optional[str] = _image_path_for_person(clean_name, idx)
        saved = False
        if item.crop_jpeg_b64:
            crop_img = _decode_jpeg_b64(item.crop_jpeg_b64)
            if crop_img is not None:
                saved = bool(cv2.imwrite(image_path, crop_img))
        if not saved:
            image_path = None

        embeddings.append(emb)
        names.append(clean_name)
        image_paths.append(image_path)
        results.append({
            "name": clean_name,
            "source_filename": item.source_filename,
            "image_path": image_path,
        })

    if not embeddings:
        raise HTTPException(
            status_code=400,
            detail=errors[0]["detail"] if errors else "No valid faces to register",
        )

    stack = np.stack(embeddings, axis=0)
    face_ids = db.add_faces_batch(
        embeddings=stack,
        person_names=names,
        image_paths=image_paths,
    )
    for i, fid in enumerate(face_ids):
        results[i]["face_id"] = int(fid)
        results[i]["person_id"] = int(fid)

    return {
        "success": True,
        "registered_count": len(face_ids),
        "count_success": len(face_ids),
        "failed_count": len(errors),
        "registered": results,
        "errors": errors or None,
        "model_tag": s.IVM_MODEL_TAG,
    }


@router.post("/register")
async def register_face(
    name: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
):
    uploads: List[UploadFile] = []
    if files:
        uploads.extend([u for u in files if u is not None])
    if file is not None:
        uploads.append(file)
    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="Cáº§n Ã­t nháº¥t má»™t áº£nh: form `file` (má»™t áº£nh) hoáº·c `files` (nhiá»u áº£nh, cÃ¹ng tÃªn form).",
        )
    force_name = (name or "").strip()
    engine = get_engine()
    db = get_face_db()
    store = get_store()
    items: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    try:
        for idx, up in enumerate(uploads):
            label = up.filename or f"#{idx}"
            try:
                display_name = force_name if force_name else _name_from_upload_filename(up.filename or "")
            except ValueError as e:
                _append_reg_error(store, errors, label, str(e), None)
                continue
            raw = await up.read()
            if not raw:
                _append_reg_error(store, errors, label, "empty file", None)
                continue
            try:
                img = _decode_upload(raw)
            except HTTPException as e:
                _append_reg_error(store, errors, label, str(e.detail), raw)
                continue
            det = engine.analyze_bgr(img)
            if not det:
                _append_reg_error(store, errors, label, "no face detected", raw)
                continue
            face = max(det, key=lambda f: f.det_score)
            path = _image_path_for_person(display_name, idx)
            cv2.imwrite(path, img)
            fid = db.add_face(face.embedding, display_name, image_path=path)
            items.append(
                {
                    "face_id": fid,
                    "filename": label,
                    "person_name": display_name,
                    "image_path": path,
                    "det_score": face.det_score,
                }
            )
    finally:
        maybe_release_global_engine_after_image_infer(log_label="register:done")

    return {
        "model_tag": s.IVM_MODEL_TAG,
        "name_override": force_name or None,
        "count_success": len(items),
        "count_error": len(errors),
        "registered": items,
        "errors": errors if errors else None,
    }


@router.get("/register/failures")
def list_registration_failures(limit: int = Query(100, ge=1, le=500)):
    rows = get_store().list_registration_failures(limit=limit)
    return {"items": rows, "count": len(rows)}


@router.get("/register/failures/{failure_id}/file")
def get_registration_failure_file(failure_id: str):
    row = get_store().get_registration_failure(failure_id)
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    p_raw = row.get("image_path")
    if not p_raw:
        raise HTTPException(status_code=404, detail="no image saved for this failure")
    p = Path(str(p_raw)).resolve()
    root = Path(s.IVM_DATA_DIR).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=403, detail="invalid path")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="file missing on disk")
    return FileResponse(p, filename=p.name)


@router.delete("/register/failures/{failure_id}")
def delete_registration_failure(failure_id: str):
    ok = get_store().delete_registration_failure(failure_id)
    if not ok:
        raise HTTPException(status_code=404, detail="not found")
    return {"ok": True, "id": failure_id}


def _effective_bulk_max_files(requested: Optional[int]) -> Optional[int]:
    cap = int(s.IVM_BULK_API_MAX_FILES)
    if cap <= 0:
        return requested
    if requested is None:
        return cap
    return min(int(requested), cap)


class RegisterFolderBody(BaseModel):
    root_path: str = Field(..., description="ThÆ° má»¥c trÃªn mÃ¡y cháº¡y API (Ä‘Æ°á»ng dáº«n local)")
    recursive: bool = True
    resume: bool = True
    clear_checkpoint: bool = False
    max_files: Optional[int] = Field(
        None,
        description="Giá»›i háº¡n sá»‘ file; server cÃ²n Ã¡p IVM_BULK_API_MAX_FILES (0 = khÃ´ng giá»›i háº¡n server)",
    )
    resume_skip_failed: Optional[bool] = None
    db_batch_size: Optional[int] = Field(
        None,
        description="KÃ­ch thÆ°á»›c batch ghi DB; None = IVM_BULK_DB_WRITE_BATCH",
    )
    progress_every: Optional[int] = Field(
        None,
        description="Sá»‘ áº£nh má»—i láº§n cáº­p nháº­t file tiáº¿n trÃ¬nh; None = 10 khi cháº¡y API",
    )
    infer_workers: Optional[int] = Field(
        None,
        ge=1,
        le=16,
        description=(
            "Sá»‘ luá»“ng infer bulk (má»—i luá»“ng má»™t FaceAnalysis ONNX). "
            "None = IVM_BULK_INFER_WORKERS (env mÃ¡y chá»§). Giá»›i tráº§n: IVM_BULK_API_MAX_INFER_WORKERS."
        ),
    )
    token: Optional[str] = Field(None, description="Báº¯t buá»™c náº¿u cáº¥u hÃ¬nh IVM_RESET_SECRET")


@router.get("/admin/register-folder/progress")
def admin_register_folder_progress() -> Dict[str, Any]:
    """Äá»c tiáº¿n trÃ¬nh Ä‘Äƒng kÃ½ thÆ° má»¥c (file JSON trÃªn server â€” giá»‘ng `/progress` app Streamlit cÅ©)."""
    p = s.IVM_REGISTER_FOLDER_PROGRESS_PATH
    idle: Dict[str, Any] = {
        "running": False,
        "phase": "idle",
        "root": "",
        "processed": 0,
        "total": 0,
        "registered": 0,
        "success": 0,
        "failed": 0,
        "skipped_checkpoint": 0,
        "progress_pct": 0.0,
        "message": "ChÆ°a cÃ³ job register-folder nÃ o ghi file tiáº¿n trÃ¬nh.",
    }
    if not p.is_file():
        return idle
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        idle["phase"] = "error"
        idle["message"] = "File tiáº¿n trÃ¬nh bá»‹ lá»—i Ä‘á»‹nh dáº¡ng JSON."
        return idle


@router.post("/admin/register-folder")
def admin_register_folder(
    body: RegisterFolderBody,
    x_ivm_reset_token: Optional[str] = Header(None, alias="X-IVM-Reset-Token"),
) -> Dict[str, Any]:
    """ÄÄƒng kÃ½ hÃ ng loáº¡t tá»« thÆ° má»¥c Ä‘Ä©a. Tráº£ vá» ngay; tiáº¿n trÃ¬nh poll qua GET `/ivm/admin/register-folder/progress`."""
    if s.IVM_RESET_SECRET:
        tok = (body.token or x_ivm_reset_token or "").strip()
        if tok != s.IVM_RESET_SECRET:
            raise HTTPException(
                status_code=403,
                detail="Cáº§n token: body.token hoáº·c header X-IVM-Reset-Token khá»›p IVM_RESET_SECRET.",
            )
    if not _bulk_folder_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="register-folder Ä‘ang cháº¡y.")

    root_path = Path(body.root_path.strip())
    try:
        root_ok = assert_bulk_root_allowed(root_path)
    except PermissionError as e:
        _bulk_folder_lock.release()
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        _bulk_folder_lock.release()
        raise HTTPException(status_code=400, detail=str(e)) from e

    max_f = _effective_bulk_max_files(body.max_files)
    batch_override = int(body.db_batch_size) if body.db_batch_size is not None else None
    if batch_override is not None:
        batch_override = max(1, min(batch_override, 4096))

    prog_every = int(body.progress_every) if body.progress_every is not None else 10
    if prog_every <= 0:
        prog_every = 10

    pj = s.IVM_REGISTER_FOLDER_PROGRESS_PATH
    nw_eff = resolve_bulk_infer_workers(body.infer_workers)
    init_stats: Dict[str, Any] = {
        "root": str(root_ok),
        "success": 0,
        "failed": 0,
        "skipped_checkpoint": 0,
        "parallel_workers": nw_eff,
        "_progress_started_at": time.time(),
    }
    write_register_folder_progress(
        pj,
        stats=init_stats,
        running=True,
        phase="queued",
        processed=0,
        total=0,
        message="Äang khá»Ÿi cháº¡y worker Ä‘Äƒng kÃ½",
    )

    def _worker() -> None:
        try:
            run_folder_register(
                root=root_ok,
                engine=None,
                db=get_face_db(),
                store=get_store(),
                recursive=bool(body.recursive),
                resume=bool(body.resume),
                resume_skip_failed=body.resume_skip_failed,
                clear_checkpoint_first=bool(body.clear_checkpoint),
                db_batch_size=batch_override,
                max_files=max_f,
                progress_every=prog_every,
                progress_json_path=pj,
                infer_workers=body.infer_workers,
            )
        finally:
            _bulk_folder_lock.release()

    threading.Thread(target=_worker, name="ivm_register_folder", daemon=True).start()
    return {
        "status": "started",
        "progress_url": "/ivm/admin/register-folder/progress",
        "progress_file": str(pj),
        "bulk_log_file": str(s.IVM_REGISTER_FOLDER_BULK_LOG),
        "root": str(root_ok),
        "infer_workers": nw_eff,
    }


@router.post("/identify_image")
async def identify_image(
    file: UploadFile = File(...),
    distance_threshold: Optional[float] = Query(None),
):
    t_req0 = time.perf_counter()
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    thr = float(distance_threshold) if distance_threshold is not None else s.IVM_DISTANCE_THRESHOLD
    try:
        t0 = time.perf_counter()
        img = _decode_upload(raw)
        decode_ms = (time.perf_counter() - t0) * 1000
        infer_prof: Dict[str, float] = {}
        faces = get_engine().analyze_bgr(img, timing_out=infer_prof)
        detect_ms = float(infer_prof.get("detect_ms", 0.0))
        embedding_ms = float(infer_prof.get("embedding_ms", 0.0))
        infer_ms = detect_ms + embedding_ms
        out, search_ms = _identify_faces_payload(faces, get_face_db(), thr)
    finally:
        maybe_release_global_engine_after_image_infer(log_label="identify_image")
    total_ms = (time.perf_counter() - t_req0) * 1000
    fname = file.filename or "image"
    fc = len(out)
    timing: Dict[str, Any] = {
        "total_ms": _round_ms(total_ms),
        "images": [
            {
                "filename": fname,
                "decode_ms": _round_ms(decode_ms),
                "detect_ms": _round_ms(detect_ms),
                "embedding_ms": _round_ms(embedding_ms),
                "infer_ms": _round_ms(infer_ms),
                "face_count": fc,
                "search_ms": _round_ms(search_ms),
            }
        ],
        "search_batch_ms": _round_ms(search_ms),
        "search_batch_face_count": fc,
        "search_batch_amortization_note": (
            "search_batch_ms is one DB call for this request; for a single image it covers all "
            f"{fc} detected face(s)."
            if fc
            else "No faces detected; search_batch not run."
        ),
        "avg_detect_ms_per_image": _round_ms(detect_ms),
        "avg_embedding_ms_per_image": _round_ms(embedding_ms),
    }
    return {"faces": out, "timing": timing}


@router.post("/identify_images")
async def identify_images(
    files: List[UploadFile] = File(...),
    distance_threshold: Optional[float] = Query(None),
    infer_workers: Optional[int] = Query(None),
):
    """Nhiá»u áº£nh má»™t request; gom embedding (cross-image recognition khi infer_workers=1 vÃ  IVM_IDENTIFY_IMAGES_MERGE_REC) rá»“i search_batch má»™t láº§n."""
    max_img = int(s.IVM_IDENTIFY_BATCH_MAX_FILES)
    if max_img > 0 and len(files) > max_img:
        raise HTTPException(
            status_code=400,
            detail=f"Tá»‘i Ä‘a {max_img} áº£nh má»—i request (IVM_IDENTIFY_BATCH_MAX_FILES). Äáº·t 0 Ä‘á»ƒ khÃ´ng giá»›i háº¡n.",
        )
    if not files:
        raise HTTPException(status_code=400, detail="Cáº§n Ã­t nháº¥t má»™t áº£nh (form field `files`).")

    t_req0 = time.perf_counter()
    thr = float(distance_threshold) if distance_threshold is not None else s.IVM_DISTANCE_THRESHOLD
    db = get_face_db()
    nw_resolved = s.resolve_identify_infer_workers(infer_workers)
    nw_pool = max(1, min(nw_resolved, len(files)))

    try:
        indexed_payloads: List[Tuple[int, str, bytes]] = []
        for idx, up in enumerate(files):
            fname = up.filename or f"image_{idx}"
            raw = await up.read()
            indexed_payloads.append((idx, fname, raw))

        results: List[Dict[str, Any]] = []
        embs_flat: List[np.ndarray] = []
        timing_images: List[Dict[str, Any]] = []
        parallel_infer_wall_ms: Optional[float] = None
        merged_recognition_ms: Optional[float] = None

        if nw_pool <= 1:
            eng = get_engine()
            if s.IVM_IDENTIFY_IMAGES_MERGE_REC:
                res_m, tim_m, emb_m, merge_ms = _identify_images_merge_recognition(indexed_payloads, eng)
                results = res_m
                timing_images = tim_m
                embs_flat = emb_m
                merged_recognition_ms = merge_ms
            else:
                for _idx, fname, raw in indexed_payloads:
                    block = _identify_decode_infer_only(fname, raw, eng)
                    results.append(block["result_entry"])
                    timing_images.append(block["timing"])
                    embs_flat.extend(block["embeddings"])
        else:
            ctx_ids = s.ivm_bulk_worker_ctx_ids()
            engines: List[InsightFaceEngine] = []
            try:
                for i in range(nw_pool):
                    engines.append(InsightFaceEngine(ctx_id=ctx_ids[i % len(ctx_ids)]))
                eq: queue.Queue[InsightFaceEngine] = queue.Queue()
                for e in engines:
                    eq.put(e)

                def _run_one(payload: Tuple[int, str, bytes]) -> Tuple[int, Dict[str, Any]]:
                    _i, fname, raw = payload
                    worker_eng = eq.get()
                    try:
                        return (_i, _identify_decode_infer_only(fname, raw, worker_eng))
                    finally:
                        eq.put(worker_eng)

                t_par0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=nw_pool) as ex:
                    futs = [ex.submit(_run_one, item) for item in indexed_payloads]
                    per_slot: List[Optional[Dict[str, Any]]] = [None] * len(indexed_payloads)
                    for fut in as_completed(futs):
                        i, block = fut.result()
                        per_slot[i] = block
                parallel_infer_wall_ms = (time.perf_counter() - t_par0) * 1000

                for i in range(len(indexed_payloads)):
                    block = per_slot[i]
                    assert block is not None
                    results.append(block["result_entry"])
                    timing_images.append(block["timing"])
                    embs_flat.extend(block["embeddings"])
            finally:
                # Pool táº¡m cho batch nÃ y â€” khÃ´ng Ä‘á»¥ng state.engine khá»Ÿi táº¡o lÃºc boot app.
                for e in engines:
                    _dispose_worker_engine(e)
                engines.clear()
                _release_bulk_infer_resources()

        search_batch_ms = 0.0
        total_faces = len(embs_flat)
        if embs_flat:
            t_s = time.perf_counter()
            mat = np.stack(embs_flat, axis=0)
            batch_matches = db.search_batch(mat, k=s.IVM_SEARCH_K, distance_threshold=thr)
            search_batch_ms = (time.perf_counter() - t_s) * 1000
            bi = 0
            for item in results:
                if item.get("error"):
                    continue
                for face in item.get("faces") or []:
                    face["matches"] = batch_matches[bi]
                    bi += 1
            del mat, batch_matches

        indexed_payloads.clear()
        embs_flat.clear()
        gpu_soft_cleanup(log_label="identify_images:post_search")

        if total_faces > 0:
            for row in timing_images:
                if row.get("error") or not row.get("face_count"):
                    row["search_ms"] = 0.0
                    continue
                fc_i = int(row["face_count"])
                row["search_ms"] = _round_ms(search_batch_ms * (fc_i / total_faces))
        else:
            for row in timing_images:
                if not row.get("error"):
                    row["search_ms"] = 0.0

        total_req_ms = (time.perf_counter() - t_req0) * 1000
        n_img_faces = sum(1 for r in timing_images if r.get("face_count"))
        note = (
            f"search_batch_ms is one DB call amortized by face count across {total_faces} face(s) "
            f"from {n_img_faces} image(s) with detections."
            if total_faces
            else "No batch search (no faces detected)."
        )
        timing: Dict[str, Any] = {
            "total_ms": _round_ms(total_req_ms),
            "images": timing_images,
            "search_batch_ms": _round_ms(search_batch_ms),
            "search_batch_face_count": total_faces,
            "search_batch_amortization_note": note,
            "infer_workers": nw_resolved,
        }
        if merged_recognition_ms is not None:
            timing["merged_recognition_ms"] = _round_ms(merged_recognition_ms)
            timing["cross_image_rec_batch"] = True
            timing["rec_get_feat_max_batch"] = int(s.IVM_REC_GET_FEAT_MAX_BATCH)
        eligible_avg = [r for r in timing_images if not r.get("error")]
        if eligible_avg:
            timing["avg_detect_ms_per_image"] = _round_ms(
                sum(float(r.get("detect_ms") or 0.0) for r in eligible_avg) / len(eligible_avg)
            )
            timing["avg_embedding_ms_per_image"] = _round_ms(
                sum(float(r.get("embedding_ms") or 0.0) for r in eligible_avg) / len(eligible_avg)
            )
        if parallel_infer_wall_ms is not None:
            timing["parallel_infer_wall_ms"] = _round_ms(parallel_infer_wall_ms)

        return {
            "model_tag": s.IVM_MODEL_TAG,
            "distance_threshold": thr,
            "results": results,
            "timing": timing,
        }
    finally:
        maybe_release_global_engine_after_image_infer(log_label="identify_images")


class RecognitionIngest(BaseModel):
    ts_utc: Optional[float] = None
    source: str = Field(default="stream")
    person_ref: str
    face_id: Optional[int] = None
    display_name: Optional[str] = None
    match_score: Optional[float] = None
    distance: Optional[float] = None
    det_score: Optional[float] = None
    recording_segment_id: Optional[int] = None
    offset_start_s: Optional[float] = None
    offset_end_s: Optional[float] = None
    gender: Optional[int] = None
    age: Optional[int] = None
    crop_jpeg_b64: Optional[str] = None
    bbox: Optional[List[float]] = None
    armed: Optional[bool] = None
    frame_armed: Optional[bool] = None
    weapon_types: Optional[List[str]] = None
    weapon_status: Optional[str] = None
    weapon_label: Optional[str] = None
    weapon_score: Optional[float] = None
    weapon_crop_jpeg_b64: Optional[str] = None
    weapon_crops_jpeg_b64: Optional[List[Dict[str, Any]]] = None
    track_scene_crop_jpeg_b64: Optional[str] = None


@router.post("/cameras/{camera_id}/events/recognition")
def ingest_recognition(camera_id: str, body: RecognitionIngest) -> Dict[str, Any]:
    ts = float(body.ts_utc) if body.ts_utc is not None else time.time()
    rec_opt = get_recorders().get(camera_id)
    seg_id = body.recording_segment_id
    off0 = body.offset_start_s
    off1 = body.offset_end_s
    if rec_opt and seg_id is None:
        sid, path, t0, _ = rec_opt.current_archive_ref()
        if sid is not None and path:
            seg_id = sid
            off0 = max(0.0, ts - t0)
            off1 = off0 + float(s.IVM_EXPORT_CUT_MIN_DURATION_S)
    eid, merged = get_store().merge_or_insert_event(
        debounce_s=s.IVM_EVENT_DEBOUNCE_S,
        ts_utc=ts,
        camera_id=camera_id,
        source=body.source,
        person_ref=body.person_ref,
        face_id=body.face_id,
        display_name=body.display_name,
        match_score=body.match_score,
        distance=body.distance,
        det_score=body.det_score,
        model_tag=s.IVM_MODEL_TAG,
        recording_segment_id=seg_id,
        offset_start_s=off0,
        offset_end_s=off1,
        gender=body.gender,
        age=body.age,
    )
    weapon_payload = None
    if body.armed is not None or body.weapon_types:
        weapon_payload = {
            "armed": bool(body.armed),
            "frame_armed": bool(body.frame_armed) if body.frame_armed is not None else bool(body.armed),
            "weapon_types": list(body.weapon_types or []),
            "weapon_status": body.weapon_status or ("co_vu_khi" if body.armed else "an_toan"),
            "weapon_label": body.weapon_label,
            "weapon_score": body.weapon_score,
        }
    tracking = get_store().apply_tracking_update(
        eid,
        merged=merged,
        det_score=body.det_score,
        crop_jpeg_b64=body.crop_jpeg_b64,
        bbox=body.bbox,
        weapon=weapon_payload,
        weapon_crop_jpeg_b64=body.weapon_crop_jpeg_b64,
        weapon_crops_jpeg_b64=body.weapon_crops_jpeg_b64,
        track_scene_crop_jpeg_b64=body.track_scene_crop_jpeg_b64,
    )
    return {
        "event_id": eid,
        "merged": merged,
        "frame_hits": tracking.get("frame_hits"),
        "crop_saved": bool(tracking.get("crop_path")),
    }
