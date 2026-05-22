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
from identity_vm_app.api.deps import get_engine, get_face_db, get_recorders, get_store
from identity_vm_app.bulk_folder_register import (
    _dispose_worker_engine,
    _release_bulk_infer_resources,
    assert_bulk_root_allowed,
    resolve_bulk_infer_workers,
    run_folder_register,
    write_register_folder_progress,
)
from identity_vm_app.engine.gpu_cleanup import (
    gpu_soft_cleanup,
    maybe_release_global_engine_after_image_infer,
)
from identity_vm_app.engine.insightface_engine import InsightFaceEngine
from identity_vm_app.data_reset import execute_clear_reports, execute_full_reset
from identity_vm_app.recorder.rolling_ffmpeg import RollingFfmpegRecorder
from identity_vm_app.services.export_cut import export_segment_cut
from identity_vm_app.services.event_crops import load_crop_bytes
from identity_vm_app.services.export_webm import build_frames_from_events, export_crops_to_webm
from identity_vm_app.camera_analyze_control import (
    get_analyze_enabled,
    set_analyze_enabled,
    snapshot_states,
)
from camera_channel_config import load_camera_channel_specs
from packages.persistence.face_database import FaceDatabase
from services.text import remove_accents

router = APIRouter(prefix="/ivm", tags=["identity-vm"])

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
    """Chuyển kết quả analyze_bgr → payload JSON; một lần search_batch cho toàn bộ mặt. Trả về (payload, search_ms)."""
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
    """Decode + analyze_bgr cho một ảnh (dùng trong identify_images tuần tự hoặc worker song song)."""
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
    Decode → detect+align → gom crop → get_feat theo lô (IVM_REC_GET_FEAT_MAX_BATCH) cho một nhóm ảnh.
    Trả (results, timing_images, embeddings_flat, merged_recognition_ms).
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
    Merge recognition cross-image; chia lô theo IVM_IDENTIFY_IMAGES_PROCESS_CHUNK để giảm peak VRAM.
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
    """Tên đối tượng = stem tên file, bỏ dấu; gạch dưới đổi thành khoảng trắng."""
    stem = Path(filename or "").stem.strip()
    if not stem:
        raise ValueError("Thiếu tên file để suy ra tên đối tượng.")
    s = remove_accents(stem)
    s = s.replace("_", " ").strip()
    s = " ".join(s.split())
    if not s:
        raise ValueError("Tên đối tượng (từ tên file) sau khi chuẩn hoá bị rỗng.")
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
    """Phát hiện mọi khuôn mặt; trả crop + embedding để UI nhập tên trước khi lưu DB."""
    uploads: List[UploadFile] = []
    if files:
        uploads.extend([u for u in files if u is not None])
    if file is not None:
        uploads.append(file)
    if not uploads:
        raise HTTPException(
            status_code=400,
            detail="Cần ít nhất một ảnh: form `file` hoặc `files`.",
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
            detail="Cần ít nhất một ảnh: form `file` (một ảnh) hoặc `files` (nhiều ảnh, cùng tên form).",
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


class ResetAllDataBody(BaseModel):
    confirm: str = Field(..., description='Phải gõ đúng DELETE_ALL')
    wipe_archive: bool = Field(False, description="Xóa luôn thư mục ghi hình archive (video)")
    token: Optional[str] = Field(None, description="Bắt buộc nếu cấu hình IVM_RESET_SECRET")


@router.post("/admin/reset-all-data")
def admin_reset_all_data(
    body: ResetAllDataBody,
    x_ivm_reset_token: Optional[str] = Header(None, alias="X-IVM-Reset-Token"),
) -> Dict[str, Any]:
    if body.confirm != "DELETE_ALL":
        raise HTTPException(
            status_code=400,
            detail='Trường confirm phải đúng chuỗi DELETE_ALL.',
        )
    if s.IVM_RESET_SECRET:
        tok = (body.token or x_ivm_reset_token or "").strip()
        if tok != s.IVM_RESET_SECRET:
            raise HTTPException(
                status_code=403,
                detail="Cần token reset: body.token hoặc header X-IVM-Reset-Token khớp IVM_RESET_SECRET.",
            )
    try:
        return execute_full_reset(wipe_archive=bool(body.wipe_archive))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


def _effective_bulk_max_files(requested: Optional[int]) -> Optional[int]:
    cap = int(s.IVM_BULK_API_MAX_FILES)
    if cap <= 0:
        return requested
    if requested is None:
        return cap
    return min(int(requested), cap)


class RegisterFolderBody(BaseModel):
    root_path: str = Field(..., description="Thư mục trên máy chạy API (đường dẫn local)")
    recursive: bool = True
    resume: bool = True
    clear_checkpoint: bool = False
    max_files: Optional[int] = Field(
        None,
        description="Giới hạn số file; server còn áp IVM_BULK_API_MAX_FILES (0 = không giới hạn server)",
    )
    resume_skip_failed: Optional[bool] = None
    db_batch_size: Optional[int] = Field(
        None,
        description="Kích thước batch ghi DB; None = IVM_BULK_DB_WRITE_BATCH",
    )
    progress_every: Optional[int] = Field(
        None,
        description="Số ảnh mỗi lần cập nhật file tiến trình; None = 10 khi chạy API",
    )
    infer_workers: Optional[int] = Field(
        None,
        ge=1,
        le=16,
        description=(
            "Số luồng infer bulk (mỗi luồng một FaceAnalysis ONNX). "
            "None = IVM_BULK_INFER_WORKERS (env máy chủ). Giới trần: IVM_BULK_API_MAX_INFER_WORKERS."
        ),
    )
    token: Optional[str] = Field(None, description="Bắt buộc nếu cấu hình IVM_RESET_SECRET")


@router.get("/admin/register-folder/progress")
def admin_register_folder_progress() -> Dict[str, Any]:
    """Đọc tiến trình đăng ký thư mục (file JSON trên server — giống `/progress` app Streamlit cũ)."""
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
        "message": "Chưa có job register-folder nào ghi file tiến trình.",
    }
    if not p.is_file():
        return idle
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        idle["phase"] = "error"
        idle["message"] = "File tiến trình bị lỗi định dạng JSON."
        return idle


@router.post("/admin/register-folder")
def admin_register_folder(
    body: RegisterFolderBody,
    x_ivm_reset_token: Optional[str] = Header(None, alias="X-IVM-Reset-Token"),
) -> Dict[str, Any]:
    """Đăng ký hàng loạt từ thư mục đĩa. Trả về ngay; tiến trình poll qua GET `/ivm/admin/register-folder/progress`."""
    if s.IVM_RESET_SECRET:
        tok = (body.token or x_ivm_reset_token or "").strip()
        if tok != s.IVM_RESET_SECRET:
            raise HTTPException(
                status_code=403,
                detail="Cần token: body.token hoặc header X-IVM-Reset-Token khớp IVM_RESET_SECRET.",
            )
    if not _bulk_folder_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="register-folder đang chạy.")

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
        message="Đang khởi chạy worker đăng ký",
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
    """Nhiều ảnh một request; gom embedding (cross-image recognition khi infer_workers=1 và IVM_IDENTIFY_IMAGES_MERGE_REC) rồi search_batch một lần."""
    max_img = int(s.IVM_IDENTIFY_BATCH_MAX_FILES)
    if max_img > 0 and len(files) > max_img:
        raise HTTPException(
            status_code=400,
            detail=f"Tối đa {max_img} ảnh mỗi request (IVM_IDENTIFY_BATCH_MAX_FILES). Đặt 0 để không giới hạn.",
        )
    if not files:
        raise HTTPException(status_code=400, detail="Cần ít nhất một ảnh (form field `files`).")

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
                # Pool tạm cho batch này — không đụng state.engine khởi tạo lúc boot app.
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


@router.get("/cameras")
def list_cameras() -> Dict[str, Any]:
    specs = load_camera_channel_specs(s.IVM_CAMERA_CONFIG)
    return {"cameras": [{"id": str(c["id"]), "source": c["source"]} for c in specs]}


class CameraAnalyzeBody(BaseModel):
    enabled: bool
    sample_fps: Optional[float] = None
    display_name: Optional[str] = None
    distance_threshold: Optional[float] = None
    save_crops: Optional[bool] = None


@router.get("/cameras/analyze")
def list_analyze_states() -> Dict[str, Any]:
    """Trạng thái nhận diện từng camera (mặc định tắt nếu chưa đặt)."""
    from identity_vm_app.services.camera_live_session import get_active_session
    from identity_vm_app.services.video_analyze_fps import sample_fps_label

    specs = load_camera_channel_specs(s.IVM_CAMERA_CONFIG)
    states: Dict[str, bool] = {}
    sessions: Dict[str, Dict[str, Any]] = {}
    for c in specs:
        cid = str(c["id"])
        en = get_analyze_enabled(cid)
        states[cid] = en
        if en:
            live = get_active_session(cid)
            if live is not None:
                sf = float(live.sample_fps)
                sessions[cid] = {
                    "job_id": live.job_id,
                    "sample_fps": sf,
                    "sample_fps_label": sample_fps_label(sf),
                }
    return {"states": states, "sessions": sessions}


@router.get("/cameras/{camera_id}/analyze")
def get_camera_analyze(camera_id: str) -> Dict[str, Any]:
    specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    return {"camera_id": camera_id, "enabled": get_analyze_enabled(camera_id)}


@router.post("/cameras/{camera_id}/analyze")
def set_camera_analyze(camera_id: str, body: CameraAnalyzeBody) -> Dict[str, Any]:
    specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")

    from identity_vm_app.camera_analyze_control import is_analyze_stopping
    from identity_vm_app.camera_recognition.activity_log import recent as activity_recent
    from identity_vm_app.camera_recognition.analyze_recording import get_visual_session
    from identity_vm_app.camera_recognition.hub import ensure_recognition_hub_started, get_recognition_hub
    from identity_vm_app.services.camera_live_session import get_active_session
    from identity_vm_app.services.video_analyze_fps import parse_sample_fps, sample_fps_label

    stream_fps = 10.0
    start_fc = 0
    if body.enabled:
        ensure_recognition_hub_started()
        w = get_recognition_hub().get_worker(camera_id)
        if w is not None:
            w.ensure_rtsp_reader()
            stream_fps = max(1.0, float(w.reader.fps_actual) or 10.0)
            start_fc = int(w.reader.frame_count)

    sf = None
    if body.enabled:
        try:
            sf = parse_sample_fps(
                body.sample_fps if body.sample_fps is not None else float(s.IVM_CAMERA_DEFAULT_SAMPLE_FPS)
            )
        except ValueError as ex:
            raise HTTPException(status_code=400, detail=str(ex)) from ex

    toggle = set_analyze_enabled(
        camera_id,
        body.enabled,
        sample_fps=sf,
        display_name=body.display_name,
        distance_threshold=body.distance_threshold,
        save_crops=body.save_crops,
        stream_fps=stream_fps,
        start_frame_count=start_fc,
    )

    w = get_recognition_hub().get_worker(camera_id)
    hub_ok = w is not None
    reader_ok = bool(w and w.reader.is_connected) if w else False
    rec = get_recorders().get(camera_id)
    archive_running = bool(rec and rec.is_running())
    live = get_active_session(camera_id) if (body.enabled or is_analyze_stopping(camera_id)) else None
    visual = get_visual_session(camera_id) if (body.enabled or is_analyze_stopping(camera_id)) else None
    sess = toggle.get("session") or {}
    queue_pending = int(toggle.get("infer_queue_pending") or 0)
    if w is not None and toggle.get("draining"):
        queue_pending = max(queue_pending, w.infer_queue_size() + (1 if w.infer_in_progress else 0))
    return {
        "camera_id": camera_id,
        "enabled": body.enabled,
        "draining": bool(toggle.get("draining")),
        "infer_queue_pending": queue_pending,
        "hub_worker_running": hub_ok,
        "reader_connected": reader_ok,
        "archive_recording": archive_running,
        "visual_recording": visual,
        "job_id": sess.get("job_id") or (live.job_id if live else None),
        "sample_fps": sf if body.enabled else None,
        "sample_fps_label": sample_fps_label(sf) if body.enabled and sf is not None else None,
        "analysis_mode": "single_thread",
        "session": sess,
        "hint": (
            "Khi BẬT: ghi archive RTSP + session.mp4 + báo cáo DB. "
            "Xem: GET /ivm/cameras/{id}/analyze/sessions"
        ),
        "recent_activity": activity_recent(camera_id, limit=5),
    }


@router.get("/cameras/{camera_id}/analyze/activity")
def get_camera_analyze_activity(
    camera_id: str,
    limit: int = Query(40, ge=1, le=200),
) -> Dict[str, Any]:
    specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    from identity_vm_app.camera_recognition.activity_log import recent as activity_recent
    from identity_vm_app.camera_recognition.hub import get_recognition_hub

    w = get_recognition_hub().get_worker(camera_id)
    return {
        "camera_id": camera_id,
        "enabled": get_analyze_enabled(camera_id),
        "hub_worker_running": w is not None,
        "reader_connected": bool(w and w.reader.is_connected) if w else False,
        "reader_fps": float(w.reader.fps_actual) if w else 0.0,
        "last_meta": w.get_meta() if w else {},
        "activity": activity_recent(camera_id, limit=limit),
    }


@router.get("/cameras/analyze/activity")
def get_all_analyze_activity(limit: int = Query(60, ge=1, le=200)) -> Dict[str, Any]:
    from identity_vm_app.camera_recognition.activity_log import recent as activity_recent

    return {
        "states": snapshot_states(),
        "activity": activity_recent(None, limit=limit),
    }


class RecorderStartBody(BaseModel):
    source_url: Optional[str] = None


@router.post("/cameras/{camera_id}/recorder/start")
def recorder_start(camera_id: str, body: Optional[RecorderStartBody] = Body(default=None)) -> Dict[str, Any]:
    specs = {str(c["id"]): c for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    src = (body.source_url if body else None) or specs[camera_id]["source"]
    if not isinstance(src, str) or not (src.startswith("rtsp://") or src.startswith("http")):
        raise HTTPException(
            status_code=400,
            detail="Recorder MVP needs RTSP/HTTP URL; set source in camera_config or pass source_url",
        )
    store = get_store()
    prev_holder: Dict[str, Optional[int]] = {"id": None}

    def hook(path: Path, started: float) -> int:
        prev = prev_holder["id"]
        if prev is not None:
            store.finalize_segment(prev, started)
        sid = store.insert_segment(camera_id, str(path), started, None)
        prev_holder["id"] = sid
        return sid

    rec = RollingFfmpegRecorder(camera_id, src, segment_hook=hook)
    get_recorders().start(camera_id, rec)
    return {"camera_id": camera_id, "started": True, "source_url": src}


@router.post("/cameras/{camera_id}/recorder/stop")
def recorder_stop(camera_id: str) -> Dict[str, Any]:
    get_recorders().stop(camera_id)
    return {"camera_id": camera_id, "stopped": True}


@router.get("/cameras/{camera_id}/recorder/status")
def recorder_status(camera_id: str) -> Dict[str, Any]:
    rec = get_recorders().get(camera_id)
    if rec is None:
        return {"camera_id": camera_id, "running": False}
    sid, path, t0, now = rec.current_archive_ref()
    return {
        "camera_id": camera_id,
        "running": rec.is_running(),
        "segment_id": sid,
        "archive_path": path,
        "segment_started_utc": t0,
        "now_utc": now,
    }


class ClearReportsBody(BaseModel):
    confirm: str = Field(..., description='Phải gõ đúng DELETE_REPORTS')
    camera_id: Optional[str] = Field(
        None,
        description="Chỉ xóa báo cáo camera này; bỏ trống = tất cả camera",
    )
    wipe_archive: bool = Field(
        False,
        description="Xóa luôn file archive RTSP và segment DB",
    )


@router.post("/reports/clear")
def clear_all_reports(body: ClearReportsBody) -> Dict[str, Any]:
    """Xóa toàn bộ báo cáo nhận diện (giữ thư viện khuôn mặt đăng ký)."""
    if body.confirm != "DELETE_REPORTS":
        raise HTTPException(
            status_code=400,
            detail='Trường confirm phải đúng chuỗi DELETE_REPORTS.',
        )
    if body.camera_id:
        specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
        if body.camera_id not in specs:
            raise HTTPException(status_code=404, detail="Unknown camera_id")
    try:
        return execute_clear_reports(
            camera_id=body.camera_id,
            wipe_archive=bool(body.wipe_archive),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@router.get("/reports/summary")
def all_cameras_report(
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """Báo cáo tổng hợp theo từng camera (kể cả camera chưa có sự kiện)."""
    now = time.time()
    to_ts_f = float(to_ts) if to_ts is not None else now
    from_ts_f = float(from_ts) if from_ts is not None else now - 86400.0
    specs = load_camera_channel_specs(s.IVM_CAMERA_CONFIG)
    cam_ids = [str(c["id"]) for c in specs]
    rows = get_store().all_cameras_report_summary(from_ts_f, to_ts_f, camera_ids=cam_ids)
    for row in rows:
        cid = str(row["camera_id"])
        row["recognition_enabled"] = get_analyze_enabled(cid)
    return {"from_ts": from_ts_f, "to_ts": to_ts_f, "cameras": rows}


@router.get("/cameras/{camera_id}/reports/summary")
def camera_report(
    camera_id: str,
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
):
    specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    now = time.time()
    to_ts = float(to_ts) if to_ts is not None else now
    from_ts = float(from_ts) if from_ts is not None else now - 86400.0
    summary, subjects = get_store().camera_report_summary(camera_id, from_ts, to_ts)
    summary["recognition_enabled"] = get_analyze_enabled(camera_id)
    return {"summary": summary, "subjects": subjects}


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


@router.get("/cameras/{camera_id}/reports/tracks")
def camera_report_tracks(
    camera_id: str,
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    limit: int = Query(300, ge=1, le=2000),
    known_only: bool = Query(False, description="Chỉ người đã định danh (bỏ unknown)"),
    person_ref: Optional[str] = None,
) -> Dict[str, Any]:
    """Báo cáo chi tiết: crop ảnh, danh tính, số frame tracking mỗi lần xuất hiện."""
    specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    now = time.time()
    to_ts_f = float(to_ts) if to_ts is not None else now
    from_ts_f = float(from_ts) if from_ts is not None else now - 86400.0
    items = get_store().list_camera_track_events(
        camera_id,
        from_ts_f,
        to_ts_f,
        limit=limit,
        known_only=known_only,
        person_ref=person_ref,
    )
    _attach_track_crop_urls(items)
    return {
        "camera_id": camera_id,
        "from_ts": from_ts_f,
        "to_ts": to_ts_f,
        "tracks": items,
    }


def _attach_track_crop_urls(items: List[Dict[str, Any]]) -> None:
    from identity_vm_app.services.weapon_crops import normalize_weapon_class

    for it in items:
        eid = str(it["event_id"])
        it["crop_url"] = f"/ivm/events/{eid}/crop.jpg" if it.get("crop_path") else None
        it["weapon_crop_url"] = (
            f"/ivm/events/{eid}/weapon-crop.jpg" if it.get("weapon_crop_path") else None
        )
        wcrops = it.get("weapon_crops") or []
        seen: set[str] = set()
        wurls: List[Dict[str, str]] = []
        for w in wcrops:
            if not isinstance(w, dict) or not w.get("path"):
                continue
            cls = normalize_weapon_class(w.get("class"))
            if cls in seen:
                continue
            seen.add(cls)
            wurls.append(
                {
                    "class": cls,
                    "url": f"/ivm/events/{eid}/weapon-crop/{cls}.jpg",
                }
            )
        for raw_t in it.get("weapon_types") or []:
            cls = normalize_weapon_class(raw_t)
            if cls in seen:
                continue
            seen.add(cls)
            wurls.append(
                {
                    "class": cls,
                    "url": f"/ivm/events/{eid}/weapon-crop/{cls}.jpg",
                }
            )
        if not wurls and it.get("weapon_crop_path"):
            wurls.append({"class": "weapon", "url": f"/ivm/events/{eid}/weapon-crop.jpg"})
        it["weapon_crop_urls"] = wurls
        it["track_scene_url"] = (
            f"/ivm/events/{eid}/track-scene.jpg" if it.get("track_scene_path") else None
        )


@router.get("/cameras/{camera_id}/reports/by-person/{person_ref}")
def reports_by_person(
    camera_id: str,
    person_ref: str,
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    limit: int = Query(2000, ge=1, le=2000),
) -> Dict[str, Any]:
    """
    Chi tiết một người đã định danh (tương tự VisionMaster get-by-track).
    Mỗi phần tử = một lần xuất hiện, có crop và frame_hits.
    """
    specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    if person_ref == "unknown":
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ người đã định danh")
    now = time.time()
    to_ts_f = float(to_ts) if to_ts is not None else now
    from_ts_f = float(from_ts) if from_ts is not None else now - 86400.0
    items = get_store().list_camera_track_events(
        camera_id,
        from_ts_f,
        to_ts_f,
        limit=limit,
        person_ref=person_ref,
        order_asc=True,
    )
    _attach_track_crop_urls(items)
    display_name = None
    if items:
        display_name = items[-1].get("display_name") or items[-1].get("identity")
    total_frames = sum(int(x.get("frame_hits") or 1) for x in items)
    return {
        "camera_id": camera_id,
        "person_ref": person_ref,
        "display_name": display_name,
        "from_ts": from_ts_f,
        "to_ts": to_ts_f,
        "appearance_count": len(items),
        "total_frames": total_frames,
        "appearances": items,
    }


@router.get("/cameras/{camera_id}/reports/by-group")
def reports_by_group(
    camera_id: str,
    group_key: str = Query(..., description="Khóa gom tên (normalize) hoặc truyền display_name"),
    display_name: Optional[str] = None,
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    limit: int = Query(2000, ge=1, le=2000),
) -> Dict[str, Any]:
    """Chi tiết một đối tượng theo tên — mọi lần xuất hiện / frame trong khoảng thời gian."""
    from identity_vm_app.report_grouping import filter_tracks_for_group, group_persons_by_display_name

    specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    now = time.time()
    to_ts_f = float(to_ts) if to_ts is not None else now
    from_ts_f = float(from_ts) if from_ts is not None else now - 86400.0
    tracks = get_store().list_camera_track_events(
        camera_id, from_ts_f, to_ts_f, limit=limit, known_only=True, order_asc=True
    )
    _attach_track_crop_urls(tracks)
    filtered = filter_tracks_for_group(
        tracks, group_key=group_key, display_name=display_name or group_key
    )
    grouped = group_persons_by_display_name(filtered)
    if not grouped:
        raise HTTPException(status_code=404, detail="Không tìm thấy đối tượng trong khoảng thời gian")
    person = grouped[0]
    items = list(person.get("events") or [])
    return {
        "camera_id": camera_id,
        "group_key": person.get("group_key"),
        "display_name": person.get("display_name"),
        "person_refs": person.get("person_refs"),
        "from_ts": from_ts_f,
        "to_ts": to_ts_f,
        "appearance_count": len(items),
        "total_frames": person.get("total_frames"),
        "appearances": items,
    }


@router.get("/cameras/{camera_id}/reports/export-webm")
def export_person_webm(
    camera_id: str,
    person_ref: Optional[str] = Query(None, description="Một person_ref (nếu không dùng group_key)"),
    group_key: Optional[str] = Query(None, description="Khóa gom theo tên hiển thị"),
    display_name: Optional[str] = Query(None, description="Tên hiển thị (gom mọi face_id cùng tên)"),
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    fps: float = Query(5.0, ge=1.0, le=30.0),
):
    """Xuất WebM từ ảnh crop — mỗi ảnh lặp theo frame_hits (shortcut tóm tắt xuất hiện)."""
    from identity_vm_app.report_grouping import filter_tracks_for_group

    specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    if not person_ref and not group_key and not display_name:
        raise HTTPException(
            status_code=400,
            detail="Cần person_ref hoặc group_key hoặc display_name",
        )
    now = time.time()
    to_ts_f = float(to_ts) if to_ts is not None else now
    from_ts_f = float(from_ts) if from_ts is not None else now - 86400.0
    if person_ref and person_ref != "unknown" and not group_key and not display_name:
        items = get_store().list_camera_track_events(
            camera_id,
            from_ts_f,
            to_ts_f,
            limit=2000,
            person_ref=person_ref,
            order_asc=True,
        )
    else:
        tracks = get_store().list_camera_track_events(
            camera_id, from_ts_f, to_ts_f, limit=2000, known_only=True, order_asc=True
        )
        items = filter_tracks_for_group(
            tracks,
            group_key=group_key,
            display_name=display_name or group_key,
        )
    frames = build_frames_from_events(items)
    if not frames:
        raise HTTPException(
            status_code=404,
            detail="Không có ảnh crop trong khoảng thời gian (cần sự kiện sau khi bật lưu crop)",
        )
    try:
        out_path = export_crops_to_webm(frames, fps=fps)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Xuất WebM thất bại: {ex}") from ex
    label = display_name or group_key or person_ref or "person"
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(label))[:40]
    return FileResponse(
        path=str(out_path),
        media_type="video/webm",
        filename=f"bao_cao_{camera_id}_{safe_name}.webm",
    )


@router.get("/events/{event_id}/crop.jpg")
def event_crop_jpeg(event_id: str):
    extra = get_store().get_event_extra(event_id)
    if not extra:
        raise HTTPException(status_code=404, detail="Event not found")
    rel = extra.get("crop_path")
    data = load_crop_bytes(str(rel) if rel else None)
    if not data:
        raise HTTPException(status_code=404, detail="Crop image not available")
    return Response(content=data, media_type="image/jpeg")


@router.get("/events/{event_id}/weapon-crop.jpg")
def event_weapon_crop_jpeg(event_id: str):
    return _event_weapon_crop_response(event_id, None)


@router.get("/events/{event_id}/weapon-crop/{weapon_class}.jpg")
def event_weapon_crop_by_class(event_id: str, weapon_class: str):
    return _event_weapon_crop_response(event_id, weapon_class)


def _event_weapon_crop_response(event_id: str, weapon_class: Optional[str]) -> Response:
    from identity_vm_app.services.weapon_crops import normalize_weapon_class

    extra = get_store().get_event_extra(event_id)
    if not extra:
        raise HTTPException(status_code=404, detail="Event not found")
    cls_q = normalize_weapon_class(weapon_class) if weapon_class else None
    for item in extra.get("weapon_crops") or []:
        if not isinstance(item, dict):
            continue
        if cls_q and normalize_weapon_class(item.get("class")) != cls_q:
            continue
        rel = item.get("path")
        data = load_crop_bytes(str(rel) if rel else None)
        if data:
            return Response(content=data, media_type="image/jpeg")
    if not cls_q:
        rel = extra.get("weapon_crop_path")
        data = load_crop_bytes(str(rel) if rel else None)
        if data:
            return Response(content=data, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Weapon crop image not available")


@router.get("/events/{event_id}/track-scene.jpg")
def event_track_scene_jpeg(event_id: str):
    extra = get_store().get_event_extra(event_id)
    if not extra:
        raise HTTPException(status_code=404, detail="Event not found")
    rel = extra.get("track_scene_path")
    data = load_crop_bytes(str(rel) if rel else None)
    if not data:
        raise HTTPException(status_code=404, detail="Track scene image not available")
    return Response(content=data, media_type="image/jpeg")


@router.get("/people/appearances")
def appearances(
    person_ref: Optional[str] = None,
    camera_id: Optional[str] = None,
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    limit: int = 200,
):
    rows = get_store().list_appearances(
        person_ref=person_ref,
        camera_id=camera_id,
        from_ts=from_ts,
        to_ts=to_ts,
        limit=min(2000, limit),
    )
    return {
        "items": [
            {
                "id": r.id,
                "ts_utc": r.ts_utc,
                "camera_id": r.camera_id,
                "person_ref": r.person_ref,
                "face_id": r.face_id,
                "display_name": r.display_name,
                "distance": r.distance,
                "det_score": r.det_score,
                "recording_segment_id": r.recording_segment_id,
                "offset_start_s": r.offset_start_s,
                "offset_end_s": r.offset_end_s,
            }
            for r in rows
        ]
    }


class ExportCutBody(BaseModel):
    event_id: Optional[str] = None
    segment_id: Optional[int] = None
    offset_start_s: Optional[float] = None
    offset_end_s: Optional[float] = None


def _resolve_cut_for_event(event_id: str) -> Any:
    """Tìm segment + offset; fallback theo ts nếu event chưa gắn archive khi ghi."""
    store = get_store()
    ev = store.get_event(event_id)
    if ev is None:
        raise HTTPException(status_code=404, detail="event not found")

    seg_id = ev.recording_segment_id
    off0 = ev.offset_start_s
    off1 = ev.offset_end_s

    if seg_id is None:
        seg_row = store.find_segment_for_timestamp(ev.camera_id, ev.ts_utc)
        if seg_row is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Không có archive cho thời điểm này. Bật nhận diện (tự ghi RTSP) "
                    "trước khi có sự kiện, hoặc POST /ivm/cameras/{id}/recorder/start."
                ),
            )
        seg_id = seg_row.id
        if off0 is None:
            off0 = max(0.0, float(ev.ts_utc) - float(seg_row.started_at_utc))
        if off1 is None:
            off1 = float(off0) + float(s.IVM_EXPORT_CUT_MIN_DURATION_S)

    seg = store.get_segment(int(seg_id))
    if seg is None or not Path(seg.path).is_file():
        raise HTTPException(status_code=404, detail="segment file not found on disk")

    if off0 is None:
        off0 = max(0.0, float(ev.ts_utc) - float(seg.started_at_utc))
    if off1 is None:
        off1 = float(off0) + float(s.IVM_EXPORT_CUT_MIN_DURATION_S)
    off1 = max(float(off1), float(off0) + 0.5)

    return export_segment_cut(
        src_path=seg.path,
        offset_start_s=float(off0),
        offset_end_s=float(off1),
    )


def _resolve_export_path(body: ExportCutBody) -> Any:
    if body.event_id:
        return _resolve_cut_for_event(body.event_id)
    if body.segment_id is not None and body.offset_start_s is not None and body.offset_end_s is not None:
        seg = get_store().get_segment(body.segment_id)
        if seg is None:
            raise HTTPException(status_code=404, detail="segment not found")
        return export_segment_cut(
            src_path=seg.path,
            offset_start_s=float(body.offset_start_s),
            offset_end_s=float(body.offset_end_s),
        )
    raise HTTPException(status_code=400, detail="Provide event_id or segment_id + offsets")


@router.get("/events/{event_id}/export-cut.mp4")
def export_cut_by_event_get(event_id: str) -> FileResponse:
    """Tải đoạn archive gắn với một lần xuất hiện (dùng trong UI chi tiết)."""
    out = _resolve_cut_for_event(event_id)
    media = "video/mp4" if str(out).lower().endswith(".mp4") else "video/x-matroska"
    return FileResponse(out, filename=out.name, media_type=media)


@router.post("/export/cut")
def export_cut(body: ExportCutBody, download: bool = Query(False)):
    out = _resolve_export_path(body)
    media = "video/mp4" if str(out).lower().endswith(".mp4") else "video/x-matroska"
    if download:
        return FileResponse(out, filename=out.name, media_type=media)
    return {"path": str(out), "filename": out.name}
