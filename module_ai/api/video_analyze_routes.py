"""API upload video → job phân tích offline (chỉ nhận diện người / khuôn mặt)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from identity_vm_app import settings as s
from module_ai.engine.yolo_person_tracker import tracker_unavailable_reason, vm_tracking_available
from identity_vm_app.services import video_analyze_media as vmedia
from identity_vm_app.services.video_analyze_media import normalize_face_thumb
from module_ai.pipelines import video_offline_analyze as va
from identity_vm_app.services.video_analyze_fps import (
    default_display_name,
    parse_sample_fps,
    sample_fps_label,
)
from module_ai.pipelines.video_face_search import search_reports_by_embedding
from identity_vm_app.services.video_report_crops import (
    crop_from_frame,
    draw_boxes_on_frame,
    encode_jpeg_bytes,
    load_frame_bgr,
    parse_box,
)
from identity_vm_app.services.video_report_merge import dedupe_reports_by_img_url
from identity_vm_app.services.video_report_vm import (
    dump_vm_person_reports,
    merge_and_dump_vm_faces_person,
)
from identity_vm_app.store.video_analyze_store import get_video_analyze_store

router = APIRouter(prefix="/ivm", tags=["video-analyze"])


class VideoJobRenameBody(BaseModel):
    display_name: str = Field(..., min_length=1, max_length=200)


def _parse_split_parts(raw: Optional[str]) -> int:
    if raw is None or str(raw).strip() == "":
        return max(1, min(4, int(s.IVM_VIDEO_ANALYZE_SPLIT_PARTS)))
    try:
        v = int(str(raw).strip())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="split_parts phải là số nguyên từ 1 đến 4",
        ) from None
    if v < 1 or v > 4:
        raise HTTPException(status_code=400, detail="split_parts phải từ 1 đến 4")
    return v


def _parse_video_job_form(
    *,
    sample_fps: Optional[str],
    display_name: Optional[str],
    distance_threshold: Optional[str],
    split_parts: Optional[str],
    original_name: str,
) -> tuple[float, str, float, int]:
    try:
        sf = parse_sample_fps(sample_fps if sample_fps is not None else s.IVM_VIDEO_ANALYZE_DEFAULT_SAMPLE_FPS)
    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    title = (display_name or "").strip() or default_display_name(original_name, sf)
    try:
        thr = (
            float(str(distance_threshold).strip())
            if distance_threshold is not None and str(distance_threshold).strip() != ""
            else float(s.IVM_DISTANCE_THRESHOLD)
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="distance_threshold không hợp lệ") from None
    sp = _parse_split_parts(split_parts)
    return sf, title, thr, sp


def _queue_video_job(
    *,
    job_id: str,
    orig: str,
    staged_path: Path,
    sf: float,
    title: str,
    thr: float,
    split_parts: int,
) -> dict:
    try:
        va.register_job(
            job_id,
            staged_path,
            original_name=orig,
            display_name=title,
            sample_fps=sf,
            split_parts=split_parts,
        )
    except Exception:
        va.release_job_slot()
        raise
    va.start_job_thread(job_id, sample_fps=sf, distance_threshold=thr)
    return {
        "job_id": job_id,
        "status": "queued",
        "display_name": title,
        "sample_fps": sf,
        "sample_fps_label": sample_fps_label(sf),
        "split_parts": split_parts,
        "split_note": (
            "Phân tích trực tiếp, không cắt file"
            if split_parts <= 1
            else f"Cắt {split_parts} đoạn song song"
        ),
    }


@router.post("/video-analyze/jobs")
async def create_video_analyze_job(
    file: UploadFile = File(...),
    sample_fps: Optional[str] = Form(None),
    display_name: Optional[str] = Form(None),
    distance_threshold: Optional[str] = Form(None),
    split_parts: Optional[str] = Form(
        None,
        description="Số luồng phân tích song song (1–4). 1 = không cắt video, phân tích trực tiếp.",
    ),
) -> dict:
    if not va.try_acquire_job_slot():
        raise HTTPException(
            status_code=429,
            detail=f"Đang có tối đa {s.IVM_VIDEO_ANALYZE_MAX_CONCURRENT} job video — thử lại sau",
        )

    max_mb = int(s.IVM_VIDEO_ANALYZE_MAX_MB)
    max_b = max_mb * 1024 * 1024 if max_mb > 0 else 0
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            buf = await file.read(1024 * 1024)
            if not buf:
                break
            total += len(buf)
            if max_b > 0 and total > max_b:
                raise HTTPException(
                    status_code=413,
                    detail=f"File vượt quá {max_mb} MB",
                )
            chunks.append(buf)
    except HTTPException:
        va.release_job_slot()
        raise
    except Exception as ex:
        va.release_job_slot()
        raise HTTPException(status_code=400, detail=f"Đọc file lỗi: {ex}") from ex

    data = b"".join(chunks)
    if not data:
        va.release_job_slot()
        raise HTTPException(status_code=400, detail="File rỗng")

    orig = file.filename or "upload.mp4"
    suf = va.upload_suffix_from_name(orig)
    if not suf:
        va.release_job_slot()
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng không hỗ trợ (chấp nhận: {', '.join(s.IVM_VIDEO_ANALYZE_ALLOWED_SUFFIXES)})",
        )

    try:
        sf, title, thr, sp = _parse_video_job_form(
            sample_fps=sample_fps,
            display_name=display_name,
            distance_threshold=distance_threshold,
            split_parts=split_parts,
            original_name=orig,
        )
    except HTTPException:
        va.release_job_slot()
        raise

    job_id = va.new_job_id()
    try:
        path = va.save_upload_bytes(data, orig, job_id)
    except Exception:
        va.release_job_slot()
        raise

    return _queue_video_job(
        job_id=job_id, orig=orig, staged_path=path, sf=sf, title=title, thr=thr, split_parts=sp
    )


@router.post("/video-analyze/jobs/from-path")
async def create_video_analyze_job_from_path(
    video_path: str = Form(..., description="Đường dẫn file video trên máy chạy API"),
    sample_fps: Optional[str] = Form(None),
    display_name: Optional[str] = Form(None),
    distance_threshold: Optional[str] = Form(None),
    split_parts: Optional[str] = Form(
        None,
        description="Số luồng phân tích (1–4). 1 = không cắt.",
    ),
) -> dict:
    """Phân tích file local — không qua upload Streamlit (file lớn)."""
    if not va.try_acquire_job_slot():
        raise HTTPException(
            status_code=429,
            detail=f"Đang có tối đa {s.IVM_VIDEO_ANALYZE_MAX_CONCURRENT} job video — thử lại sau",
        )

    raw = (video_path or "").strip().strip('"')
    if not raw:
        va.release_job_slot()
        raise HTTPException(status_code=400, detail="Thiếu video_path")

    try:
        sf, title, thr, sp = _parse_video_job_form(
            sample_fps=sample_fps,
            display_name=display_name,
            distance_threshold=distance_threshold,
            split_parts=split_parts,
            original_name=Path(raw).name,
        )
    except HTTPException:
        va.release_job_slot()
        raise

    src = Path(raw)
    job_id = va.new_job_id()
    try:
        staged = va.stage_local_video(src, job_id)
    except FileNotFoundError as ex:
        va.release_job_slot()
        raise HTTPException(status_code=404, detail=str(ex)) from ex
    except PermissionError as ex:
        va.release_job_slot()
        raise HTTPException(status_code=403, detail=str(ex)) from ex
    except ValueError as ex:
        va.release_job_slot()
        raise HTTPException(status_code=400, detail=str(ex)) from ex
    except OSError as ex:
        va.release_job_slot()
        raise HTTPException(status_code=400, detail=f"Không đọc được file: {ex}") from ex

    return _queue_video_job(
        job_id=job_id,
        orig=src.name,
        staged_path=staged,
        sf=sf,
        title=title,
        thr=thr,
        split_parts=sp,
    )


@router.get("/video-analyze/jobs")
def list_video_analyze_jobs(limit: int = Query(50, ge=1, le=200)) -> dict:
    jobs = get_video_analyze_store().list_jobs(limit=limit)
    for j in jobs:
        sf = float(j.get("sample_fps") or 0)
        j["sample_fps_label"] = sample_fps_label(sf)
        j["title"] = get_video_analyze_store().job_title(j)
        jid = str(j.get("id") or "")
        j["is_active"] = va.job_is_active(jid) if jid else False
        fa = j.get("feature_analyze") if isinstance(j.get("feature_analyze"), dict) else {}
        sp = fa.get("split_parts")
        if sp is not None:
            try:
                sp_i = max(1, min(4, int(sp)))
            except (TypeError, ValueError):
                sp_i = None
            if sp_i is not None:
                j["split_parts"] = sp_i
                j["split_parts_label"] = (
                    "1 luồng (không cắt)" if sp_i <= 1 else f"{sp_i} luồng"
                )
    return {"jobs": jobs}


@router.get("/video-analyze/jobs/{job_id}")
def get_video_analyze_job(job_id: str) -> dict:
    j = va.get_job(job_id)
    if j is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    db = get_video_analyze_store().get_job(job_id) or {}
    sf = float(db.get("sample_fps") or 0)
    j["sample_fps_label"] = sample_fps_label(sf)
    j["title"] = get_video_analyze_store().job_title(db)
    return j


@router.patch("/video-analyze/jobs/{job_id}")
def rename_video_analyze_job(job_id: str, body: VideoJobRenameBody) -> dict:
    if get_video_analyze_store().get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    if not va.rename_job(job_id, body.display_name):
        raise HTTPException(status_code=400, detail="Không đổi được tên")
    return {"ok": True, "job_id": job_id, "display_name": body.display_name.strip()}


@router.get("/video-analyze/config")
def video_analyze_config() -> dict:
    from module_ai.camera.weapon import weapon_detection_available

    max_mb = int(s.IVM_VIDEO_ANALYZE_MAX_MB)
    max_dur = float(s.IVM_VIDEO_ANALYZE_MAX_DURATION_S)
    return {
        "save_crops": bool(s.IVM_VIDEO_ANALYZE_SAVE_CROPS),
        "weapon_detection_enabled": weapon_detection_available(),
        "yolo_tracking": bool(s.IVM_VIDEO_ANALYZE_USE_YOLO_TRACKING),
        "yolo_available": vm_tracking_available(),
        "yolo_unavailable_reason": tracker_unavailable_reason(),
        "tracker": str(s.IVM_VIDEO_ANALYZE_TRACKER),
        "yolo_model": str(s.IVM_VIDEO_ANALYZE_YOLO_MODEL),
        "persist_embeddings": bool(s.IVM_VIDEO_ANALYZE_PERSIST_EMBEDDINGS),
        "max_upload_mb": max_mb,
        "max_upload_unlimited": max_mb <= 0,
        "max_duration_s": max_dur,
        "max_duration_unlimited": max_dur <= 0,
        "local_path_roots": [str(p) for p in s.IVM_VIDEO_ANALYZE_LOCAL_PATH_ROOTS],
        "default_split_parts": max(1, min(4, int(s.IVM_VIDEO_ANALYZE_SPLIT_PARTS))),
        "max_split_parts": 4,
        "split_parts_choices": [1, 2, 3, 4],
    }


@router.get("/video-analyze/jobs/{job_id}/reports/tracks")
def get_video_track_reports(
    job_id: str,
    include_clip: bool = Query(
        False,
        description="False: một dòng mỗi id_tracking (tên = định danh khuôn mặt). True: tách theo clip.",
    ),
) -> dict:
    """Báo cáo tổng hợp theo ByteTrack id_tracking; track_names = top N tên nghi ngờ."""
    if get_video_analyze_store().get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    rows = get_video_analyze_store().list_person_reports_merged(
        job_id, include_clip=include_clip
    )
    return {"job_id": job_id, "merged_by": "clip" if include_clip else "tracking", "reports": rows}


@router.get("/video-analyze/jobs/{job_id}/reports/persons")
def get_video_person_reports(
    job_id: str,
    merged: bool = Query(True, description="Gom báo cáo (mặc định theo id_tracking)"),
    include_clip: bool = Query(
        False,
        description="Khi merged=true: False = gom theo track; True = gom track+clip",
    ),
    start_time_s: float = Query(0.0, ge=0.0),
    end_time_s: float = Query(0.0, ge=0.0),
    gender: Optional[int] = Query(None),
    limit: int = Query(5000, ge=1, le=20000),
) -> dict:
    if get_video_analyze_store().get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    store = get_video_analyze_store()
    if merged:
        rows = store.list_person_reports_merged(job_id, include_clip=include_clip)
    else:
        rows = store.list_person_reports(
            job_id, start_time_s=start_time_s, end_time_s=end_time_s, gender=gender, limit=limit
        )
    return {"job_id": job_id, "merged": merged, "reports": rows}


@router.get("/video-analyze/jobs/{job_id}/reports/persons/sub-data")
def get_video_person_sub_data(
    job_id: str,
    video_clip: int = Query(..., ge=1),
    start_time_s: float = Query(0.0, ge=0.0),
    end_time_s: float = Query(0.0, ge=0.0),
) -> dict:
    """Sub-data theo video_clip — giống get_sub_data_video_clip_view VideoMaster."""
    if get_video_analyze_store().get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    store = get_video_analyze_store()
    job = store.get_job(job_id) or {}
    raw = store.list_person_reports_by_clip(
        job_id, video_clip, start_time_s=start_time_s, end_time_s=end_time_s
    )
    return {
        "job_id": job_id,
        "video_clip": video_clip,
        "persons": merge_and_dump_vm_faces_person(raw, job=job),
        "reports": dump_vm_person_reports(dedupe_reports_by_img_url(raw), job=job),
    }


@router.get("/video-analyze/reports/persons/{report_id}/face-image")
def get_report_face_image(report_id: str):
    row = get_video_analyze_store().get_person_report(report_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy báo cáo")
    rel = row.get("face_img")
    if rel:
        fp = vmedia.resolve_media_path(str(rel))
        if fp and fp.is_file():
            saved = cv2.imread(str(fp))
            if saved is not None and saved.size:
                thumb = normalize_face_thumb(saved)
                return Response(content=encode_jpeg_bytes(thumb), media_type="image/jpeg")
    frame = load_frame_bgr(row.get("img_url"))
    if frame is None:
        raise HTTPException(status_code=404, detail="Không có khung gốc")
    crop = crop_from_frame(frame, row.get("box_face"), pad=0.18)
    if crop is None:
        raise HTTPException(status_code=404, detail="Không crop được mặt")
    thumb = normalize_face_thumb(crop)
    return Response(content=encode_jpeg_bytes(thumb), media_type="image/jpeg")


@router.get("/video-analyze/reports/persons/{report_id}/person-image")
def get_report_person_image(report_id: str):
    row = get_video_analyze_store().get_person_report(report_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy báo cáo")
    rel = row.get("person_img")
    if rel:
        fp = vmedia.resolve_media_path(str(rel))
        if fp and fp.is_file():
            return FileResponse(str(fp), media_type="image/jpeg")
    frame = load_frame_bgr(row.get("img_url"))
    if frame is None:
        raise HTTPException(status_code=404, detail="Không có khung gốc")
    crop = crop_from_frame(frame, row.get("box_person"), pad=0.05)
    if crop is None:
        raise HTTPException(status_code=404, detail="Không crop được người")
    return Response(content=encode_jpeg_bytes(crop), media_type="image/jpeg")


@router.get("/video-analyze/reports/persons/{report_id}/weapon-image")
def get_report_weapon_image(report_id: str, weapon_class: Optional[str] = Query(None)):
    from module_ai.pipelines.weapon_crops import (
        normalize_weapon_class,
        parse_weapon_crops_json,
        render_weapon_bbox_crop_bgr,
    )

    row = get_video_analyze_store().get_person_report(report_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy báo cáo")
    cls_q = normalize_weapon_class(weapon_class) if weapon_class else None
    if cls_q:
        for item in parse_weapon_crops_json(row.get("weapon_crops_json")):
            if normalize_weapon_class(item.get("class")) == cls_q and item.get("path"):
                fp = vmedia.resolve_media_path(str(item["path"]))
                if fp and fp.is_file():
                    return FileResponse(str(fp), media_type="image/jpeg")
    rel = row.get("weapon_img")
    if rel and not cls_q:
        fp = vmedia.resolve_media_path(str(rel))
        if fp and fp.is_file():
            return FileResponse(str(fp), media_type="image/jpeg")
    if not int(row.get("armed") or 0):
        raise HTTPException(status_code=404, detail="Không có vũ khí trên báo cáo này")
    frame = load_frame_bgr(row.get("img_url"))
    if frame is None:
        raise HTTPException(status_code=404, detail="Không có khung gốc")
    from identity_vm_app.services.video_report_crops import parse_weapon_boxes

    weapons = parse_weapon_boxes(row.get("weapon_boxes_json"))
    if not weapons:
        raise HTTPException(status_code=404, detail="Không có crop vũ khí")
    crop = render_weapon_bbox_crop_bgr(frame, weapons, weapon_class=cls_q)
    if crop is None:
        raise HTTPException(status_code=404, detail="Không render được crop vũ khí")
    return Response(content=encode_jpeg_bytes(crop), media_type="image/jpeg")


@router.get("/video-analyze/reports/persons/{report_id}/track-scene-image")
def get_report_track_scene_image(report_id: str):
    row = get_video_analyze_store().get_person_report(report_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy báo cáo")
    frame = load_frame_bgr(row.get("img_url"))
    if frame is None:
        raise HTTPException(status_code=404, detail="Không có khung gốc")
    from identity_vm_app.services.video_report_crops import parse_weapon_boxes
    from module_ai.pipelines.weapon_crops import render_track_scene_crop_bgr

    pb = parse_box(row.get("box_person"))
    if pb is None:
        raise HTTPException(status_code=404, detail="Không có box người")
    weapons = parse_weapon_boxes(row.get("weapon_boxes_json"))
    fb = parse_box(row.get("box_face"))
    crop = render_track_scene_crop_bgr(frame, pb, weapons, fb)
    if crop is None:
        raise HTTPException(status_code=404, detail="Không render được scene track")
    return Response(content=encode_jpeg_bytes(crop), media_type="image/jpeg")


@router.get("/video-analyze/reports/persons/{report_id}/draw-box")
def get_report_draw_box(report_id: str):
    row = get_video_analyze_store().get_person_report(report_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy báo cáo")
    frame = load_frame_bgr(row.get("img_url"))
    if frame is None:
        raise HTTPException(status_code=404, detail="Không có khung gốc")
    drawn = draw_boxes_on_frame(
        frame,
        box_person=row.get("box_person"),
        box_face=row.get("box_face"),
        weapon_boxes=row.get("weapon_boxes_json"),
    )
    return Response(content=encode_jpeg_bytes(drawn), media_type="image/jpeg")


@router.post("/video-analyze/reports/search-faces")
async def search_faces_in_job(
    job_id: str = Form(...),
    file: UploadFile = File(...),
    min_percent: float = Form(0.0),
    limit: int = Form(50),
) -> dict:
    if not bool(s.IVM_VIDEO_ANALYZE_PERSIST_EMBEDDINGS):
        raise HTTPException(
            status_code=400,
            detail="Cần bật IVM_VIDEO_ANALYZE_PERSIST_EMBEDDINGS=1 khi phân tích để tìm trong DB",
        )
    store = get_video_analyze_store()
    if store.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File rỗng")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        img = cv2.imread(tmp_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Không đọc được ảnh")
        from identity_vm_app.api.deps import get_engine

        engine = get_engine()
        _ms, aligned, _meta = engine.detect_align_faces(img)
        if not aligned:
            return {"job_id": job_id, "hits": []}
        feats, _ = engine.embed_aligned_crops(aligned[:1])
        query = np.asarray(feats[0], dtype=np.float32).reshape(-1)
        rows = store.list_person_reports_with_embeddings(job_id)
        hits = search_reports_by_embedding(query, rows, min_percent=min_percent, limit=limit)
        merged = merge_reports(hits)
        return {"job_id": job_id, "hits": hits, "persons_merged": merged}
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass


@router.get("/video-analyze/jobs/{job_id}/reports/export.csv")
def export_person_reports_csv(
    job_id: str,
    merged: bool = Query(True),
    include_clip: bool = Query(False),
) -> Response:
    if get_video_analyze_store().get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    store = get_video_analyze_store()
    if merged:
        rows = store.list_person_reports_merged(job_id, include_clip=include_clip)
    else:
        rows = store.list_person_reports(job_id, limit=50000)
    lines = ["id_tracking,track_name,clips,time_start_s,time_end_s,hit_count"]
    for r in rows:
        t0 = r.get("time_analyze") or r.get("time_analyze_s") or 0
        t1 = r.get("end_time") or t0
        cmin = r.get("video_clip_min") or r.get("video_clip") or 1
        cmax = r.get("video_clip_max") or r.get("video_clip") or cmin
        clips = str(cmin) if int(cmin) == int(cmax) else f"{cmin}-{cmax}"
        names = r.get("track_names")
        if isinstance(names, list) and names:
            name = " | ".join(str(x) for x in names if str(x).strip())
        else:
            name = str(
                r.get("track_name_label") or r.get("track_name") or r.get("display_name") or ""
            )
        name = name.replace('"', "'")
        lines.append(
            f"{r.get('id_tracking')},\"{name}\",{clips},"
            f"{t0},{t1},{r.get('hit_count', 1)}"
        )
    body = "\n".join(lines).encode("utf-8-sig")
    return Response(
        content=body,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="ivm_report_{job_id[:8]}.csv"'},
    )


@router.delete("/video-analyze/jobs/{job_id}/reports")
def delete_video_analyze_reports(job_id: str) -> dict:
    if get_video_analyze_store().get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    out = va.delete_job_reports(job_id)
    if not out.get("ok"):
        reason = out.get("reason")
        if reason == "job_running":
            raise HTTPException(status_code=409, detail="Job đang chạy — không xóa báo cáo được")
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    return out


@router.get("/video-analyze/media")
def get_video_analyze_media(path: str = Query(..., description="Đường dẫn tương đối trong IVM_VIDEO_ANALYZE_DIR")):
    fp = vmedia.resolve_media_path(path)
    if fp is None or not fp.is_file():
        raise HTTPException(status_code=404, detail="Không tìm thấy file ảnh")
    return FileResponse(str(fp), media_type="image/jpeg")


@router.delete("/video-analyze/jobs/{job_id}")
def delete_video_analyze_job(job_id: str) -> dict:
    if get_video_analyze_store().get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    if va.job_is_active(job_id):
        raise HTTPException(status_code=409, detail="Job đang chạy — dừng hoặc đợi xong rồi xóa")
    ok = va.delete_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Không tìm thấy job")
    return {"ok": True, "job_id": job_id}
