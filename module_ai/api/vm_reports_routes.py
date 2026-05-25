"""
API báo cáo tương thích VideoMaster (đường dẫn /ivm/api/reports/* và /ivm/reports/*).
job_id IVM tương đương video_id VideoMaster.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response

from identity_vm_app import settings as s
from identity_vm_app.api.deps import get_engine
from identity_vm_app.services import video_analyze_media as vmedia
from module_ai.pipelines.video_face_search import search_reports_by_embedding
from identity_vm_app.services.video_report_crops import (
    crop_from_frame,
    draw_all_boxes_on_frame,
    draw_boxes_on_frame,
    encode_jpeg_bytes,
    load_frame_bgr,
)
from identity_vm_app.services.video_report_merge import (
    dedupe_reports_by_img_url,
    filter_segments_by_track_total_frames,
    merge_reports_vm,
)
from identity_vm_app.services.video_report_vm import (
    dump_vm_person_reports,
    extract_colors_from_reports,
    merge_and_dump_vm_faces_person,
    to_vm_person_row,
    vm_draw_box_url,
    vm_person_image_url,
)
from identity_vm_app.services.video_track_segments import (
    build_track_segment_video,
    list_track_appearance_segments,
)
from identity_vm_app.store.video_analyze_store import get_video_analyze_store

router = APIRouter(tags=["reports-vm"])


def _api_base(request: Request) -> str:
    base = (s.IVM_VIDEO_ANALYZE_HTTP_BASE or "").strip().rstrip("/")
    if base:
        return base
    return str(request.base_url).rstrip("/")


def _job_or_404(job_id: str) -> dict:
    job = get_video_analyze_store().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy video/job")
    return job


@router.get("/ivm/api/reports/faces-person")
def api_faces_person_report(
    request: Request,
    video_ids: List[str] = Query(..., description="Danh sách job_id (video_id)"),
    start_time: float = Query(0, alias="start_time"),
    end_time: float = Query(0, alias="end_time"),
    gender: Optional[int] = Query(None),
    start_age: int = Query(0),
    end_age: int = Query(1000),
) -> List[dict]:
    """Giống VideoMaster GET faces-person — list merge theo video_id:id_tracking."""
    store = get_video_analyze_store()
    raw = store.list_faces_person_reports(
        video_ids,
        start_time_s=float(start_time),
        end_time_s=float(end_time),
        gender=gender,
        start_age=start_age,
        end_age=end_age,
    )
    if not raw:
        return []
    jid = video_ids[0] if len(video_ids) == 1 else None
    job = store.get_job(jid) if jid else None
    return merge_and_dump_vm_faces_person(raw, job=job, api_base=_api_base(request))


@router.get("/ivm/api/reports/persons")
def api_persons_report(
    video_ids: List[str] = Query(...),
    start_time: float = Query(0),
    end_time: float = Query(0),
    gender: Optional[int] = Query(None),
    sleeve_length: Optional[str] = Query(None),
    type_of_lower_body_clothing: Optional[str] = Query(None),
    length_of_lower_body_clothing: Optional[str] = Query(None),
    carrying_handbag: Optional[str] = Query(None),
    wearing_hat: Optional[str] = Query(None),
    color: Optional[str] = Query(None),
    mask: Optional[int] = Query(None),
    start_age: int = Query(0),
    end_age: int = Query(1000),
) -> List[dict]:
    """Giống VideoMaster GET persons — tóm tắt theo video_clip."""
    return get_video_analyze_store().list_person_clips_summary(
        video_ids,
        start_time_s=float(start_time),
        end_time_s=float(end_time),
        gender=gender,
        start_age=start_age,
        end_age=end_age,
        sleeve_length=sleeve_length,
        type_of_lower_body_clothing=type_of_lower_body_clothing,
        length_of_lower_body_clothing=length_of_lower_body_clothing,
        carrying_handbag=carrying_handbag,
        wearing_hat=wearing_hat,
        color=color,
        mask=mask,
    )


@router.get("/ivm/api/reports/track-segments")
def api_track_appearance_segments(
    request: Request,
    video_ids: List[str] = Query(..., description="job_id (video_id)"),
    gap_s: float = Query(0, ge=0, description="0 = tự tính từ sample_fps job"),
) -> List[dict]:
    """IVM: đoạn xuất hiện theo id_tracking (mở rộng; VM không có route này)."""
    if not video_ids:
        return []
    job_id = video_ids[0]
    _job_or_404(job_id)
    api = _api_base(request)
    gap = float(gap_s) if gap_s > 0 else None
    segments = filter_segments_by_track_total_frames(
        list_track_appearance_segments(job_id, gap_s=gap)
    )
    out: List[dict] = []
    for seg in segments:
        item = dict(seg)
        rid = item.get("first_frame_report_id") or item.get("report_id")
        if api and rid:
            item["thumb_image_url"] = vm_draw_box_url(api, str(rid))
            item["draw_box_url"] = item["thumb_image_url"]
            item["track_image_url"] = vm_person_image_url(api, str(rid))
            item["person_image_url"] = item["track_image_url"]
        out.append(item)
    return out


@router.get("/ivm/api/reports/track-segment-video/{video_id}/{id_tracking}")
def api_track_segment_video(
    video_id: str,
    id_tracking: int,
    segment_index: int = Query(0, ge=0),
    draw_boxes: bool = Query(True),
    rebuild: bool = Query(False, description="1 = bỏ cache, ghép lại"),
):
    _job_or_404(video_id)
    try:
        path = build_track_segment_video(
            video_id,
            int(id_tracking),
            segment_index=int(segment_index),
            draw_boxes=bool(draw_boxes),
            force_rebuild=bool(rebuild),
        )
    except ValueError as ex:
        raise HTTPException(status_code=404, detail=str(ex)) from ex
    except RuntimeError as ex:
        raise HTTPException(status_code=500, detail=str(ex)) from ex
    return FileResponse(
        str(path),
        media_type="video/mp4",
        filename=f"track_{id_tracking}_seg{segment_index}.mp4",
        headers={"Accept-Ranges": "none", "Cache-Control": "private, max-age=3600"},
    )


@router.get("/ivm/api/reports/get-sub-data-video-clip/{video_id}/{video_clip}")
def api_sub_data_video_clip(
    request: Request,
    video_id: str,
    video_clip: int,
    start_time: float = Query(0),
    end_time: float = Query(0),
) -> dict:
    _job_or_404(video_id)
    store = get_video_analyze_store()
    job = store.get_job(video_id) or {}
    api = _api_base(request)
    raw = store.list_person_reports_by_clip(
        video_id,
        video_clip,
        start_time_s=float(start_time),
        end_time_s=float(end_time),
        use_vm_time_filter=True,
    )
    persons = merge_and_dump_vm_faces_person(raw, job=job, api_base=api)
    reports = dump_vm_person_reports(dedupe_reports_by_img_url(raw), job=job, api_base=api)
    return {"persons": persons, "reports": reports}


@router.get("/ivm/reports/get-person-sub-data/{video_id}/{id_tracking}")
def api_person_sub_data(
    request: Request,
    video_id: str,
    id_tracking: int,
    start_time: float = Query(0),
    end_time: float = Query(0),
) -> dict:
    _job_or_404(video_id)
    store = get_video_analyze_store()
    job = store.get_job(video_id) or {}
    raw = store.list_person_reports_by_tracking(
        video_id,
        id_tracking,
        start_time_s=float(start_time),
        end_time_s=float(end_time),
    )
    return {
        "colors": extract_colors_from_reports(raw),
        "subdata": dump_vm_person_reports(raw, job=job, api_base=_api_base(request)),
    }


@router.get("/ivm/reports/get-face-image/{report_id}")
def reports_get_face_image(report_id: str):
    from module_ai.api.video_analyze_routes import get_report_face_image

    return get_report_face_image(report_id)


@router.get("/ivm/reports/get-person-image/{report_id}")
def reports_get_person_image(report_id: str):
    from module_ai.api.video_analyze_routes import get_report_person_image

    return get_report_person_image(report_id)


@router.get("/ivm/reports/get-weapon-image/{report_id}")
def reports_get_weapon_image(report_id: str, weapon_class: Optional[str] = None):
    from module_ai.api.video_analyze_routes import get_report_weapon_image

    return get_report_weapon_image(report_id, weapon_class=weapon_class)


@router.get("/ivm/reports/get-track-scene-image/{report_id}")
def reports_get_track_scene_image(report_id: str):
    from module_ai.api.video_analyze_routes import get_report_track_scene_image

    return get_report_track_scene_image(report_id)


@router.get("/ivm/reports/draw-box-all-person/{report_id}")
def reports_draw_box_all_person(report_id: str):
    store = get_video_analyze_store()
    row = store.get_person_report(report_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy báo cáo")
    img_url = row.get("img_url")
    frame = load_frame_bgr(img_url)
    if frame is None:
        raise HTTPException(status_code=404, detail="Không có khung gốc")
    same_frame = [
        r
        for r in store.list_person_reports_by_clip(
            str(row.get("job_id") or ""),
            int(row.get("video_clip") or 1),
            limit=500,
        )
        if str(r.get("img_url") or "") == str(img_url or "")
    ]
    drawn = draw_all_boxes_on_frame(frame, same_frame)
    return Response(content=encode_jpeg_bytes(drawn), media_type="image/jpeg")


@router.get("/ivm/reports/draw-box-person/{report_id}")
def reports_draw_box_person(report_id: str):
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


@router.post("/ivm/api/reports/search-faces-person")
async def api_search_faces_person(
    request: Request,
    video_ids: List[str] = Form(...),
    img: UploadFile = File(...),
    percent: int = Form(50),
    start_time: float = Form(0),
    end_time: float = Form(0),
) -> dict:
    if not video_ids:
        raise HTTPException(status_code=400, detail="Thiếu video_ids")
    data = await img.read()
    if not data:
        raise HTTPException(status_code=400, detail="Thiếu ảnh")
    store = get_video_analyze_store()
    for jid in video_ids:
        if store.get_job(jid) is None:
            raise HTTPException(status_code=404, detail=f"Không tìm thấy job {jid}")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        bgr = cv2.imread(tmp_path)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Không đọc được ảnh")
        engine = get_engine()
        _ms, aligned, _meta = engine.detect_align_faces(bgr)
        if not aligned:
            return {"message": "ok", "persons": [], "hits": []}
        feats, _ = engine.embed_aligned_crops(aligned[:1])
        query = np.asarray(feats[0], dtype=np.float32).reshape(-1)
        all_rows: List[dict] = []
        for jid in video_ids:
            rows = store.list_faces_person_reports(
                [jid], start_time_s=float(start_time), end_time_s=float(end_time)
            )
            all_rows.extend(rows)
        hits_raw = search_reports_by_embedding(
            query, all_rows, min_percent=float(percent), limit=200
        )
        for h in hits_raw:
            h.pop("features_face", None)
            h.pop("features_person", None)
            h.pop("match_candidates_json", None)
        job = store.get_job(video_ids[0]) if video_ids else None
        api = _api_base(request)
        persons = merge_and_dump_vm_faces_person(hits_raw, job=job, api_base=api)
        for p in persons:
            p["percent"] = p.get("percent") or next(
                (h.get("percent") for h in hits_raw if str(h.get("id")) == str(p.get("id"))),
                None,
            )
        return {
            "message": "ok",
            "persons": persons,
            "hits": dump_vm_person_reports(hits_raw, job=job, api_base=api),
        }
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass
