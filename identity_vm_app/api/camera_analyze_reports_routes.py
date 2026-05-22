"""API báo cáo phiên nhận diện camera live."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from identity_vm_app import settings as s
from camera_channel_config import load_camera_channel_specs
from identity_vm_app.services.video_report_vm import merge_and_dump_vm_faces_person
from identity_vm_app.services.video_track_segments import build_track_segment_video, list_track_appearance_segments
from identity_vm_app.store.video_analyze_store import get_video_analyze_store

router = APIRouter(prefix="/ivm", tags=["camera-analyze-reports"])


def _camera_ok(camera_id: str) -> None:
    specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")


def _api_base(request: Request) -> str:
    base = (s.IVM_VIDEO_ANALYZE_HTTP_BASE or "").strip().rstrip("/")
    if base:
        return base
    return str(request.base_url).rstrip("/")


def _resolve_session_mp4(job: Dict[str, Any]) -> Optional[Path]:
    vp = str(job.get("video_path") or "").strip()
    if vp:
        p = Path(vp)
        if p.is_file():
            return p
    cam = str(job.get("camera_id") or "")
    jid = str(job.get("id") or "")
    if cam and jid:
        from identity_vm_app.services.camera_session_media import session_mp4_path

        cand = session_mp4_path(cam, jid)
        if cand.is_file():
            return cand
        web = cand.with_name(cand.stem + "_web.mp4")
        if web.is_file():
            return web
    return None


@router.get("/cameras/{camera_id}/analyze/sessions")
def list_camera_sessions(
    camera_id: str,
    from_ts: float = Query(0, alias="from_ts"),
    to_ts: float = Query(0, alias="to_ts"),
    limit: int = Query(50, ge=1, le=200),
) -> Dict[str, Any]:
    _camera_ok(camera_id)
    store = get_video_analyze_store()
    rows = store.list_camera_live_sessions(
        camera_id, from_ts=float(from_ts), to_ts=float(to_ts), limit=limit
    )
    for j in rows:
        j["title"] = store.job_title(j)
        counts = store.count_reports(str(j.get("id") or ""))
        j["report_counts"] = counts
    return {"camera_id": camera_id, "sessions": rows}


@router.get("/cameras/{camera_id}/analyze/sessions/{job_id}")
def get_camera_session(camera_id: str, job_id: str) -> Dict[str, Any]:
    _camera_ok(camera_id)
    store = get_video_analyze_store()
    job = store.get_job(job_id)
    if job is None or str(job.get("camera_id") or "") != camera_id:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên")
    if str(job.get("source_type") or "") != "camera_live":
        raise HTTPException(status_code=404, detail="Job không phải phiên camera live")
    job["title"] = store.job_title(job)
    job["report_counts"] = store.count_reports(job_id)
    job["has_video"] = _resolve_session_mp4(job) is not None
    return job


@router.get("/cameras/{camera_id}/analyze/sessions/{job_id}/session.mp4")
def get_camera_session_mp4(camera_id: str, job_id: str) -> FileResponse:
    _camera_ok(camera_id)
    store = get_video_analyze_store()
    job = store.get_job(job_id)
    if job is None or str(job.get("camera_id") or "") != camera_id:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên")
    path = _resolve_session_mp4(job)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Chưa có video phiên")
    return FileResponse(str(path), media_type="video/mp4", filename="session.mp4")


@router.get("/cameras/{camera_id}/reports/faces-person")
def camera_faces_person_report(
    request: Request,
    camera_id: str,
    session_ids: List[str] = Query(default=[], alias="session_ids"),
    from_ts: float = Query(0, alias="from_ts"),
    to_ts: float = Query(0, alias="to_ts"),
    gender: Optional[int] = Query(None),
    start_age: int = Query(0),
    end_age: int = Query(1000),
) -> List[dict]:
    _camera_ok(camera_id)
    store = get_video_analyze_store()
    job_ids: List[str] = list(session_ids) if session_ids else []
    if not job_ids:
        sessions = store.list_camera_live_sessions(
            camera_id, from_ts=float(from_ts), to_ts=float(to_ts), limit=200
        )
        job_ids = [str(j["id"]) for j in sessions if j.get("id")]
    if not job_ids:
        return []
    for jid in job_ids:
        job = store.get_job(jid)
        if job is None or str(job.get("camera_id") or "") != camera_id:
            raise HTTPException(status_code=400, detail=f"session_ids không hợp lệ: {jid}")
    raw = store.list_faces_person_reports(
        job_ids,
        start_time_s=0.0,
        end_time_s=0.0,
        gender=gender,
        start_age=start_age,
        end_age=end_age,
        use_vm_time_filter=False,
    )
    if not raw:
        return []
    job = store.get_job(job_ids[0]) if len(job_ids) == 1 else None
    return merge_and_dump_vm_faces_person(raw, job=job, api_base=_api_base(request), min_track_frames=1)


@router.get("/cameras/{camera_id}/reports/track-segments")
def camera_track_segments(
    camera_id: str,
    job_id: str = Query(..., alias="job_id"),
    id_tracking: int = Query(..., alias="id_tracking"),
    min_frames: int = Query(1, ge=1),
) -> Dict[str, Any]:
    _camera_ok(camera_id)
    store = get_video_analyze_store()
    job = store.get_job(job_id)
    if job is None or str(job.get("camera_id") or "") != camera_id:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên")
    segments = list_track_appearance_segments(job_id, int(id_tracking), min_frames=min_frames)
    return {
        "camera_id": camera_id,
        "job_id": job_id,
        "id_tracking": int(id_tracking),
        "segments": segments,
    }


@router.get("/cameras/{camera_id}/reports/track-segments/video")
def camera_track_segment_video(
    camera_id: str,
    job_id: str = Query(...),
    id_tracking: int = Query(...),
    segment_index: int = Query(0, ge=0),
) -> FileResponse:
    _camera_ok(camera_id)
    store = get_video_analyze_store()
    job = store.get_job(job_id)
    if job is None or str(job.get("camera_id") or "") != camera_id:
        raise HTTPException(status_code=404, detail="Không tìm thấy phiên")
    try:
        path = build_track_segment_video(
            job_id,
            int(id_tracking),
            segment_index=int(segment_index),
        )
    except ValueError as ex:
        raise HTTPException(status_code=404, detail=str(ex)) from ex
    return FileResponse(str(path), media_type="video/mp4")
