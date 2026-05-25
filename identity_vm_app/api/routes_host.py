from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, File, Header, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from identity_vm_app import settings as s
from identity_vm_app.api.deps import get_recorders, get_store
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

router = APIRouter(prefix="/ivm", tags=["identity-vm-host"])

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
    """Tráº¡ng thÃ¡i nháº­n diá»‡n tá»«ng camera (máº·c Ä‘á»‹nh táº¯t náº¿u chÆ°a Ä‘áº·t)."""
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
    from module_ai.camera.activity_log import recent as activity_recent
    from module_ai.camera.analyze_recording import get_visual_session
    from module_ai.camera.hub import ensure_recognition_hub_started, get_recognition_hub
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
            "Khi Báº¬T: ghi archive RTSP + session.mp4 + bÃ¡o cÃ¡o DB. "
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
    from module_ai.camera.activity_log import recent as activity_recent
    from module_ai.camera.hub import get_recognition_hub

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
    from module_ai.camera.activity_log import recent as activity_recent

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
    confirm: str = Field(..., description='Pháº£i gÃµ Ä‘Ãºng DELETE_REPORTS')
    camera_id: Optional[str] = Field(
        None,
        description="Chá»‰ xÃ³a bÃ¡o cÃ¡o camera nÃ y; bá» trá»‘ng = táº¥t cáº£ camera",
    )
    wipe_archive: bool = Field(
        False,
        description="XÃ³a luÃ´n file archive RTSP vÃ  segment DB",
    )


@router.post("/reports/clear")
def clear_all_reports(body: ClearReportsBody) -> Dict[str, Any]:
    """XÃ³a toÃ n bá»™ bÃ¡o cÃ¡o nháº­n diá»‡n (giá»¯ thÆ° viá»‡n khuÃ´n máº·t Ä‘Äƒng kÃ½)."""
    if body.confirm != "DELETE_REPORTS":
        raise HTTPException(
            status_code=400,
            detail='TrÆ°á»ng confirm pháº£i Ä‘Ãºng chuá»—i DELETE_REPORTS.',
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
    """BÃ¡o cÃ¡o tá»•ng há»£p theo tá»«ng camera (ká»ƒ cáº£ camera chÆ°a cÃ³ sá»± kiá»‡n)."""
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


@router.get("/cameras/{camera_id}/reports/tracks")
def camera_report_tracks(
    camera_id: str,
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    limit: int = Query(300, ge=1, le=2000),
    known_only: bool = Query(False, description="Chá»‰ ngÆ°á»i Ä‘Ã£ Ä‘á»‹nh danh (bá» unknown)"),
    person_ref: Optional[str] = None,
) -> Dict[str, Any]:
    """BÃ¡o cÃ¡o chi tiáº¿t: crop áº£nh, danh tÃ­nh, sá»‘ frame tracking má»—i láº§n xuáº¥t hiá»‡n."""
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
    from module_ai.pipelines.weapon_crops import normalize_weapon_class

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
    Chi tiáº¿t má»™t ngÆ°á»i Ä‘Ã£ Ä‘á»‹nh danh (tÆ°Æ¡ng tá»± VisionMaster get-by-track).
    Má»—i pháº§n tá»­ = má»™t láº§n xuáº¥t hiá»‡n, cÃ³ crop vÃ  frame_hits.
    """
    specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    if person_ref == "unknown":
        raise HTTPException(status_code=400, detail="Chá»‰ há»— trá»£ ngÆ°á»i Ä‘Ã£ Ä‘á»‹nh danh")
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
    group_key: str = Query(..., description="KhÃ³a gom tÃªn (normalize) hoáº·c truyá»n display_name"),
    display_name: Optional[str] = None,
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    limit: int = Query(2000, ge=1, le=2000),
) -> Dict[str, Any]:
    """Chi tiáº¿t má»™t Ä‘á»‘i tÆ°á»£ng theo tÃªn â€” má»i láº§n xuáº¥t hiá»‡n / frame trong khoáº£ng thá»i gian."""
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
        raise HTTPException(status_code=404, detail="KhÃ´ng tÃ¬m tháº¥y Ä‘á»‘i tÆ°á»£ng trong khoáº£ng thá»i gian")
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
    person_ref: Optional[str] = Query(None, description="Má»™t person_ref (náº¿u khÃ´ng dÃ¹ng group_key)"),
    group_key: Optional[str] = Query(None, description="KhÃ³a gom theo tÃªn hiá»ƒn thá»‹"),
    display_name: Optional[str] = Query(None, description="TÃªn hiá»ƒn thá»‹ (gom má»i face_id cÃ¹ng tÃªn)"),
    from_ts: Optional[float] = None,
    to_ts: Optional[float] = None,
    fps: float = Query(5.0, ge=1.0, le=30.0),
):
    """Xuáº¥t WebM tá»« áº£nh crop â€” má»—i áº£nh láº·p theo frame_hits (shortcut tÃ³m táº¯t xuáº¥t hiá»‡n)."""
    from identity_vm_app.report_grouping import filter_tracks_for_group

    specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    if not person_ref and not group_key and not display_name:
        raise HTTPException(
            status_code=400,
            detail="Cáº§n person_ref hoáº·c group_key hoáº·c display_name",
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
            detail="KhÃ´ng cÃ³ áº£nh crop trong khoáº£ng thá»i gian (cáº§n sá»± kiá»‡n sau khi báº­t lÆ°u crop)",
        )
    try:
        out_path = export_crops_to_webm(frames, fps=fps)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Xuáº¥t WebM tháº¥t báº¡i: {ex}") from ex
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
    from module_ai.pipelines.weapon_crops import normalize_weapon_class

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
    """TÃ¬m segment + offset; fallback theo ts náº¿u event chÆ°a gáº¯n archive khi ghi."""
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
                    "KhÃ´ng cÃ³ archive cho thá»i Ä‘iá»ƒm nÃ y. Báº­t nháº­n diá»‡n (tá»± ghi RTSP) "
                    "trÆ°á»›c khi cÃ³ sá»± kiá»‡n, hoáº·c POST /ivm/cameras/{id}/recorder/start."
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
    """Táº£i Ä‘oáº¡n archive gáº¯n vá»›i má»™t láº§n xuáº¥t hiá»‡n (dÃ¹ng trong UI chi tiáº¿t)."""
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
