"""MJPEG / snapshot khung sau nhận diện (overlay bbox) — khi analyze BẬT."""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from pathlib import Path

from fastapi.responses import FileResponse, Response, StreamingResponse

from camera_channel_config import load_camera_channel_specs
from identity_vm_app import settings as s
from identity_vm_app.camera_analyze_control import get_analyze_enabled
from identity_vm_app.api.deps import get_recorders
from module_ai.camera.analyze_recording import (
    get_visual_session,
    list_visual_sessions,
)
from module_ai.camera.hub import get_recognition_hub
from module_ai.camera.weapon import weapon_detection_available

router = APIRouter(prefix="/ivm", tags=["camera-infer"])


def _worker_or_404(camera_id: str):
    specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    w = get_recognition_hub().get_worker(camera_id)
    if w is None:
        raise HTTPException(
            status_code=503,
            detail="Recognition hub chưa chạy — khởi động python main.py (không dùng --no-camera)",
        )
    return w


async def _mjpeg_stream(camera_id: str) -> AsyncIterator[bytes]:
    poll = float(s.IVM_INFER_MJPEG_POLL_S)
    w = _worker_or_404(camera_id)
    while True:
        jpg = w.get_display_jpeg()
        if jpg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
            )
        import asyncio

        await asyncio.sleep(poll)


@router.get("/cameras/{camera_id}/infer/mjpeg")
async def infer_mjpeg(camera_id: str) -> StreamingResponse:
    """Luồng MJPEG có bbox — cập nhật sau mỗi lần infer (khi nhận diện BẬT)."""
    _worker_or_404(camera_id)
    return StreamingResponse(
        _mjpeg_stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/cameras/{camera_id}/infer/snapshot.jpg")
def infer_snapshot(camera_id: str) -> Response:
    w = _worker_or_404(camera_id)
    jpg = w.get_display_jpeg()
    if not jpg:
        raise HTTPException(status_code=503, detail="Chưa có khung hiển thị — đợi camera kết nối")
    return Response(content=jpg, media_type="image/jpeg")


@router.get("/cameras/{camera_id}/infer/status")
def infer_status(camera_id: str) -> dict:
    w = _worker_or_404(camera_id)
    meta = w.get_meta()
    meta["recognition_enabled"] = get_analyze_enabled(camera_id)
    return {
        "camera_id": camera_id,
        "reader_connected": w.reader.is_connected,
        "reader_fps": w.reader.fps_actual,
        "frame_count": w.reader.frame_count,
        "weapon_detection_enabled": weapon_detection_available(),
        "recognition": meta,
    }


@router.get("/cameras/{camera_id}/analyze/visual/sessions")
def list_analyze_visual_sessions(camera_id: str, limit: int = 20) -> dict:
    """Danh sách file video overlay đã ghi khi nhận diện BẬT."""
    specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    rec = get_recorders().get(camera_id)
    archive = None
    if rec is not None:
        sid, path, t0, now = rec.current_archive_ref()
        archive = {
            "running": rec.is_running(),
            "segment_id": sid,
            "archive_path": path,
            "segment_started_utc": t0,
        }
    return {
        "camera_id": camera_id,
        "recognition_enabled": get_analyze_enabled(camera_id),
        "archive": archive,
        "visual_sessions": list_visual_sessions(camera_id, limit=limit),
    }


@router.get("/cameras/{camera_id}/analyze/visual/{session_id}.mp4")
def download_analyze_visual_mp4(camera_id: str, session_id: str) -> FileResponse:
    """Tải video overlay (bbox + tên) của một phiên nhận diện."""
    specs = {str(c["id"]) for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}
    if camera_id not in specs:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    live = get_visual_session(camera_id)
    if live and str(live.get("session_id")) == session_id and live.get("recording"):
        p = Path(str(live.get("path", "")))
        if p.is_file() and p.stat().st_size > 0:
            return FileResponse(p, media_type="video/mp4", filename=p.name)
        raise HTTPException(
            status_code=409,
            detail="Phiên đang ghi — tắt nhận diện để đóng file, rồi tải lại.",
        )
    import os

    root = Path(os.getenv("IVM_ANALYZE_VISUAL_DIR", str(s.IVM_DATA_DIR / "analyze_visual"))).resolve()
    p = (root / camera_id / f"session_{session_id}.mp4").resolve()
    if not p.is_file() or p.stat().st_size <= 0:
        raise HTTPException(status_code=404, detail="Không tìm thấy video phiên này")
    from identity_vm_app.services.visual_mp4 import browser_mp4_path

    web = browser_mp4_path(p)
    serve = web if web.is_file() and web.stat().st_size > 0 else p
    return FileResponse(serve, media_type="video/mp4", filename=serve.name)


@router.get("/cameras/infer/status")
def infer_hub_status() -> dict:
    hub = get_recognition_hub()
    out = hub.status()
    for row in out.get("cameras") or []:
        cid = str(row.get("camera_id", ""))
        if cid:
            row["recognition_enabled"] = get_analyze_enabled(cid)
    return out
