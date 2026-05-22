"""MJPEG xem trước — luồng độc lập với nhận diện (worker `camera_pipeline`)."""

from __future__ import annotations

import threading
import time
from typing import Any, Iterator

import cv2
import numpy as np
from camera_channel_config import load_camera_channel_specs
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from identity_vm_app import settings as s
from identity_vm_app.preview.mjpeg_hub import get_preview_hub
from identity_vm_app.preview.preview_api_helpers import (
    ensure_preview_stream,
    get_preview_stream_if_active,
    warm_all_preview_streams,
)
from identity_vm_app.preview.snapshot_grid_html import build_snapshot_grid_html

router = APIRouter(prefix="/ivm/preview", tags=["preview"])

_PLACEHOLDER_JPEG: bytes | None = None
_warm_lock = threading.Lock()
_warm_running = False


def _header_latin1(value: str, max_len: int = 200) -> str:
    """Starlette encodes response header values as latin-1; strip/safe-fail Unicode."""
    if not value:
        return ""
    return value.encode("latin-1", errors="replace").decode("latin-1")[:max_len]


def _placeholder_jpeg() -> bytes:
    global _PLACEHOLDER_JPEG
    if _PLACEHOLDER_JPEG is None:
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        _PLACEHOLDER_JPEG = buf.tobytes() if ok else b""
    return _PLACEHOLDER_JPEG


def _spec_sources() -> dict[str, Any]:
    return {str(c["id"]): c["source"] for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}


@router.get("/status")
def preview_status() -> dict:
    """Camera có trong config."""
    return {"cameras": list(_spec_sources().keys())}


def _warm_background(sources: dict[str, Any]) -> None:
    global _warm_running
    try:
        warm_all_preview_streams(sources)
    finally:
        with _warm_lock:
            _warm_running = False


@router.get("/grid")
def snapshot_grid(
    request: Request,
    cols: int = Query(2, ge=1, le=6, description="Số cột lưới (2–6)"),
) -> HTMLResponse:
    """Trang HTML lưới — polling snapshot từng camera (dùng trong iframe Streamlit)."""
    camera_ids = list(_spec_sources().keys())
    api_base = str(request.base_url).rstrip("/")
    page, _ = build_snapshot_grid_html(
        api_base,
        camera_ids,
        cols,
        poll_fps=float(s.IVM_PREVIEW_GRID_POLL_FPS),
    )
    return HTMLResponse(
        content=page,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@router.post("/warm")
def warm_previews() -> dict:
    """Bật reader cho mọi camera (chạy nền — không chặn snapshot polling)."""
    global _warm_running
    sources = _spec_sources()
    ids = list(sources.keys())
    with _warm_lock:
        if _warm_running:
            return {"status": "already_warming", "camera_ids": ids, "count": len(ids)}
        _warm_running = True
    threading.Thread(
        target=_warm_background,
        args=(sources,),
        name="ivm-preview-warm",
        daemon=True,
    ).start()
    return {"status": "warming", "camera_ids": ids, "count": len(ids)}


@router.get("/{camera_id}/snapshot.jpg")
def snapshot_jpg(
    camera_id: str,
    wait_s: float = Query(0.0, ge=0.0, le=float(s.IVM_PREVIEW_SNAPSHOT_MAX_WAIT_S)),
) -> Response:
    """Một khung JPEG mới nhất — lưới nhiều camera (polling), không giữ kết nối MJPEG."""
    sources = _spec_sources()
    if camera_id not in sources:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    stream = get_preview_stream_if_active(camera_id, sources[camera_id])
    if stream is None:
        return Response(
            content=_placeholder_jpeg(),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-IVM-Preview-Error": "preview_off",
            },
        )
    wait_s = min(float(wait_s), float(s.IVM_PREVIEW_SNAPSHOT_MAX_WAIT_S))
    deadline = time.monotonic() + wait_s if wait_s > 0 else time.monotonic()
    while True:
        j = stream.get_jpeg()
        if j:
            return Response(
                content=j,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
            )
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    err = stream.error_message()
    body = _placeholder_jpeg()
    headers = {
        "Cache-Control": "no-store",
        "X-IVM-Preview-Error": _header_latin1(err or "no_frame"),
    }
    return Response(content=body, media_type="image/jpeg", headers=headers)


@router.get("/{camera_id}/mjpeg")
def mjpeg_stream(camera_id: str) -> StreamingResponse:
    """multipart/x-mixed-replace — dùng cho phóng to / xem mượt một camera."""
    sources = _spec_sources()
    if camera_id not in sources:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    stream = ensure_preview_stream(camera_id, sources[camera_id])

    out_period = 1.0 / max(5.0, min(60.0, float(s.IVM_PREVIEW_MJPEG_OUT_FPS)))

    def gen() -> Iterator[bytes]:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 8.0:
            j = stream.get_jpeg()
            if j:
                break
            time.sleep(0.04)
        while True:
            j = stream.get_jpeg()
            if j:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
            time.sleep(out_period)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{camera_id}/stop")
def stop_preview(camera_id: str) -> dict:
    get_preview_hub().stop(camera_id)
    return {"camera_id": camera_id, "stopped": True}
