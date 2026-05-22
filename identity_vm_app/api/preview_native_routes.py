"""Xem trước MJPEG qua `packages.camera_stream.StableCameraReader` (test ổn định kết nối)."""

from __future__ import annotations

import time
from typing import Any, Iterator

from camera_channel_config import load_camera_channel_specs
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from identity_vm_app import settings as s
from identity_vm_app.preview.native_reader_hub import get_native_preview_hub

router = APIRouter(prefix="/ivm/preview_native", tags=["preview-native"])


def _spec_sources() -> dict[str, Any]:
    return {str(c["id"]): c["source"] for c in load_camera_channel_specs(s.IVM_CAMERA_CONFIG)}


@router.get("/status")
def native_preview_status() -> dict:
    """Trạng thái reader nội bộ (FPS, số frame, lỗi)."""
    sources = _spec_sources()
    hub = get_native_preview_hub()
    out: list[dict[str, Any]] = []
    for cid, src in sources.items():
        r = hub.get(cid)
        if r is None:
            out.append(
                {
                    "camera_id": cid,
                    "running": False,
                    "connected": False,
                    "source": src,
                }
            )
        else:
            out.append(
                {
                    "camera_id": cid,
                    "running": r.is_running,
                    "connected": r.is_connected,
                    "fps": r.fps_actual,
                    "frames": r.frame_count,
                    "opens_ok": r.connect_success_count,
                    "error": r.last_error(),
                    "source": src,
                }
            )
    return {"cameras": out}


@router.get("/{camera_id}/mjpeg")
def native_mjpeg_stream(camera_id: str) -> StreamingResponse:
    sources = _spec_sources()
    if camera_id not in sources:
        raise HTTPException(status_code=404, detail="Unknown camera_id")
    hub = get_native_preview_hub()
    reader = hub.ensure(camera_id, sources[camera_id])

    q = int(s.IVM_PREVIEW_JPEG_QUALITY)
    out_period = 1.0 / max(5.0, min(60.0, float(s.IVM_PREVIEW_MJPEG_OUT_FPS)))

    def gen() -> Iterator[bytes]:
        t0 = time.monotonic()
        while time.monotonic() - t0 < 8.0:
            j = reader.get_jpeg(quality=q)
            if j:
                break
            time.sleep(0.04)
        while True:
            j = reader.get_jpeg(quality=q)
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
def stop_native_preview(camera_id: str) -> dict:
    get_native_preview_hub().stop(camera_id)
    return {"camera_id": camera_id, "stopped": True, "mode": "native_reader"}
