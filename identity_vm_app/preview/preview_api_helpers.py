"""Tham số chung ensure/warm preview — tránh lặp trong routes."""

from __future__ import annotations

from typing import Any

from identity_vm_app import settings as s
from identity_vm_app.preview.mjpeg_hub import CameraPreviewStream, get_preview_hub


def preview_ensure_kwargs() -> dict[str, Any]:
    return {
        "jpeg_quality": s.IVM_PREVIEW_JPEG_QUALITY,
        "capture_fps": s.IVM_PREVIEW_CAPTURE_FPS,
        "reconnect_delay_s": s.IVM_PREVIEW_RECONNECT_DELAY_S,
        "read_fails_before_reconnect": s.IVM_PREVIEW_READ_FAILS_BEFORE_RECONNECT,
        "open_backoff_cap_s": s.IVM_PREVIEW_OPEN_BACKOFF_CAP_S,
        "cap_prop_buffersize": s.IVM_CAP_PROP_BUFFERSIZE,
    }


def ensure_preview_stream(camera_id: str, source: Any) -> CameraPreviewStream:
    return get_preview_hub().ensure(camera_id, source, **preview_ensure_kwargs())


def get_preview_stream_if_active(camera_id: str, source: Any) -> CameraPreviewStream | None:
    """Chỉ trả stream đang chạy, hoặc mở mới khi đã warm (armed)."""
    hub = get_preview_hub()
    running = hub.get_running(camera_id)
    if running is not None:
        return running
    if hub.is_armed():
        return ensure_preview_stream(camera_id, source)
    return None


def warm_all_preview_streams(sources: dict[str, Any]) -> int:
    return get_preview_hub().warm_all(
        sources,
        stagger_s=float(s.IVM_PREVIEW_WARM_STAGGER_S),
        **preview_ensure_kwargs(),
    )
