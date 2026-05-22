"""Thiết lập OPENCV_FFMPEG_CAPTURE_OPTIONS trước khi `import cv2` — RTSP/HEVC ổn định hơn."""

from __future__ import annotations

import os


def apply_ffmpeg_capture_env() -> None:
    """
    Nếu process đã set OPENCV_FFMPEG_CAPTURE_OPTIONS thì giữ nguyên.
    Ngược lại gán từ `identity_vm_app.settings.IVM_FFMPEG_CAPTURE_OPTIONS`.
    Gọi càng sớm càng tốt (main.py, streamlit, trước import cv2).
    """
    if os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "").strip():
        return
    from identity_vm_app import settings as s

    opts = (getattr(s, "IVM_FFMPEG_CAPTURE_OPTIONS", None) or "").strip()
    if opts:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
