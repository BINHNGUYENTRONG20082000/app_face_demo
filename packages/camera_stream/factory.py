"""Chọn reader OpenCV hoặc FFmpeg theo nguồn và cấu hình."""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

from identity_vm_app import settings as s
from packages.camera_stream.config import StreamConnectionConfig
from packages.camera_stream.ffmpeg_common import ffmpeg_cli_available, rtsp_url
from packages.camera_stream.ffmpeg_reader import FfmpegCameraReader
from packages.camera_stream.opencv_reader import StableCameraReader

logger = logging.getLogger("camera_stream.factory")

ReaderT = Union[StableCameraReader, FfmpegCameraReader]


def should_use_ffmpeg_reader(source: Any) -> bool:
    if not bool(getattr(s, "IVM_CAMERA_READ_VIA_FFMPEG", True)):
        return False
    if rtsp_url(source) is None:
        return False
    return ffmpeg_cli_available()


def create_camera_reader(
    camera_id: str,
    source: Any,
    *,
    config: Optional[StreamConnectionConfig] = None,
) -> ReaderT:
    """RTSP/HTTP → FFmpeg (mặc định); webcam/số/file → OpenCV."""
    if should_use_ffmpeg_reader(source):
        logger.info("[%s] Dùng FfmpegCameraReader cho %s", camera_id, str(source)[:80])
        return FfmpegCameraReader(camera_id, source, config=config)
    logger.info("[%s] Dùng StableCameraReader (OpenCV)", camera_id)
    return StableCameraReader(camera_id, source, config=config)
