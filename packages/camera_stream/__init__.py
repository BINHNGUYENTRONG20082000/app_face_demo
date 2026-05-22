"""
Tầng kết nối camera / stream — tách khỏi nhận diện.

Dùng trước: ổn định RTSP hoặc webcam; sau đó worker nhận diện chỉ consume
`get_frame()` / `get_jpeg()` (FFmpeg hoặc OpenCV).
"""

from __future__ import annotations

from packages.camera_stream.config import StreamConnectionConfig
from packages.camera_stream.factory import create_camera_reader, should_use_ffmpeg_reader
from packages.camera_stream.ffmpeg_reader import FfmpegCameraReader
from packages.camera_stream.opencv_reader import FrameDecodedCallback, StableCameraReader

__all__ = [
    "StreamConnectionConfig",
    "StableCameraReader",
    "FfmpegCameraReader",
    "FrameDecodedCallback",
    "create_camera_reader",
    "should_use_ffmpeg_reader",
]
