"""Host app settings + re-export AI settings from module_ai.config.settings."""

from __future__ import annotations

import os
from pathlib import Path

from module_ai.config.settings import *  # noqa: F403
from module_ai.config.settings import (  # noqa: F401
    LEGACY_MODEL_DIR,
    MODEL_DIR,
    MODULE_AI_ROOT,
    REPO_ROOT,
    ivm_bulk_worker_ctx_ids,
    ivm_report_min_track_frames,
    resolve_identify_infer_workers,
    resolve_model_file,
)

def _truthy(name: str, default: str = "") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return float(str(raw).strip())


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    return int(str(raw).strip())


# Data layout (host)
IVM_DATA_DIR = Path(os.getenv("IVM_DATA_DIR", str(REPO_ROOT / "identity_vm_data"))).resolve()
IVM_FACE_DB_DIR = Path(os.getenv("IVM_FACE_DB_DIR", str(IVM_DATA_DIR / "face_db")))
IVM_SQLITE_PATH = Path(os.getenv("IVM_SQLITE_PATH", str(IVM_DATA_DIR / "identity_vm.sqlite3")))
IVM_ARCHIVE_ROOT = Path(os.getenv("IVM_ARCHIVE_ROOT", str(IVM_DATA_DIR / "archive")))

# API (host)
IVM_API_HOST = os.getenv("IVM_API_HOST", "0.0.0.0")
IVM_API_PORT = _int("IVM_API_PORT", 8010)

# Recorder / export (host)
IVM_FFMPEG_BIN = os.getenv("IVM_FFMPEG_BIN", "ffmpeg")
IVM_SEGMENT_SECONDS = _int("IVM_SEGMENT_SECONDS", 3600)
IVM_EXPORT_CACHE_DIR = Path(os.getenv("IVM_EXPORT_CACHE_DIR", str(IVM_DATA_DIR / "export_cache")))
IVM_EXPORT_CUT_MIN_DURATION_S = max(0.5, min(60.0, _float("IVM_EXPORT_CUT_MIN_DURATION_S", 3.0)))

# Video analyze storage (host paths; pipeline settings live in module_ai)
IVM_VIDEO_ANALYZE_DIR = Path(
    os.getenv("IVM_VIDEO_ANALYZE_DIR", str(IVM_DATA_DIR / "video_analyze"))
).resolve()
os.environ.setdefault("IVM_VIDEO_ANALYZE_MAX_MB", "0")
os.environ.setdefault("IVM_VIDEO_ANALYZE_MAX_DURATION_S", "0")
IVM_VIDEO_ANALYZE_MAX_MB = max(0, _int("IVM_VIDEO_ANALYZE_MAX_MB", 0))
IVM_VIDEO_ANALYZE_MAX_DURATION_S = max(0.0, _float("IVM_VIDEO_ANALYZE_MAX_DURATION_S", 0.0))
IVM_VIDEO_ANALYZE_LOCAL_PATH_ROOTS: tuple[Path, ...] = tuple(
    Path(p.strip()).resolve()
    for p in os.getenv("IVM_VIDEO_ANALYZE_LOCAL_PATH_ROOTS", "").split(",")
    if p.strip()
)
IVM_VIDEO_ANALYZE_ALLOWED_SUFFIXES = tuple(
    x.strip().lower()
    for x in os.getenv("IVM_VIDEO_ANALYZE_ALLOWED_SUFFIXES", ".mp4,.avi,.mov,.mkv,.webm").split(",")
    if x.strip()
)
IVM_VIDEO_ANALYZE_HTTP_BASE = os.getenv(
    "IVM_VIDEO_ANALYZE_HTTP_BASE", f"http://127.0.0.1:{IVM_API_PORT}"
).rstrip("/")

IVM_REGISTER_FOLDER_PROGRESS_PATH = Path(
    os.getenv(
        "IVM_REGISTER_FOLDER_PROGRESS_PATH",
        str(IVM_DATA_DIR / "register_folder_progress.json"),
    )
).resolve()
IVM_REGISTER_FOLDER_BULK_LOG = Path(
    os.getenv(
        "IVM_REGISTER_FOLDER_BULK_LOG",
        str(IVM_DATA_DIR / "register_folder_bulk.log"),
    )
).resolve()

IVM_CAMERA_CONFIG = os.getenv("IVM_CAMERA_CONFIG", str(REPO_ROOT / "camera_config.json"))

_IVM_FFMPEG_DEFAULT = (
    "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|reorder_queue_size;0|stimeout;8000000"
)
IVM_FFMPEG_CAPTURE_OPTIONS = os.getenv("IVM_FFMPEG_CAPTURE_OPTIONS", _IVM_FFMPEG_DEFAULT).strip()
IVM_CAP_PROP_BUFFERSIZE = max(1, min(16, _int("IVM_CAP_PROP_BUFFERSIZE", 2)))

# Preview MJPEG (host)
IVM_PREVIEW_JPEG_QUALITY = max(40, min(95, _int("IVM_PREVIEW_JPEG_QUALITY", 78)))
IVM_PREVIEW_CAPTURE_FPS = max(5.0, min(60.0, _float("IVM_PREVIEW_CAPTURE_FPS", 20.0)))
IVM_PREVIEW_MJPEG_OUT_FPS = max(5.0, min(60.0, _float("IVM_PREVIEW_MJPEG_OUT_FPS", 22.0)))
IVM_PREVIEW_RECONNECT_DELAY_S = max(0.2, min(120.0, _float("IVM_PREVIEW_RECONNECT_DELAY_S", 2.5)))
IVM_PREVIEW_READ_FAILS_BEFORE_RECONNECT = max(1, min(60, _int("IVM_PREVIEW_READ_FAILS_BEFORE_RECONNECT", 4)))
IVM_PREVIEW_OPEN_BACKOFF_CAP_S = max(5.0, min(300.0, _float("IVM_PREVIEW_OPEN_BACKOFF_CAP_S", 60.0)))
IVM_PREVIEW_DECODE_VIA_FFMPEG = _truthy("IVM_PREVIEW_DECODE_VIA_FFMPEG", "1")
IVM_PREVIEW_FFMPEG_MAX_HEIGHT = max(240, min(2160, _int("IVM_PREVIEW_FFMPEG_MAX_HEIGHT", 720)))
IVM_PREVIEW_WARM_STAGGER_S = max(0.05, min(5.0, _float("IVM_PREVIEW_WARM_STAGGER_S", 0.35)))
IVM_PREVIEW_SNAPSHOT_MAX_WAIT_S = max(0.5, min(30.0, _float("IVM_PREVIEW_SNAPSHOT_MAX_WAIT_S", 6.0)))
IVM_PREVIEW_GRID_POLL_FPS = max(2.0, min(15.0, _float("IVM_PREVIEW_GRID_POLL_FPS", 8.0)))
IVM_PREVIEW_GRID_INITIAL_WAIT_S = max(0.0, min(10.0, _float("IVM_PREVIEW_GRID_INITIAL_WAIT_S", 2.0)))

IVM_AUTO_START_CAMERA_WORKERS = _truthy("IVM_AUTO_START_CAMERA_WORKERS", "0")
IVM_STREAM_FRAME_QUEUE_MAX = max(0, min(64, _int("IVM_STREAM_FRAME_QUEUE_MAX", 8)))
IVM_PIPELINE_START_STAGGER_S = max(0.0, min(10.0, _float("IVM_PIPELINE_START_STAGGER_S", 0.4)))

IVM_CAMERA_SESSION_DIR = Path(
    os.getenv("IVM_CAMERA_SESSION_DIR", str(IVM_DATA_DIR / "camera_sessions"))
)
IVM_CAMERA_SESSION_STREAM_RECORD = _truthy("IVM_CAMERA_SESSION_STREAM_RECORD", "1")
IVM_CAMERA_SESSION_OVERLAY_LIVE = _truthy("IVM_CAMERA_SESSION_OVERLAY_LIVE", "0")
IVM_CAMERA_LEGACY_EVENTS = _truthy("IVM_CAMERA_LEGACY_EVENTS", "0")
IVM_REPORT_ALL_INFER_FRAMES = _truthy("IVM_REPORT_ALL_INFER_FRAMES", "1")
IVM_ANALYZE_AUTO_ARCHIVE = _truthy("IVM_ANALYZE_AUTO_ARCHIVE", "1")
IVM_ANALYZE_RECORD_VISUAL = _truthy("IVM_ANALYZE_RECORD_VISUAL", "1")
IVM_ANALYZE_VISUAL_FPS = max(0.0, min(60.0, _float("IVM_ANALYZE_VISUAL_FPS", 0.0)))

IVM_PIPELINE_READ_FAILS_BEFORE_RECONNECT = max(1, min(30, _int("IVM_PIPELINE_READ_FAILS_BEFORE_RECONNECT", 4)))
IVM_RTSP_STALE_RECONNECT_S = max(0.0, _float("IVM_RTSP_STALE_RECONNECT_S", 20.0))
IVM_RTSP_PROACTIVE_RECONNECT_S = max(0.0, _float("IVM_RTSP_PROACTIVE_RECONNECT_S", 0.0))
IVM_RTSP_RECONNECT_DELAY_S = max(0.5, min(30.0, _float("IVM_RTSP_RECONNECT_DELAY_S", 2.5)))
IVM_CAMERA_READ_VIA_FFMPEG = _truthy("IVM_CAMERA_READ_VIA_FFMPEG", "0")
IVM_CAMERA_FFMPEG_READ_MAX_HEIGHT = max(0, _int("IVM_CAMERA_FFMPEG_READ_MAX_HEIGHT", 0))
IVM_RESET_SECRET = os.getenv("IVM_RESET_SECRET", "").strip()
