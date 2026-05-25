"""AI / model / inference settings — override via IVM_* environment variables."""

from __future__ import annotations

import logging
import os
from pathlib import Path

_log = logging.getLogger(__name__)

MODULE_AI_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = MODULE_AI_ROOT.parent
MODEL_DIR = MODULE_AI_ROOT / "models"
LEGACY_MODEL_DIR = REPO_ROOT / "identity_vm_app" / "modelAi"


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


def resolve_model_file(names: tuple[str, ...], default_name: str) -> str:
    """Resolve weight path: module_ai/models, then legacy identity_vm_app/modelAi."""
    for base in (MODEL_DIR, LEGACY_MODEL_DIR):
        for name in names:
            p = base / name
            if p.is_file():
                if base == LEGACY_MODEL_DIR:
                    _log.warning(
                        "Model weight %s loaded from deprecated path %s — move to %s",
                        name,
                        LEGACY_MODEL_DIR,
                        MODEL_DIR,
                    )
                return str(p.resolve())
    return str((MODEL_DIR / default_name).resolve())


def _default_video_yolo_model() -> str:
    return resolve_model_file(
        ("yolo26s.pt", "yolo26m.pt", "yolo26s.engine", "yolo26s.onnx"),
        "yolo26s.pt",
    )


def _default_pose_model() -> str:
    return resolve_model_file(("yolo26m-pose.pt", "yolo26n-pose.pt"), "yolo26m-pose.pt")


# InsightFace — pack ONNX: {IVM_INSIGHTFACE_ROOT}/models/{IVM_INSIGHTFACE_MODEL_NAME}/
# Mặc định root = module_ai/ → E:/app_face/module_ai/models/buffalo_l/
IVM_INSIGHTFACE_MODEL_NAME = os.getenv("IVM_INSIGHTFACE_MODEL_NAME", "buffalo_l")
IVM_INSIGHTFACE_ROOT = os.getenv("IVM_INSIGHTFACE_ROOT", "").strip() or str(MODULE_AI_ROOT.resolve())
# 0 = không tự tải từ GitHub khi thiếu pack (báo lỗi + đường dẫn local)
IVM_INSIGHTFACE_AUTO_DOWNLOAD = _truthy("IVM_INSIGHTFACE_AUTO_DOWNLOAD", "0")


def insightface_pack_dir(model_name: str | None = None) -> Path:
    """Thư mục chứa *.onnx của pack (theo quy ước InsightFace: root/models/name/)."""
    name = (model_name or IVM_INSIGHTFACE_MODEL_NAME).strip()
    return (Path(IVM_INSIGHTFACE_ROOT).expanduser().resolve() / "models" / name)


def validate_insightface_pack(
    model_name: str | None = None,
    *,
    require_recognition: bool = True,
) -> Path:
    """
    Kiểm tra pack local trước khi load FaceAnalysis.
    Cần ít nhất det_10g.onnx + w600k_r50.onnx (detection + recognition).
    """
    pack = insightface_pack_dir(model_name)
    if not pack.is_dir():
        raise FileNotFoundError(
            f"Không thấy thư mục InsightFace pack: {pack}\n"
            f"Đặt file .onnx vào đó (ví dụ giải nén buffalo_l.zip) hoặc đặt IVM_INSIGHTFACE_ROOT."
        )
    onnx_files = sorted(pack.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(
            f"Không có file .onnx trong {pack}. "
            "Copy pack buffalo_l (det_10g.onnx, w600k_r50.onnx, …) vào thư mục này."
        )
    names = {p.name.lower() for p in onnx_files}
    missing: list[str] = []
    if not any(n == "det_10g.onnx" or n.startswith("det_") for n in names):
        missing.append("detection (cần det_10g.onnx)")
    if require_recognition and not any(
        n == "w600k_r50.onnx" or "w600k" in n or "r50" in n for n in names
    ):
        missing.append("recognition (cần w600k_r50.onnx)")
    if missing:
        raise FileNotFoundError(
            f"Pack InsightFace thiếu module: {', '.join(missing)} trong {pack}. "
            f"Có: {[p.name for p in onnx_files]}"
        )
    return pack
IVM_INSIGHTFACE_PROVIDERS = [
    p.strip()
    for p in os.getenv(
        "IVM_INSIGHTFACE_PROVIDERS",
        "CUDAExecutionProvider,CPUExecutionProvider",
    ).split(",")
    if p.strip()
]
IVM_CTX_ID = _int("IVM_CTX_ID", 0)
IVM_DET_SIZE = (_int("IVM_DET_W", 640), _int("IVM_DET_H", 640))
IVM_DET_THRESH = _float("IVM_DET_THRESH", 0.5)
IVM_MODEL_TAG = os.getenv("IVM_MODEL_TAG", IVM_INSIGHTFACE_MODEL_NAME)

# Recognition DB
IVM_DISTANCE_THRESHOLD = _float("IVM_DISTANCE_THRESHOLD", 0.7)
IVM_SEARCH_K = _int("IVM_SEARCH_K", 5)
IVM_IDENTIFY_BATCH_MAX_FILES = max(0, _int("IVM_IDENTIFY_BATCH_MAX_FILES", 0))
IVM_IDENTIFY_BATCH_INFER_WORKERS = max(1, min(16, _int("IVM_IDENTIFY_BATCH_INFER_WORKERS", 1)))
IVM_IDENTIFY_BATCH_API_MAX_INFER_WORKERS = max(
    1, min(16, _int("IVM_IDENTIFY_BATCH_API_MAX_INFER_WORKERS", 16))
)
IVM_REC_GET_FEAT_MAX_BATCH = max(1, min(256, _int("IVM_REC_GET_FEAT_MAX_BATCH", 64)))
IVM_IDENTIFY_IMAGES_MERGE_REC = _truthy("IVM_IDENTIFY_IMAGES_MERGE_REC", "1")
IVM_IDENTIFY_IMAGES_PROCESS_CHUNK = max(0, _int("IVM_IDENTIFY_IMAGES_PROCESS_CHUNK", 8))
IVM_USE_FAISS = _truthy("IVM_USE_FAISS", "0")
IVM_LOG_UNKNOWN_EVENTS = _truthy("IVM_LOG_UNKNOWN_EVENTS", "1")
IVM_EVENT_DEBOUNCE_S = _float("IVM_EVENT_DEBOUNCE_S", 3.0)

# Video offline analyze
IVM_VIDEO_ANALYZE_USE_YOLO_TRACKING = _truthy("IVM_VIDEO_ANALYZE_USE_YOLO_TRACKING", "1")
IVM_VIDEO_ANALYZE_YOLO_MODEL = os.getenv(
    "IVM_VIDEO_ANALYZE_YOLO_MODEL", _default_video_yolo_model()
).strip()
IVM_VIDEO_ANALYZE_TRACKER = os.getenv("IVM_VIDEO_ANALYZE_TRACKER", "bytetrack.yaml").strip()
IVM_VIDEO_ANALYZE_YOLO_CONF = max(0.05, min(0.99, _float("IVM_VIDEO_ANALYZE_YOLO_CONF", 0.5)))
IVM_VIDEO_ANALYZE_YOLO_IMGSZ = max(320, min(1280, _int("IVM_VIDEO_ANALYZE_YOLO_IMGSZ", 640)))
IVM_FACE_ASSIGN_IOU_MIN = max(0.0, min(0.95, _float("IVM_FACE_ASSIGN_IOU_MIN", 0.05)))
IVM_FACE_ASSIGN_CENTER_IN_PERSON = _truthy("IVM_FACE_ASSIGN_CENTER_IN_PERSON", "1")
IVM_FACE_POSE_REFINE = _truthy("IVM_FACE_POSE_REFINE", "1")
IVM_FACE_POSE_MODEL = os.getenv("IVM_FACE_POSE_MODEL", _default_pose_model()).strip()
IVM_FACE_POSE_CONF = max(0.05, min(0.99, _float("IVM_FACE_POSE_CONF", 0.40)))
IVM_FACE_POSE_IMGSZ = max(320, min(1280, _int("IVM_FACE_POSE_IMGSZ", 640)))
IVM_FACE_POSE_KP_CONF = max(0.1, min(0.99, _float("IVM_FACE_POSE_KP_CONF", 0.40)))
IVM_FACE_POSE_MIN_FACES = max(2, min(8, _int("IVM_FACE_POSE_MIN_FACES", 2)))
IVM_FACE_POSE_DET_IOU_MIN = max(0.01, min(0.5, _float("IVM_FACE_POSE_DET_IOU_MIN", 0.05)))
IVM_TRACK_SUSPECT_NAMES_LIMIT = max(1, min(10, _int("IVM_TRACK_SUSPECT_NAMES_LIMIT", 5)))
IVM_REPORT_MIN_TRACK_FRAMES = max(0, _int("IVM_REPORT_MIN_TRACK_FRAMES", 5))
IVM_VIDEO_ANALYZE_PERSIST_EMBEDDINGS = _truthy("IVM_VIDEO_ANALYZE_PERSIST_EMBEDDINGS", "1")
IVM_EVENT_SAVE_CROPS = _truthy("IVM_EVENT_SAVE_CROPS", "0")
IVM_VIDEO_ANALYZE_EMBED_BATCH = max(1, min(256, _int("IVM_VIDEO_ANALYZE_EMBED_BATCH", 32)))
IVM_VIDEO_ANALYZE_DB_BATCH = max(1, min(512, _int("IVM_VIDEO_ANALYZE_DB_BATCH", 48)))
IVM_VIDEO_ANALYZE_PROGRESS_EVERY = max(1, _int("IVM_VIDEO_ANALYZE_PROGRESS_EVERY", 5))
IVM_VIDEO_ANALYZE_JPEG_QUALITY = max(50, min(95, _int("IVM_VIDEO_ANALYZE_JPEG_QUALITY", 82)))
IVM_TRACK_SEGMENT_SLIDESHOW_MS = max(80, min(800, _int("IVM_TRACK_SEGMENT_SLIDESHOW_MS", 200)))
IVM_TRACK_SEGMENT_ENCODE_FPS_MIN = max(5.0, min(60.0, _float("IVM_TRACK_SEGMENT_ENCODE_FPS_MIN", 15.0)))
IVM_TRACK_SEGMENT_ENCODE_FPS_MAX = max(10.0, min(60.0, _float("IVM_TRACK_SEGMENT_ENCODE_FPS_MAX", 30.0)))
IVM_VIDEO_ANALYZE_SAVE_CROPS = _truthy("IVM_VIDEO_ANALYZE_SAVE_CROPS", "0")
IVM_VIDEO_ANALYZE_DEFAULT_SAMPLE_FPS = _float("IVM_VIDEO_ANALYZE_DEFAULT_SAMPLE_FPS", 0.0)
IVM_VIDEO_ANALYZE_MAX_CONCURRENT = max(1, min(8, _int("IVM_VIDEO_ANALYZE_MAX_CONCURRENT", 2)))
IVM_VIDEO_ANALYZE_SPLIT_PARTS = max(1, min(8, _int("IVM_VIDEO_ANALYZE_SPLIT_PARTS", 4)))
IVM_VIDEO_ANALYZE_SPLIT_GPU_ENCODE = _truthy("IVM_VIDEO_ANALYZE_SPLIT_GPU_ENCODE", "0")
IVM_VIDEO_ANALYZE_TIMELINE_MAX = max(20, min(500, _int("IVM_VIDEO_ANALYZE_TIMELINE_MAX", 200)))

# Camera config + RTSP pipeline (module_ai.camera hub/worker)
IVM_CAMERA_CONFIG = os.getenv("IVM_CAMERA_CONFIG", str(REPO_ROOT / "camera_config.json"))
IVM_API_PORT = _int("IVM_API_PORT", 8010)
IVM_PIPELINE_START_STAGGER_S = max(0.0, min(10.0, _float("IVM_PIPELINE_START_STAGGER_S", 0.4)))
IVM_CAP_PROP_BUFFERSIZE = max(1, min(16, _int("IVM_CAP_PROP_BUFFERSIZE", 2)))
IVM_PIPELINE_READ_FAILS_BEFORE_RECONNECT = max(1, min(30, _int("IVM_PIPELINE_READ_FAILS_BEFORE_RECONNECT", 4)))
IVM_RTSP_STALE_RECONNECT_S = max(0.0, _float("IVM_RTSP_STALE_RECONNECT_S", 20.0))
IVM_RTSP_PROACTIVE_RECONNECT_S = max(0.0, _float("IVM_RTSP_PROACTIVE_RECONNECT_S", 0.0))
IVM_RTSP_RECONNECT_DELAY_S = max(0.5, min(30.0, _float("IVM_RTSP_RECONNECT_DELAY_S", 2.5)))
IVM_CAMERA_LEGACY_EVENTS = _truthy("IVM_CAMERA_LEGACY_EVENTS", "0")
IVM_DATA_DIR = Path(os.getenv("IVM_DATA_DIR", str(REPO_ROOT / "identity_vm_data"))).resolve()
IVM_ANALYZE_VISUAL_FPS = max(0.0, min(60.0, _float("IVM_ANALYZE_VISUAL_FPS", 0.0)))
IVM_ANALYZE_RECORD_VISUAL = _truthy("IVM_ANALYZE_RECORD_VISUAL", "1")
IVM_ANALYZE_AUTO_ARCHIVE = _truthy("IVM_ANALYZE_AUTO_ARCHIVE", "1")
IVM_EXPORT_CUT_MIN_DURATION_S = max(0.5, min(60.0, _float("IVM_EXPORT_CUT_MIN_DURATION_S", 3.0)))

# Live camera infer
IVM_ANALYZE_EVEN_FRAMES_ONLY = _truthy("IVM_ANALYZE_EVEN_FRAMES_ONLY", "1")
IVM_ANALYZE_TARGET_FPS = max(0.5, min(30.0, _float("IVM_ANALYZE_TARGET_FPS", 5.0)))
IVM_CAMERA_DEFAULT_SAMPLE_FPS = max(0.0, _float("IVM_CAMERA_DEFAULT_SAMPLE_FPS", 5.0))
IVM_ANALYZE_INTERVAL_S = _float("IVM_ANALYZE_INTERVAL_S", 0.0)
IVM_ANALYZE_MAX_FRAME_WIDTH = max(0, _int("IVM_ANALYZE_MAX_FRAME_WIDTH", 960))
IVM_ANALYZE_JPEG_QUALITY = max(40, min(100, _int("IVM_ANALYZE_JPEG_QUALITY", 85)))
IVM_ANALYZE_IDENTIFY_TIMEOUT_S = max(5.0, min(300.0, _float("IVM_ANALYZE_IDENTIFY_TIMEOUT_S", 60.0)))
IVM_ANALYZE_INGEST_TIMEOUT_S = max(2.0, min(120.0, _float("IVM_ANALYZE_INGEST_TIMEOUT_S", 15.0)))
IVM_USE_IN_PROCESS_INFER = _truthy("IVM_USE_IN_PROCESS_INFER", "1")
IVM_USE_PERSON_FIRST_PIPELINE = _truthy("IVM_USE_PERSON_FIRST_PIPELINE", "1")
IVM_INFER_DISPLAY_JPEG_QUALITY = max(50, min(95, _int("IVM_INFER_DISPLAY_JPEG_QUALITY", 80)))
IVM_INFER_MJPEG_POLL_S = max(0.02, min(0.5, _float("IVM_INFER_MJPEG_POLL_S", 0.08)))

# Weapon detection
IVM_WEAPON_ENABLED = _truthy("IVM_WEAPON_ENABLED", "1")
IVM_WEAPON_MODEL = os.getenv("IVM_WEAPON_MODEL", resolve_model_file(("weapon_detect.pt",), "weapon_detect.pt"))
IVM_WEAPON_IMGSZ = max(320, min(1280, _int("IVM_WEAPON_IMGSZ", 640)))
IVM_WEAPON_DEVICE = os.getenv("IVM_WEAPON_DEVICE", "0").strip() or "0"
IVM_WEAPON_ROI_PAD_RATIO = max(0.0, min(0.5, _float("IVM_WEAPON_ROI_PAD_RATIO", 0.12)))
IVM_WEAPON_MATCH_IOU_MIN = max(0.0, min(0.5, _float("IVM_WEAPON_MATCH_IOU_MIN", 0.01)))
IVM_WEAPON_POSE_MODEL = os.getenv("IVM_WEAPON_POSE_MODEL", _default_pose_model()).strip()
IVM_WEAPON_WARMUP = _truthy("IVM_WEAPON_WARMUP", "1")
IVM_WEAPON_BATCH_SIZE = max(1, min(32, _int("IVM_WEAPON_BATCH_SIZE", 8)))
IVM_WEAPON_MEMORY_FRAMES = max(0, _int("IVM_WEAPON_MEMORY_FRAMES", 10))
IVM_WEAPON_INPUT_CONF = max(0.05, min(0.99, _float("IVM_WEAPON_INPUT_CONF", 0.50)))
IVM_WEAPON_VOTER_WINDOW = max(1, min(30, _int("IVM_WEAPON_VOTER_WINDOW", 5)))
IVM_WEAPON_VOTER_THRESHOLD = max(0.05, min(0.99, _float("IVM_WEAPON_VOTER_THRESHOLD", 0.40)))
IVM_WEAPON_ARMED_MIN_FRAMES = max(1, _int("IVM_WEAPON_ARMED_MIN_FRAMES", 10))
IVM_WEAPON_DANGEROUS_MIN_FRAMES = max(1, _int("IVM_WEAPON_DANGEROUS_MIN_FRAMES", 20))
IVM_WEAPON_ALERT_ENABLED = _truthy("IVM_WEAPON_ALERT_ENABLED", "1")
IVM_WEAPON_ALERT_MIN_FRAMES = max(1, _int("IVM_WEAPON_ALERT_MIN_FRAMES", 5))
IVM_WEAPON_ALERT_HISTORY_PER_CAMERA = max(1, min(10, _int("IVM_WEAPON_ALERT_HISTORY_PER_CAMERA", 3)))
IVM_WEAPON_ALERT_THUMB_MAX_WIDTH = max(120, min(960, _int("IVM_WEAPON_ALERT_THUMB_MAX_WIDTH", 480)))
IVM_WEAPON_ALERT_FULL_MAX_WIDTH = max(0, min(3840, _int("IVM_WEAPON_ALERT_FULL_MAX_WIDTH", 0)))
IVM_WEAPON_ALERT_SNAPSHOT_JPEG_QUALITY = max(60, min(98, _int("IVM_WEAPON_ALERT_SNAPSHOT_JPEG_QUALITY", 90)))
IVM_WEAPON_EVERY_N_CYCLES = max(1, _int("IVM_WEAPON_EVERY_N_CYCLES", 1))
IVM_WEAPON_RELEASE_AFTER_USE = _truthy("IVM_WEAPON_RELEASE_AFTER_USE", "1")
IVM_GPU_SOFT_CLEANUP_ON_PROCESS_DONE = _truthy(
    "IVM_GPU_SOFT_CLEANUP_ON_PROCESS_DONE",
    os.environ.get("IVM_GPU_SOFT_CLEANUP_AFTER_INFER", "1"),
)
IVM_GPU_SOFT_CLEANUP_AFTER_INFER = IVM_GPU_SOFT_CLEANUP_ON_PROCESS_DONE
IVM_GPU_RELEASE_ENGINE_AFTER_IMAGE_INFER = _truthy("IVM_GPU_RELEASE_ENGINE_AFTER_IMAGE_INFER", "1")

# Camera dedicated models
IVM_CAMERA_DEDICATED_MODELS = _truthy("IVM_CAMERA_DEDICATED_MODELS", "1")
IVM_SHARED_YOLO_PERSON = _truthy("IVM_SHARED_YOLO_PERSON", "0")
IVM_CAMERA_INFER_QUEUE_MAX = max(0, _int("IVM_CAMERA_INFER_QUEUE_MAX", 0))
IVM_CAMERA_INFER_QUEUE_WARN_DEPTH = max(1, _int("IVM_CAMERA_INFER_QUEUE_WARN_DEPTH", 30))
IVM_CAMERA_INFER_QUEUE_DRAIN_TIMEOUT_S = max(5.0, min(3600.0, _float("IVM_CAMERA_INFER_QUEUE_DRAIN_TIMEOUT_S", 300.0)))
IVM_CAMERA_DRAIN_INFER_ON_STOP = _truthy("IVM_CAMERA_DRAIN_INFER_ON_STOP", "1")
IVM_INFER_STATS_EVERY_N = max(0, _int("IVM_INFER_STATS_EVERY_N", 20))
IVM_INFER_STATS_FIRST_N = max(1, min(100, _int("IVM_INFER_STATS_FIRST_N", 5)))
IVM_CAMERA_HUB_STATUS_LOG_INTERVAL_S = max(0.0, _float("IVM_CAMERA_HUB_STATUS_LOG_INTERVAL_S", 20.0))
IVM_CAMERA_INFER_OK_LOG_QUEUE_THROTTLE = max(0, _int("IVM_CAMERA_INFER_OK_LOG_QUEUE_THROTTLE", 40))
IVM_CAMERA_INFER_OK_LOG_EVERY_N_WHEN_BUSY = max(1, _int("IVM_CAMERA_INFER_OK_LOG_EVERY_N_WHEN_BUSY", 30))

# Bulk register
IVM_BULK_DB_WRITE_BATCH = max(1, _int("IVM_BULK_DB_WRITE_BATCH", 64))
IVM_BULK_MULTI_THREAD_DB_FLUSH = max(1, _int("IVM_BULK_MULTI_THREAD_DB_FLUSH", 1000))
IVM_BULK_ALLOWED_ROOTS: list[str] = [
    p.strip()
    for p in os.getenv("IVM_BULK_ALLOWED_ROOTS", "").split(",")
    if p.strip()
]
IVM_BULK_API_MAX_FILES = _int("IVM_BULK_API_MAX_FILES", 0)
IVM_BULK_RESUME_SKIP_FAILED = _truthy("IVM_BULK_RESUME_SKIP_FAILED", "1")
IVM_BULK_FAILURE_SAMPLE_MAX_BYTES = _int("IVM_BULK_FAILURE_SAMPLE_MAX_BYTES", 5_000_000)
IVM_BULK_PREFETCH = max(1, _int("IVM_BULK_PREFETCH", 4))
IVM_BULK_INFER_WORKERS = max(1, min(16, _int("IVM_BULK_INFER_WORKERS", 1)))
IVM_BULK_API_MAX_INFER_WORKERS = max(1, min(16, _int("IVM_BULK_API_MAX_INFER_WORKERS", 16)))
IVM_BULK_MAX_SORT_FILES = max(0, _int("IVM_BULK_MAX_SORT_FILES", 250_000))
IVM_BULK_CHECKPOINT_LOOKUP_CHUNK = max(50, min(900, _int("IVM_BULK_CHECKPOINT_LOOKUP_CHUNK", 900)))


def ivm_report_min_track_frames() -> int:
    base = int(IVM_REPORT_MIN_TRACK_FRAMES)
    if base <= 0:
        return 0
    return max(10, base)


def resolve_identify_infer_workers(requested: int | None) -> int:
    cap_w = max(1, min(16, int(IVM_IDENTIFY_BATCH_API_MAX_INFER_WORKERS)))
    env_w = max(1, min(cap_w, int(IVM_IDENTIFY_BATCH_INFER_WORKERS)))
    if requested is None:
        return env_w
    return max(1, min(cap_w, int(requested)))


def ivm_bulk_worker_ctx_ids() -> list[int]:
    raw = os.getenv("IVM_BULK_WORKER_CTX_IDS", "").strip()
    if not raw:
        return [IVM_CTX_ID]
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            ids.append(int(part))
    return ids if ids else [IVM_CTX_ID]
