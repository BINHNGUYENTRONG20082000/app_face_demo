"""Cấu hình app tách biệt với `config.py` của dự án cũ — override bằng biến môi trường IVM_* hoặc mặc định dưới đây."""

from __future__ import annotations

import os
from pathlib import Path


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


REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_video_yolo_model() -> str:
    """YOLO26 — ưu tiên identity_vm_app/modelAi/."""
    model_dir = Path(__file__).resolve().parent / "modelAi"
    for name in ("yolo26s.pt", "yolo26m.pt", "yolo26s.engine", "yolo26s.onnx"):
        p = model_dir / name
        if p.is_file():
            return str(p.resolve())
    return str((model_dir / "yolo26s.pt").resolve())


def _default_pose_model() -> str:
    """YOLO pose — ưu tiên identity_vm_app/modelAi/."""
    model_dir = Path(__file__).resolve().parent / "modelAi"
    for name in ("yolo26m-pose.pt", "yolo26n-pose.pt"):
        p = model_dir / name
        if p.is_file():
            return str(p.resolve())
    return str((model_dir / "yolo26m-pose.pt").resolve())


# Data layout (tách hẳn face_db / sqlite / archive khỏi instance cũ)
IVM_DATA_DIR = Path(os.getenv("IVM_DATA_DIR", str(REPO_ROOT / "identity_vm_data"))).resolve()
IVM_FACE_DB_DIR = Path(os.getenv("IVM_FACE_DB_DIR", str(IVM_DATA_DIR / "face_db")))
IVM_SQLITE_PATH = Path(os.getenv("IVM_SQLITE_PATH", str(IVM_DATA_DIR / "identity_vm.sqlite3")))
IVM_ARCHIVE_ROOT = Path(os.getenv("IVM_ARCHIVE_ROOT", str(IVM_DATA_DIR / "archive")))

# API
IVM_API_HOST = os.getenv("IVM_API_HOST", "0.0.0.0")
IVM_API_PORT = _int("IVM_API_PORT", 8010)

# InsightFace pack ( buffalo_l | antelopev2 | ... — phụ thuộc model zoo đã tải)
IVM_INSIGHTFACE_MODEL_NAME = os.getenv("IVM_INSIGHTFACE_MODEL_NAME", "buffalo_l")
IVM_INSIGHTFACE_ROOT = os.getenv("IVM_INSIGHTFACE_ROOT", "").strip() or os.path.join(
    os.path.expanduser("~"), ".insightface", "models"
)
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
# POST /ivm/identify_images — giới hạn số ảnh mỗi request (0 = không giới hạn; >0 để tránh RAM/timeout).
IVM_IDENTIFY_BATCH_MAX_FILES = max(0, _int("IVM_IDENTIFY_BATCH_MAX_FILES", 0))
# Decode+infer song song (mỗi worker một InsightFaceEngine); 1 = tuần tự như cũ.
IVM_IDENTIFY_BATCH_INFER_WORKERS = max(1, min(16, _int("IVM_IDENTIFY_BATCH_INFER_WORKERS", 1)))
# Trần cho query/UI `infer_workers` (client không vượt số này).
IVM_IDENTIFY_BATCH_API_MAX_INFER_WORKERS = max(
    1, min(16, _int("IVM_IDENTIFY_BATCH_API_MAX_INFER_WORKERS", 16))
)
# Số crop tối đa mỗi lần ONNX recognition `get_feat` (analyze_bgr + gom cross-image identify_images).
IVM_REC_GET_FEAT_MAX_BATCH = max(1, min(256, _int("IVM_REC_GET_FEAT_MAX_BATCH", 64)))
# POST /identify_images với 1 engine: detect từng ảnh rồi gom toàn bộ crop → get_feat theo lô (0 = tắt, như cũ từng ảnh).
IVM_IDENTIFY_IMAGES_MERGE_REC = _truthy("IVM_IDENTIFY_IMAGES_MERGE_REC", "1")
# Xử lý N ảnh/lượt trong merge (giảm peak VRAM khi upload batch). 0 = gom toàn bộ một lượt (nhanh hơn nếu đủ VRAM).
IVM_IDENTIFY_IMAGES_PROCESS_CHUNK = max(0, _int("IVM_IDENTIFY_IMAGES_PROCESS_CHUNK", 8))
IVM_USE_FAISS = _truthy("IVM_USE_FAISS", "0")
IVM_LOG_UNKNOWN_EVENTS = _truthy("IVM_LOG_UNKNOWN_EVENTS", "1")
# Gộp các ingest liên tiếp cùng người trên cùng camera (giây); 0 = tắt
IVM_EVENT_DEBOUNCE_S = _float("IVM_EVENT_DEBOUNCE_S", 3.0)

# Recorder
IVM_FFMPEG_BIN = os.getenv("IVM_FFMPEG_BIN", "ffmpeg")
IVM_SEGMENT_SECONDS = _int("IVM_SEGMENT_SECONDS", 3600)

# Export cuts
IVM_EXPORT_CACHE_DIR = Path(os.getenv("IVM_EXPORT_CACHE_DIR", str(IVM_DATA_DIR / "export_cache")))
IVM_EXPORT_CUT_MIN_DURATION_S = max(0.5, min(60.0, _float("IVM_EXPORT_CUT_MIN_DURATION_S", 3.0)))

# Phân tích video offline (upload → job → báo cáo tổng hợp; không ghi SQLite mặc định)
IVM_VIDEO_ANALYZE_DIR = Path(
    os.getenv("IVM_VIDEO_ANALYZE_DIR", str(IVM_DATA_DIR / "video_analyze"))
).resolve()
# 0 = không giới hạn dung lượng / thời lượng video upload phân tích.
os.environ.setdefault("IVM_VIDEO_ANALYZE_MAX_MB", "0")
os.environ.setdefault("IVM_VIDEO_ANALYZE_MAX_DURATION_S", "0")
IVM_VIDEO_ANALYZE_MAX_MB = max(0, _int("IVM_VIDEO_ANALYZE_MAX_MB", 0))
IVM_VIDEO_ANALYZE_MAX_DURATION_S = max(0.0, _float("IVM_VIDEO_ANALYZE_MAX_DURATION_S", 0.0))
# POST /video-analyze/jobs/from-path — rỗng = mọi đường dẫn file tồn tại trên máy API.
IVM_VIDEO_ANALYZE_LOCAL_PATH_ROOTS: tuple[Path, ...] = tuple(
    Path(p.strip()).resolve()
    for p in os.getenv("IVM_VIDEO_ANALYZE_LOCAL_PATH_ROOTS", "").split(",")
    if p.strip()
)
# 0 = full frame; hoặc 5 / 10 / 15 (xem video_analyze_fps.ALLOWED_SAMPLE_FPS).
IVM_VIDEO_ANALYZE_DEFAULT_SAMPLE_FPS = _float("IVM_VIDEO_ANALYZE_DEFAULT_SAMPLE_FPS", 0.0)
# Số job video chạy song song (mỗi job còn spawn IVM_VIDEO_ANALYZE_SPLIT_PARTS thread + engine).
IVM_VIDEO_ANALYZE_MAX_CONCURRENT = max(1, min(8, _int("IVM_VIDEO_ANALYZE_MAX_CONCURRENT", 2)))
# Chia 1 video thành N đoạn ffmpeg → N luồng analyze song song (giống TOTAL_THREADS_ANALYZE VideoMaster).
IVM_VIDEO_ANALYZE_SPLIT_PARTS = max(1, min(8, _int("IVM_VIDEO_ANALYZE_SPLIT_PARTS", 4)))
# Thử h264_nvenc trước khi cắt (0 = chỉ libx264).
IVM_VIDEO_ANALYZE_SPLIT_GPU_ENCODE = _truthy("IVM_VIDEO_ANALYZE_SPLIT_GPU_ENCODE", "0")
IVM_VIDEO_ANALYZE_TIMELINE_MAX = max(20, min(500, _int("IVM_VIDEO_ANALYZE_TIMELINE_MAX", 200)))
# VideoMaster: chỉ lưu full frame (img_url); crop lấy on-demand qua API. 1 = ghi thêm face_imgs_*.jpg (chậm).
IVM_VIDEO_ANALYZE_SAVE_CROPS = _truthy("IVM_VIDEO_ANALYZE_SAVE_CROPS", "0")
# YOLO (ultralytics) + ByteTrack — luồng chính phân tích video (bắt buộc khi bật).
IVM_VIDEO_ANALYZE_USE_YOLO_TRACKING = _truthy("IVM_VIDEO_ANALYZE_USE_YOLO_TRACKING", "1")
IVM_VIDEO_ANALYZE_YOLO_MODEL = os.getenv(
    "IVM_VIDEO_ANALYZE_YOLO_MODEL", _default_video_yolo_model()
).strip()
IVM_VIDEO_ANALYZE_TRACKER = os.getenv("IVM_VIDEO_ANALYZE_TRACKER", "bytetrack.yaml").strip()
IVM_VIDEO_ANALYZE_YOLO_CONF = max(0.05, min(0.99, _float("IVM_VIDEO_ANALYZE_YOLO_CONF", 0.5)))
IVM_VIDEO_ANALYZE_YOLO_IMGSZ = max(320, min(1280, _int("IVM_VIDEO_ANALYZE_YOLO_IMGSZ", 640)))
# Ghép mặt ↔ box người: IoU tối thiểu; tâm mặt nằm trong box người → cho phép nhiều mặt / một track.
IVM_FACE_ASSIGN_IOU_MIN = max(0.0, min(0.95, _float("IVM_FACE_ASSIGN_IOU_MIN", 0.05)))
IVM_FACE_ASSIGN_CENTER_IN_PERSON = _truthy("IVM_FACE_ASSIGN_CENTER_IN_PERSON", "1")
# Hướng B: chỉ chạy YOLO-pose khi một box detect có >= N mặt (tiết kiệm GPU).
IVM_FACE_POSE_REFINE = _truthy("IVM_FACE_POSE_REFINE", "1")
IVM_FACE_POSE_MODEL = os.getenv("IVM_FACE_POSE_MODEL", _default_pose_model()).strip()
IVM_FACE_POSE_CONF = max(0.05, min(0.99, _float("IVM_FACE_POSE_CONF", 0.40)))
IVM_FACE_POSE_IMGSZ = max(320, min(1280, _int("IVM_FACE_POSE_IMGSZ", 640)))
IVM_FACE_POSE_KP_CONF = max(0.1, min(0.99, _float("IVM_FACE_POSE_KP_CONF", 0.40)))
IVM_FACE_POSE_MIN_FACES = max(2, min(8, _int("IVM_FACE_POSE_MIN_FACES", 2)))
IVM_FACE_POSE_DET_IOU_MIN = max(0.01, min(0.5, _float("IVM_FACE_POSE_DET_IOU_MIN", 0.05)))
# Top tên nghi ngờ gắn mỗi id_tracking (video merge + camera overlay).
IVM_TRACK_SUSPECT_NAMES_LIMIT = max(1, min(10, _int("IVM_TRACK_SUSPECT_NAMES_LIMIT", 5)))
# Báo cáo faces-person: bỏ track có ít hơn N khung mẫu (0 = không lọc).
IVM_REPORT_MIN_TRACK_FRAMES = max(0, _int("IVM_REPORT_MIN_TRACK_FRAMES", 5))


def ivm_report_min_track_frames() -> int:
    """Ngưỡng hiển thị track trên UI/API báo cáo — tối thiểu 10 khung (cùng quy tắc vũ khí)."""
    base = int(IVM_REPORT_MIN_TRACK_FRAMES)
    if base <= 0:
        return 0
    return max(10, base)
# Lưu features_face (np.array_str) — bắt buộc cho báo cáo / tìm mặt giống VideoMaster.
IVM_VIDEO_ANALYZE_PERSIST_EMBEDDINGS = _truthy("IVM_VIDEO_ANALYZE_PERSIST_EMBEDDINGS", "1")
# Camera: không ghi event_crops/*.jpg (crop qua archive + bbox trong extra_json khi cần).
IVM_EVENT_SAVE_CROPS = _truthy("IVM_EVENT_SAVE_CROPS", "0")
# Gom ONNX recognition N crop trước khi search DB (một segment).
IVM_VIDEO_ANALYZE_EMBED_BATCH = max(1, min(256, _int("IVM_VIDEO_ANALYZE_EMBED_BATCH", 32)))
# Ghi SQLite theo lô thay vì từng dòng.
IVM_VIDEO_ANALYZE_DB_BATCH = max(1, min(512, _int("IVM_VIDEO_ANALYZE_DB_BATCH", 48)))
# Cập nhật progress mỗi N khung mẫu (giảm lock SQLite).
IVM_VIDEO_ANALYZE_PROGRESS_EVERY = max(1, _int("IVM_VIDEO_ANALYZE_PROGRESS_EVERY", 5))
# JPEG analyze (85=nhanh hon, 90=mac dinh cu).
IVM_VIDEO_ANALYZE_JPEG_QUALITY = max(50, min(95, _int("IVM_VIDEO_ANALYZE_JPEG_QUALITY", 82)))
# Đoạn xuất hiện — slideshow UI (ms/khung, giống VideoMaster FE FRAME_INTERVAL=200).
IVM_TRACK_SEGMENT_SLIDESHOW_MS = max(80, min(800, _int("IVM_TRACK_SEGMENT_SLIDESHOW_MS", 200)))
# FPS ghép MP4 từ khung mẫu (ưu tiên fps video gốc trong job, kẹp min–max).
IVM_TRACK_SEGMENT_ENCODE_FPS_MIN = max(5.0, min(60.0, _float("IVM_TRACK_SEGMENT_ENCODE_FPS_MIN", 15.0)))
IVM_TRACK_SEGMENT_ENCODE_FPS_MAX = max(10.0, min(60.0, _float("IVM_TRACK_SEGMENT_ENCODE_FPS_MAX", 30.0)))
IVM_VIDEO_ANALYZE_ALLOWED_SUFFIXES = tuple(
    x.strip().lower()
    for x in os.getenv("IVM_VIDEO_ANALYZE_ALLOWED_SUFFIXES", ".mp4,.avi,.mov,.mkv,.webm").split(",")
    if x.strip()
)
# (Cũ) HTTP fallback từng khung — phân tích video không dùng; giữ biến cho tương thích cấu hình cũ.
IVM_VIDEO_ANALYZE_HTTP_BASE = os.getenv(
    "IVM_VIDEO_ANALYZE_HTTP_BASE", f"http://127.0.0.1:{IVM_API_PORT}"
).rstrip("/")

# Tiến trình ĐK thư mục lớn (poll từ UI / GET admin)
IVM_REGISTER_FOLDER_PROGRESS_PATH = Path(
    os.getenv(
        "IVM_REGISTER_FOLDER_PROGRESS_PATH",
        str(IVM_DATA_DIR / "register_folder_progress.json"),
    )
).resolve()
# Log text tiến trình bulk (mốc job + throughput từng worker) — xem bằng terminal server hoặc `type`/`tail` file này.
IVM_REGISTER_FOLDER_BULK_LOG = Path(
    os.getenv(
        "IVM_REGISTER_FOLDER_BULK_LOG",
        str(IVM_DATA_DIR / "register_folder_bulk.log"),
    )
).resolve()

# Camera list (mặc định đọc camera_config.json ở root repo)
IVM_CAMERA_CONFIG = os.getenv("IVM_CAMERA_CONFIG", str(REPO_ROOT / "camera_config.json"))

# OpenCV/FFmpeg khi mở RTSP: OpenCV 4 dùng chuỗi "key;val|key2;val2".
# Giúp giảm trễ và lỗi giải HEVC (Could not find ref with POC / Duplicate POC) khi mạng kém.
# Override toàn bộ: biến môi trường IVM_FFMPEG_CAPTURE_OPTIONS hoặc OPENCV_FFMPEG_CAPTURE_OPTIONS.
_IVM_FFMPEG_DEFAULT = (
    "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000|reorder_queue_size;0|stimeout;8000000"
)
IVM_FFMPEG_CAPTURE_OPTIONS = os.getenv("IVM_FFMPEG_CAPTURE_OPTIONS", _IVM_FFMPEG_DEFAULT).strip()
# Hàng đợi khung OpenCV: 1 = trễ thấp; 2 thường ổn định hơn với H.265/RTSP lỗi packet (đổi IVM_CAP_PROP_BUFFERSIZE=1 nếu cần tối thiểu độ trễ)
IVM_CAP_PROP_BUFFERSIZE = max(1, min(16, _int("IVM_CAP_PROP_BUFFERSIZE", 2)))

# Xem trước MJPEG (thread riêng trong API — không dùng worker nhận diện, không chặn Streamlit)
IVM_PREVIEW_JPEG_QUALITY = max(40, min(95, _int("IVM_PREVIEW_JPEG_QUALITY", 78)))
IVM_PREVIEW_CAPTURE_FPS = max(5.0, min(60.0, _float("IVM_PREVIEW_CAPTURE_FPS", 20.0)))
IVM_PREVIEW_MJPEG_OUT_FPS = max(5.0, min(60.0, _float("IVM_PREVIEW_MJPEG_OUT_FPS", 22.0)))
# Khi đọc frame lỗi / RTSP rớt: đóng capture và mở lại (backoff giữa các lần thử mở)
IVM_PREVIEW_RECONNECT_DELAY_S = max(0.2, min(120.0, _float("IVM_PREVIEW_RECONNECT_DELAY_S", 2.5)))
IVM_PREVIEW_READ_FAILS_BEFORE_RECONNECT = max(1, min(60, _int("IVM_PREVIEW_READ_FAILS_BEFORE_RECONNECT", 4)))
IVM_PREVIEW_OPEN_BACKOFF_CAP_S = max(5.0, min(300.0, _float("IVM_PREVIEW_OPEN_BACKOFF_CAP_S", 60.0)))
# Preview RTSP/HTTP: giải bằng FFmpeg → raw BGR (tránh HEVC trong OpenCV). Tắt: IVM_PREVIEW_DECODE_VIA_FFMPEG=0
IVM_PREVIEW_DECODE_VIA_FFMPEG = _truthy("IVM_PREVIEW_DECODE_VIA_FFMPEG", "1")
IVM_PREVIEW_FFMPEG_MAX_HEIGHT = max(240, min(2160, _int("IVM_PREVIEW_FFMPEG_MAX_HEIGHT", 720)))
# Lưới nhiều camera: warm reader lần lượt (giây giữa mỗi camera) + snapshot JPEG (không giữ 50 kết nối MJPEG)
IVM_PREVIEW_WARM_STAGGER_S = max(0.05, min(5.0, _float("IVM_PREVIEW_WARM_STAGGER_S", 0.35)))
IVM_PREVIEW_SNAPSHOT_MAX_WAIT_S = max(0.5, min(30.0, _float("IVM_PREVIEW_SNAPSHOT_MAX_WAIT_S", 6.0)))
# Lưới UI: FPS polling mỗi ô (chỉ lấy JPEG đã có sẵn — wait_s=0 sau khung đầu)
IVM_PREVIEW_GRID_POLL_FPS = max(2.0, min(15.0, _float("IVM_PREVIEW_GRID_POLL_FPS", 8.0)))
IVM_PREVIEW_GRID_INITIAL_WAIT_S = max(0.0, min(10.0, _float("IVM_PREVIEW_GRID_INITIAL_WAIT_S", 2.0)))

# Tự khởi động worker đọc RTSP cho mọi camera khi API startup (tốn CPU — 10 camera ≈ 10 luồng decode).
# Mặc định tắt; bật khi deploy: IVM_AUTO_START_CAMERA_WORKERS=1. Hoặc IVM_NO_CAMERA_WORKERS=1 để chặn.
IVM_AUTO_START_CAMERA_WORKERS = _truthy("IVM_AUTO_START_CAMERA_WORKERS", "0")

# Hàng đợi frame từ RTSP cho worker nhận diện (0 = chỉ khung mới nhất, infer chậm có thể bỏ qua số đếm).
# >0 = tối đa N bản copy chờ xử lý; khi đầy thread đọc chặn (backpressure) — RAM ~ N × kích thước ảnh.
IVM_STREAM_FRAME_QUEUE_MAX = max(0, min(64, _int("IVM_STREAM_FRAME_QUEUE_MAX", 8)))
# Khởi động worker pipeline lần lượt (giây) — tránh 10 camera mở RTSP cùng lúc
IVM_PIPELINE_START_STAGGER_S = max(0.0, min(10.0, _float("IVM_PIPELINE_START_STAGGER_S", 0.4)))

# Thư mục phiên nhận diện camera live (job DB + session.mp4 + crop)
IVM_CAMERA_SESSION_DIR = Path(
    os.getenv("IVM_CAMERA_SESSION_DIR", str(IVM_DATA_DIR / "camera_sessions"))
)
# Ghi full luồng camera khi BẬT nhận diện (ffmpeg RTSP song song hoặc VideoWriter từ reader).
IVM_CAMERA_SESSION_STREAM_RECORD = _truthy("IVM_CAMERA_SESSION_STREAM_RECORD", "1")
# Ghi overlay bbox vào session_overlay.mp4 (tách khỏi video gốc; visualize sau giống upload).
IVM_CAMERA_SESSION_OVERLAY_LIVE = _truthy("IVM_CAMERA_SESSION_OVERLAY_LIVE", "0")
# =1: worker vẫn ghi recognition_events (song song tạm); mặc định tắt — chỉ video_person_reports
IVM_CAMERA_LEGACY_EVENTS = _truthy("IVM_CAMERA_LEGACY_EVENTS", "0")
# =1: mỗi frame infer → video_person_reports (kể cả không người / người không mặt); id_tracking=0 = mốc frame
IVM_REPORT_ALL_INFER_FRAMES = _truthy("IVM_REPORT_ALL_INFER_FRAMES", "1")

# Khi BẬT nhận diện: tự ghi archive RTSP (ffmpeg segment) + video mp4 có bbox
IVM_ANALYZE_AUTO_ARCHIVE = _truthy("IVM_ANALYZE_AUTO_ARCHIVE", "1")
IVM_ANALYZE_RECORD_VISUAL = _truthy("IVM_ANALYZE_RECORD_VISUAL", "1")
# FPS header file mp4 phân tích. 0 = theo fps ước lượng từ camera (5–60); >0 = cố định (tối đa 60).
IVM_ANALYZE_VISUAL_FPS = max(0.0, min(60.0, _float("IVM_ANALYZE_VISUAL_FPS", 0.0)))

# Nhận diện realtime theo camera
# IVM_ANALYZE_EVEN_FRAMES_ONLY=1 (mặc định): chỉ infer khi frame_count chẵn (2, 4, 6, …).
# Preview + video phân tích vẫn cập nhật mỗi khung mới (bbox/HUD từ infer gần nhất trên frame lẻ).
# Tắt (=0): giới hạn tốc độ infer bằng IVM_ANALYZE_TARGET_FPS / IVM_ANALYZE_INTERVAL_S; giữa hai lần infer vẫn vẽ khung mới.
IVM_ANALYZE_EVEN_FRAMES_ONLY = _truthy("IVM_ANALYZE_EVEN_FRAMES_ONLY", "1")
IVM_ANALYZE_TARGET_FPS = max(0.5, min(30.0, _float("IVM_ANALYZE_TARGET_FPS", 5.0)))
# FPS mẫu mặc định khi bật analyze camera live (VisionMaster: 5)
IVM_CAMERA_DEFAULT_SAMPLE_FPS = max(0.0, _float("IVM_CAMERA_DEFAULT_SAMPLE_FPS", 5.0))
IVM_ANALYZE_INTERVAL_S = _float("IVM_ANALYZE_INTERVAL_S", 0.0)
# 0 = infer trên full độ rộng ảnh (chậm hơn); >0 = thu nhỏ trước khi ONNX (bbox vẫn scale về full cho overlay/video).
IVM_ANALYZE_MAX_FRAME_WIDTH = max(0, _int("IVM_ANALYZE_MAX_FRAME_WIDTH", 960))
IVM_ANALYZE_JPEG_QUALITY = max(40, min(100, _int("IVM_ANALYZE_JPEG_QUALITY", 85)))
IVM_ANALYZE_IDENTIFY_TIMEOUT_S = max(5.0, min(300.0, _float("IVM_ANALYZE_IDENTIFY_TIMEOUT_S", 60.0)))
IVM_ANALYZE_INGEST_TIMEOUT_S = max(2.0, min(120.0, _float("IVM_ANALYZE_INGEST_TIMEOUT_S", 15.0)))
# In-process infer (cùng process với API — python main.py); tắt nếu worker chạy process riêng
IVM_USE_IN_PROCESS_INFER = _truthy("IVM_USE_IN_PROCESS_INFER", "1")
# Camera + video: YOLO person+ByteTrack trước, nhánh InsightFace detect/recognition (giống VideoMaster).
IVM_USE_PERSON_FIRST_PIPELINE = _truthy("IVM_USE_PERSON_FIRST_PIPELINE", "1")
IVM_INFER_DISPLAY_JPEG_QUALITY = max(50, min(95, _int("IVM_INFER_DISPLAY_JPEG_QUALITY", 80)))
IVM_INFER_MJPEG_POLL_S = max(0.02, min(0.5, _float("IVM_INFER_MJPEG_POLL_S", 0.08)))

# Phát hiện vũ khí (YOLO weapon_detect.pt: gun, knife) — cùng chu kỳ nhận diện khi BẬT analyze
IVM_WEAPON_ENABLED = _truthy("IVM_WEAPON_ENABLED", "1")
IVM_WEAPON_MODEL = os.getenv(
    "IVM_WEAPON_MODEL",
    str(Path(__file__).resolve().parent / "modelAi" / "weapon_detect.pt"),
)
IVM_WEAPON_IMGSZ = max(320, min(1280, _int("IVM_WEAPON_IMGSZ", 640)))
IVM_WEAPON_DEVICE = os.getenv("IVM_WEAPON_DEVICE", "0").strip() or "0"
# Padding ROI người trước khi chạy weapon_detect (chỉ sau khi đã có box track)
IVM_WEAPON_ROI_PAD_RATIO = max(0.0, min(0.5, _float("IVM_WEAPON_ROI_PAD_RATIO", 0.12)))
# Ghép bbox vũ khí (toàn khung) → box người đã track
IVM_WEAPON_MATCH_IOU_MIN = max(0.0, min(0.5, _float("IVM_WEAPON_MATCH_IOU_MIN", 0.01)))
# Alias cũ (pose stack) — không dùng khi chỉ chạy weapon_detect.pt
IVM_WEAPON_POSE_MODEL = os.getenv("IVM_WEAPON_POSE_MODEL", _default_pose_model()).strip()
IVM_WEAPON_WARMUP = _truthy("IVM_WEAPON_WARMUP", "1")
IVM_WEAPON_BATCH_SIZE = max(1, min(32, _int("IVM_WEAPON_BATCH_SIZE", 8)))
IVM_WEAPON_MEMORY_FRAMES = max(0, _int("IVM_WEAPON_MEMORY_FRAMES", 10))
IVM_WEAPON_INPUT_CONF = max(0.05, min(0.99, _float("IVM_WEAPON_INPUT_CONF", 0.50)))
IVM_WEAPON_VOTER_WINDOW = max(1, min(30, _int("IVM_WEAPON_VOTER_WINDOW", 5)))
IVM_WEAPON_VOTER_THRESHOLD = max(0.05, min(0.99, _float("IVM_WEAPON_VOTER_THRESHOLD", 0.40)))
# Track / phiên: > N frame phát hiện vũ khí → coi là có vũ khí; > M frame → nguy hiểm
IVM_WEAPON_ARMED_MIN_FRAMES = max(1, _int("IVM_WEAPON_ARMED_MIN_FRAMES", 10))
IVM_WEAPON_DANGEROUS_MIN_FRAMES = max(1, _int("IVM_WEAPON_DANGEROUS_MIN_FRAMES", 20))
# Cảnh báo live (camera BẬT nhận diện): track > N frame có det vũ khí → log/UI (mặc định 5)
IVM_WEAPON_ALERT_ENABLED = _truthy("IVM_WEAPON_ALERT_ENABLED", "1")
IVM_WEAPON_ALERT_MIN_FRAMES = max(1, _int("IVM_WEAPON_ALERT_MIN_FRAMES", 5))
# Lịch sử ảnh cảnh báo / camera trên UI (thumb + phóng to)
IVM_WEAPON_ALERT_HISTORY_PER_CAMERA = max(1, min(10, _int("IVM_WEAPON_ALERT_HISTORY_PER_CAMERA", 3)))
# Thumb ~480px; 0 = không resize ảnh phóng to (giữ độ phân giải khung gốc)
IVM_WEAPON_ALERT_THUMB_MAX_WIDTH = max(120, min(960, _int("IVM_WEAPON_ALERT_THUMB_MAX_WIDTH", 480)))
IVM_WEAPON_ALERT_FULL_MAX_WIDTH = max(0, min(3840, _int("IVM_WEAPON_ALERT_FULL_MAX_WIDTH", 0)))
IVM_WEAPON_ALERT_SNAPSHOT_JPEG_QUALITY = max(60, min(98, _int("IVM_WEAPON_ALERT_SNAPSHOT_JPEG_QUALITY", 90)))
# Chỉ chạy weapon mỗi N chu kỳ infer (1 = mỗi lần infer)
# Live camera (person-first): luôn chạy weapon mỗi khung sample — giống video offline. Chỉ legacy non-live có thể nhảy chu kỳ.
IVM_WEAPON_EVERY_N_CYCLES = max(1, _int("IVM_WEAPON_EVERY_N_CYCLES", 1))
# Mặc định: load khi infer, giải phóng GPU/RAM ngay sau mỗi lần dùng.
# =0: giữ model vũ khí trong RAM khi nhận diện BẬT (chỉ release khi TẮT analyze).
IVM_WEAPON_RELEASE_AFTER_USE = _truthy("IVM_WEAPON_RELEASE_AFTER_USE", "1")
# Dọn VRAM khi tiến trình xong (request API, tắt nhận diện camera, job bulk) — không sau mỗi frame/ảnh
IVM_GPU_SOFT_CLEANUP_ON_PROCESS_DONE = _truthy(
    "IVM_GPU_SOFT_CLEANUP_ON_PROCESS_DONE",
    os.environ.get("IVM_GPU_SOFT_CLEANUP_AFTER_INFER", "1"),
)
IVM_GPU_SOFT_CLEANUP_AFTER_INFER = IVM_GPU_SOFT_CLEANUP_ON_PROCESS_DONE  # alias cũ
# =1: sau POST identify/register ảnh — unload InsightFace toàn cục (VRAM ~ lúc boot trước khi nạp model).
# Request kế tiếp tự load lại qua get_engine(). Boot cũng hoãn nạp model khi bật cờ này.
IVM_GPU_RELEASE_ENGINE_AFTER_IMAGE_INFER = _truthy("IVM_GPU_RELEASE_ENGINE_AFTER_IMAGE_INFER", "1")

# Camera — VisionMaster: mỗi camera một thread + một bộ model riêng (không skip frame sample)
# Worker nhận diện: RTSP (opencv_reader) chỉ mở khi BẬT analyze. Hiển thị lưới/MJPEG: /ivm/preview (hub riêng).
IVM_CAMERA_DEDICATED_MODELS = _truthy("IVM_CAMERA_DEDICATED_MODELS", "1")
# YOLO person riêng từng camera (1 = shared weights — không khuyến nghị multi-camera)
IVM_SHARED_YOLO_PERSON = _truthy("IVM_SHARED_YOLO_PERSON", "0")
# Hàng đợi infer riêng từng camera — giữ mọi khung sample (0 = không giới hạn)
IVM_CAMERA_INFER_QUEUE_MAX = max(0, _int("IVM_CAMERA_INFER_QUEUE_MAX", 0))
IVM_CAMERA_INFER_QUEUE_WARN_DEPTH = max(1, _int("IVM_CAMERA_INFER_QUEUE_WARN_DEPTH", 30))
IVM_CAMERA_INFER_QUEUE_DRAIN_TIMEOUT_S = max(5.0, min(3600.0, _float("IVM_CAMERA_INFER_QUEUE_DRAIN_TIMEOUT_S", 300.0)))
# =1 khi TẮT nhận diện: infer hết queue (chậm). =0 (mặc định): xóa queue, chỉ chờ khung đang infer (nếu có) rồi unload model.
IVM_CAMERA_DRAIN_INFER_ON_STOP = _truthy("IVM_CAMERA_DRAIN_INFER_ON_STOP", "1")
# Log TB infer ms mỗi N khung đã xử lý ra terminal (0 = tắt)
IVM_INFER_STATS_EVERY_N = max(0, _int("IVM_INFER_STATS_EVERY_N", 20))
# Lô đầu sau khi BẬT analyze: log sớm hơn để thấy bottleneck ngay (≤ EVERY_N)
IVM_INFER_STATS_FIRST_N = max(1, min(100, _int("IVM_INFER_STATS_FIRST_N", 5)))
# Mỗi N giây in 1 dòng tổng hợp mọi camera đang analyze (queue, infer) — dễ thấy cam0 khi cam8 spam log. 0 = tắt.
IVM_CAMERA_HUB_STATUS_LOG_INTERVAL_S = max(0.0, _float("IVM_CAMERA_HUB_STATUS_LOG_INTERVAL_S", 20.0))
# Khi queue >= ngưỡng: chỉ log infer_ok mỗi N khung (giảm tràn terminal). 0 = luôn log.
IVM_CAMERA_INFER_OK_LOG_QUEUE_THROTTLE = max(0, _int("IVM_CAMERA_INFER_OK_LOG_QUEUE_THROTTLE", 40))
IVM_CAMERA_INFER_OK_LOG_EVERY_N_WHEN_BUSY = max(1, _int("IVM_CAMERA_INFER_OK_LOG_EVERY_N_WHEN_BUSY", 30))

# Camera pipeline (worker RTSP → API) — số frame lỗi liên tiếp trước khi đóng và mở lại RTSP
IVM_PIPELINE_READ_FAILS_BEFORE_RECONNECT = max(1, min(30, _int("IVM_PIPELINE_READ_FAILS_BEFORE_RECONNECT", 4)))
# Không có frame decode > X giây → reconnect ngay (OpenCV RTSP chạy lâu hay timeout ~30s). 0 = tắt.
IVM_RTSP_STALE_RECONNECT_S = max(0.0, _float("IVM_RTSP_STALE_RECONNECT_S", 20.0))
# Reconnect chủ động sau Y giây một phiên RTSP (0 = tắt; ví dụ 1800 = 30 phút).
IVM_RTSP_PROACTIVE_RECONNECT_S = max(0.0, _float("IVM_RTSP_PROACTIVE_RECONNECT_S", 0.0))
IVM_RTSP_RECONNECT_DELAY_S = max(0.5, min(30.0, _float("IVM_RTSP_RECONNECT_DELAY_S", 2.5)))
# Nhận diện camera: RTSP/HTTP đọc bằng FFmpeg subprocess (0 = OpenCV StableCameraReader, mặc định để test/so sánh).
IVM_CAMERA_READ_VIA_FFMPEG = _truthy("IVM_CAMERA_READ_VIA_FFMPEG", "0")
# 0 = full độ phân giải nguồn; >0 = scale max chiều cao (giảm CPU)
IVM_CAMERA_FFMPEG_READ_MAX_HEIGHT = max(0, _int("IVM_CAMERA_FFMPEG_READ_MAX_HEIGHT", 0))
# Nếu đặt biến môi trường, reset-all yêu cầu khớp token (header X-IVM-Reset-Token hoặc body)
IVM_RESET_SECRET = os.getenv("IVM_RESET_SECRET", "").strip()

# Đăng ký hàng loạt từ thư mục (CLI / admin API) — phù hợp thư mục lớn (~10^6 ảnh)
# Số ảnh gom trước mỗi lần gọi FaceDatabase.add_faces_batch (giảm số lần save/index)
IVM_BULK_DB_WRITE_BATCH = max(1, _int("IVM_BULK_DB_WRITE_BATCH", 64))
# Khi infer nhiều luồng: mỗi luồng gom tối thiểu N mặt thành công rồi mới ghi DB (max với db_batch_size; giảm tranh lock / lỗi ghi).
IVM_BULK_MULTI_THREAD_DB_FLUSH = max(1, _int("IVM_BULK_MULTI_THREAD_DB_FLUSH", 1000))
# Danh sách thư mục được phép làm root khi gọi API (đường dẫn tuyệt đối, phân tách bằng dấu phẩy).
# Rỗng = không kiểm tra (chỉ nên dùng trong mạng tin cậy).
IVM_BULK_ALLOWED_ROOTS: list[str] = [
    p.strip()
    for p in os.getenv("IVM_BULK_ALLOWED_ROOTS", "").split(",")
    if p.strip()
]
# Giới hạn số file mỗi request POST register-folder (0 = không giới hạn). Mặc định 0 vì đây là API admin;
# trên môi trường không tin cậy hãy đặt ví dụ IVM_BULK_API_MAX_FILES=50000.
IVM_BULK_API_MAX_FILES = _int("IVM_BULK_API_MAX_FILES", 0)
# Khi resume, có bỏ qua các path đã đánh dấu lỗi trước đó trong checkpoint không
IVM_BULK_RESUME_SKIP_FAILED = _truthy("IVM_BULK_RESUME_SKIP_FAILED", "1")
# Khi ghi registration_failures, chỉ lưu bản sao ảnh nếu file không quá lớn (0 = không lưu ảnh, chỉ log path)
IVM_BULK_FAILURE_SAMPLE_MAX_BYTES = _int("IVM_BULK_FAILURE_SAMPLE_MAX_BYTES", 5_000_000)
# Độ sâu hàng đợi prefetch decode — tăng nếu SSD nhanh / ảnh nhỏ
IVM_BULK_PREFETCH = max(1, _int("IVM_BULK_PREFETCH", 4))
# Số worker infer song song (mỗi worker một FaceAnalysis + VRAM riêng).
# Mặc định 1: ONNX/CUDA thường không ổn khi nhiều session cùng GPU — thư mục rất lớn dễ treo/lỗi.
# Tăng IVM_BULK_INFER_WORKERS=2..4 nếu máy ổn định và cần tốc độ.
IVM_BULK_INFER_WORKERS = max(1, min(16, _int("IVM_BULK_INFER_WORKERS", 1)))
# Giới trần cho request API/UI `/admin/register-folder` (bulk infer_worker gửi từ client không vượt số này).
IVM_BULK_API_MAX_INFER_WORKERS = max(1, min(16, _int("IVM_BULK_API_MAX_INFER_WORKERS", 16)))
# Nếu số file sau quét vượt ngưỡng → không sort lexicographic (sort hàng trăm nghìn path tốn RAM/CPU).
IVM_BULK_MAX_SORT_FILES = max(0, _int("IVM_BULK_MAX_SORT_FILES", 250_000))
# Kích thước lô SELECT checkpoint (giữ ≤ ~900 — giới hạn biến bind SQLite).
IVM_BULK_CHECKPOINT_LOOKUP_CHUNK = max(50, min(900, _int("IVM_BULK_CHECKPOINT_LOOKUP_CHUNK", 900)))


def resolve_identify_infer_workers(requested: int | None) -> int:
    """
    Clamp số worker infer cho POST /identify_images:
    requested None → IVM_IDENTIFY_BATCH_INFER_WORKERS (env).
    Luôn ≤ IVM_IDENTIFY_BATCH_API_MAX_INFER_WORKERS và ≤ 16.
    """
    cap_w = max(1, min(16, int(IVM_IDENTIFY_BATCH_API_MAX_INFER_WORKERS)))
    env_w = max(1, min(cap_w, int(IVM_IDENTIFY_BATCH_INFER_WORKERS)))
    if requested is None:
        return env_w
    return max(1, min(cap_w, int(requested)))


def ivm_bulk_worker_ctx_ids() -> list[int]:
    """ctx_id InsightFace cho từng worker (lặp vòng nếu ít id hơn số worker)."""
    raw = os.getenv("IVM_BULK_WORKER_CTX_IDS", "").strip()
    if not raw:
        return [IVM_CTX_ID]
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            ids.append(int(part))
    return ids if ids else [IVM_CTX_ID]
