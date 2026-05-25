# Identity VM App

Nền tảng nhận diện khuôn mặt đa camera: InsightFace + YOLO tracking + phát hiện vũ khí, FastAPI backend và giao diện Streamlit.

## Tính năng

- Nhận diện / đăng ký khuôn mặt qua API và UI
- Camera RTSP trực tiếp: preview MJPEG, bật/tắt phân tích theo từng camera
- Ghi archive liên tục, cắt clip theo sự kiện
- Phân tích video offline (upload hoặc đường dẫn local)
- Báo cáo, cảnh báo vũ khí, bulk register từ thư mục

## Yêu cầu hệ thống

- Python 3.10+
- NVIDIA GPU + CUDA (khuyến nghị cho ONNX Runtime GPU và YOLO)
- [FFmpeg](https://ffmpeg.org/) có trong `PATH`
- Windows hoặc Linux

## Cài đặt nhanh

```powershell
# 1. Clone repo và vào thư mục dự án
cd app_face

# 2. Tạo virtualenv và cài dependency
.\setup.ps1

# 3. Sao chép cấu hình camera mẫu (chỉnh RTSP/webcam của bạn)
copy camera_config.example.json camera_config.json

# 4. Đặt model YOLO/vũ khí vào module_ai\models\
#    (xem module_ai\models\README.md)
```

Trên Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate   # hoặc: . .venv/bin/activate
pip install -r requirements.txt
cp camera_config.example.json camera_config.json
```

## Chạy ứng dụng

**Terminal 1 — API + worker camera:**

```bash
python main.py
```

Chỉ chạy API (không worker RTSP):

```bash
python main.py --no-camera
```

**Terminal 2 — Giao diện Streamlit (đầy đủ):**

```bash
streamlit run identity_vm_app/streamlit_test.py --server.port 8510
```

Hoặc dashboard gọn hơn:

```bash
streamlit run ui.py --server.port 8510
```

Trên Windows có thể dùng:

```bat
scripts\run_identity_vm_api.bat
scripts\run_identity_vm_ui.bat
```

## Địa chỉ mặc định

| Dịch vụ | URL |
|---------|-----|
| API | http://127.0.0.1:8010 |
| Health | http://127.0.0.1:8010/ivm/health |
| API docs | http://127.0.0.1:8010/docs |
| Streamlit UI | http://127.0.0.1:8510 |

Đổi cổng API: biến môi trường `IVM_API_PORT` hoặc `python main.py --port 8010`.

## Cấu trúc dự án

**Backend AI (bàn giao):** xem [`module_ai/README.md`](module_ai/README.md) — engine, persistence, camera infer, pipelines, API nhận diện.

```text
app_face/
├── module_ai/                # Backend ML/inference + API AI (mới)
│   ├── engine/               # InsightFace, YOLO, GPU cleanup
│   ├── persistence/          # FaceDatabase
│   ├── camera/               # Worker RTSP + infer
│   ├── pipelines/            # Video offline, weapon crops
│   ├── jobs/                 # Bulk register
│   ├── models/               # Trọng số YOLO/vũ khí (local, không commit)
│   └── api/                  # face_routes, people, infer, video-analyze, …
├── identity_vm_app/          # Host: SQLite, recorder, preview, routes_host
│   ├── api/                  # app.py, routes_host, preview routes
│   ├── preview/              # MJPEG preview hub
│   ├── recorder/             # Archive FFmpeg
│   ├── services/             # Export, camera session (không infer core)
│   ├── store/                # SQLite
│   ├── engine/               # Shim → module_ai.engine
│   ├── camera_recognition/   # Shim → module_ai.camera
│   └── streamlit_test.py     # UI đầy đủ
├── packages/
│   ├── persistence/          # Shim → module_ai.persistence
│   └── camera_stream/        # RTSP reader ổn định
├── services/text.py          # Shim → module_ai.utils.text
├── camera_pipeline/          # CLI worker camera độc lập
├── scripts/                  # Script test / launcher .bat
├── tests/                    # Pytest
├── main.py                   # Entry point API + camera workers
├── ui.py                     # Launcher Streamlit dashboard
├── camera_config.json        # Cấu hình camera (local, không commit)
├── camera_config.example.json
├── requirements.txt
└── setup.ps1
```

## Dữ liệu runtime

Mặc định lưu tại `identity_vm_data/` (gitignore):

- `face_db/` — embeddings + metadata
- `identity_vm.sqlite3` — sự kiện, báo cáo
- `archive/` — segment video camera
- `video_analyze/` — job phân tích video

Đổi thư mục: `IVM_DATA_DIR`.

## Biến môi trường thường dùng

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `IVM_API_PORT` | `8010` | Cổng FastAPI |
| `IVM_DATA_DIR` | `./identity_vm_data` | Thư mục dữ liệu |
| `IVM_INSIGHTFACE_MODEL_NAME` | `buffalo_l` | Pack InsightFace |
| `IVM_DISTANCE_THRESHOLD` | `0.7` | Ngưỡng cosine distance |
| `IVM_CAMERA_CONFIG` | `./camera_config.json` | File cấu hình camera |
| `IVM_WEAPON_ENABLED` | `1` | Bật phát hiện vũ khí |
| `IVM_USE_FAISS` | `0` | Dùng FAISS thay sklearn search |

Xem thêm: `identity_vm_app/settings.py` (host), `module_ai/config/settings.py` (AI).

## API chính

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/ivm/health` | Trạng thái API |
| POST | `/ivm/register` | Đăng ký khuôn mặt |
| POST | `/ivm/identify` | Nhận diện ảnh |
| GET | `/ivm/people` | Danh sách người đã đăng ký |
| POST | `/ivm/cameras/{id}/analyze` | Bật/tắt phân tích camera |
| GET | `/ivm/preview/{id}/mjpeg` | Stream preview |
| GET | `/ivm/cameras/{id}/infer/mjpeg` | Stream có overlay nhận diện |

## Kiểm thử

```bash
python -m pytest tests/test_bulk_folder_register_flow.py tests/test_weapon_alerts.py tests/test_identify_infer_workers.py -q
python main.py --no-camera
curl -s http://127.0.0.1:8010/ivm/health
python scripts/smoke_ivm_cameras.py
```

Chi tiết smoke AI: [`module_ai/README.md`](module_ai/README.md#kiểm-thử--test-plan).

Worker camera riêng (khi API đã chạy):

```bash
python -m camera_pipeline --api http://127.0.0.1:8010
```

## Ghi chú

- File `camera_config.json` chứa thông tin RTSP — **không commit**; dùng `camera_config.example.json` làm mẫu.
- Model `.pt` / `.onnx` lớn — đặt local theo `module_ai/models/README.md` (legacy `identity_vm_app/modelAi/` vẫn fallback một release).
- InsightFace `buffalo_l`: đặt vào `module_ai/models/buffalo_l/` (mặc định không tự tải; xem `module_ai/models/README.md`).
