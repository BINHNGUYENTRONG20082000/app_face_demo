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

# 4. Đặt model YOLO/vũ khí vào identity_vm_app\modelAi\
#    (xem identity_vm_app\modelAi\README.md)
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

```text
app_face/
├── identity_vm_app/          # Ứng dụng chính
│   ├── api/                  # FastAPI routes
│   ├── camera_recognition/   # Worker RTSP + infer
│   ├── engine/               # InsightFace, YOLO, GPU cleanup
│   ├── preview/              # MJPEG preview hub
│   ├── recorder/             # Archive FFmpeg
│   ├── services/             # Video analyze, export, reports
│   ├── store/                # SQLite
│   ├── modelAi/              # Trọng số YOLO/vũ khí (local, không commit)
│   ├── streamlit_test.py     # UI đầy đủ
│   └── camera_dashboard.py   # UI dashboard
├── packages/
│   ├── persistence/          # FaceDatabase (embeddings)
│   └── camera_stream/        # RTSP reader ổn định
├── services/text.py          # Chuẩn hoá tên tiếng Việt
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

Xem thêm trong `identity_vm_app/settings.py`.

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
python -m pytest tests/ -q
python scripts/smoke_ivm_cameras.py
```

Worker camera riêng (khi API đã chạy):

```bash
python -m camera_pipeline --api http://127.0.0.1:8010
```

## Ghi chú

- File `camera_config.json` chứa thông tin RTSP — **không commit**; dùng `camera_config.example.json` làm mẫu.
- Model `.pt` / `.onnx` lớn — đặt local theo hướng dẫn `identity_vm_app/modelAi/README.md`.
- InsightFace tải model `buffalo_l` lần đầu chạy (~300 MB).
