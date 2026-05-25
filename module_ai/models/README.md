# Model weights

Đặt các file trọng số vào thư mục này trước khi chạy app.

## YOLO / vũ khí (file `.pt` ở ngay trong `models/`)

| File | Mục đích |
|------|----------|
| `yolo26s.pt` | YOLO person tracking (mặc định) |
| `yolo26m.pt` | YOLO person (tuỳ chọn, nặng hơn) |
| `yolo26m-pose.pt` | YOLO pose refine khi nhiều mặt trong một box |
| `weapon_detect.pt` | Phát hiện vũ khí (gun, knife) |

## InsightFace `buffalo_l` (thư mục con, không tự tải mặc định)

Đặt pack vào:

```text
module_ai/models/buffalo_l/
├── det_10g.onnx
├── w600k_r50.onnx
└── … (các file .onnx khác của pack, tuỳ chọn)
```

App dùng `IVM_INSIGHTFACE_ROOT` = thư mục `module_ai/` (mặc định), InsightFace đọc `{root}/models/buffalo_l/*.onnx`.

| Biến môi trường | Mặc định | Mô tả |
|----------------|----------|--------|
| `IVM_INSIGHTFACE_ROOT` | `E:\app_face\module_ai` | Thư mục gốc (cha của `models/buffalo_l`) |
| `IVM_INSIGHTFACE_MODEL_NAME` | `buffalo_l` | Tên pack |
| `IVM_INSIGHTFACE_AUTO_DOWNLOAD` | `0` | `1` = cho phép tải từ GitHub nếu thiếu pack |

Các file `.pt` / `.onnx` lớn không được commit vào git (xem `.gitignore`).
