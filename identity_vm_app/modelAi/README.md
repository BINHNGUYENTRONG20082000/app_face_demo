# Model weights

Đặt các file trọng số YOLO/vũ khí vào thư mục này trước khi chạy app.

| File | Mục đích |
|------|----------|
| `yolo26s.pt` | YOLO person tracking (mặc định) |
| `yolo26m.pt` | YOLO person (tuỳ chọn, nặng hơn) |
| `yolo26m-pose.pt` | YOLO pose refine khi nhiều mặt trong một box |
| `weapon_detect.pt` | Phát hiện vũ khí (gun, knife) |

InsightFace pack `buffalo_l` được tải tự động vào `~/.insightface/models/buffalo_l` khi chạy lần đầu, hoặc chỉ định qua biến môi trường `IVM_INSIGHTFACE_ROOT`.

Các file `.pt` không được commit vào git (xem `.gitignore`).
