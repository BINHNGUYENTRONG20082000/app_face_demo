#!/usr/bin/env python
"""
Entry backend chính cho Streamlit `identity_vm_app/streamlit_test.py`:

  Terminal 1:  python main.py
  Terminal 2:  streamlit run identity_vm_app/streamlit_test.py --server.port 8510

API Identity VM mặc định: http://127.0.0.1:8010 (đổi bằng IVM_API_PORT).
  Xem trước MJPEG (thread riêng): GET /ivm/preview/<camera_id>/mjpeg
  Nhận diện + overlay (khi BẬT analyze): GET /ivm/cameras/<camera_id>/infer/mjpeg

Tuỳ chọn:
  python main.py --no-camera
  python main.py --port 8010 --interval 1.5
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from identity_vm_app.preview.ffmpeg_env import apply_ffmpeg_capture_env

apply_ffmpeg_capture_env()

import requests
import uvicorn

from identity_vm_app.api.app import create_app

# Uvicorn / tham chiếu từ bên ngoài: uvicorn main:app
app = create_app()


def _wait_for_api(base: str, timeout_s: float = 180.0) -> bool:
    deadline = time.monotonic() + timeout_s
    url = f"{base.rstrip('/')}/ivm/health"
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.25)
    return False


def _setup_logging() -> None:
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    for name in ("camera_recognition", "camera_recognition.worker", "camera_recognition.activity"):
        logging.getLogger(name).setLevel(logging.INFO)


def main() -> None:
    import os

    from identity_vm_app import settings as s
    from identity_vm_app.camera_recognition.hub import (
        ensure_recognition_hub_started,
        shutdown_recognition_hub,
    )

    _setup_logging()
    p = argparse.ArgumentParser(description="Identity VM — API + worker camera")
    p.add_argument("--host", default=s.IVM_API_HOST, help="Bind API")
    p.add_argument("--port", type=int, default=s.IVM_API_PORT, help="Cổng API (mặc định IVM_API_PORT)")
    p.add_argument("--no-camera", action="store_true", help="Chỉ API, không worker RTSP")
    p.add_argument("--config", default=None, help="camera_config.json")
    p.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Giây giữa các lần infer (mặc định 1/IVM_ANALYZE_TARGET_FPS, ~10 fps)",
    )
    p.add_argument("--reconnect-max", type=int, default=0, help="0 = reconnect không giới hạn")
    args = p.parse_args()
    if args.no_camera:
        os.environ["IVM_NO_CAMERA_WORKERS"] = "1"

    api_worker_base = f"http://127.0.0.1:{args.port}"

    def _run_uvicorn() -> None:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    th = threading.Thread(target=_run_uvicorn, name="uvicorn", daemon=True)
    th.start()

    print(
        "[main.py] Đang đợi API… (log InsightFace: \"=== IVM InsightFace ONNX ===\")",
        flush=True,
    )
    if not _wait_for_api(api_worker_base):
        print("[main.py] Lỗi: API không phản hồi /ivm/health", file=sys.stderr)
        sys.exit(1)
    print(f"[main.py] API: {api_worker_base}/ivm/health — Streamlit: streamlit run identity_vm_app/streamlit_test.py --server.port 8510", flush=True)

    if args.no_camera:
        print("[main.py] --no-camera — Ctrl+C để thoát.", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\n[main.py] Đã dừng.", flush=True)
        return

    hub = ensure_recognition_hub_started(
        api_base=api_worker_base,
        config_path=args.config,
        interval_s=args.interval,
        threshold=None,
    )
    n = len(hub.list_camera_ids())
    print(
        f"[main.py] Worker nhận diện: {n} camera (RTSP chỉ khi BẬT analyze; "
        f"hiển thị: POST /ivm/preview/warm). BẬT/TẮT: POST /ivm/cameras/{{id}}/analyze. Ctrl+C để dừng.",
        flush=True,
    )
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[main.py] Đang dừng worker…", flush=True)
    finally:
        shutdown_recognition_hub()
        print("[main.py] Đã dừng.", flush=True)


if __name__ == "__main__":
    main()
