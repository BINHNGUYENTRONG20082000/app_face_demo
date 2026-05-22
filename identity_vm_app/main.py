from __future__ import annotations

import sys
from pathlib import Path

# Cho phép: cd identity_vm_app && python main.py (không cần đứng ở E:\app_face)
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse

import uvicorn
from identity_vm_app import settings as s
from identity_vm_app.api.app import create_app


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
    _setup_logging()
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=s.IVM_API_HOST)
    p.add_argument("--port", type=int, default=s.IVM_API_PORT)
    args = p.parse_args()
    print(
        "[identity_vm_app] Sau khi uvicorn khởi động, tìm khối log "
        "\"=== IVM InsightFace ONNX ===\" (đường dẫn .onnx + ort_session_providers GPU/CPU).",
        flush=True,
    )
    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
