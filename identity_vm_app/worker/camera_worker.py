"""CLI một camera — dùng chung pipeline ổn định (`camera_pipeline`)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from camera_pipeline.runner import run_service
from identity_vm_app import settings as s


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    default_api = f"http://127.0.0.1:{s.IVM_API_PORT}"
    p = argparse.ArgumentParser()
    p.add_argument("--camera", required=True, help="camera_id trong camera_config.json")
    p.add_argument("--api", default=default_api, help="Base URL Identity VM API")
    p.add_argument("--interval", type=float, default=1.0, help="Giây tối thiểu giữa các lần infer")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--config", default=None, help="camera_config.json (mặc định IVM_CAMERA_CONFIG)")
    p.add_argument("--skip-health-check", action="store_true")
    args = p.parse_args()

    run_service(
        api_base=args.api,
        config_path=args.config,
        camera_ids=[args.camera],
        interval_s=args.interval,
        threshold=args.threshold,
        jpeg_quality=85,
        reconnect_delay=5.0,
        reconnect_max=0,
        skip_health=args.skip_health_check,
    )


if __name__ == "__main__":
    main()
