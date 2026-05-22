"""
Worker đa camera → Identity VM (luồng nhận diện kiểu VisionMaster).

Chạy khi API đã sẵn sàng (thường `python main.py` gộp API + worker).
CLI độc lập: `python -m camera_pipeline --api http://127.0.0.1:8010`
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from identity_vm_app.preview.ffmpeg_env import apply_ffmpeg_capture_env

apply_ffmpeg_capture_env()

import requests

from camera_channel_config import load_camera_channel_specs
from identity_vm_app import settings as s
from identity_vm_app.camera_recognition.hub import shutdown_recognition_hub, start_recognition_hub

logger = logging.getLogger("camera_pipeline")


def _health_check(api_base: str, timeout: float = 5.0) -> bool:
    url = f"{api_base.rstrip('/')}/ivm/health"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            logger.info("API OK %s", url)
            return True
    except requests.RequestException as e:
        logger.error("Health check failed %s: %s", url, e)
    return False


def run_service(
    *,
    api_base: str,
    config_path: Optional[str],
    camera_ids: Optional[Sequence[str]],
    interval_s: Optional[float],
    threshold: Optional[float],
    jpeg_quality: int,
    reconnect_delay: float,
    reconnect_max: int,
    skip_health: bool,
) -> None:
    _ = jpeg_quality, reconnect_delay, reconnect_max
    if not skip_health and not _health_check(api_base):
        raise SystemExit(
            "Identity VM API không phản hồi. Chạy trước: python main.py "
            "(hoặc --skip-health-check)"
        )

    specs = load_camera_channel_specs(config_path or s.IVM_CAMERA_CONFIG)
    by_id = {str(c["id"]): c for c in specs}
    if camera_ids:
        wanted = [str(x) for x in camera_ids]
        missing = [x for x in wanted if x not in by_id]
        if missing:
            raise SystemExit(f"Unknown camera id in config: {missing}")
        picked = [by_id[x] for x in wanted]
    else:
        picked = specs

    if os.getenv("IVM_USE_IN_PROCESS_INFER", "").strip() == "":
        os.environ.setdefault("IVM_USE_IN_PROCESS_INFER", "0")

    eff_interval = interval_s if interval_s is not None and interval_s > 0 else None
    start_recognition_hub(
        picked,
        api_base=api_base,
        interval_s=eff_interval,
        threshold=threshold,
    )
    logger.info("Running %d recognition worker(s) — Ctrl+C to stop", len(picked))
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Stopping…")
    finally:
        shutdown_recognition_hub()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    default_api = f"http://127.0.0.1:{s.IVM_API_PORT}"
    p = argparse.ArgumentParser(description="Camera pipeline → Identity VM (recognition hub)")
    p.add_argument("--api", default=default_api, help="Base URL (vd http://127.0.0.1:8010)")
    p.add_argument("--config", default=None, help="Đường dẫn camera_config.json")
    p.add_argument("--camera", dest="cameras", action="append", metavar="ID", help="Chỉ chạy camera_id")
    p.add_argument("--interval", type=float, default=None, help="Giây giữa hai lần infer (mặc định 1/target_fps)")
    p.add_argument("--threshold", type=float, default=None, help="distance_threshold")
    p.add_argument("--jpeg-quality", type=int, default=85, metavar="Q", help="(legacy, ignored)")
    p.add_argument("--reconnect-delay", type=float, default=5.0, help="(legacy, ignored)")
    p.add_argument("--reconnect-max", type=int, default=0, help="(legacy, ignored)")
    p.add_argument("--skip-health-check", action="store_true")
    args = p.parse_args()

    run_service(
        api_base=args.api,
        config_path=args.config,
        camera_ids=args.cameras,
        interval_s=args.interval,
        threshold=args.threshold,
        jpeg_quality=max(1, min(100, args.jpeg_quality)),
        reconnect_delay=args.reconnect_delay,
        reconnect_max=max(0, args.reconnect_max),
        skip_health=args.skip_health_check,
    )


if __name__ == "__main__":
    main()
