"""Smoke test: camera_config load + /ivm/cameras (chạy: python scripts/smoke_ivm_cameras.py)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from camera_channel_config import load_camera_channel_specs
from identity_vm_app.api.routes import list_cameras


def main() -> int:
    d = {f"camera_source{i}": f"rtsp://x/{i}" for i in range(12)}
    fd, path = tempfile.mkstemp(suffix=".json", text=True)
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f)
        specs = load_camera_channel_specs(path)
        assert len(specs) == 12, len(specs)
        assert [s["id"] for s in specs[:3]] == ["cam0", "cam1", "cam2"]
        print("OK load_camera_channel_specs (12 numbered keys)")
    finally:
        os.unlink(path)

    cfg = REPO / "camera_config.json"
    real = load_camera_channel_specs(str(cfg))
    print(f"OK real {cfg.name}: n={len(real)} ids={[c['id'] for c in real]}")

    api_payload = list_cameras()
    n = len(api_payload.get("cameras") or [])
    if n != len(real):
        print(f"FAIL: list_cameras n={n} != file specs n={len(real)}")
        return 1
    print(f"OK list_cameras() matches file ({n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
