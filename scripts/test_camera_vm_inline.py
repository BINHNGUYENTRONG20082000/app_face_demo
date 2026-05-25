"""Unit test VisionMaster per-camera: slot sample + không dùng global GPU skip."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from module_ai.camera.worker import CameraRecognitionConfig, CameraRecognitionWorker  # noqa: E402


def test_vm_sample_slot() -> None:
    ok, key = CameraRecognitionWorker.vm_sample_slot(100, 100, 25.0, 5.0)
    assert ok and key == 0
    ok, key = CameraRecognitionWorker.vm_sample_slot(101, 100, 25.0, 5.0)
    assert not ok
    ok, key = CameraRecognitionWorker.vm_sample_slot(105, 100, 25.0, 5.0)
    assert ok and key == 1


def test_worker_has_dedicated_stack() -> None:
    cfg = CameraRecognitionConfig(camera_id="vm_cam", source=0, api_base="http://127.0.0.1:8010")
    w = CameraRecognitionWorker(cfg)
    assert w._models.camera_id == "vm_cam"
    assert w.infer_queue_size() == 0


def main() -> int:
    test_vm_sample_slot()
    test_worker_has_dedicated_stack()
    print("OK camera_vm_per_camera")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
