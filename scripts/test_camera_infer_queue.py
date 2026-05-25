"""Unit test hàng đợi infer camera — giữ mọi khung sample (VisionMaster per-camera)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from module_ai.camera.worker import (  # noqa: E402
    CameraRecognitionConfig,
    CameraRecognitionWorker,
    _QueuedInferFrame,
)


def test_live_slot_enqueue_by_infer_key() -> None:
    cfg = CameraRecognitionConfig(camera_id="qcam", source=0, api_base="http://127.0.0.1:8010")
    w = CameraRecognitionWorker(cfg)
    live = MagicMock()
    live.job_id = "live-qcam-abc"
    live.sample_fps = 10.0
    live.start_frame_count = 100
    live.session_start_utc = time.time()
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    w._maybe_enqueue_live(frame, 100, 25.0, live)
    assert w.infer_queue_size() == 1
    w._maybe_enqueue_live(frame, 100, 25.0, live)
    assert w.infer_queue_size() == 1
    w._maybe_enqueue_live(frame, 102, 25.0, live)
    assert w.infer_queue_size() == 2


def test_wait_drain() -> None:
    cfg = CameraRecognitionConfig(camera_id="qcam2", source=0, api_base="http://127.0.0.1:8010")
    w = CameraRecognitionWorker(cfg)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    w._enqueue_infer(
        _QueuedInferFrame(
            frame_bgr=frame,
            frame_count=1,
            stream_fps=25.0,
            job_id="j1",
            live_mode=True,
            infer_key=0,
        )
    )
    assert w.infer_queue_size() == 1

    def drain() -> None:
        time.sleep(0.05)
        w._dequeue_infer()

    import threading

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    assert w.wait_infer_queue_drained(timeout=2.0) is True
    t.join(timeout=1.0)


def test_dedicated_model_stack_lazy() -> None:
    cfg = CameraRecognitionConfig(camera_id="qcam3", source=0, api_base="http://127.0.0.1:8010")
    w = CameraRecognitionWorker(cfg)
    assert not w._models.loaded
    fake_engine = MagicMock()
    fake_db = MagicMock()
    with patch(
        "module_ai.engine.insightface_engine.InsightFaceEngine",
        return_value=fake_engine,
    ):
        with patch("identity_vm_app.state.state") as st:
            st.face_db = fake_db
            st.engine = None
            with patch(
                "module_ai.engine.yolo_person_tracker.vm_tracking_available",
                return_value=False,
            ):
                eng, trk, db = w._models.ensure_loaded()
                assert eng is fake_engine and db is fake_db and trk is None
                assert w._models.loaded


def main() -> int:
    test_live_slot_enqueue_by_infer_key()
    test_wait_drain()
    test_dedicated_model_stack_lazy()
    print("OK camera_infer_queue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
