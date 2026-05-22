"""Live sample enqueue — không bỏ slot theo sample_fps."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from identity_vm_app.camera_recognition.worker import CameraRecognitionConfig, CameraRecognitionWorker
from identity_vm_app.services.video_analyze_fps import frame_skip_for_sample


class TestVmSampleSlot(unittest.TestCase):
    def test_skip_matches_offline_formula(self) -> None:
        self.assertEqual(frame_skip_for_sample(25.0, 5.0), 5)
        self.assertEqual(frame_skip_for_sample(25.0, 0.0), 1)

    def test_every_sample_key_enqueued_once(self) -> None:
        start_fc = 100
        stream_fps = 25.0
        sample_fps = 5.0
        skip = frame_skip_for_sample(stream_fps, sample_fps)
        keys = []
        for fc in range(start_fc, start_fc + skip * 4 + 1):
            slot, key = CameraRecognitionWorker.vm_sample_slot(
                fc, start_fc, stream_fps, sample_fps
            )
            if slot:
                keys.append(key)
        self.assertEqual(keys, [0, 1, 2, 3, 4])


class TestReaderCallbackEnqueue(unittest.TestCase):
    def test_callback_enqueues_all_sample_slots(self) -> None:
        cfg = CameraRecognitionConfig(
            camera_id="cam-test",
            source="rtsp://dummy",
            use_in_process=True,
        )
        worker = CameraRecognitionWorker(cfg)
        worker._stop.clear()

        live = MagicMock()
        live.job_id = "live-cam-test-abc"
        live.start_frame_count = 0
        live.sample_fps = 5.0

        enqueued: list[int] = []

        def fake_enqueue(item) -> None:
            enqueued.append(int(item.infer_key))

        worker._enqueue_infer = fake_enqueue  # type: ignore[method-assign]

        stream_fps = 25.0
        skip = frame_skip_for_sample(stream_fps, 5.0)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        with patch(
            "identity_vm_app.camera_recognition.worker.get_analyze_enabled",
            return_value=True,
        ), patch(
            "identity_vm_app.services.camera_live_session.get_active_session",
            return_value=live,
        ):
            for fc in range(0, skip * 3 + 1):
                worker._on_reader_frame_decoded(fc, frame, stream_fps)

        self.assertEqual(enqueued, [0, 1, 2, 3])
        self.assertEqual(worker._last_enqueued_key, 3)
