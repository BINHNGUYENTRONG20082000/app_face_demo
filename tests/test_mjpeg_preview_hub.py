"""Unit tests cho luồng preview MJPEG (mock OpenCV — không cần camera thật)."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class MockVideoCapture:
    """Giả lập cv2.VideoCapture: queue các kết quả read()."""

    def __init__(self, reads: list[tuple[bool, np.ndarray | None]], *, opened: bool = True) -> None:
        self._reads = list(reads)
        self._opened = opened

    def isOpened(self) -> bool:
        return self._opened

    def set(self, *_a: object, **_k: object) -> None:
        pass

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._reads:
            return False, None
        return self._reads.pop(0)

    def release(self) -> None:
        self._opened = False


class TestMjpegPreviewHub(unittest.TestCase):
    def tearDown(self) -> None:
        from identity_vm_app.preview.mjpeg_hub import shutdown_preview_hub

        shutdown_preview_hub()

    def test_reconnect_after_consecutive_bad_reads(self) -> None:
        """Sau N frame lỗi liên tiếp phải release và mở lại capture."""
        from identity_vm_app.preview import mjpeg_hub as mh

        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        first_cap = MockVideoCapture([(False, None)] * 4)
        second_cap = MockVideoCapture([(True, frame)] * 20)

        caps: list[MockVideoCapture] = [first_cap, second_cap]

        def capture_factory(*_a: object, **_k: object) -> MockVideoCapture:
            return caps.pop(0)

        with patch.object(mh.cv2, "VideoCapture", side_effect=capture_factory):
            st = mh.CameraPreviewStream(
                "cam0",
                "rtsp://example/stream",
                jpeg_quality=80,
                capture_fps=60.0,
                reconnect_delay_s=0.02,
                read_fails_before_reconnect=4,
                open_backoff_cap_s=5.0,
                use_ffmpeg=False,
            )
            st.start()
            try:
                deadline = time.monotonic() + 5.0
                got = b""
                while time.monotonic() < deadline:
                    got = st.get_jpeg()
                    if got and len(got) > 30:
                        break
                    time.sleep(0.03)
                self.assertTrue(got and len(got) > 30, "Phải có JPEG sau khi reconnect")
                self.assertFalse(first_cap.isOpened(), "Capture cũ phải được release sau lỗi đọc")
            finally:
                st.stop()

    def test_preview_hub_starts_single_thread_per_camera(self) -> None:
        from identity_vm_app.preview import mjpeg_hub as mh

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        cap = MockVideoCapture([(True, frame)] * 50)

        with patch.object(mh.cv2, "VideoCapture", return_value=cap):
            hub = mh.PreviewHub()
            s1 = hub.ensure(
                "c1",
                "x",
                jpeg_quality=75,
                capture_fps=30.0,
                reconnect_delay_s=1.0,
                read_fails_before_reconnect=3,
                open_backoff_cap_s=10.0,
            )
            s2 = hub.ensure(
                "c1",
                "x",
                jpeg_quality=75,
                capture_fps=30.0,
                reconnect_delay_s=1.0,
                read_fails_before_reconnect=3,
                open_backoff_cap_s=10.0,
            )
            self.assertIs(s1, s2)

            deadline = time.monotonic() + 3.0
            buf = b""
            while time.monotonic() < deadline:
                buf = s1.get_jpeg()
                if buf:
                    break
                time.sleep(0.05)
            self.assertTrue(len(buf) > 20)
            hub.stop("c1")
