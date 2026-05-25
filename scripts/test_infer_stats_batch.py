"""Test thống kê TB infer mỗi N khung."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ["IVM_INFER_STATS_EVERY_N"] = "3"
os.environ["IVM_INFER_STATS_FIRST_N"] = "3"

from module_ai.camera.worker import CameraRecognitionConfig, CameraRecognitionWorker  # noqa: E402


def main() -> int:
    cfg = CameraRecognitionConfig(camera_id="stat_cam", source=0, api_base="http://127.0.0.1:8010")
    w = CameraRecognitionWorker(cfg)
    messages: list[str] = []

    with patch(
        "module_ai.camera.worker.log_activity",
        side_effect=lambda _cid, _ev, msg, **_: messages.append(msg),
    ):
        with patch(
            "module_ai.camera.worker.weapon_detection_available",
            return_value=False,
        ):
            for i in range(1, 4):
                w._accumulate_infer_stats(
                    i,
                    {
                        "person_track_ms": 120.0,
                        "detect_ms": 10.0,
                        "pose_refine_ms": 30.0,
                        "embedding_ms": 1.0,
                        "search_ms": 0.0,
                        "weapon_ms": 26.0,
                    },
                    187.0,
                )

    assert len(messages) == 1, messages
    assert "📊 [lô đầu]" in messages[0]
    assert "TB 3 khung" in messages[0]
    assert "person_track 120" in messages[0]
    assert "pose 30" in messages[0]
    assert "face_det 10" in messages[0]
    assert "nghẽn:" in messages[0]
    assert w._stats_count == 0
    print("OK infer_stats_batch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
