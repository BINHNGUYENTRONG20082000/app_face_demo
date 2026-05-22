"""Test BẬT/TẮT analyze qua camera_analyze_control (không RTSP)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TEST_ROOT = REPO / "data_test_live"
TEST_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["IVM_SQLITE_PATH"] = str(TEST_ROOT / "test_toggle.sqlite3")
os.environ["IVM_CAMERA_SESSION_DIR"] = str(TEST_ROOT / "camera_sessions_toggle")
os.environ["IVM_NO_CAMERA_WORKERS"] = "1"
os.environ["IVM_ANALYZE_RECORD_VISUAL"] = "0"
os.environ["IVM_ANALYZE_AUTO_ARCHIVE"] = "0"

import identity_vm_app.store.video_analyze_store as vas

vas._store_singleton = None

from identity_vm_app.camera_analyze_control import (  # noqa: E402
    get_analyze_enabled,
    set_analyze_enabled,
)
from identity_vm_app.services.camera_live_session import get_active_session  # noqa: E402
from identity_vm_app.store.video_analyze_store import get_video_analyze_store  # noqa: E402


def main() -> int:
    cam = "cam0"
    assert not get_analyze_enabled(cam)

    out = set_analyze_enabled(
        cam,
        True,
        sample_fps=10,
        display_name="Toggle test",
        distance_threshold=0.5,
        save_crops=False,
        stream_fps=20.0,
        start_frame_count=0,
    )
    assert get_analyze_enabled(cam)
    assert get_active_session(cam) is not None
    job_id = out["session"]["job_id"]
    assert job_id.startswith("live-cam0-")

    out2 = set_analyze_enabled(cam, False)
    assert not get_analyze_enabled(cam)
    assert get_active_session(cam) is None
    job = get_video_analyze_store().get_job(job_id)
    assert job and job.get("status_name") == "done", job

    print("OK toggle: set_analyze_enabled ON/OFF + live session lifecycle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
