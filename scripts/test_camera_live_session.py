"""Unit test luồng phiên camera live (không cần RTSP)."""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TEST_ROOT = REPO / "data_test_live"
TEST_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["IVM_SQLITE_PATH"] = str(TEST_ROOT / "test_live.sqlite3")
os.environ["IVM_CAMERA_SESSION_DIR"] = str(TEST_ROOT / "camera_sessions")
os.environ["IVM_NO_CAMERA_WORKERS"] = "1"
os.environ["IVM_ANALYZE_RECORD_VISUAL"] = "0"
os.environ["IVM_CAMERA_SESSION_STREAM_RECORD"] = "0"

import identity_vm_app.store.video_analyze_store as vas

vas._store_singleton = None

from identity_vm_app.services.camera_live_session import (  # noqa: E402
    get_active_session,
    on_frame,
    start_live_session,
    stop_live_session,
)
from identity_vm_app.services.video_analyze_fps import frame_skip_for_sample  # noqa: E402
from identity_vm_app.store.video_analyze_store import (  # noqa: E402
    VA_STATUS_COMPLETED,
    get_video_analyze_store,
)


def main() -> int:
    store = get_video_analyze_store()

    job_upload = str(uuid.uuid4())
    store.insert_job(
        job_id=job_upload,
        original_name="v.mp4",
        display_name="v",
        video_path="/tmp/x.mp4",
        thumb_path=None,
        feature_analyze={},
        sample_fps=10,
    )
    j = store.get_job(job_upload)
    assert j and j.get("source_type") == "upload", j

    sess = start_live_session(
        "cam_test",
        sample_fps=10,
        display_name="Test phiên",
        stream_fps=25.0,
        start_frame_count=100,
    )
    assert get_active_session("cam_test") is not None
    assert sess.job_id.startswith("live-cam_test-")

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assignments = [
        {
            "person_track": [10.0, 20.0, 200.0, 400.0, 1],
            "has_face": True,
            "weapon": {"armed": False, "weapon_types": []},
            "face_meta": {"bbox": [50, 60, 120, 140], "det_score": 0.9},
        }
    ]
    faces = [
        {
            "bbox": [50, 60, 120, 140],
            "det_score": 0.9,
            "matches": [{"face_id": 1, "name": "Alice", "distance": 0.3}],
        }
    ]
    n = on_frame(
        "cam_test",
        frame,
        frame_count=110,
        assignments=assignments,
        faces_with_matches=faces,
        visual_bgr=None,
        wall_t_s=0.0,
    )
    assert n >= 1, "expected report rows"

    stop = stop_live_session("cam_test")
    assert stop and stop.get("job_id") == sess.job_id
    assert get_active_session("cam_test") is None

    job = store.get_job(sess.job_id)
    assert job.get("status") == VA_STATUS_COMPLETED or job.get("status_name") == "done"
    assert job.get("camera_id") == "cam_test"
    assert job.get("source_type") == "camera_live"
    counts = store.count_reports(sess.job_id)
    assert counts["person_reports"] >= 1, counts

    rows = store.list_person_reports(sess.job_id)
    assert abs(float(rows[0]["time_analyze_s"]) - 0.0) < 1e-6

    skip = frame_skip_for_sample(25, 10)
    assert skip in (2, 3), skip

    stuck_id = str(uuid.uuid4())
    store.insert_camera_live_job(
        stuck_id,
        camera_id="cam_x",
        video_path="/x.mp4",
        display_name="x",
        sample_fps=10,
        feature_analyze={},
        session_start_utc=time.time(),
    )
    n_stuck = store.cleanup_stuck_camera_live_jobs()
    assert n_stuck >= 1, n_stuck

    print("OK unit: live session DB + on_frame + stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
