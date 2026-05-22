"""Test ghi full stream phiên camera (frame writer, không RTSP)."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TEST_ROOT = REPO / "data_test_stream"
TEST_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["IVM_SQLITE_PATH"] = str(TEST_ROOT / "test_stream.sqlite3")
os.environ["IVM_CAMERA_SESSION_DIR"] = str(TEST_ROOT / "camera_sessions")
os.environ["IVM_NO_CAMERA_WORKERS"] = "1"
os.environ["IVM_CAMERA_SESSION_STREAM_RECORD"] = "1"
os.environ["IVM_CAMERA_SESSION_OVERLAY_LIVE"] = "0"
os.environ["IVM_ANALYZE_RECORD_VISUAL"] = "0"

import identity_vm_app.store.video_analyze_store as vas

vas._store_singleton = None

from identity_vm_app.services.camera_live_session import (  # noqa: E402
    on_frame,
    record_stream_frame,
    start_live_session,
    stop_live_session,
)
from identity_vm_app.services.camera_session_media import session_mp4_path  # noqa: E402
from identity_vm_app.store.video_analyze_store import get_video_analyze_store  # noqa: E402


def main() -> int:
    store = get_video_analyze_store()
    sess = start_live_session(
        "cam_stream",
        sample_fps=5,
        stream_fps=10.0,
        source=0,
    )
    assert sess.record_mode == "frame", sess.record_mode

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(12):
        record_stream_frame("cam_stream", frame, stream_fps=10.0)

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
            "matches": [{"face_id": 1, "name": "Bob", "distance": 0.3}],
        }
    ]
    n = on_frame(
        "cam_stream",
        frame,
        frame_count=5,
        assignments=assignments,
        faces_with_matches=faces,
        wall_t_s=2.5,
    )
    assert n >= 1

    stop = stop_live_session("cam_stream")
    assert stop and stop.get("record_mode") == "frame"

    mp4 = session_mp4_path("cam_stream", sess.job_id)
    assert mp4.is_file() and mp4.stat().st_size > 0, mp4

    rows = store.list_person_reports(sess.job_id)
    assert abs(float(rows[0]["time_analyze_s"]) - 2.5) < 1e-6

    job = store.get_job(sess.job_id)
    assert job.get("duration_s", 0) > 0

    stuck = str(uuid.uuid4())
    store.insert_camera_live_job(
        stuck,
        camera_id="x",
        video_path="/x.mp4",
        display_name="x",
        sample_fps=5,
        feature_analyze={},
        session_start_utc=0,
    )
    assert store.cleanup_stuck_camera_live_jobs() >= 1

    print("OK unit: stream record frame writer + wall timeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
