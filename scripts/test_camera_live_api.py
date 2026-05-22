"""Test API routes phiên camera (FastAPI TestClient)."""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TEST_ROOT = REPO / "data_test_live"
TEST_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["IVM_SQLITE_PATH"] = str(TEST_ROOT / "test_live_api.sqlite3")
os.environ["IVM_CAMERA_SESSION_DIR"] = str(TEST_ROOT / "camera_sessions_api")
os.environ["IVM_NO_CAMERA_WORKERS"] = "1"

import identity_vm_app.store.video_analyze_store as vas

vas._store_singleton = None

from fastapi import FastAPI
from fastapi.testclient import TestClient  # noqa: E402

from identity_vm_app.api.camera_analyze_reports_routes import router  # noqa: E402
from identity_vm_app.store.video_analyze_store import get_video_analyze_store  # noqa: E402


def main() -> int:
    store = get_video_analyze_store()
    cam = "cam0"
    job_id = f"live-{cam}-{uuid.uuid4().hex[:8]}"
    now = time.time()
    mp4 = TEST_ROOT / "camera_sessions_api" / cam / job_id / "session.mp4"
    mp4.parent.mkdir(parents=True, exist_ok=True)
    mp4.write_bytes(b"\x00" * 64)

    store.insert_camera_live_job(
        job_id,
        camera_id=cam,
        video_path=str(mp4),
        display_name="API test",
        sample_fps=10,
        feature_analyze={"pipeline": "camera_live"},
        session_start_utc=now - 60,
    )
    store.finalize_camera_live_job(
        job_id,
        video_path=str(mp4),
        thumb_path=None,
        duration_s=30.0,
        analyze_fps=10.0,
        total_sample_frames=5,
        session_end_utc=now,
    )
    store.insert_person_reports_batch(
        [
            {
                "job_id": job_id,
                "time_analyze_s": 0.5,
                "frame_index": 10,
                "sample_index": 0,
                "img_url": str(mp4),
                "id_tracking": 1,
                "video_clip": 1,
                "display_name": "Bob",
                "box_person": "[0,0,10,10]",
                "features_face": "[0.1 0.2]",
                "box_face": "[1,2,3,4]",
            }
        ]
    )

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    r = client.get(f"/ivm/cameras/{cam}/analyze/sessions", params={"limit": 5})
    assert r.status_code == 200, r.text
    sessions = r.json().get("sessions") or []
    assert any(s.get("id") == job_id for s in sessions), sessions

    r2 = client.get(f"/ivm/cameras/{cam}/analyze/sessions/{job_id}")
    assert r2.status_code == 200, r2.text
    assert r2.json().get("id") == job_id

    r3 = client.get(f"/ivm/cameras/{cam}/analyze/sessions/{job_id}/session.mp4")
    assert r3.status_code == 200, r3.status_code

    r4 = client.get(
        f"/ivm/cameras/{cam}/reports/faces-person",
        params={"session_ids": job_id},
    )
    assert r4.status_code == 200, r4.text
    fp = r4.json()
    assert isinstance(fp, list) and len(fp) >= 1, fp

    print("OK api: sessions + faces-person + session.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
