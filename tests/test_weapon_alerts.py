"""Tests cảnh báo vũ khí theo track (live camera)."""

from __future__ import annotations

from module_ai.camera import weapon_alerts
from identity_vm_app.services.weapon_track_status import weapon_should_alert


def test_weapon_should_alert_more_than_five_frames():
    assert not weapon_should_alert(5)
    assert weapon_should_alert(6)


def test_emit_weapon_alert_once_per_track_per_job():
    weapon_alerts.reset_weapon_alerts("cam0")
    faces = [
        {
            "track_id": 7,
            "weapon": {
                "weapon_armed_frames": 6,
                "weapon_types": ["gun"],
                "weapon_alert": True,
            },
        }
    ]
    import numpy as np

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    a1 = weapon_alerts.emit_weapon_track_alerts(
        "cam0",
        job_id="job-a",
        faces=faces,
        armed_persons=[],
        frame_bgr=frame,
    )
    assert a1[0].get("thumb_jpeg_b64")
    assert a1[0].get("alert_id")
    hist = weapon_alerts.weapon_alert_history_by_camera("cam0")
    assert len(hist.get("cam0") or []) == 1
    a2 = weapon_alerts.emit_weapon_track_alerts(
        "cam0",
        job_id="job-a",
        faces=faces,
        armed_persons=[],
        frame_bgr=frame,
    )
    assert len(a1) == 1
    assert a1[0]["track_id"] == 7
    assert len(a2) == 0
    weapon_alerts.reset_weapon_alerts("cam0", job_id="job-b")
    a3 = weapon_alerts.emit_weapon_track_alerts(
        "cam0",
        job_id="job-b",
        faces=faces,
        armed_persons=[],
    )
    assert len(a3) == 1
