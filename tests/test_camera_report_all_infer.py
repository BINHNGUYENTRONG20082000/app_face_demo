"""Ghi báo cáo mọi frame infer (IVM_REPORT_ALL_INFER_FRAMES)."""

from __future__ import annotations

import numpy as np

from identity_vm_app.services.camera_report_writer import (
    FRAME_MARKER_TRACKING_ID,
    build_person_report_rows_from_assignments,
)


def test_empty_assignments_logs_frame_marker(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "identity_vm_app.services.camera_report_writer.s.IVM_REPORT_ALL_INFER_FRAMES",
        True,
    )
    monkeypatch.setattr(
        "identity_vm_app.services.camera_report_writer.cmedia.save_root_frame",
        lambda *_a, **_k: "/fake/root.jpg",
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    rows = build_person_report_rows_from_assignments(
        "cam1",
        "job1",
        frame,
        t_s=1.0,
        frame_index=10,
        sample_index=2,
        video_clip=1,
        assignments=[],
        faces_with_matches=[],
        log_all_infer_frames=True,
    )
    assert len(rows) == 1
    assert rows[0]["id_tracking"] == FRAME_MARKER_TRACKING_ID
    assert rows[0]["frame_index"] == 10
    assert rows[0]["sample_index"] == 2
    assert rows[0]["img_url"] == "/fake/root.jpg"


def test_person_without_face_logged_when_log_all(monkeypatch):
    monkeypatch.setattr(
        "identity_vm_app.services.camera_report_writer.s.IVM_REPORT_ALL_INFER_FRAMES",
        True,
    )
    monkeypatch.setattr(
        "identity_vm_app.services.camera_report_writer.cmedia.save_root_frame",
        lambda *_a, **_k: "/fake/root.jpg",
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    assignments = [
        {
            "person_track": [10, 20, 30, 40, 7],
            "has_face": False,
            "weapon": {"armed": False},
        }
    ]
    rows = build_person_report_rows_from_assignments(
        "cam1",
        "job1",
        frame,
        t_s=2.0,
        frame_index=4,
        sample_index=1,
        video_clip=1,
        assignments=assignments,
        faces_with_matches=[],
        log_all_infer_frames=True,
    )
    assert len(rows) == 1
    assert rows[0]["id_tracking"] == 7
    assert rows[0]["box_person"] == "[10, 20, 30, 40]"
    assert rows[0]["face_id"] is None


def test_legacy_skip_person_without_face_or_weapon(monkeypatch):
    monkeypatch.setattr(
        "identity_vm_app.services.camera_report_writer.s.IVM_REPORT_ALL_INFER_FRAMES",
        False,
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    assignments = [
        {
            "person_track": [0, 0, 10, 10, 1],
            "has_face": False,
            "weapon": {"armed": False},
        }
    ]
    rows = build_person_report_rows_from_assignments(
        "cam1",
        "job1",
        frame,
        t_s=0.0,
        frame_index=0,
        sample_index=0,
        video_clip=1,
        assignments=assignments,
        faces_with_matches=[],
        log_all_infer_frames=False,
    )
    assert rows == []
