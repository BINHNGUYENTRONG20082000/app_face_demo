"""Kiểm tra map frame_index → box giữ ổn định (không bật/tắt từng khung)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from identity_vm_app.services.video_track_segments import (  # noqa: E402
    row_for_video_frame_index,
)


def _rows(fi_list):
    return [
        {
            "frame_index": fi,
            "time_analyze_s": fi / 25.0,
            "box_person": f"[{fi},{fi},{fi+10},{fi+10}]",
        }
        for fi in [0, 25, 50, 75]
    ]


def test_hold_box_between_samples_no_flicker():
    timeline = _rows([0, 25, 50, 75])
    r10 = row_for_video_frame_index(timeline, 10)
    r24 = row_for_video_frame_index(timeline, 24)
    r25 = row_for_video_frame_index(timeline, 25)
    assert r10["frame_index"] == 0
    assert r24["frame_index"] == 0
    assert r25["frame_index"] == 25
    assert r10["box_person"] == r24["box_person"]


def test_exact_sample_uses_that_row():
    timeline = _rows([0, 25, 50, 75])
    r50 = row_for_video_frame_index(timeline, 50)
    assert r50["frame_index"] == 50


def test_fi_now_sequence_monotonic_hold():
    """Mô phỏng 30 khung video @25fps — row_id chỉ đổi khi qua mẫu, không nhảy mỗi khung."""
    timeline = _rows([0, 25, 50, 75])
    fi_anchor = 0
    src_fps = 25.0
    cap_fps = 25.0
    row_ids = []
    for written in range(30):
        fi_now = fi_anchor + int(round(written * src_fps / cap_fps))
        row = row_for_video_frame_index(timeline, fi_now)
        row_ids.append(int(row["frame_index"]))
    changes = [i for i in range(1, len(row_ids)) if row_ids[i] != row_ids[i - 1]]
    assert changes == [25, 50]
    assert row_ids[0] == row_ids[24] == 0
    assert row_ids[25] == row_ids[49] == 25
