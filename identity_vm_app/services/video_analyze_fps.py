"""Chế độ FPS lấy mẫu video: 0 = full frame, 5 / 10 / 15."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

ALLOWED_SAMPLE_FPS: Tuple[float, ...] = (0.0, 5.0, 10.0, 15.0)


def parse_sample_fps(raw: object) -> float:
    if raw is None or str(raw).strip() == "":
        return 0.0
    try:
        v = float(str(raw).strip())
    except ValueError as ex:
        raise ValueError("sample_fps không hợp lệ") from ex
    for allowed in ALLOWED_SAMPLE_FPS:
        if abs(v - allowed) < 1e-6:
            return allowed
    raise ValueError("sample_fps phải là một trong: 0 (full frame), 5, 10, 15")


def frame_skip_for_sample(video_fps: float, sample_fps: float) -> int:
    """0 → mọi khung (skip=1); ngược lại ~ video_fps / sample_fps."""
    if sample_fps <= 0:
        return 1
    vf = max(1.0, float(video_fps))
    return max(1, int(round(vf / float(sample_fps))))


def sample_fps_label(sample_fps: float) -> str:
    if sample_fps <= 0:
        return "Full frame"
    return f"{int(sample_fps)} FPS"


def default_display_name(original_name: str, sample_fps: float) -> str:
    stem = Path(original_name or "video").stem or "video"
    return f"{stem} — {sample_fps_label(sample_fps)}"
