"""Ghép ảnh crop thành WebM (giống VisionMaster export_video_view)."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from identity_vm_app import settings as s
from identity_vm_app.services.event_crops import load_crop_bytes


def build_frames_from_events(
    items: Sequence[Dict[str, Any]],
) -> List[Tuple[bytes, int]]:
    """(jpeg_bytes, repeat_count) theo thứ tự items (đã sort asc cho export)."""
    frames: List[Tuple[bytes, int]] = []
    for it in items:
        rel = it.get("crop_path")
        if not rel:
            continue
        data = load_crop_bytes(str(rel))
        if not data:
            continue
        repeat = max(1, int(it.get("frame_hits") or 1))
        frames.append((data, repeat))
    return frames


def export_crops_to_webm(
    frames: Sequence[Tuple[bytes, int]],
    *,
    out_path: Optional[Path] = None,
    fps: float = 5.0,
) -> Path:
    """
    frames: (jpeg_bytes, repeat_count) — mỗi ảnh lặp repeat_count lần (≈ số frame tracking).
    Trả về đường dẫn file .webm.
    """
    if not frames:
        raise ValueError("Không có ảnh để xuất video")

    s.IVM_EXPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or s.IVM_EXPORT_CACHE_DIR / f"crops_{int(time.time())}_{uuid.uuid4().hex[:8]}.webm"

    first_arr = np.frombuffer(frames[0][0], dtype=np.uint8)
    first_img = cv2.imdecode(first_arr, cv2.IMREAD_COLOR)
    if first_img is None:
        raise ValueError("Không decode được ảnh crop đầu tiên")
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"vp80")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError("Không mở được VideoWriter WebM (cần OpenCV hỗ trợ vp80)")

    try:
        for jpeg_bytes, repeat in frames:
            arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[0] != h or img.shape[1] != w:
                img = cv2.resize(img, (w, h))
            for _ in range(max(1, int(repeat))):
                writer.write(img)
    finally:
        writer.release()

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError("Xuất WebM thất bại (file rỗng)")
    return out_path
