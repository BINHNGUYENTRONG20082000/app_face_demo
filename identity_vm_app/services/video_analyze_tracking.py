"""Tracking id + video_clip khi phân tích video (tương tự VideoMaster)."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    bx1, by1, bx2, by2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


class BboxTracker:
    """Gán id_tracking ổn định qua IoU bbox giữa các khung mẫu."""

    def __init__(self, *, iou_threshold: float = 0.25) -> None:
        self._iou_threshold = float(iou_threshold)
        self._tracks: Dict[int, List[float]] = {}
        self._next_id = 1

    def assign(self, bbox_xyxy: List[float]) -> int:
        best_id: Optional[int] = None
        best_iou = 0.0
        for tid, lb in self._tracks.items():
            iou = bbox_iou(bbox_xyxy, lb)
            if iou > best_iou:
                best_iou = iou
                best_id = tid
        if best_id is not None and best_iou >= self._iou_threshold:
            self._tracks[best_id] = list(bbox_xyxy[:4])
            return best_id
        tid = self._next_id
        self._next_id += 1
        self._tracks[tid] = list(bbox_xyxy[:4])
        return tid


class VideoClipCounter:
    """Tăng video_clip sau N khung liên tiếp không có detection (VideoMaster: 10)."""

    def __init__(self, *, max_miss: int = 10, initial_clip: int = 1) -> None:
        self._max_miss = max(1, int(max_miss))
        self._miss = 0
        self._clip = max(1, int(initial_clip))

    @property
    def clip(self) -> int:
        return self._clip

    def on_frame(self, had_detection: bool) -> int:
        if had_detection:
            self._miss = 0
        else:
            self._miss += 1
            if self._miss >= self._max_miss:
                self._clip += 1
                self._miss = 0
        return self._clip
