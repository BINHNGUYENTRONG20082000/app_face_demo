"""
Hướng B: chỉ chạy YOLO-pose khi một box người (detect+track) có >= N mặt.
Tách/ghép mặt theo keypoint đầu (mũi, mắt, tai) thay vì box thân rộng.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from identity_vm_app import settings as s
from identity_vm_app.engine.yolo_person_tracker import _bbox_center_xy, _bbox_iou, _center_inside_box

# COCO 17 — đầu
_HEAD_KP_IDX = (0, 1, 2, 3, 4)

_pose_refiner: Optional["PoseFaceRefiner"] = None
_pose_refiner_lock = threading.Lock()
_pose_unavailable_reason: Optional[str] = None


def _resolve_pose_model_path() -> str:
    raw = (s.IVM_FACE_POSE_MODEL or "").strip()
    if raw:
        p = Path(raw)
        if p.is_file():
            return str(p.resolve())
    repo = Path(__file__).resolve().parents[2]
    search_dirs = (
        repo / "Module" / "model",
        Path(__file__).resolve().parents[1] / "modelAi",
    )
    names = (
        "yolo26n-pose.engine",
        "yolo26n-pose.pt",
        "yolo26m-pose.engine",
        "yolo26m-pose.pt",
    )
    for model_dir in search_dirs:
        for name in names:
            cand = model_dir / name
            if cand.is_file():
                return str(cand.resolve())
    return raw or str(repo / "Module" / "model" / "yolo26n-pose.engine")


def pose_refine_available() -> bool:
    global _pose_unavailable_reason
    if not bool(s.IVM_FACE_POSE_REFINE):
        _pose_unavailable_reason = "IVM_FACE_POSE_REFINE=0"
        return False
    try:
        from ultralytics import YOLO  # noqa: F401
    except ImportError as ex:
        _pose_unavailable_reason = f"Cần ultralytics: {ex}"
        return False
    mp = Path(_resolve_pose_model_path())
    if mp.is_file():
        return True
    if str(mp).endswith(".pt"):
        return True
    _pose_unavailable_reason = f"Không tìm thấy pose model: {mp}"
    return False


def pose_refine_unavailable_reason() -> Optional[str]:
    pose_refine_available()
    return _pose_unavailable_reason


def get_pose_refiner() -> Optional["PoseFaceRefiner"]:
    global _pose_refiner, _pose_unavailable_reason
    if not pose_refine_available():
        return None
    with _pose_refiner_lock:
        if _pose_refiner is None:
            try:
                _pose_refiner = PoseFaceRefiner()
            except Exception as ex:
                _pose_unavailable_reason = str(ex)
                return None
        return _pose_refiner


def release_pose_refiner() -> None:
    global _pose_refiner
    with _pose_refiner_lock:
        if _pose_refiner is not None:
            _pose_refiner.dispose()
            _pose_refiner = None


def _head_from_keypoints(
    keypoints: Optional[np.ndarray],
    person_bbox: List[float],
    *,
    kp_conf: float,
) -> Tuple[Optional[Tuple[float, float]], Optional[List[float]]]:
    if keypoints is None:
        return None, None
    kps = np.asarray(keypoints, dtype=np.float32)
    if kps.ndim == 1 and kps.size >= 3:
        kps = kps.reshape(-1, 3)
    if kps.ndim != 2 or kps.shape[1] < 3:
        return None, None

    pts: List[Tuple[float, float]] = []
    for idx in _HEAD_KP_IDX:
        if idx >= kps.shape[0]:
            continue
        x, y, c = float(kps[idx, 0]), float(kps[idx, 1]), float(kps[idx, 2])
        if c >= kp_conf:
            pts.append((x, y))

    if not pts:
        return None, None

    hx = sum(p[0] for p in pts) / len(pts)
    hy = sum(p[1] for p in pts) / len(pts)
    head_xy = (hx, hy)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pb = [float(person_bbox[0]), float(person_bbox[1]), float(person_bbox[2]), float(person_bbox[3])]
    ph = max(1.0, pb[3] - pb[1])
    pad = max(8.0, ph * 0.12)
    if x2 - x1 < pad:
        cx = (x1 + x2) * 0.5
        x1, x2 = cx - pad, cx + pad
    if y2 - y1 < pad:
        cy = (y1 + y2) * 0.5
        y1, y2 = cy - pad, cy + pad
    head_bbox = [x1 - pad * 0.3, y1 - pad * 0.5, x2 + pad * 0.3, y2 + pad * 0.2]
    return head_xy, head_bbox


def _pose_overlaps_detect(
    pose_bbox: List[float],
    det_bbox: List[float],
    *,
    iou_min: float,
) -> bool:
    if _bbox_iou(pose_bbox, det_bbox) >= iou_min:
        return True
    cx, cy = _bbox_center_xy(pose_bbox)
    return _center_inside_box((cx, cy), det_bbox, margin=0.0)


def _make_person_track_from_pose(
    pose_bbox: List[int],
    source_track: List[Any],
) -> List[Any]:
    tid = int(source_track[4]) if len(source_track) > 4 else 0
    cid = int(source_track[5]) if len(source_track) > 5 else 0
    return [
        int(pose_bbox[0]),
        int(pose_bbox[1]),
        int(pose_bbox[2]),
        int(pose_bbox[3]),
        tid,
        cid,
    ]


def _match_faces_to_pose_candidates(
    face_indices: List[int],
    face_boxes: List[List[float]],
    candidates: List[Dict[str, Any]],
) -> Dict[int, int]:
    """face_index -> candidate index (1:1)."""
    if not face_indices or not candidates:
        return {}

    n_f = len(face_indices)
    n_p = len(candidates)
    cost = np.full((n_f, n_p), 1e6, dtype=np.float64)

    for i, fi in enumerate(face_indices):
        if fi < 0 or fi >= len(face_boxes):
            continue
        fc = _bbox_center_xy(face_boxes[fi])
        fb = face_boxes[fi]
        for j, cand in enumerate(candidates):
            hp = cand.get("head_xy")
            if hp is None:
                hp = _bbox_center_xy(cand["bbox"])
            dist = math.hypot(fc[0] - hp[0], fc[1] - hp[1])
            hb = cand.get("head_bbox")
            iou_h = _bbox_iou(hb, fb) if hb else 0.0
            cost[i, j] = dist - iou_h * 80.0

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost)
        out: Dict[int, int] = {}
        for row, col in zip(row_ind, col_ind):
            fi = int(face_indices[int(row)])
            if cost[int(row), int(col)] < 1e5:
                out[fi] = int(col)
        return out
    except ImportError:
        used_p: set[int] = set()
        out = {}
        for fi in sorted(face_indices, key=lambda x: face_boxes[x][2] - face_boxes[x][0], reverse=True):
            fc = _bbox_center_xy(face_boxes[fi])
            best_j, best_c = None, 1e9
            for j, cand in enumerate(candidates):
                if j in used_p:
                    continue
                hp = cand.get("head_xy") or _bbox_center_xy(cand["bbox"])
                c = math.hypot(fc[0] - hp[0], fc[1] - hp[1])
                if c < best_c:
                    best_c, best_j = c, j
            if best_j is not None:
                used_p.add(best_j)
                out[fi] = best_j
        return out


def refine_grouped_with_pose(
    frame_bgr: np.ndarray,
    grouped: List[Tuple[List[Any], List[int]]],
    face_boxes: List[List[float]],
    *,
    min_faces: Optional[int] = None,
    det_iou_min: Optional[float] = None,
) -> Tuple[List[Tuple[List[Any], List[int]]], float]:
    """
    Tách bucket nhiều mặt / một detect track bằng pose (nếu đủ instance pose).
    Trả (grouped_mới, pose_ms).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return grouped, 0.0

    thr = int(min_faces if min_faces is not None else s.IVM_FACE_POSE_MIN_FACES)
    thr = max(2, thr)
    iou_det = float(det_iou_min if det_iou_min is not None else s.IVM_FACE_POSE_DET_IOU_MIN)

    conflict = [i for i, (_, fis) in enumerate(grouped) if len(fis) >= thr]
    if not conflict:
        return grouped, 0.0

    refiner = get_pose_refiner()
    if refiner is None:
        return grouped, 0.0

    t0 = time.perf_counter()
    pose_persons = refiner.detect_persons(frame_bgr)
    pose_ms = (time.perf_counter() - t0) * 1000.0
    if not pose_persons:
        return grouped, pose_ms

    new_grouped: List[Tuple[List[Any], List[int]]] = []
    for i, (pt, fis) in enumerate(grouped):
        if i not in conflict or len(fis) < thr:
            new_grouped.append((pt, list(fis)))
            continue

        det_box = list(pt[:4])
        candidates = [
            p
            for p in pose_persons
            if _pose_overlaps_detect(list(p["bbox"]), det_box, iou_min=iou_det)
        ]
        if len(candidates) < 2:
            new_grouped.append((pt, list(fis)))
            continue

        mapping = _match_faces_to_pose_candidates(fis, face_boxes, candidates)
        assigned: set[int] = set()
        for fi in fis:
            cj = mapping.get(fi)
            if cj is None:
                continue
            cand = candidates[cj]
            pose_pt = _make_person_track_from_pose(cand["bbox"], pt)
            new_grouped.append((pose_pt, [int(fi)]))
            assigned.add(int(fi))

        leftover = [int(fi) for fi in fis if int(fi) not in assigned]
        if leftover:
            new_grouped.append((pt, leftover))

    return new_grouped, pose_ms


class PoseFaceRefiner:
    """YOLO-pose một lần / khung khi cần tinh chỉnh ghép mặt."""

    def __init__(self) -> None:
        from ultralytics import YOLO

        self._model_path = _resolve_pose_model_path()
        self._conf = float(s.IVM_FACE_POSE_CONF)
        self._imgsz = int(s.IVM_FACE_POSE_IMGSZ)
        self._kp_conf = float(s.IVM_FACE_POSE_KP_CONF)
        self._model = YOLO(self._model_path)
        self._lock = threading.Lock()

    def detect_persons(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        with self._lock:
            results = self._model.predict(
                frame_bgr,
                conf=self._conf,
                imgsz=self._imgsz,
                classes=[0],
                verbose=False,
            )
        if not results:
            return []
        r0 = results[0]
        boxes = r0.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
        kp_all = None
        if r0.keypoints is not None and r0.keypoints.data is not None:
            kp_all = r0.keypoints.data
            if hasattr(kp_all, "cpu"):
                kp_all = kp_all.cpu().numpy()

        out: List[Dict[str, Any]] = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = (
                int(xyxy[i][0]),
                int(xyxy[i][1]),
                int(xyxy[i][2]),
                int(xyxy[i][3]),
            )
            bbox = [x1, y1, x2, y2]
            kps = kp_all[i] if kp_all is not None and i < len(kp_all) else None
            head_xy, head_bbox = _head_from_keypoints(kps, bbox, kp_conf=self._kp_conf)
            out.append(
                {
                    "bbox": bbox,
                    "keypoints": kps,
                    "head_xy": head_xy,
                    "head_bbox": head_bbox,
                }
            )
        return out

    def dispose(self) -> None:
        try:
            from identity_vm_app.engine.gpu_cleanup import dispose_ultralytics_yolo

            dispose_ultralytics_yolo(self._model)
        except Exception:
            pass
        try:
            del self._model
        except Exception:
            pass
