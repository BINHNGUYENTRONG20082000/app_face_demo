"""
Theo dõi người: YOLO (ultralytics) + ByteTrack.
Mỗi đoạn video nên dùng create_person_tracker() riêng (persist qua khung trong đoạn).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from module_ai.config import settings as s

_tracker_unavailable_reason: Optional[str] = None
_ultralytics_ok: Optional[bool] = None
_shared_yolo_model: Any = None
_shared_yolo_lock = threading.Lock()
_tracker_states: Dict[str, list] = {}


def _check_ultralytics() -> bool:
    global _ultralytics_ok, _tracker_unavailable_reason
    if _ultralytics_ok is not None:
        return _ultralytics_ok
    try:
        from ultralytics import YOLO  # noqa: F401
        _ultralytics_ok = True
    except ImportError as ex:
        _ultralytics_ok = False
        _tracker_unavailable_reason = f"Cần cài ultralytics: pip install ultralytics ({ex})"
    return bool(_ultralytics_ok)


def _resolve_model_path() -> str:
    raw = (s.IVM_VIDEO_ANALYZE_YOLO_MODEL or "yolo26m.pt").strip()
    p = Path(raw)
    if p.is_file():
        return str(p.resolve())
    search_dirs = (
        s.MODEL_DIR,
        s.LEGACY_MODEL_DIR,
        Path(__file__).resolve().parents[2] / "Module" / "model",
    )
    names = (Path(raw).name,) if raw else ()
    names = names + ("yolo26m.pt", "yolo26m.engine", "yolo26m.onnx")
    for model_dir in search_dirs:
        for name in names:
            cand = model_dir / name
            if cand.is_file():
                return str(cand.resolve())
    return raw


def vm_tracking_available() -> bool:
    """ByteTrack + ultralytics sẵn sàng."""
    global _tracker_unavailable_reason
    if not bool(s.IVM_VIDEO_ANALYZE_USE_YOLO_TRACKING):
        return False
    if not _check_ultralytics():
        return False
    model = _resolve_model_path()
    mp = Path(model)
    if mp.suffix.lower() in (".pt", ".engine", ".onnx") and mp.is_file():
        return True
    # yolo26m.pt — ultralytics tự tải khi load
    if not mp.is_file() and model.endswith(".pt"):
        return True
    _tracker_unavailable_reason = f"Không tìm thấy model: {model}"
    return False


def tracker_unavailable_reason() -> Optional[str]:
    vm_tracking_available()
    return _tracker_unavailable_reason


def create_person_tracker(camera_id: str = "default") -> "ByteTrackPersonTracker":
    """Một instance / camera (ByteTrack persist trong phiên camera)."""
    if not vm_tracking_available():
        raise RuntimeError(tracker_unavailable_reason() or "Person tracker không khả dụng")
    return ByteTrackPersonTracker(camera_id=camera_id)


def _get_shared_yolo_model() -> Any:
    global _shared_yolo_model
    with _shared_yolo_lock:
        if _shared_yolo_model is not None:
            return _shared_yolo_model
    import logging
    import time

    log = logging.getLogger("identity_vm_app.yolo_person_tracker")
    log.info("Đang load YOLO person dùng chung (lần đầu) — mọi camera chờ lock này…")
    t0 = time.perf_counter()
    from ultralytics import YOLO

    loaded = YOLO(_resolve_model_path())
    with _shared_yolo_lock:
        if _shared_yolo_model is None:
            _shared_yolo_model = loaded
        log.info("YOLO person dùng chung sẵn sàng (%.1fs)", time.perf_counter() - t0)
        return _shared_yolo_model


def get_yolo_person_tracker() -> Optional["ByteTrackPersonTracker"]:
    """Giữ API cũ — chỉ dùng khi một luồng; phân tích video nên create_person_tracker()."""
    try:
        return create_person_tracker()
    except RuntimeError:
        return None


def _bbox_center_xy(xyxy: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _center_inside_box(center: Tuple[float, float], person_xyxy: List[float], *, margin: float = 0.0) -> bool:
    cx, cy = center
    x1, y1, x2, y2 = (float(person_xyxy[0]), float(person_xyxy[1]), float(person_xyxy[2]), float(person_xyxy[3]))
    return (x1 - margin) <= cx <= (x2 + margin) and (y1 - margin) <= cy <= (y2 + margin)


def _face_matches_person(
    person_xyxy: List[float],
    face_xyxy: List[float],
    *,
    iou_min: float,
    use_center_in_person: bool,
) -> bool:
    iou = _bbox_iou(person_xyxy, face_xyxy)
    if iou >= iou_min:
        return True
    if use_center_in_person:
        return _center_inside_box(_bbox_center_xy(face_xyxy), person_xyxy)
    return False


def assign_faces_to_person_tracks_multi(
    person_tracks: List[List[Any]],
    face_boxes: List[List[float]],
    *,
    iou_min: float = 0.05,
    use_center_in_person: bool = True,
) -> List[Tuple[List[Any], List[int]]]:
    """
    Mỗi person track → danh sách index mặt (0..n) nằm trong / chồng box người.
    Mỗi mặt chỉ gán một person (person có IoU / chứa tâm mặt tốt nhất).
  """
    if not person_tracks:
        return []
    if not face_boxes:
        return [(list(pt), []) for pt in person_tracks]

    n_p = len(person_tracks)
    n_f = len(face_boxes)
    per_person: List[List[int]] = [[] for _ in range(n_p)]

    for j, fb in enumerate(face_boxes):
        cx, cy = _bbox_center_xy(fb)
        best_i: Optional[int] = None
        best_score = -1.0
        for i, pt in enumerate(person_tracks):
            pb = list(pt[:4])
            if not _face_matches_person(pb, fb, iou_min=iou_min, use_center_in_person=use_center_in_person):
                continue
            iou = _bbox_iou(pb, fb)
            inside = use_center_in_person and _center_inside_box((cx, cy), pb)
            score = iou + (0.25 if inside else 0.0)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is not None:
            per_person[best_i].append(j)

    for i in range(n_p):
        per_person[i].sort()
    return [(list(person_tracks[i]), per_person[i]) for i in range(n_p)]


def assign_faces_to_person_tracks(
    person_tracks: List[List[Any]],
    face_boxes: List[List[float]],
    *,
    iou_min: float = 0.0,
) -> List[Tuple[List[Any], Optional[int]]]:
    """
    Tương thích cũ: mỗi person tối đa một mặt (mặt đầu trong danh sách multi).
    """
    grouped = assign_faces_to_person_tracks_multi(
        person_tracks,
        face_boxes,
        iou_min=max(iou_min, 0.05),
        use_center_in_person=True,
    )
    out: List[Tuple[List[Any], Optional[int]]] = []
    for pt, face_idxs in grouped:
        out.append((pt, face_idxs[0] if face_idxs else None))
    return out


def match_faces_to_persons(
    person_tracks: List[List[Any]],
    face_boxes: List[List[float]],
    *,
    iou_min: float = 0.0,
) -> List[Tuple[Optional[List[Any]], List[float]]]:
    """
    Ghép face → person qua IoU + Hungarian.
    person_track: [x1,y1,x2,y2, track_id, class_id]
    """
    if not face_boxes:
        return []
    if not person_tracks:
        return [(None, fb) for fb in face_boxes]

    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        pairs: List[Tuple[Optional[List[Any]], List[float]]] = []
        used: set[int] = set()
        for fb in face_boxes:
            best_i, best_iou = None, 0.0
            for i, pt in enumerate(person_tracks):
                if i in used:
                    continue
                iou = _bbox_iou(list(pt[:4]), fb)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i is not None and best_iou > iou_min:
                used.add(best_i)
                pairs.append((person_tracks[best_i], fb))
            else:
                pairs.append((None, fb))
        return pairs

    n_p = len(person_tracks)
    n_f = len(face_boxes)
    iou_mat = np.zeros((n_p, n_f), dtype=np.float32)
    for i, pt in enumerate(person_tracks):
        pb = list(pt[:4])
        for j, fb in enumerate(face_boxes):
            iou_mat[i, j] = _bbox_iou(pb, fb)
    row_ind, col_ind = linear_sum_assignment(-iou_mat)
    face_to_person: Dict[int, Optional[List[Any]]] = {j: None for j in range(n_f)}
    for row, col in zip(row_ind, col_ind):
        if iou_mat[row, col] > iou_min:
            face_to_person[int(col)] = person_tracks[int(row)]
    return [(face_to_person[j], face_boxes[j]) for j in range(n_f)]


def _bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    bx1, by1, bx2, by2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


class ByteTrackPersonTracker:
    """YOLO detect + ByteTrack (ultralytics)."""

    def __init__(self, *, camera_id: str = "default") -> None:
        self._camera_id = str(camera_id)
        self._tracker = str(s.IVM_VIDEO_ANALYZE_TRACKER or "bytetrack.yaml")
        self._conf = float(s.IVM_VIDEO_ANALYZE_YOLO_CONF)
        self._imgsz = int(s.IVM_VIDEO_ANALYZE_YOLO_IMGSZ)
        self._owned_model = False
        if bool(s.IVM_SHARED_YOLO_PERSON):
            self._model = _get_shared_yolo_model()
        else:
            from ultralytics import YOLO

            self._model_path = _resolve_model_path()
            self._model = YOLO(self._model_path)
            self._owned_model = True
        self._lock = threading.Lock()

    def _restore_tracker_state(self) -> None:
        if not bool(s.IVM_SHARED_YOLO_PERSON):
            return
        pred = getattr(self._model, "predictor", None)
        saved = _tracker_states.get(self._camera_id)
        if pred is not None:
            pred.trackers = saved if saved is not None else []

    def _save_tracker_state(self) -> None:
        if not bool(s.IVM_SHARED_YOLO_PERSON):
            return
        pred = getattr(self._model, "predictor", None)
        if pred is not None and hasattr(pred, "trackers"):
            _tracker_states[self._camera_id] = pred.trackers

    def detect_person_tracks(self, frame_bgr: np.ndarray) -> List[List[Any]]:
        """Trả list [x1, y1, x2, y2, track_id, class_id] (class_id=0 person)."""
        with self._lock:
            if bool(s.IVM_SHARED_YOLO_PERSON):
                with _shared_yolo_lock:
                    self._restore_tracker_state()
                    try:
                        results = self._model.track(
                            frame_bgr,
                            persist=True,
                            tracker=self._tracker,
                            classes=[0],
                            conf=self._conf,
                            imgsz=self._imgsz,
                            verbose=False,
                        )
                    finally:
                        self._save_tracker_state()
            else:
                results = self._model.track(
                    frame_bgr,
                    persist=True,
                    tracker=self._tracker,
                    classes=[0],
                    conf=self._conf,
                    imgsz=self._imgsz,
                    verbose=False,
                )
        out: List[List[Any]] = []
        if not results:
            return out
        r0 = results[0]
        boxes = r0.boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
        ids = boxes.id
        if ids is None:
            return out
        if hasattr(ids, "cpu"):
            ids = ids.cpu().numpy()
        clss = boxes.cls
        if hasattr(clss, "cpu"):
            clss = clss.cpu().numpy()
        for i in range(len(boxes)):
            cid = int(clss[i]) if clss is not None else 0
            if cid != 0:
                continue
            x1, y1, x2, y2 = (int(xyxy[i][0]), int(xyxy[i][1]), int(xyxy[i][2]), int(xyxy[i][3]))
            tid = int(ids[i])
            out.append([x1, y1, x2, y2, tid, cid])
        return out

    def reset(self) -> None:
        """Xóa trạng thái ByteTrack giữa các đoạn."""
        if bool(s.IVM_SHARED_YOLO_PERSON):
            _tracker_states.pop(self._camera_id, None)
        pred = getattr(self._model, "predictor", None)
        if pred is not None:
            try:
                pred.trackers = []
            except Exception:
                pass
            if self._owned_model:
                self._model.predictor = None

    def dispose(self) -> None:
        self.reset()
        if self._owned_model:
            try:
                del self._model
            except Exception:
                pass


# Alias tương thích code cũ
YoloPersonTracker = ByteTrackPersonTracker
