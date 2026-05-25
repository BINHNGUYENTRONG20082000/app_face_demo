"""Phát hiện vũ khí trên toàn khung, ghép vào người đã track (gun/knife)."""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from module_ai.config import settings as s

logger = logging.getLogger("camera_recognition.weapon")

WEAPON_CLASS_NAMES: Dict[int, str] = {0: "gun", 1: "knife"}

_UNARMED: Dict[str, Any] = {
    "armed": False,
    "frame_armed": False,
    "voted_armed": False,
    "memory_armed": False,
    "weapon_types": [],
    "weapon_status": "an_toan",
    "weapon_label": "Không vũ khí",
    "weapon_score": 0.0,
    "person_bbox": [],
    "weapons": [],
}

_detector_lock = threading.Lock()
_detectors: Dict[str, "_FrameWeaponDetector"] = {}
_voters: Dict[str, "_TemporalVoter"] = {}


class _TemporalVoter:
    def __init__(self, window: int, threshold: float) -> None:
        self._window = max(1, int(window))
        self._threshold = float(threshold)
        self._scores: Dict[int, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )

    def update(self, track_id: int, score: float) -> None:
        self._scores[int(track_id)].append(float(score))

    def is_armed(self, track_id: int) -> bool:
        hist = self._scores.get(int(track_id))
        if not hist:
            return False
        return (sum(hist) / len(hist)) >= self._threshold


class _FrameWeaponDetector:
    """YOLO weapon trên toàn khung → ghép bbox vào từng person track."""

    def __init__(self) -> None:
        from ultralytics import YOLO

        model_path = str(s.IVM_WEAPON_MODEL)
        self._model = YOLO(model_path, task="detect")
        self._names = _extract_class_names(self._model)
        self._conf = float(s.IVM_WEAPON_INPUT_CONF)
        self._imgsz = int(s.IVM_WEAPON_IMGSZ)
        self._device = s.IVM_WEAPON_DEVICE
        self._match_iou = float(s.IVM_WEAPON_MATCH_IOU_MIN)
        self._memory_frames = max(0, int(s.IVM_WEAPON_MEMORY_FRAMES))
        self._weapon_armed_frame_counts: Dict[int, int] = {}
        self._frame_count = 0
        self._dangerous_tracks: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        if s.IVM_WEAPON_WARMUP:
            dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
            self._model.predict(
                dummy,
                conf=self._conf,
                imgsz=self._imgsz,
                device=self._device,
                verbose=False,
            )
        logger.info("Weapon full-frame YOLO: %s names=%s", model_path, self._names)

    def detect_on_person_tracks(
        self,
        frame_bgr: np.ndarray,
        person_tracks: List[List[Any]],
        voter: Optional[_TemporalVoter],
    ) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, float]]:
        if not person_tracks:
            return {}, {"weapon": 0.0, "total": 0.0}

        H, W = frame_bgr.shape[:2]
        self._frame_count += 1
        t0 = time.perf_counter()

        raw_weapons = self._detect_weapons_full_frame(frame_bgr)
        weapon_by_track = self._match_weapons_to_person_tracks(
            raw_weapons, person_tracks, W, H
        )

        weapon_ms = (time.perf_counter() - t0) * 1000.0
        if raw_weapons:
            logger.debug(
                "weapon frame: %d det → %d track armed (persons=%d)",
                len(raw_weapons),
                sum(1 for ws in weapon_by_track.values() if ws),
                len(person_tracks),
            )

        visible_ids: set[int] = set()
        out: Dict[int, Dict[str, Any]] = {}

        for pt in person_tracks:
            if len(pt) < 5:
                continue
            tid = int(pt[4])
            visible_ids.add(tid)
            pb = list(_clamp_box(int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]), W, H))
            weapons = list(weapon_by_track.get(tid) or [])
            person_row = {
                "person_id": tid,
                "person_bbox": pb,
                "weapons": weapons,
                "frame_armed": len(weapons) > 0,
            }
            out[tid] = _finalize_person_weapon(person_row, voter, self)

        for lost_tid in list(self._dangerous_tracks.keys()):
            if lost_tid not in visible_ids:
                del self._dangerous_tracks[lost_tid]
        for lost_tid in list(self._weapon_armed_frame_counts.keys()):
            if lost_tid not in visible_ids:
                del self._weapon_armed_frame_counts[lost_tid]

        timing = {"weapon": round(weapon_ms, 2), "total": round(weapon_ms, 2)}
        return out, timing

    def _detect_weapons_full_frame(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        with self._lock:
            results = self._model.predict(
                frame_bgr,
                conf=self._conf,
                imgsz=self._imgsz,
                device=self._device,
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
        confs = boxes.conf
        if hasattr(confs, "cpu"):
            confs = confs.cpu().numpy()
        clss = boxes.cls
        if hasattr(clss, "cpu"):
            clss = clss.cpu().numpy()

        out: List[Dict[str, Any]] = []
        for i in range(len(boxes)):
            cls_id = int(clss[i]) if clss is not None else 0
            cls_name = self._names.get(cls_id, WEAPON_CLASS_NAMES.get(cls_id, "weapon"))
            conf = float(confs[i]) if confs is not None else 0.0
            out.append(
                {
                    "bbox": [
                        int(xyxy[i][0]),
                        int(xyxy[i][1]),
                        int(xyxy[i][2]),
                        int(xyxy[i][3]),
                    ],
                    "class": cls_name,
                    "conf": round(conf, 3),
                    "fusion_score": round(conf, 3),
                }
            )
        return out

    def _match_weapons_to_person_tracks(
        self,
        raw_weapons: List[Dict[str, Any]],
        person_tracks: List[List[Any]],
        W: int,
        H: int,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Mỗi vũ khí gán vào track người có IoU / tâm vũ khí nằm trong box người tốt nhất."""
        persons: List[Tuple[int, List[int]]] = []
        for pt in person_tracks:
            if len(pt) < 5:
                continue
            tid = int(pt[4])
            pb = _clamp_box(int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]), W, H)
            persons.append((tid, pb))

        weapon_by_track: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        used_weapon_idx: set[int] = set()

        for w_idx, w in enumerate(raw_weapons):
            wb = w.get("bbox") or []
            if len(wb) < 4:
                continue
            best_tid: Optional[int] = None
            best_score = 0.0
            for tid, pb in persons:
                iou = _bbox_iou(list(wb), list(pb))
                center_hit = _weapon_center_in_person(list(wb), list(pb))
                score = iou + (0.2 if center_hit else 0.0)
                if score > best_score:
                    best_score = score
                    best_tid = tid
            if best_tid is None:
                continue
            pb_match = next((pb for t, pb in persons if t == best_tid), None)
            if pb_match is None:
                continue
            if best_score < self._match_iou and not _weapon_center_in_person(
                list(wb), list(pb_match)
            ):
                continue
            weapon_by_track[best_tid].append(w)
            used_weapon_idx.add(w_idx)

        return dict(weapon_by_track)


def _weapon_center_in_person(weapon_bbox: List[float], person_bbox: List[int]) -> bool:
    wx = (float(weapon_bbox[0]) + float(weapon_bbox[2])) / 2.0
    wy = (float(weapon_bbox[1]) + float(weapon_bbox[3])) / 2.0
    px1, py1, px2, py2 = person_bbox
    return px1 <= wx <= px2 and py1 <= wy <= py2


def _finalize_person_weapon(
    person: Dict[str, Any],
    voter: Optional[_TemporalVoter],
    detector: _FrameWeaponDetector,
) -> Dict[str, Any]:
    from identity_vm_app.services.weapon_track_status import classify_weapon_by_frame_count

    track_id = int(person["person_id"])
    weapons = list(person.get("weapons") or [])
    raw_frame_armed = bool(person.get("frame_armed")) or bool(weapons)

    if raw_frame_armed:
        detector._weapon_armed_frame_counts[track_id] = (
            int(detector._weapon_armed_frame_counts.get(track_id) or 0) + 1
        )

    armed_frame_count = int(detector._weapon_armed_frame_counts.get(track_id) or 0)
    types = sorted({str(w.get("class")) for w in weapons if w.get("class")})
    cls = classify_weapon_by_frame_count(armed_frame_count, types)
    armed = bool(cls["armed"])
    dangerous = bool(cls["dangerous"])
    from identity_vm_app.services.weapon_track_status import weapon_should_alert

    weapon_alert = bool(weapon_should_alert(armed_frame_count))

    voted_armed = False
    if voter is not None:
        best_score = max((float(w.get("fusion_score") or 0) for w in weapons), default=0.0)
        voter.update(track_id, best_score)
        voted_armed = voter.is_armed(track_id)

    memory_armed = False
    if raw_frame_armed:
        detector._dangerous_tracks[track_id] = {
            "last_seen_frame": detector._frame_count,
            "score": max((float(w.get("fusion_score") or 0) for w in weapons), default=0.0),
        }
    elif track_id in detector._dangerous_tracks:
        gap = detector._frame_count - detector._dangerous_tracks[track_id]["last_seen_frame"]
        if gap <= detector._memory_frames:
            memory_armed = True
        else:
            del detector._dangerous_tracks[track_id]

    return {
        "armed": armed,
        "dangerous": dangerous,
        "weapon_alert": weapon_alert,
        "frame_armed": raw_frame_armed,
        "voted_armed": voted_armed,
        "memory_armed": memory_armed,
        "weapon_armed_frames": armed_frame_count,
        "weapon_types": types,
        "weapon_status": cls["weapon_status"],
        "weapon_label": cls["weapon_label"],
        "weapon_score": max((float(w.get("fusion_score") or 0) for w in weapons), default=0.0)
        if weapons
        else 0.0,
        "person_bbox": list(person.get("person_bbox") or []),
        "weapons": weapons,
    }


def _extract_class_names(model: Any) -> Dict[int, str]:
    try:
        names = model.names
        if isinstance(names, dict) and names:
            return {int(k): str(v) for k, v in names.items()}
    except Exception:
        pass
    return dict(WEAPON_CLASS_NAMES)


def _clamp_box(x1: int, y1: int, x2: int, y2: int, W: int, H: int) -> List[int]:
    return [
        max(0, min(x1, W - 1)),
        max(0, min(y1, H - 1)),
        max(1, min(x2, W)),
        max(1, min(y2, H)),
    ]


def weapon_detection_available() -> bool:
    if not s.IVM_WEAPON_ENABLED:
        return False
    return Path(s.IVM_WEAPON_MODEL).is_file()


def weapon_info_for_track(
    weapon_by_track: Dict[int, Dict[str, Any]],
    track_id: Optional[int],
    *,
    person_bbox: Optional[List[int]] = None,
) -> Dict[str, Any]:
    if track_id is None:
        info = dict(_UNARMED)
        if person_bbox:
            info["person_bbox"] = list(person_bbox)
        return info
    w = weapon_by_track.get(int(track_id))
    if not w:
        info = dict(_UNARMED)
        if person_bbox:
            info["person_bbox"] = list(person_bbox)
        return info
    return dict(w)


def _get_stack(camera_id: str) -> Tuple[_FrameWeaponDetector, _TemporalVoter]:
    cam = str(camera_id)
    with _detector_lock:
        if cam in _detectors:
            return _detectors[cam], _voters[cam]
    logger.info("[%s] Loading weapon full-frame detector (có thể tạm dừng log camera khác)…", cam)
    t0 = time.perf_counter()
    new_det = _FrameWeaponDetector()
    new_voter = _TemporalVoter(
        window=s.IVM_WEAPON_VOTER_WINDOW,
        threshold=s.IVM_WEAPON_VOTER_THRESHOLD,
    )
    with _detector_lock:
        if cam not in _detectors:
            _detectors[cam] = new_det
            _voters[cam] = new_voter
        else:
            new_det = _detectors[cam]
            new_voter = _voters[cam]
    logger.info(
        "[%s] Weapon detector ready (%.1fs)",
        cam,
        time.perf_counter() - t0,
    )
    return new_det, new_voter


def detect_weapons_on_person_tracks(
    frame_bgr: np.ndarray,
    camera_id: str,
    person_tracks: List[List[Any]],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, float]]:
    """
    Sau khi có person_tracks: kiểm tra vũ khí trong ROI từng người.
    Trả {track_id: weapon_info}, timing_ms.
    """
    if not weapon_detection_available() or not person_tracks:
        return {}, {}
    try:
        detector, voter = _get_stack(camera_id)
        return detector.detect_on_person_tracks(frame_bgr, person_tracks, voter)
    except Exception as ex:
        logger.warning("[%s] weapon detect failed: %s", camera_id, ex)
        return {}, {"error": 0.0}


def detect_weapons_on_frame(
    frame_bgr: np.ndarray,
    camera_id: str,
    *,
    person_tracks: Optional[List[List[Any]]] = None,
    person_tracker: Optional[Any] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, float]]:
    """Tương thích cũ — yêu cầu đã có person_tracks (không detect toàn khung độc lập)."""
    tracks = list(person_tracks or [])
    if not tracks and person_tracker is not None:
        try:
            tracks = person_tracker.detect_person_tracks(frame_bgr) or []
        except Exception as ex:
            logger.debug("[%s] person_tracks skipped: %s", camera_id, ex)
    if not tracks:
        return None, {}

    weapon_by_track, timing = detect_weapons_on_person_tracks(frame_bgr, camera_id, tracks)
    H, W = frame_bgr.shape[:2]
    persons = []
    for pt in tracks:
        if len(pt) < 5:
            continue
        tid = int(pt[4])
        pb = list(_clamp_box(int(pt[0]), int(pt[1]), int(pt[2]), int(pt[3]), W, H))
        w = weapon_by_track.get(tid, dict(_UNARMED))
        persons.append(
            {
                "person_id": tid,
                "person_bbox": pb,
                "armed": w.get("armed"),
                "frame_armed": w.get("frame_armed"),
                "voted_armed": w.get("voted_armed"),
                "memory_armed": w.get("memory_armed"),
                "weapons": w.get("weapons") or [],
            }
        )
    armed = sum(1 for p in persons if p.get("armed"))
    dangerous = sum(1 for p in persons if p.get("dangerous"))
    scene_status = "DANGEROUS" if dangerous else ("ARMED" if armed else "SAFE")
    return (
        {
            "image_status": scene_status,
            "timing_ms": timing,
            "persons": persons,
            "summary": {"total_persons": len(persons), "armed_persons": armed},
        },
        timing,
    )


def _bbox_iou(a: List[float], b: List[int]) -> float:
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


def _face_center_in_person(face_bbox: List[float], person_bbox: List[int]) -> bool:
    fx = (float(face_bbox[0]) + float(face_bbox[2])) / 2.0
    fy = (float(face_bbox[1]) + float(face_bbox[3])) / 2.0
    px1, py1, px2, py2 = person_bbox
    return px1 <= fx <= px2 and py1 <= fy <= py2


def match_weapon_to_faces(
    faces: List[Dict[str, Any]],
    weapon_result: Optional[Dict[str, Any]],
    *,
    min_iou: float = 0.02,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Gắn weapon theo track_id hoặc person_bbox (fallback IoU mặt)."""
    persons = (weapon_result or {}).get("persons") or []
    by_track: Dict[int, Dict[str, Any]] = {}
    for p in persons:
        tid = p.get("person_id")
        if tid is not None:
            by_track[int(tid)] = p

    scene_armed = int((weapon_result or {}).get("summary", {}).get("armed_persons") or 0)
    image_status = str((weapon_result or {}).get("image_status") or "SAFE")
    timing = (weapon_result or {}).get("timing_ms") or {}
    scene = {
        "image_status": image_status,
        "armed_persons": scene_armed,
        "total_persons": int((weapon_result or {}).get("summary", {}).get("total_persons") or len(persons)),
        "weapon_ms": float(timing.get("weapon") or timing.get("total") or 0.0),
        "weapon_timing_ms": timing,
    }

    if not faces:
        return faces, scene

    out: List[Dict[str, Any]] = []
    for face in faces:
        tid = face.get("track_id")
        pb = face.get("person_bbox")
        weapon_info = dict(_UNARMED)
        if tid is not None and int(tid) in by_track:
            weapon_info = _weapon_from_person_row(by_track[int(tid)])
        elif pb and len(pb) >= 4 and persons:
            best_p = None
            best_score = 0.0
            for p in persons:
                pbb = p.get("person_bbox")
                if not pbb or len(pbb) < 4:
                    continue
                fb = face.get("bbox") or []
                if len(fb) < 4:
                    continue
                iou = _bbox_iou(list(fb), list(pbb))
                center_hit = _face_center_in_person(list(fb), list(pbb))
                score = iou + (0.15 if center_hit else 0.0)
                if score > best_score:
                    best_score = score
                    best_p = p
            if best_p and (
                best_score >= min_iou
                or _face_center_in_person(list(face.get("bbox") or []), list(best_p["person_bbox"]))
            ):
                weapon_info = _weapon_from_person_row(best_p)
        out.append({**face, "weapon": weapon_info})
    return out, scene


def _weapon_from_person_row(person: Dict[str, Any]) -> Dict[str, Any]:
    weapons = person.get("weapons") or []
    types = sorted({str(w.get("class")) for w in weapons if w.get("class")})
    armed = bool(person.get("armed"))
    dangerous = bool(person.get("dangerous"))
    label = str(person.get("weapon_label") or "Không vũ khí")
    status = str(person.get("weapon_status") or ("co_vu_khi" if armed else "an_toan"))
    if not label or label == "Không vũ khí":
        if armed and types:
            label = f"Có vũ khí ({', '.join(types)})"
        elif armed:
            label = "Có vũ khí"
    return {
        "armed": armed,
        "dangerous": dangerous,
        "weapon_alert": bool(person.get("weapon_alert")),
        "frame_armed": bool(person.get("frame_armed")),
        "voted_armed": bool(person.get("voted_armed")),
        "memory_armed": bool(person.get("memory_armed")),
        "weapon_armed_frames": int(person.get("weapon_armed_frames") or 0),
        "weapon_types": types,
        "weapon_status": status,
        "weapon_label": label,
        "weapon_score": max((float(w.get("fusion_score") or 0) for w in weapons), default=0.0)
        if weapons
        else 0.0,
        "person_bbox": list(person.get("person_bbox") or []),
        "weapons": list(weapons),
    }


def release_weapon_detector(camera_id: str) -> bool:
    det = None
    with _detector_lock:
        det = _detectors.pop(camera_id, None)
        _voters.pop(camera_id, None)
    if det is None:
        return False
    _dispose_detector_instance(det)
    del det
    logger.info("[%s] Đã giải phóng model vũ khí", camera_id)
    return True


def release_weapon_detectors() -> None:
    with _detector_lock:
        items = list(_detectors.items())
        _detectors.clear()
        _voters.clear()
    for _cid, det in items:
        _dispose_detector_instance(det)
    if items:
        from module_ai.engine.gpu_cleanup import maybe_gpu_soft_cleanup

        maybe_gpu_soft_cleanup(log_label="weapon:all")
        logger.info("Đã giải phóng %d model vũ khí", len(items))


def _dispose_detector_instance(det: Any) -> None:
    from module_ai.engine.gpu_cleanup import dispose_ultralytics_yolo

    if det is None:
        return
    try:
        if hasattr(det, "_model"):
            dispose_ultralytics_yolo(det._model)
            det._model = None
    except Exception:
        pass
