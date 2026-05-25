"""YOLO person + track (chính) → vũ khí trên ROI người → mặt → recognition."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from module_ai.config import settings as s
from module_ai.camera.infer import resize_for_analyze, scale_bbox_xyxy
from module_ai.camera.weapon import (
    detect_weapons_on_person_tracks,
    weapon_detection_available,
    weapon_info_for_track,
)
from module_ai.engine.insightface_engine import InsightFaceEngine
from module_ai.engine.yolo_person_tracker import assign_faces_to_person_tracks_multi


def detect_faces_on_frame(
    engine: InsightFaceEngine,
    frame_bgr: np.ndarray,
    *,
    max_w: int,
) -> Tuple[float, List[np.ndarray], List[Dict[str, Any]]]:
    """
    Detect + align mặt trên khung (resize theo max_w).
    Trả (detect_ms, aligned_crops, meta[]) với meta[i]: bbox_xyxy full frame, det_score.
    """
    small, sx, sy = resize_for_analyze(frame_bgr, max_w)
    inv_sx, inv_sy = (1.0 / sx), (1.0 / sy)

    detect_ms, aligned, det_meta = engine.detect_align_faces(small)

    face_metas: List[Dict[str, Any]] = []
    for j, (bbox, det_score) in enumerate(det_meta):
        face_metas.append(
            {
                "bbox": scale_bbox_xyxy(list(bbox[:4]), inv_sx, inv_sy),
                "det_score": float(det_score),
                "aligned_index": j,
                "aligned_crop": aligned[j] if j < len(aligned) else None,
            }
        )
    return detect_ms, aligned, face_metas


def plan_person_face_assignments(
    person_tracks: List[List[Any]],
    face_metas: List[Dict[str, Any]],
    *,
    frame_bgr: Optional[np.ndarray] = None,
    iou_min: Optional[float] = None,
    use_center_in_person: Optional[bool] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Một entry cho mỗi cặp (person track, face).
    Nếu một box detect có >= IVM_FACE_POSE_MIN_FACES mặt và bật pose refine,
    chạy YOLO-pose để tách theo keypoint đầu (hướng B).
    Trả (assignments, pose_ms).
    """
    if iou_min is None:
        iou_min = float(s.IVM_FACE_ASSIGN_IOU_MIN)
    if use_center_in_person is None:
        use_center_in_person = bool(s.IVM_FACE_ASSIGN_CENTER_IN_PERSON)

    face_boxes = [list(m["bbox"][:4]) for m in face_metas if m.get("bbox")]
    grouped = assign_faces_to_person_tracks_multi(
        person_tracks,
        face_boxes,
        iou_min=float(iou_min),
        use_center_in_person=bool(use_center_in_person),
    )

    pose_ms = 0.0
    if frame_bgr is not None and bool(s.IVM_FACE_POSE_REFINE):
        from module_ai.engine.pose_face_refine import refine_grouped_with_pose

        grouped, pose_ms = refine_grouped_with_pose(frame_bgr, grouped, face_boxes)

    out: List[Dict[str, Any]] = []
    for person_track, face_indices in grouped:
        if not face_indices:
            continue
        for fi in face_indices:
            if fi < 0 or fi >= len(face_metas):
                continue
            fm = face_metas[int(fi)]
            crop = fm.get("aligned_crop")
            if crop is None:
                continue
            out.append(
                {
                    "person_track": person_track,
                    "face_meta": fm,
                    "face_index": int(fi),
                    "has_face": True,
                }
            )
    return out, float(pose_ms)


def run_person_track_weapon_face_pipeline(
    frame_bgr: np.ndarray,
    *,
    person_tracker: Any,
    engine: InsightFaceEngine,
    camera_id: str = "default",
    max_w: int,
    run_weapon: bool = True,
) -> Tuple[List[List[Any]], Dict[int, Dict[str, Any]], List[Dict[str, Any]], float, Dict[str, float]]:
    """
    Luồng chính một khung:
      1) YOLO person + ByteTrack
      2) Weapon toàn khung → ghép vào track (nếu bật)
      3) Detect/align mặt + ghép vào track

    Trả: person_tracks, weapon_by_track, assignments, detect_ms, timing_partial.
    """
    t0 = time.perf_counter()
    try:
        person_tracks = person_tracker.detect_person_tracks(frame_bgr) or []
    except Exception:
        person_tracks = []
    person_ms = (time.perf_counter() - t0) * 1000.0

    timing: Dict[str, float] = {"person_track_ms": person_ms}
    weapon_by_track: Dict[int, Dict[str, Any]] = {}
    if person_tracks and run_weapon and weapon_detection_available():
        weapon_by_track, w_timing = detect_weapons_on_person_tracks(
            frame_bgr, camera_id, person_tracks
        )
        timing["weapon_ms"] = float(w_timing.get("total") or w_timing.get("weapon") or 0.0)

    if not person_tracks:
        return [], weapon_by_track, [], 0.0, timing

    detect_ms, _aligned, face_metas = detect_faces_on_frame(engine, frame_bgr, max_w=max_w)
    timing["detect_ms"] = float(detect_ms)

    assignments, pose_ms = plan_person_face_assignments(
        person_tracks, face_metas, frame_bgr=frame_bgr
    )
    timing["pose_refine_ms"] = float(pose_ms)

    for asn in assignments:
        pt = asn.get("person_track") or []
        tid = int(pt[4]) if len(pt) > 4 else None
        pb = list(pt[:4]) if len(pt) >= 4 else None
        asn["weapon"] = weapon_info_for_track(weapon_by_track, tid, person_bbox=pb)
        asn["track_id"] = tid

    return person_tracks, weapon_by_track, assignments, float(detect_ms), timing


def weapon_types_json(weapon: Optional[Dict[str, Any]]) -> Optional[str]:
    if not weapon:
        return None
    types = weapon.get("weapon_types") or []
    if not types:
        return None
    return json.dumps(list(types), ensure_ascii=False)
