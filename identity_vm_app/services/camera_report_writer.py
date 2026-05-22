"""Ghi báo cáo video_person_reports cho phiên camera live."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Optional

import numpy as np

from identity_vm_app import settings as s
from identity_vm_app.services import camera_session_media as cmedia
from identity_vm_app.services.video_match_candidates import serialize_match_candidates

# Mốc frame infer không có detection (tránh trùng ByteTrack id thật).
FRAME_MARKER_TRACKING_ID = 0


def _features_face_str(embedding: Any) -> Optional[str]:
    if embedding is None:
        return None
    try:
        return np.array_str(np.asarray(embedding, dtype=np.float32).reshape(-1))
    except Exception:
        return None


def _weapon_boxes_json(weapon: Optional[Dict[str, Any]]) -> Optional[str]:
    if not weapon:
        return None
    items = weapon.get("weapons") or []
    if not items:
        return None
    payload = []
    for w in items:
        bb = w.get("bbox")
        if not bb or len(bb) < 4:
            continue
        payload.append(
            {
                "bbox": [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
                "class": str(w.get("class") or "weapon"),
                "conf": float(w.get("conf") or w.get("fusion_score") or 0.0),
            }
        )
    return json.dumps(payload, ensure_ascii=False) if payload else None


def parse_box_literal(box_raw: Any) -> Optional[List[int]]:
    if box_raw is None:
        return None
    if isinstance(box_raw, (list, tuple)) and len(box_raw) >= 4:
        return [int(box_raw[0]), int(box_raw[1]), int(box_raw[2]), int(box_raw[3])]
    raw = str(box_raw).strip()
    if not raw:
        return None
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, (list, tuple)) and len(val) >= 4:
            return [int(val[0]), int(val[1]), int(val[2]), int(val[3])]
    except (SyntaxError, ValueError, TypeError):
        pass
    return None


def person_report_row(
    camera_id: str,
    job_id: str,
    *,
    t_s: float,
    frame_index: int,
    sample_index: int,
    video_clip: int,
    img_url: str,
    id_tracking: int,
    box_person: str,
    face: Optional[Dict[str, Any]] = None,
    weapon: Optional[Dict[str, Any]] = None,
    persist_embeddings: bool = False,
    save_crops: bool = False,
    frame: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    bb: List[float] = []
    face_img = None
    face_id = None
    display_name = None
    distance = None
    match_score = None
    match_candidates_json = None
    det_score = None
    gender = None
    age = None
    features_face = None
    box_face = None
    person_img = None
    pb_list = parse_box_literal(box_person)

    if face is not None:
        bb = list((face.get("bbox") or [])[:4])
        if len(bb) >= 4 and save_crops and frame is not None:
            face_img = cmedia.save_face_crop(camera_id, job_id, frame, bb)
        ms = face.get("matches") or []
        if ms:
            m0 = ms[0]
            face_id = m0.get("face_id")
            display_name = m0.get("name") or m0.get("display_name")
            distance = m0.get("distance")
            if distance is not None:
                match_score = 1.0 - float(distance)
        match_candidates_json = serialize_match_candidates(ms, limit=int(s.IVM_SEARCH_K))
        det_score = face.get("det_score")
        gender = face.get("gender")
        age = face.get("age")
        if persist_embeddings:
            features_face = _features_face_str(face.get("embedding"))
        box_face = f"{bb}" if len(bb) >= 4 else None

    armed = 0
    weapon_status = None
    weapon_label = None
    weapon_types_json = None
    weapon_score = None
    weapon_img = None
    weapon_crops_json = None
    weapon_boxes_json = None
    if weapon:
        frame_armed = bool(weapon.get("frame_armed")) or bool(weapon.get("weapons"))
        armed = 1 if frame_armed else 0
        weapon_status = weapon.get("weapon_status")
        weapon_label = weapon.get("weapon_label")
        types = weapon.get("weapon_types") or []
        if types:
            weapon_types_json = json.dumps(list(types), ensure_ascii=False)
        weapon_score = weapon.get("weapon_score")
        weapon_boxes_json = _weapon_boxes_json(weapon)
        if save_crops and frame is not None and pb_list and len(pb_list) >= 4:
            person_img = cmedia.save_person_crop(camera_id, job_id, frame, pb_list)
            if frame_armed:
                weapon_img, weapon_crops_json = cmedia.save_weapon_crops_for_session(
                    camera_id,
                    job_id,
                    frame,
                    list(weapon.get("weapons") or []),
                )

    return {
        "job_id": job_id,
        "time_analyze_s": float(t_s),
        "frame_index": frame_index,
        "sample_index": sample_index,
        "img_url": img_url,
        "id_tracking": id_tracking,
        "video_clip": video_clip,
        "face_id": face_id,
        "display_name": display_name,
        "distance": distance,
        "match_score": match_score,
        "match_candidates_json": match_candidates_json,
        "det_score": det_score,
        "gender": gender,
        "age": age,
        "box_face": box_face,
        "face_img": face_img,
        "box_person": box_person,
        "person_img": person_img,
        "features_face": features_face,
        "armed": armed,
        "weapon_status": weapon_status,
        "weapon_label": weapon_label,
        "weapon_types_json": weapon_types_json,
        "weapon_score": weapon_score,
        "weapon_img": weapon_img,
        "weapon_crops_json": weapon_crops_json,
        "weapon_boxes_json": weapon_boxes_json,
    }


def build_person_report_rows_from_assignments(
    camera_id: str,
    job_id: str,
    frame: np.ndarray,
    *,
    t_s: float,
    frame_index: int,
    sample_index: int,
    video_clip: int,
    assignments: List[Dict[str, Any]],
    faces_with_matches: List[Dict[str, Any]],
    img_url: Optional[str] = None,
    persist_embeddings: bool = False,
    save_crops: bool = False,
    log_all_infer_frames: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Ghi báo cáo DB cho mỗi frame infer (tùy IVM_REPORT_ALL_INFER_FRAMES)."""
    log_all = (
        bool(s.IVM_REPORT_ALL_INFER_FRAMES)
        if log_all_infer_frames is None
        else bool(log_all_infer_frames)
    )
    if not assignments:
        if not log_all:
            return []
        if img_url is None:
            img_url = cmedia.save_root_frame(camera_id, job_id, frame)
        return [
            person_report_row(
                camera_id,
                job_id,
                t_s=t_s,
                frame_index=frame_index,
                sample_index=sample_index,
                video_clip=video_clip,
                img_url=img_url,
                id_tracking=FRAME_MARKER_TRACKING_ID,
                box_person="[]",
                frame=frame,
            )
        ]
    if img_url is None:
        img_url = cmedia.save_root_frame(camera_id, job_id, frame)
    rows: List[Dict[str, Any]] = []
    fi = 0
    for asn in assignments:
        pt = asn.get("person_track") or []
        if len(pt) < 5:
            continue
        pb = list(pt[:4])
        local_tid = int(pt[4])
        weapon = asn.get("weapon")
        face_entry = None
        if asn.get("has_face"):
            if fi < len(faces_with_matches):
                face_entry = faces_with_matches[fi]
            fi += 1
            if not log_all:
                if not face_entry:
                    continue
        elif not log_all and not (weapon or {}).get("armed"):
            continue
        rows.append(
            person_report_row(
                camera_id,
                job_id,
                t_s=t_s,
                frame_index=frame_index,
                sample_index=sample_index,
                video_clip=video_clip,
                img_url=img_url,
                id_tracking=local_tid,
                box_person=f"{pb}",
                face=face_entry,
                weapon=weapon,
                persist_embeddings=persist_embeddings,
                save_crops=save_crops,
                frame=frame,
            )
        )
    return rows
