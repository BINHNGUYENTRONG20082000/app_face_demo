"""Infer + tra cứu + ingest sự kiện (in-process hoặc HTTP fallback)."""

from __future__ import annotations

import base64
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

from identity_vm_app import settings as s
from identity_vm_app.services.weapon_crops import (
    encode_track_scene_crop_b64,
    encode_weapon_bbox_crop_b64,
    encode_weapon_bbox_crops_b64,
)

logger = logging.getLogger("camera_recognition.infer")

_infer_lock = threading.Lock()


def resize_for_analyze(frame_bgr: np.ndarray, max_w: int) -> Tuple[np.ndarray, float, float]:
    h, w = frame_bgr.shape[:2]
    if max_w <= 0 or w <= max_w:
        return frame_bgr, 1.0, 1.0
    scale = max_w / float(w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, scale, scale


def scale_bbox_xyxy(bbox: List[float], inv_sx: float, inv_sy: float) -> List[float]:
    x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    return [x1 * inv_sx, y1 * inv_sy, x2 * inv_sx, y2 * inv_sy]


def _crop_face_jpeg_b64(frame: np.ndarray, bbox_xyxy: List[float], *, quality: int = 85) -> Optional[str]:
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy[:4]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        pad = 0.18
        xi = max(0, int(x1 - bw * pad))
        yi = max(0, int(y1 - bh * pad))
        xa = min(w, int(x2 + bw * pad))
        ya = min(h, int(y2 + bh * pad))
        if xa <= xi or ya <= yi:
            return None
        crop = frame[yi:ya, xi:xa]
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return None


def _weapon_ingest_crops_b64(
    frame_full: np.ndarray,
    *,
    person_bbox: Optional[List[float]],
    face_bbox: Optional[List[float]],
    weapons: List[Dict[str, Any]],
    quality: int,
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """(danh sách crop bbox theo loại gun/knife, ảnh scene)."""
    wlist = list(weapons or [])
    bbox_crops = encode_weapon_bbox_crops_b64(frame_full, wlist, quality=quality) if wlist else []
    scene_b64 = None
    if person_bbox and len(person_bbox) >= 4:
        scene_b64 = encode_track_scene_crop_b64(
            frame_full,
            [int(person_bbox[0]), int(person_bbox[1]), int(person_bbox[2]), int(person_bbox[3])],
            wlist,
            list(face_bbox) if face_bbox and len(face_bbox) >= 4 else None,
            quality=quality,
        )
    return bbox_crops, scene_b64


def _ingest_payload_for_face(
    face: Dict[str, Any],
    *,
    log_unknown: bool,
) -> Optional[Dict[str, Any]]:
    matches = face.get("matches") or []
    if not matches:
        if not log_unknown:
            return None
        return {
            "source": "stream",
            "person_ref": "unknown",
            "distance": None,
            "det_score": face.get("det_score"),
            "gender": face.get("gender"),
            "age": face.get("age"),
        }
    m0 = matches[0]
    fid = m0.get("face_id")
    pname = m0.get("name")
    dist = m0.get("distance")
    return {
        "source": "stream",
        "person_ref": str(int(fid)) if fid is not None else "unknown",
        "face_id": int(fid) if fid is not None else None,
        "display_name": pname,
        "distance": dist,
        "match_score": (1.0 - float(dist)) if dist is not None else None,
        "det_score": face.get("det_score"),
        "gender": face.get("gender"),
        "age": face.get("age"),
    }


def _faces_to_identify_payload(faces_det: List[Any], db: Any, thr: float) -> Tuple[List[Dict[str, Any]], float]:
    import time as _time

    if not faces_det:
        return [], 0.0
    t0 = _time.perf_counter()
    embs = np.stack(
        [np.asarray(f.embedding, dtype=np.float32).reshape(-1) for f in faces_det],
        axis=0,
    )
    all_matches = db.search_batch(embs, k=s.IVM_SEARCH_K, distance_threshold=thr)
    search_ms = (_time.perf_counter() - t0) * 1000.0
    out: List[Dict[str, Any]] = []
    for f, matches in zip(faces_det, all_matches):
        entry: Dict[str, Any] = {
            "bbox": f.bbox.tolist(),
            "det_score": f.det_score,
            "gender": f.gender,
            "age": f.age,
            "matches": matches,
        }
        emb = getattr(f, "embedding", None)
        if emb is not None:
            entry["embedding"] = np.asarray(emb, dtype=np.float32).reshape(-1)
        out.append(entry)
    return out, search_ms


def identify_frame_person_first(
    frame_bgr: np.ndarray,
    *,
    engine: Any,
    face_db: Any,
    person_tracker: Any,
    thr: float,
    max_width: int,
    camera_id: str = "default",
    run_weapon: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Luồng chính: person+track → weapon ROI từng người → mặt → embed/search.
    Mỗi face có person_bbox, track_id, weapon. Người có vũ khí không mặt → timing['armed_persons'].
    """
    from identity_vm_app.camera_recognition.weapon import weapon_detection_available
    from identity_vm_app.services.video_person_face_pipeline import (
        run_person_track_weapon_face_pipeline,
    )

    do_weapon = bool(run_weapon) and weapon_detection_available()
    person_tracks, weapon_by_track, assignments, _detect_ms, timing = (
        run_person_track_weapon_face_pipeline(
            frame_bgr,
            person_tracker=person_tracker,
            engine=engine,
            camera_id=camera_id,
            max_w=max_width,
            run_weapon=do_weapon,
        )
    )

    if not person_tracks:
        timing.setdefault("embedding_ms", 0.0)
        timing.setdefault("search_ms", 0.0)
        timing["infer_ms"] = float(timing.get("person_track_ms", 0.0))
        timing["n_persons"] = 0
        timing["armed_persons"] = []
        return [], timing

    crops: List[np.ndarray] = []
    for asn in assignments:
        if not asn.get("has_face"):
            continue
        crop = (asn.get("face_meta") or {}).get("aligned_crop")
        if crop is not None:
            crops.append(crop)
        else:
            asn["has_face"] = False

    matches_batch: List[List[Dict[str, Any]]] = []
    embed_ms = 0.0
    search_ms = 0.0
    if crops:
        feats, embed_ms = engine.embed_aligned_crops(crops, max_batch=int(s.IVM_REC_GET_FEAT_MAX_BATCH))
        mat = np.stack(
            [np.asarray(feats[j], dtype=np.float32).reshape(-1) for j in range(feats.shape[0])],
            axis=0,
        )
        t_s = time.perf_counter()
        matches_batch = face_db.search_batch(mat, k=s.IVM_SEARCH_K, distance_threshold=thr)
        search_ms = (time.perf_counter() - t_s) * 1000.0

    out: List[Dict[str, Any]] = []
    tracks_with_face: set[int] = set()
    mi = 0
    for asn in assignments:
        if not asn.get("has_face"):
            continue
        pt = asn.get("person_track") or []
        fm = asn.get("face_meta") or {}
        bb = fm.get("bbox")
        if bb is None or len(bb) < 4:
            continue
        tid = int(pt[4]) if len(pt) > 4 else None
        if tid is not None:
            tracks_with_face.add(tid)
        entry: Dict[str, Any] = {
            "bbox": list(bb[:4]),
            "det_score": fm.get("det_score"),
            "gender": None,
            "age": None,
            "matches": matches_batch[mi] if mi < len(matches_batch) else [],
            "person_bbox": list(pt[:4]),
            "track_id": tid,
            "weapon": asn.get("weapon"),
        }
        mi += 1
        out.append(entry)

    armed_persons: List[Dict[str, Any]] = []
    for pt in person_tracks:
        if len(pt) < 5:
            continue
        tid = int(pt[4])
        if tid in tracks_with_face:
            continue
        from identity_vm_app.camera_recognition.weapon import weapon_info_for_track

        weapon = weapon_info_for_track(weapon_by_track, tid, person_bbox=list(pt[:4]))
        if not weapon.get("armed"):
            continue
        armed_persons.append(
            {
                "track_id": tid,
                "person_bbox": list(pt[:4]),
                "weapon": weapon,
            }
        )

    armed_n = sum(1 for p in weapon_by_track.values() if p.get("armed"))
    dangerous_n = sum(1 for p in weapon_by_track.values() if p.get("dangerous"))
    alert_n = sum(1 for p in weapon_by_track.values() if p.get("weapon_alert"))
    scene_status = "DANGEROUS" if dangerous_n else ("ARMED" if armed_n else "SAFE")
    timing.update(
        {
            "embedding_ms": float(embed_ms),
            "search_ms": float(search_ms),
            "n_persons": len(person_tracks),
            "armed_persons": armed_persons,
            "armed_tracks": armed_n,
            "weapon_alert_tracks": alert_n,
            "weapon_track_rows": [
                {"track_id": int(tid), **dict(w)} for tid, w in weapon_by_track.items()
            ],
            "weapon_scene": {
                "image_status": scene_status,
                "armed_persons": armed_n,
                "dangerous_persons": dangerous_n,
                "alert_persons": alert_n,
                "total_persons": len(person_tracks),
            },
        }
    )
    timing["infer_ms"] = (
        float(timing.get("person_track_ms", 0.0))
        + float(timing.get("detect_ms", 0.0))
        + float(timing.get("pose_refine_ms", 0.0))
        + float(timing.get("weapon_ms", 0.0))
        + float(embed_ms)
        + float(search_ms)
    )
    return out, timing


def identify_frame_person_first_detailed(
    frame_bgr: np.ndarray,
    *,
    engine: Any,
    face_db: Any,
    person_tracker: Any,
    thr: float,
    max_width: int,
    camera_id: str = "default",
    run_weapon: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
    """
    Giống identify_frame_person_first nhưng trả thêm assignments + faces_with_matches
    để ghi video_person_reports.
    """
    from identity_vm_app.camera_recognition.weapon import weapon_detection_available
    from identity_vm_app.services.video_person_face_pipeline import (
        run_person_track_weapon_face_pipeline,
    )

    do_weapon = bool(run_weapon) and weapon_detection_available()
    person_tracks, weapon_by_track, assignments, _detect_ms, timing = (
        run_person_track_weapon_face_pipeline(
            frame_bgr,
            person_tracker=person_tracker,
            engine=engine,
            camera_id=camera_id,
            max_w=max_width,
            run_weapon=do_weapon,
        )
    )

    if not person_tracks:
        timing.setdefault("embedding_ms", 0.0)
        timing.setdefault("search_ms", 0.0)
        timing["infer_ms"] = float(timing.get("person_track_ms", 0.0))
        timing["n_persons"] = 0
        timing["armed_persons"] = []
        return [], assignments, [], timing

    crops: List[np.ndarray] = []
    for asn in assignments:
        if not asn.get("has_face"):
            continue
        crop = (asn.get("face_meta") or {}).get("aligned_crop")
        if crop is not None:
            crops.append(crop)
        else:
            asn["has_face"] = False

    matches_batch: List[List[Dict[str, Any]]] = []
    embed_ms = 0.0
    search_ms = 0.0
    feats: Optional[np.ndarray] = None
    if crops:
        feats, embed_ms = engine.embed_aligned_crops(crops, max_batch=int(s.IVM_REC_GET_FEAT_MAX_BATCH))
        mat = np.stack(
            [np.asarray(feats[j], dtype=np.float32).reshape(-1) for j in range(feats.shape[0])],
            axis=0,
        )
        t_s = time.perf_counter()
        matches_batch = face_db.search_batch(mat, k=s.IVM_SEARCH_K, distance_threshold=thr)
        search_ms = (time.perf_counter() - t_s) * 1000.0

    faces_display: List[Dict[str, Any]] = []
    faces_with_matches: List[Dict[str, Any]] = []
    tracks_with_face: set[int] = set()
    embed_i = 0
    for asn in assignments:
        if not asn.get("has_face"):
            continue
        pt = asn.get("person_track") or []
        fm = asn.get("face_meta") or {}
        bb = fm.get("bbox")
        if bb is None or len(bb) < 4:
            continue
        tid = int(pt[4]) if len(pt) > 4 else None
        if tid is not None:
            tracks_with_face.add(tid)
        matches = matches_batch[embed_i] if embed_i < len(matches_batch) else []
        face_entry: Dict[str, Any] = {
            "bbox": list(bb[:4]),
            "det_score": fm.get("det_score"),
            "gender": None,
            "age": None,
            "matches": matches,
            "person_bbox": list(pt[:4]),
            "track_id": tid,
            "weapon": asn.get("weapon"),
        }
        if feats is not None and embed_i < feats.shape[0]:
            face_entry["embedding"] = np.asarray(feats[embed_i], dtype=np.float32).reshape(-1)
        else:
            emb = fm.get("embedding")
            if emb is not None:
                face_entry["embedding"] = np.asarray(emb, dtype=np.float32).reshape(-1)
        embed_i += 1
        faces_with_matches.append(face_entry)
        faces_display.append(dict(face_entry))

    armed_persons: List[Dict[str, Any]] = []
    for pt in person_tracks:
        if len(pt) < 5:
            continue
        tid = int(pt[4])
        if tid in tracks_with_face:
            continue
        from identity_vm_app.camera_recognition.weapon import weapon_info_for_track

        weapon = weapon_info_for_track(weapon_by_track, tid, person_bbox=list(pt[:4]))
        if not weapon.get("armed"):
            continue
        armed_persons.append(
            {
                "track_id": tid,
                "person_bbox": list(pt[:4]),
                "weapon": weapon,
            }
        )

    armed_n = sum(1 for p in weapon_by_track.values() if p.get("armed"))
    dangerous_n = sum(1 for p in weapon_by_track.values() if p.get("dangerous"))
    alert_n = sum(1 for p in weapon_by_track.values() if p.get("weapon_alert"))
    scene_status = "DANGEROUS" if dangerous_n else ("ARMED" if armed_n else "SAFE")
    timing.update(
        {
            "embedding_ms": float(embed_ms),
            "search_ms": float(search_ms),
            "n_persons": len(person_tracks),
            "armed_persons": armed_persons,
            "armed_tracks": armed_n,
            "weapon_alert_tracks": alert_n,
            "weapon_track_rows": [
                {"track_id": int(tid), **dict(w)} for tid, w in weapon_by_track.items()
            ],
            "weapon_scene": {
                "image_status": scene_status,
                "armed_persons": armed_n,
                "dangerous_persons": dangerous_n,
                "alert_persons": alert_n,
                "total_persons": len(person_tracks),
            },
        }
    )
    timing["infer_ms"] = (
        float(timing.get("person_track_ms", 0.0))
        + float(timing.get("detect_ms", 0.0))
        + float(timing.get("pose_refine_ms", 0.0))
        + float(timing.get("weapon_ms", 0.0))
        + float(embed_ms)
        + float(search_ms)
    )
    return faces_display, assignments, faces_with_matches, timing


def identify_frame_in_process(
    frame_bgr: np.ndarray,
    thr: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    from identity_vm_app.api.deps import get_engine, get_face_db

    infer_prof: Dict[str, float] = {}
    with _infer_lock:
        faces_det = get_engine().analyze_bgr(frame_bgr, timing_out=infer_prof)
        payload, search_ms = _faces_to_identify_payload(faces_det, get_face_db(), thr)
    timing = {
        "detect_ms": float(infer_prof.get("detect_ms", 0.0)),
        "embedding_ms": float(infer_prof.get("embedding_ms", 0.0)),
        "search_ms": float(search_ms),
    }
    return payload, timing


def identify_frame_via_http(
    frame_bgr: np.ndarray,
    api_base: str,
    thr: float,
    *,
    timeout_s: float,
    jpeg_quality: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    enc_ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not enc_ok:
        raise RuntimeError("JPEG encode failed")
    url = f"{api_base.rstrip('/')}/ivm/identify_image"
    t0 = time.perf_counter()
    r = requests.post(
        url,
        files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
        params={"distance_threshold": thr},
        timeout=timeout_s,
    )
    r.raise_for_status()
    data = r.json()
    elapsed = (time.perf_counter() - t0) * 1000.0
    timing_img = ((data.get("timing") or {}).get("images") or [{}])[0]
    return data.get("faces") or [], {
        "detect_ms": float(timing_img.get("detect_ms", 0.0)),
        "embedding_ms": float(timing_img.get("embedding_ms", 0.0)),
        "search_ms": float(timing_img.get("search_ms", 0.0)),
        "http_ms": elapsed,
    }


def identify_and_scale(
    frame_bgr: np.ndarray,
    *,
    api_base: str,
    thr: float,
    max_width: int,
    use_in_process: bool,
    http_timeout_s: float,
    jpeg_quality: int,
    person_tracker: Optional[Any] = None,
    camera_id: str = "default",
    run_weapon: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, float], float, float]:
    t0 = time.perf_counter()
    use_person_first = bool(s.IVM_USE_PERSON_FIRST_PIPELINE) and person_tracker is not None
    try:
        if use_in_process:
            from identity_vm_app.state import state

            if state.engine is not None and state.face_db is not None:
                if use_person_first:
                    faces, timing = identify_frame_person_first(
                        frame_bgr,
                        engine=state.engine,
                        face_db=state.face_db,
                        person_tracker=person_tracker,
                        thr=thr,
                        max_width=max_width,
                        camera_id=camera_id,
                        run_weapon=run_weapon,
                    )
                    timing["infer_ms"] = (time.perf_counter() - t0) * 1000.0
                    return faces, timing, 1.0, 1.0
                small, sx, sy = resize_for_analyze(frame_bgr, max_width)
                inv_sx, inv_sy = (1.0 / sx), (1.0 / sy)
                faces, timing = identify_frame_in_process(small, thr)
                scaled: List[Dict[str, Any]] = []
                for f in faces:
                    bb = f.get("bbox")
                    if bb and len(bb) >= 4:
                        f = {**f, "bbox": scale_bbox_xyxy(list(bb), inv_sx, inv_sy)}
                    scaled.append(f)
                timing["infer_ms"] = (time.perf_counter() - t0) * 1000.0
                return scaled, timing, inv_sx, inv_sy
            faces, timing = identify_frame_via_http(
                frame_bgr, api_base, thr, timeout_s=http_timeout_s, jpeg_quality=jpeg_quality
            )
            timing["infer_ms"] = (time.perf_counter() - t0) * 1000.0
            return faces, timing, 1.0, 1.0
        faces, timing = identify_frame_via_http(
            frame_bgr, api_base, thr, timeout_s=http_timeout_s, jpeg_quality=jpeg_quality
        )
    except Exception:
        raise
    timing["infer_ms"] = (time.perf_counter() - t0) * 1000.0
    return faces, timing, 1.0, 1.0


def identify_scaled_with_engine(
    frame_bgr: np.ndarray,
    *,
    engine: Any,
    face_db: Any,
    thr: float,
    max_width: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Infer một khung với engine/db cho sẵn (luồng split video — không dùng khóa toàn cục)."""
    small, sx, sy = resize_for_analyze(frame_bgr, max_width)
    inv_sx, inv_sy = (1.0 / sx), (1.0 / sy)
    t0 = time.perf_counter()
    infer_prof: Dict[str, float] = {}
    faces_det = engine.analyze_bgr(small, timing_out=infer_prof)
    payload, search_ms = _faces_to_identify_payload(faces_det, face_db, thr)
    timing = {
        "detect_ms": float(infer_prof.get("detect_ms", 0.0)),
        "embedding_ms": float(infer_prof.get("embedding_ms", 0.0)),
        "search_ms": float(search_ms),
    }
    scaled: List[Dict[str, Any]] = []
    for f in payload:
        bb = f.get("bbox")
        if bb and len(bb) >= 4:
            f = {**f, "bbox": scale_bbox_xyxy(list(bb), inv_sx, inv_sy)}
        scaled.append(f)
    timing["infer_ms"] = (time.perf_counter() - t0) * 1000.0
    return scaled, timing


def identify_and_scale_in_process(
    frame_bgr: np.ndarray,
    *,
    thr: float,
    max_width: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Nhận diện một khung BGR chỉ trong process API (InsightFace + FaceDatabase).
    Dùng cho phân tích video — không gọi HTTP /identify_image.
  """
    from identity_vm_app.state import state

    if state.engine is None or state.face_db is None:
        raise RuntimeError(
            "Engine nhận diện chưa sẵn sàng trong process API. "
            "Chạy `python identity_vm_app/main.py` và đợi model InsightFace load xong."
        )

    small, sx, sy = resize_for_analyze(frame_bgr, max_width)
    inv_sx, inv_sy = (1.0 / sx), (1.0 / sy)
    t0 = time.perf_counter()
    faces, timing = identify_frame_in_process(small, thr)
    scaled: List[Dict[str, Any]] = []
    for f in faces:
        bb = f.get("bbox")
        if bb and len(bb) >= 4:
            f = {**f, "bbox": scale_bbox_xyxy(list(bb), inv_sx, inv_sy)}
        scaled.append(f)
    timing["infer_ms"] = (time.perf_counter() - t0) * 1000.0
    return scaled, timing


def identify_and_scale_detailed(
    frame_bgr: np.ndarray,
    *,
    api_base: str,
    thr: float,
    max_width: int,
    use_in_process: bool,
    http_timeout_s: float,
    jpeg_quality: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Giống identify_and_scale; face dict có thêm embedding khi in-process."""
    faces, timing, _, _ = identify_and_scale(
        frame_bgr,
        api_base=api_base,
        thr=thr,
        max_width=max_width,
        use_in_process=use_in_process,
        http_timeout_s=http_timeout_s,
        jpeg_quality=jpeg_quality,
    )
    return faces, timing


def ingest_armed_persons(
    camera_id: str,
    frame_full: np.ndarray,
    armed_persons: List[Dict[str, Any]],
    *,
    api_base: str,
    use_in_process: bool,
    ingest_timeout_s: float,
    jpeg_quality: int,
) -> int:
    """Ghi event người có vũ khí nhưng không detect được mặt (person_ref=track:id)."""
    if not armed_persons:
        return 0
    n = 0
    if use_in_process:
        try:
            from identity_vm_app.api.deps import get_recorders, get_store
            from identity_vm_app.recorder.registry import RecorderRegistry
            from identity_vm_app.store.sqlite_store import IdentityVmStore

            store: IdentityVmStore = get_store()
            recorders: RecorderRegistry = get_recorders()
            rec_opt = recorders.get(camera_id)
            ts = time.time()
            for person in armed_persons:
                weapon = person.get("weapon") or {}
                if not weapon.get("armed"):
                    continue
                tid = person.get("track_id")
                pb = person.get("person_bbox") or weapon.get("person_bbox")
                wlist = list(weapon.get("weapons") or [])
                weapon_crops_b64, scene_crop_b64 = _weapon_ingest_crops_b64(
                    frame_full,
                    person_bbox=list(pb) if pb else None,
                    face_bbox=None,
                    weapons=wlist,
                    quality=jpeg_quality,
                )
                seg_id = None
                off0 = off1 = None
                if rec_opt:
                    sid, path, t0, _ = rec_opt.current_archive_ref()
                    if sid is not None and path:
                        seg_id = sid
                        off0 = max(0.0, ts - t0)
                        off1 = off0 + float(s.IVM_EXPORT_CUT_MIN_DURATION_S)
                eid, merged = store.merge_or_insert_event(
                    debounce_s=s.IVM_EVENT_DEBOUNCE_S,
                    ts_utc=ts,
                    camera_id=camera_id,
                    source="stream_weapon",
                    person_ref=f"track:{tid}" if tid is not None else "armed_unknown",
                    face_id=None,
                    display_name=weapon.get("weapon_label"),
                    match_score=None,
                    distance=None,
                    det_score=None,
                    model_tag=s.IVM_MODEL_TAG,
                    recording_segment_id=seg_id,
                    offset_start_s=off0,
                    offset_end_s=off1,
                )
                store.apply_tracking_update(
                    eid,
                    merged=merged,
                    weapon=weapon,
                    weapon_crops_jpeg_b64=weapon_crops_b64 or None,
                    track_scene_crop_jpeg_b64=scene_crop_b64,
                )
                n += 1
            return n
        except Exception as ex:
            logger.debug("[%s] in-process armed ingest fallback HTTP: %s", camera_id, ex)

    base = api_base.rstrip("/")
    session = requests.Session()
    for person in armed_persons:
        weapon = person.get("weapon") or {}
        if not weapon.get("armed"):
            continue
        tid = person.get("track_id")
        pb = person.get("person_bbox") or weapon.get("person_bbox")
        payload: Dict[str, Any] = {
            "source": "stream_weapon",
            "person_ref": f"track:{tid}" if tid is not None else "armed_unknown",
            "display_name": weapon.get("weapon_label"),
            "armed": True,
            "weapon_types": list(weapon.get("weapon_types") or []),
            "weapon_status": weapon.get("weapon_status"),
            "weapon_label": weapon.get("weapon_label"),
            "weapon_score": weapon.get("weapon_score"),
        }
        wlist = list(weapon.get("weapons") or [])
        w_crops, scene_crop = _weapon_ingest_crops_b64(
            frame_full,
            person_bbox=list(pb) if pb else None,
            face_bbox=None,
            weapons=wlist,
            quality=jpeg_quality,
        )
        if w_crops:
            payload["weapon_crops_jpeg_b64"] = w_crops
        if scene_crop:
            payload["track_scene_crop_jpeg_b64"] = scene_crop
        url = f"{base}/ivm/cameras/{camera_id}/events/recognition"
        try:
            session.post(url, json=payload, timeout=ingest_timeout_s)
            n += 1
        except requests.RequestException as e:
            logger.warning("[%s] armed ingest failed: %s", camera_id, e)
    return n


def ingest_faces(
    camera_id: str,
    frame_full: np.ndarray,
    faces: List[Dict[str, Any]],
    *,
    api_base: str,
    use_in_process: bool,
    ingest_timeout_s: float,
    jpeg_quality: int,
    log_unknown: bool,
) -> int:
    """Ghi recognition_events; trả số event đã gửi."""
    n = 0
    if use_in_process:
        try:
            from identity_vm_app.api.deps import get_recorders, get_store
            from identity_vm_app.recorder.registry import RecorderRegistry
            from identity_vm_app.store.sqlite_store import IdentityVmStore

            store: IdentityVmStore = get_store()
            recorders: RecorderRegistry = get_recorders()
            rec_opt = recorders.get(camera_id)
            ts = time.time()
            for face in faces:
                payload = _ingest_payload_for_face(face, log_unknown=log_unknown)
                if payload is None:
                    continue
                bb = face.get("bbox")
                if bb and len(bb) >= 4:
                    crop_b64 = _crop_face_jpeg_b64(frame_full, list(bb), quality=jpeg_quality)
                    if crop_b64:
                        payload["crop_jpeg_b64"] = crop_b64
                        payload["bbox"] = [float(x) for x in bb[:4]]
                weapon = face.get("weapon") or {}
                scene_crop_b64 = None
                pb = weapon.get("person_bbox")
                wlist = list(weapon.get("weapons") or [])
                face_bb = face.get("bbox")
                if pb and len(pb) >= 4:
                    scene_crop_b64 = encode_track_scene_crop_b64(
                        frame_full,
                        [int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])],
                        wlist,
                        list(face_bb) if face_bb and len(face_bb) >= 4 else None,
                        quality=jpeg_quality,
                    )
                weapon_crops_b64: List[Dict[str, str]] = []
                frame_armed = bool(weapon.get("frame_armed")) or bool(wlist)
                if frame_armed and wlist:
                    weapon_crops_b64 = encode_weapon_bbox_crops_b64(
                        frame_full, wlist, quality=jpeg_quality
                    )
                seg_id = None
                off0 = off1 = None
                if rec_opt:
                    sid, path, t0, _ = rec_opt.current_archive_ref()
                    if sid is not None and path:
                        seg_id = sid
                        off0 = max(0.0, ts - t0)
                        off1 = off0 + float(s.IVM_EXPORT_CUT_MIN_DURATION_S)
                eid, merged = store.merge_or_insert_event(
                    debounce_s=s.IVM_EVENT_DEBOUNCE_S,
                    ts_utc=ts,
                    camera_id=camera_id,
                    source=str(payload.get("source", "stream")),
                    person_ref=str(payload["person_ref"]),
                    face_id=payload.get("face_id"),
                    display_name=payload.get("display_name"),
                    match_score=payload.get("match_score"),
                    distance=payload.get("distance"),
                    det_score=payload.get("det_score"),
                    model_tag=s.IVM_MODEL_TAG,
                    recording_segment_id=seg_id,
                    offset_start_s=off0,
                    offset_end_s=off1,
                    gender=payload.get("gender"),
                    age=payload.get("age"),
                )
                store.apply_tracking_update(
                    eid,
                    merged=merged,
                    det_score=payload.get("det_score"),
                    crop_jpeg_b64=payload.get("crop_jpeg_b64"),
                    bbox=payload.get("bbox"),
                    weapon=face.get("weapon"),
                    weapon_crops_jpeg_b64=weapon_crops_b64 or None,
                    track_scene_crop_jpeg_b64=scene_crop_b64,
                )
                n += 1
            return n
        except Exception as ex:
            logger.debug("[%s] in-process ingest fallback HTTP: %s", camera_id, ex)

    base = api_base.rstrip("/")
    session = requests.Session()
    for face in faces:
        payload = _ingest_payload_for_face(face, log_unknown=log_unknown)
        if payload is None:
            continue
        bb = face.get("bbox")
        if bb and len(bb) >= 4:
            crop_b64 = _crop_face_jpeg_b64(frame_full, list(bb), quality=jpeg_quality)
            if crop_b64:
                payload["crop_jpeg_b64"] = crop_b64
                payload["bbox"] = [float(x) for x in bb[:4]]
        weapon = face.get("weapon") or {}
        if weapon:
            payload["armed"] = bool(weapon.get("armed"))
            payload["frame_armed"] = bool(weapon.get("frame_armed"))
            payload["weapon_types"] = list(weapon.get("weapon_types") or [])
            payload["weapon_status"] = weapon.get("weapon_status")
            payload["weapon_label"] = weapon.get("weapon_label")
            payload["weapon_score"] = weapon.get("weapon_score")
            pb = weapon.get("person_bbox")
            wlist = list(weapon.get("weapons") or [])
            if pb and len(pb) >= 4:
                scene_crop = encode_track_scene_crop_b64(
                    frame_full,
                    [int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3])],
                    wlist,
                    list(bb) if bb and len(bb) >= 4 else None,
                    quality=jpeg_quality,
                )
                if scene_crop:
                    payload["track_scene_crop_jpeg_b64"] = scene_crop
            if (weapon.get("frame_armed") or wlist) and wlist:
                w_crops = encode_weapon_bbox_crops_b64(
                    frame_full, wlist, quality=jpeg_quality
                )
                if w_crops:
                    payload["weapon_crops_jpeg_b64"] = w_crops
        url = f"{base}/ivm/cameras/{camera_id}/events/recognition"
        try:
            session.post(url, json=payload, timeout=ingest_timeout_s)
            n += 1
        except requests.RequestException as e:
            logger.warning("[%s] ingest failed: %s", camera_id, e)
    return n
