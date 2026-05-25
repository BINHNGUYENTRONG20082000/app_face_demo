"""Cảnh báo vũ khí theo track khi camera live BẬT nhận diện (> N frame det)."""

from __future__ import annotations

import base64
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from module_ai.config import settings as s
from module_ai.camera.activity_log import record as log_activity
from identity_vm_app.services.weapon_track_status import weapon_alert_min_frames, weapon_should_alert

_lock = threading.Lock()
# scope_key = "{camera_id}:{job_id}" → track_id đã cảnh báo trong phiên
_fired: Dict[str, Set[int]] = {}
_recent: Deque[Dict[str, Any]] = deque(maxlen=120)
_active_by_camera: Dict[str, List[Dict[str, Any]]] = {}
# camera_id → tối đa N cảnh báo (mỗi lần fire 1 ảnh)
_history_by_camera: Dict[str, Deque[Dict[str, Any]]] = {}
_full_jpeg_by_alert_id: Dict[str, bytes] = {}


def _scope_key(camera_id: str, job_id: Optional[str]) -> str:
    return f"{str(camera_id)}:{str(job_id or 'live')}"


def make_alert_id(camera_id: str, ts_utc: float, track_id: int) -> str:
    return f"{camera_id}:{int(float(ts_utc) * 1000)}:{int(track_id)}"


def _history_deque(camera_id: str) -> Deque[Dict[str, Any]]:
    cam = str(camera_id)
    if cam not in _history_by_camera:
        n = max(1, int(s.IVM_WEAPON_ALERT_HISTORY_PER_CAMERA))
        _history_by_camera[cam] = deque(maxlen=n)
    return _history_by_camera[cam]


def reset_weapon_alerts(camera_id: str, *, job_id: Optional[str] = None) -> None:
    """Xóa trạng thái cảnh báo (phiên mới hoặc tắt nhận diện)."""
    cam = str(camera_id)
    with _lock:
        if job_id is not None:
            _fired.pop(_scope_key(cam, job_id), None)
            return
        prefix = f"{cam}:"
        for key in list(_fired.keys()):
            if key.startswith(prefix):
                _fired.pop(key, None)
        _active_by_camera.pop(cam, None)
        hist = _history_by_camera.pop(cam, None)
        if hist:
            for row in hist:
                aid = str(row.get("alert_id") or "")
                if aid:
                    _full_jpeg_by_alert_id.pop(aid, None)


def weapon_alert_history_by_camera(
    camera_id: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Lịch sử ảnh cảnh báo (mới nhất trước) — có thumb_jpeg_b64, không gửi full trong JSON."""
    with _lock:
        if camera_id is not None:
            hist = list(_history_deque(str(camera_id)))
            return {str(camera_id): [_public_row(dict(r)) for r in reversed(hist)]}
        out: Dict[str, List[Dict[str, Any]]] = {}
        for cam, dq in _history_by_camera.items():
            out[cam] = [_public_row(dict(r)) for r in reversed(list(dq))]
        return out


def get_alert_full_jpeg(alert_id: str) -> Optional[bytes]:
    with _lock:
        return _full_jpeg_by_alert_id.get(str(alert_id))


def recent_weapon_alerts(
    camera_id: Optional[str] = None,
    *,
    limit: int = 30,
    since_ts: float = 0.0,
) -> List[Dict[str, Any]]:
    lim = max(1, min(200, int(limit)))
    since = float(since_ts or 0.0)
    with _lock:
        rows = list(_recent)
    out: List[Dict[str, Any]] = []
    for row in reversed(rows):
        if since > 0 and float(row.get("ts_utc") or 0) < since:
            continue
        if camera_id is not None and str(row.get("camera_id")) != str(camera_id):
            continue
        out.append(_public_row(dict(row)))
        if len(out) >= lim:
            break
    return out


def active_weapon_alerts_by_camera() -> Dict[str, List[Dict[str, Any]]]:
    with _lock:
        return {
            cam: [_public_row(dict(r)) for r in list(rows)]
            for cam, rows in _active_by_camera.items()
        }


def _public_row(row: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(row)
    item["message"] = _alert_message(item)
    item["has_thumb"] = bool(item.get("thumb_jpeg_b64"))
    item["has_full"] = bool(item.get("alert_id"))
    return item


def _alert_message(row: Dict[str, Any]) -> str:
    tid = row.get("track_id")
    types = row.get("weapon_types") or []
    types_txt = ", ".join(str(t) for t in types) if types else "vũ khí"
    n = int(row.get("weapon_armed_frames") or 0)
    thr = weapon_alert_min_frames()
    return (
        f"Track #{tid}: phát hiện {types_txt} "
        f"> {thr} frame mẫu ({n} frame)"
    )


def _encode_alert_images(
    frame_bgr: np.ndarray,
    *,
    track_id: int,
    person_bbox: Optional[List[int]] = None,
) -> Tuple[Optional[str], Optional[bytes]]:
    """Thumb base64 cho poll UI + full JPEG bytes cho phóng to."""
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return None, None
    img = frame_bgr.copy()
    if person_bbox and len(person_bbox) >= 4:
        x1, y1, x2, y2 = (int(person_bbox[0]), int(person_bbox[1]), int(person_bbox[2]), int(person_bbox[3]))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 100, 255), 3)
        cv2.putText(
            img,
            f"ALERT #{track_id}",
            (max(4, x1), max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 100, 255),
            2,
            cv2.LINE_AA,
        )
    q = int(s.IVM_WEAPON_ALERT_SNAPSHOT_JPEG_QUALITY)
    thumb_w = int(s.IVM_WEAPON_ALERT_THUMB_MAX_WIDTH)
    full_w = int(s.IVM_WEAPON_ALERT_FULL_MAX_WIDTH)
    thumb_b64 = _jpeg_b64_resize(img, thumb_w, q)
    full_q = max(q, 92)
    full_bytes = _jpeg_bytes_resize(img, full_w, full_q) if full_w > 0 else _jpeg_bytes_native(img, full_q)
    return thumb_b64, full_bytes


def _jpeg_b64_resize(frame_bgr: np.ndarray, max_w: int, quality: int) -> Optional[str]:
    buf = _jpeg_bytes_resize(frame_bgr, max_w, quality)
    if not buf:
        return None
    return base64.b64encode(buf).decode("ascii")


def _jpeg_bytes_native(frame_bgr: np.ndarray, quality: int) -> Optional[bytes]:
    """JPEG full độ phân giải gốc — không resize (tránh mờ khi phóng to)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    q = max(60, min(98, int(quality)))
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), q, int(cv2.IMWRITE_JPEG_OPTIMIZE), 1],
    )
    return buf.tobytes() if ok else None


def _jpeg_bytes_resize(frame_bgr: np.ndarray, max_w: int, quality: int) -> Optional[bytes]:
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    mw = max(64, int(max_w))
    if w > mw:
        scale = mw / float(w)
        nh = max(1, int(round(h * scale)))
        small = cv2.resize(frame_bgr, (mw, nh), interpolation=cv2.INTER_AREA)
    else:
        small = frame_bgr
    q = max(60, min(98, int(quality)))
    ok, buf = cv2.imencode(
        ".jpg",
        small,
        [int(cv2.IMWRITE_JPEG_QUALITY), q, int(cv2.IMWRITE_JPEG_OPTIMIZE), 1],
    )
    return buf.tobytes() if ok else None


def _collect_track_weapons(
    faces: List[Dict[str, Any]],
    armed_persons: List[Dict[str, Any]],
    weapon_track_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[int, Dict[str, Any]]:
    by_tid: Dict[int, Dict[str, Any]] = {}
    for face in faces:
        tid = face.get("track_id")
        if tid is None:
            continue
        w = face.get("weapon") or {}
        if w:
            by_tid[int(tid)] = w
    for person in armed_persons:
        tid = person.get("track_id")
        if tid is None:
            continue
        w = person.get("weapon") or {}
        if w:
            by_tid[int(tid)] = w
    for row in weapon_track_rows or []:
        tid = row.get("track_id")
        if tid is None:
            continue
        w = row if "weapon_armed_frames" in row else (row.get("weapon") or row)
        if w:
            by_tid[int(tid)] = w
    return by_tid


def emit_weapon_track_alerts(
    camera_id: str,
    *,
    job_id: Optional[str],
    faces: List[Dict[str, Any]],
    armed_persons: List[Dict[str, Any]],
    weapon_track_rows: Optional[List[Dict[str, Any]]] = None,
    frame_bgr: Optional[np.ndarray] = None,
    frame_count: Optional[int] = None,
    t_analyze_s: Optional[float] = None,
    sample_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not bool(s.IVM_WEAPON_ALERT_ENABLED):
        return []

    tracks = _collect_track_weapons(faces, armed_persons, weapon_track_rows)
    if not tracks:
        return []

    scope = _scope_key(camera_id, job_id)
    thr = weapon_alert_min_frames()
    fired: List[Dict[str, Any]] = []

    with _lock:
        seen = _fired.setdefault(scope, set())
        hist = _history_deque(camera_id)
        for tid, weapon in tracks.items():
            n_frames = int(weapon.get("weapon_armed_frames") or 0)
            if not weapon_should_alert(n_frames):
                continue
            if int(tid) in seen:
                continue
            seen.add(int(tid))
            types = list(weapon.get("weapon_types") or [])
            types_txt = ", ".join(types) if types else "vũ khí"
            ts = time.time()
            alert_id = make_alert_id(camera_id, ts, int(tid))
            pb = list(weapon.get("person_bbox") or [])
            thumb_b64, full_bytes = (None, None)
            if frame_bgr is not None:
                thumb_b64, full_bytes = _encode_alert_images(
                    frame_bgr, track_id=int(tid), person_bbox=pb
                )
            if full_bytes:
                _full_jpeg_by_alert_id[alert_id] = full_bytes
            msg = (
                f"⚠ CẢNH BÁO track #{tid}: phát hiện {types_txt} "
                f"> {thr} frame mẫu ({n_frames} frame)"
            )
            row = {
                "alert_id": alert_id,
                "ts_utc": ts,
                "camera_id": str(camera_id),
                "job_id": job_id,
                "track_id": int(tid),
                "weapon_armed_frames": n_frames,
                "weapon_types": types,
                "weapon_status": weapon.get("weapon_status"),
                "weapon_label": weapon.get("weapon_label"),
                "frame_count": frame_count,
                "t_analyze_s": t_analyze_s,
                "sample_index": sample_index,
                "thumb_jpeg_b64": thumb_b64,
            }
            row["message"] = msg
            fired.append(row)
            _recent.append(dict(row))
            _active_by_camera.setdefault(str(camera_id), []).append(dict(row))
            if len(_active_by_camera[str(camera_id)]) > 20:
                _active_by_camera[str(camera_id)] = _active_by_camera[str(camera_id)][-20:]
            if len(hist) >= hist.maxlen and hist:
                old = hist[0]
                old_id = str(old.get("alert_id") or "")
                if old_id:
                    _full_jpeg_by_alert_id.pop(old_id, None)
            hist.append(dict(row))
            log_activity(
                str(camera_id),
                "weapon_alert",
                msg,
                level="warning",
                extra={k: v for k, v in row.items() if k != "thumb_jpeg_b64"},
            )

    return fired
