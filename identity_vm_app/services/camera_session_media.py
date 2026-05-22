"""Lưu ảnh / video phiên nhận diện camera live."""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from identity_vm_app import settings as s
from identity_vm_app.services.video_analyze_media import (
    FACE_THUMB_PX,
    normalize_face_thumb,
)


def session_root(camera_id: str, job_id: str) -> Path:
    p = Path(s.IVM_CAMERA_SESSION_DIR) / str(camera_id) / str(job_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def session_mp4_path(camera_id: str, job_id: str) -> Path:
    return session_root(camera_id, job_id) / "session.mp4"


def session_raw_mkv_path(camera_id: str, job_id: str) -> Path:
    return session_root(camera_id, job_id) / "session_raw.mkv"


def session_overlay_mp4_path(camera_id: str, job_id: str) -> Path:
    return session_root(camera_id, job_id) / "session_overlay.mp4"


def session_thumb_path(camera_id: str, job_id: str) -> Path:
    return session_root(camera_id, job_id) / "thumb.jpg"


def analyze_dir(camera_id: str, job_id: str) -> Path:
    p = session_root(camera_id, job_id) / "analyze"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _jpeg_q() -> int:
    return int(s.IVM_VIDEO_ANALYZE_JPEG_QUALITY)


def _crop_xyxy(frame: np.ndarray, bbox: List[float], *, pad: float) -> Optional[np.ndarray]:
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        xi = max(0, int(x1 - bw * pad))
        yi = max(0, int(y1 - bh * pad))
        xa = min(w, int(x2 + bw * pad))
        ya = min(h, int(y2 + bh * pad))
        if xa <= xi or ya <= yi:
            return None
        return frame[yi:ya, xi:xa].copy()
    except Exception:
        return None


def save_root_frame(camera_id: str, job_id: str, frame_bgr: np.ndarray, *, prefix: str = "root_imgs") -> str:
    adir = analyze_dir(camera_id, job_id)
    name = f"{prefix}_{int(time.time() * 1000)}_{secrets.token_hex(4)}.jpg"
    fp = adir / name
    cv2.imwrite(str(fp), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _jpeg_q()])
    return str(fp.resolve())


def save_face_crop(
    camera_id: str, job_id: str, frame_bgr: np.ndarray, bbox_xyxy: List[float]
) -> Optional[str]:
    crop = _crop_xyxy(frame_bgr, bbox_xyxy, pad=0.18)
    if crop is None:
        return None
    crop = normalize_face_thumb(crop, out_size=FACE_THUMB_PX)
    adir = analyze_dir(camera_id, job_id)
    name = f"face_imgs_{int(time.time() * 1000)}_{secrets.token_hex(4)}.jpg"
    fp = adir / name
    cv2.imwrite(str(fp), crop, [int(cv2.IMWRITE_JPEG_QUALITY), _jpeg_q()])
    return str(fp.resolve())


def save_person_crop(
    camera_id: str, job_id: str, frame_bgr: np.ndarray, bbox_xyxy: List[float]
) -> Optional[str]:
    crop = _crop_xyxy(frame_bgr, bbox_xyxy, pad=0.05)
    if crop is None:
        return None
    adir = analyze_dir(camera_id, job_id)
    name = f"person_imgs_{int(time.time() * 1000)}_{secrets.token_hex(4)}.jpg"
    fp = adir / name
    cv2.imwrite(str(fp), crop, [int(cv2.IMWRITE_JPEG_QUALITY), _jpeg_q()])
    return str(fp.resolve())


def save_weapon_crops_for_session(
    camera_id: str,
    job_id: str,
    frame_bgr: np.ndarray,
    weapons: List[dict],
) -> tuple[Optional[str], Optional[str]]:
    """Trả (primary_abs_path, weapon_crops_json) — path tuyệt đối."""
    import json

    from identity_vm_app.services.weapon_crops import (
        normalize_weapon_class,
        render_weapon_bbox_crop_one_bgr,
        weapons_best_per_class,
    )

    items = weapons_best_per_class(weapons)
    if not items:
        return None, None
    adir = analyze_dir(camera_id, job_id)
    ts = int(time.time() * 1000)
    saved: List[dict] = []
    primary: Optional[str] = None
    for wb in items:
        crop = render_weapon_bbox_crop_one_bgr(frame_bgr, wb)
        if crop is None:
            continue
        cls = normalize_weapon_class(wb.get("class"))
        name = f"weapon_{cls}_{ts}_{secrets.token_hex(4)}.jpg"
        fp = adir / name
        cv2.imwrite(str(fp), crop, [int(cv2.IMWRITE_JPEG_QUALITY), _jpeg_q()])
        abs_path = str(fp.resolve())
        saved.append({"class": cls, "path": abs_path})
        if primary is None:
            primary = abs_path
    if not saved:
        return None, None
    return primary, json.dumps(saved, ensure_ascii=False)


def write_session_thumb(camera_id: str, job_id: str, frame_bgr: np.ndarray) -> str:
    fp = session_thumb_path(camera_id, job_id)
    small = frame_bgr
    h, w = frame_bgr.shape[:2]
    if w > 480:
        scale = 480.0 / w
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    cv2.imwrite(str(fp), small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return str(fp.resolve())
