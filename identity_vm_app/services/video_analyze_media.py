"""Lưu ảnh phân tích video (root / face) — cấu trúc giống VideoMaster analyze/."""

from __future__ import annotations

import secrets
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from identity_vm_app import settings as s

# Khớp _face_display_crop (routes.py) — ảnh đăng ký face DB.
FACE_THUMB_PX = 160


def normalize_face_thumb(img: np.ndarray, *, out_size: int = FACE_THUMB_PX) -> np.ndarray:
    """Resize crop mặt báo cáo về vuông out_size×out_size (giống thumbnail DB)."""
    if img is None or img.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    return cv2.resize(img, (out_size, out_size))


def job_analyze_dir(job_id: str) -> Path:
    p = Path(s.IVM_VIDEO_ANALYZE_DIR) / "jobs" / job_id / "analyze"
    p.mkdir(parents=True, exist_ok=True)
    return p


def job_thumb_path(job_id: str) -> Path:
    p = Path(s.IVM_VIDEO_ANALYZE_DIR) / "jobs" / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p / "thumb.jpg"


def _rel(path: Path) -> str:
    root = Path(s.IVM_VIDEO_ANALYZE_DIR).resolve()
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _jpeg_q() -> int:
    return int(s.IVM_VIDEO_ANALYZE_JPEG_QUALITY)


def save_root_frame(job_id: str, frame_bgr: np.ndarray, *, prefix: str = "root_imgs") -> str:
    adir = job_analyze_dir(job_id)
    name = f"{prefix}_{int(time.time() * 1000)}_{secrets.token_hex(4)}.jpg"
    fp = adir / name
    cv2.imwrite(str(fp), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _jpeg_q()])
    return _rel(fp)


def save_face_crop(job_id: str, frame_bgr: np.ndarray, bbox_xyxy: List[float]) -> Optional[str]:
    crop = _crop_xyxy(frame_bgr, bbox_xyxy, pad=0.18)
    if crop is None:
        return None
    crop = normalize_face_thumb(crop)
    adir = job_analyze_dir(job_id)
    name = f"face_imgs_{int(time.time() * 1000)}_{secrets.token_hex(4)}.jpg"
    fp = adir / name
    cv2.imwrite(str(fp), crop, [int(cv2.IMWRITE_JPEG_QUALITY), _jpeg_q()])
    return _rel(fp)


def save_person_crop(job_id: str, frame_bgr: np.ndarray, bbox_xyxy: List[float]) -> Optional[str]:
    crop = _crop_xyxy(frame_bgr, bbox_xyxy, pad=0.05)
    if crop is None:
        return None
    adir = job_analyze_dir(job_id)
    name = f"person_imgs_{int(time.time() * 1000)}_{secrets.token_hex(4)}.jpg"
    fp = adir / name
    cv2.imwrite(str(fp), crop, [int(cv2.IMWRITE_JPEG_QUALITY), _jpeg_q()])
    return _rel(fp)


def save_weapon_bbox_crops(
    job_id: str,
    frame_bgr: np.ndarray,
    weapons: List[dict],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Lưu một crop mỗi loại (gun, knife, …).
    Trả (weapon_img chính — tương thích cũ, weapon_crops_json).
    """
    import json

    from identity_vm_app.services.weapon_crops import (
        normalize_weapon_class,
        render_weapon_bbox_crop_one_bgr,
        weapons_best_per_class,
    )

    items = weapons_best_per_class(weapons)
    if not items:
        return None, None
    adir = job_analyze_dir(job_id)
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
        rel = _rel(fp)
        saved.append({"class": cls, "path": rel})
        if primary is None:
            primary = rel
    if not saved:
        return None, None
    return primary, json.dumps(saved, ensure_ascii=False)


def save_weapon_scene_crop(
    job_id: str,
    frame_bgr: np.ndarray,
    person_bbox: List[float],
    weapons: List[dict],
) -> Optional[str]:
    del person_bbox
    primary, _ = save_weapon_bbox_crops(job_id, frame_bgr, weapons)
    return primary


def resolve_media_path(rel_or_abs: str) -> Optional[Path]:
    if not rel_or_abs:
        return None
    p = Path(rel_or_abs)
    if p.is_file():
        return p
    for root in (
        Path(s.IVM_VIDEO_ANALYZE_DIR).resolve(),
        Path(s.IVM_CAMERA_SESSION_DIR).resolve(),
    ):
        cand = (root / str(rel_or_abs).replace("\\", "/").lstrip("/")).resolve()
        try:
            cand.relative_to(root)
        except ValueError:
            continue
        if cand.is_file():
            return cand
    return None


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


def clear_analyze_artifacts(job_id: str) -> int:
    """Xóa ảnh báo cáo trong analyze/ (giữ source + thumb). Trả số file đã xóa."""
    adir = job_analyze_dir(job_id)
    if not adir.is_dir():
        return 0
    n = 0
    for fp in adir.glob("*"):
        if fp.is_file():
            try:
                fp.unlink()
                n += 1
            except OSError:
                pass
    return n


def write_thumb(job_id: str, frame_bgr: np.ndarray) -> str:
    fp = job_thumb_path(job_id)
    small = frame_bgr
    h, w = frame_bgr.shape[:2]
    if w > 480:
        scale = 480.0 / w
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    cv2.imwrite(str(fp), small, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return _rel(fp)
