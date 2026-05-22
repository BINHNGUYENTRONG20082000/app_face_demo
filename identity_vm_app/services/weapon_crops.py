"""Crop vũ khí / scene track (person + face + weapon boxes)."""

from __future__ import annotations

import base64
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from identity_vm_app import settings as s
from identity_vm_app.services.event_crops import CROPS_DIR, crop_weapon_file_for_event


def _clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    return (
        max(0, min(x1, w - 1)),
        max(0, min(y1, h - 1)),
        max(1, min(x2, w)),
        max(1, min(y2, h)),
    )


def normalize_weapon_class(class_name: Any) -> str:
    cls = str(class_name or "weapon").strip().lower()
    if cls in ("gun", "knife"):
        return cls
    return "weapon"


def _weapon_score(wb: Dict[str, Any]) -> float:
    return float(wb.get("conf") or wb.get("fusion_score") or 0.0)


def weapons_best_per_class(weapons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mỗi loại (gun, knife, …) giữ một bbox có điểm cao nhất — để crop riêng từng ảnh."""
    best: Dict[str, Dict[str, Any]] = {}
    for wb in weapons or []:
        bb = wb.get("bbox")
        if not bb or len(bb) < 4:
            continue
        cls = normalize_weapon_class(wb.get("class"))
        if cls not in best or _weapon_score(wb) > _weapon_score(best[cls]):
            best[cls] = wb
    order: List[str] = []
    for k in ("gun", "knife"):
        if k in best:
            order.append(k)
    for k in sorted(best.keys()):
        if k not in order:
            order.append(k)
    return [best[k] for k in order]


def pick_primary_weapon(weapons: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    items = weapons_best_per_class(weapons)
    return items[0] if items else None


def render_weapon_bbox_crop_one_bgr(
    frame_bgr: np.ndarray,
    weapon: Dict[str, Any],
    *,
    pad_ratio: Optional[float] = None,
) -> Optional[np.ndarray]:
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    bb = weapon.get("bbox")
    if not bb or len(bb) < 4:
        return None

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    pad = float(s.IVM_WEAPON_ROI_PAD_RATIO if pad_ratio is None else pad_ratio)
    xi, yi, xa, ya = _clamp_box(
        int(x1 - bw * pad),
        int(y1 - bh * pad),
        int(x2 + bw * pad),
        int(y2 + bh * pad),
        w,
        h,
    )
    if xa <= xi or ya <= yi:
        return None
    return frame_bgr[yi:ya, xi:xa].copy()


def render_weapon_bbox_crop_bgr(
    frame_bgr: np.ndarray,
    weapons: List[Dict[str, Any]],
    *,
    pad_ratio: Optional[float] = None,
    weapon_class: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Một crop: theo `weapon_class` hoặc vũ khí điểm cao nhất."""
    items = weapons_best_per_class(weapons)
    if not items:
        return None
    if weapon_class:
        cls = normalize_weapon_class(weapon_class)
        for wb in items:
            if normalize_weapon_class(wb.get("class")) == cls:
                return render_weapon_bbox_crop_one_bgr(frame_bgr, wb, pad_ratio=pad_ratio)
        return None
    return render_weapon_bbox_crop_one_bgr(frame_bgr, items[0], pad_ratio=pad_ratio)


def encode_weapon_bbox_crops_b64(
    frame_bgr: np.ndarray,
    weapons: List[Dict[str, Any]],
    *,
    quality: int = 88,
    pad_ratio: Optional[float] = None,
) -> List[Dict[str, str]]:
    """Danh sách {class, jpeg_b64} — một entry mỗi loại gun/knife/…"""
    out: List[Dict[str, str]] = []
    for wb in weapons_best_per_class(weapons):
        crop = render_weapon_bbox_crop_one_bgr(frame_bgr, wb, pad_ratio=pad_ratio)
        if crop is None:
            continue
        ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            continue
        cls = normalize_weapon_class(wb.get("class"))
        out.append(
            {
                "class": cls,
                "jpeg_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
            }
        )
    return out


def encode_weapon_bbox_crop_b64(
    frame_bgr: np.ndarray,
    weapons: List[Dict[str, Any]],
    *,
    quality: int = 88,
    pad_ratio: Optional[float] = None,
    weapon_class: Optional[str] = None,
) -> Optional[str]:
    crop = render_weapon_bbox_crop_bgr(
        frame_bgr, weapons, pad_ratio=pad_ratio, weapon_class=weapon_class
    )
    if crop is None:
        return None
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def parse_weapon_crops_json(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    try:
        val = json.loads(str(raw))
        return [x for x in val if isinstance(x, dict)] if isinstance(val, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _draw_box_on_crop(
    crop: np.ndarray,
    bbox: Sequence[float],
    origin_x: int,
    origin_y: int,
    *,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if not bbox or len(bbox) < 4:
        return
    ch, cw = crop.shape[:2]
    x1, y1, x2, y2 = (
        int(bbox[0]) - origin_x,
        int(bbox[1]) - origin_y,
        int(bbox[2]) - origin_x,
        int(bbox[3]) - origin_y,
    )
    x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, cw, ch)
    cv2.rectangle(crop, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.putText(
        crop,
        label,
        (max(2, x1), max(14, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def render_track_scene_crop_bgr(
    frame_bgr: np.ndarray,
    person_bbox: List[int],
    weapons: List[Dict[str, Any]],
    face_bbox: Optional[List[float]] = None,
    *,
    pad_ratio: float = 0.15,
) -> Optional[np.ndarray]:
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    if not person_bbox or len(person_bbox) < 4:
        return None

    h, w = frame_bgr.shape[:2]
    px1, py1, px2, py2 = (int(person_bbox[0]), int(person_bbox[1]), int(person_bbox[2]), int(person_bbox[3]))
    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)
    pad_x = int(pw * pad_ratio)
    pad_y = int(ph * pad_ratio)
    xi, yi, xa, ya = _clamp_box(px1 - pad_x, py1 - pad_y, px2 + pad_x, py2 + pad_y, w, h)
    if xa <= xi or ya <= yi:
        return None

    crop = frame_bgr[yi:ya, xi:xa].copy()
    _draw_box_on_crop(crop, person_bbox, xi, yi, color=(255, 180, 0), label="Person")
    if face_bbox and len(face_bbox) >= 4:
        _draw_box_on_crop(crop, face_bbox, xi, yi, color=(0, 0, 255), label="Face")

    for wb in weapons_best_per_class(weapons):
        bb = wb.get("bbox")
        if not bb or len(bb) < 4:
            continue
        cls = normalize_weapon_class(wb.get("class"))
        conf = _weapon_score(wb)
        color = (0, 255, 0) if cls == "gun" else (0, 165, 255)
        _draw_box_on_crop(crop, bb, xi, yi, color=color, label=f"{cls} {conf:.2f}")

    return crop


def encode_track_scene_crop_b64(
    frame_bgr: np.ndarray,
    person_bbox: List[int],
    weapons: List[Dict[str, Any]],
    face_bbox: Optional[List[float]] = None,
    *,
    quality: int = 88,
    pad_ratio: float = 0.15,
) -> Optional[str]:
    crop = render_track_scene_crop_bgr(
        frame_bgr, person_bbox, weapons, face_bbox, pad_ratio=pad_ratio
    )
    if crop is None:
        return None
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def render_weapon_scene_crop_bgr(
    frame_bgr: np.ndarray,
    person_bbox: List[int],
    weapons: List[Dict[str, Any]],
    *,
    pad_ratio: float = 0.15,
) -> Optional[np.ndarray]:
    return render_track_scene_crop_bgr(frame_bgr, person_bbox, weapons, None, pad_ratio=pad_ratio)


def encode_weapon_scene_crop_b64(
    frame_bgr: np.ndarray,
    person_bbox: List[int],
    weapons: List[Dict[str, Any]],
    *,
    quality: int = 88,
) -> Optional[str]:
    return encode_track_scene_crop_b64(frame_bgr, person_bbox, weapons, None, quality=quality)


def weapon_crop_relpath(event_id: str, weapon_class: str) -> str:
    cls = normalize_weapon_class(weapon_class)
    safe = re.sub(r"[^a-z0-9_-]+", "_", cls) or "weapon"
    return f"event_crops/{event_id}_weapon_{safe}.jpg"


def save_weapon_crop_jpeg(event_id: str, jpeg_bytes: bytes, weapon_class: str = "") -> str:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    cls = normalize_weapon_class(weapon_class) if weapon_class else "weapon"
    fp = crop_weapon_file_for_event(event_id, cls)
    fp.write_bytes(jpeg_bytes)
    return weapon_crop_relpath(event_id, cls)


def save_track_scene_crop_jpeg(event_id: str, jpeg_bytes: bytes) -> str:
    from identity_vm_app.services.event_crops import crop_scene_file_for_event

    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    fp = crop_scene_file_for_event(event_id)
    fp.write_bytes(jpeg_bytes)
    return f"event_crops/{event_id}_scene.jpg"
