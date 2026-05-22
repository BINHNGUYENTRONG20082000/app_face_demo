"""Lưu ảnh crop khuôn mặt gắn với recognition_events."""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Optional

from identity_vm_app import settings as s

CROPS_DIR = Path(s.IVM_DATA_DIR) / "event_crops"


def crop_file_for_event(event_id: str) -> Path:
    return CROPS_DIR / f"{event_id}.jpg"


def crop_weapon_file_for_event(event_id: str, weapon_class: str = "") -> Path:
    if not (weapon_class or "").strip():
        legacy = CROPS_DIR / f"{event_id}_weapon.jpg"
        if legacy.is_file():
            return legacy
    cls = (weapon_class or "weapon").strip().lower()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in cls) or "weapon"
    return CROPS_DIR / f"{event_id}_weapon_{safe}.jpg"


def crop_scene_file_for_event(event_id: str) -> Path:
    return CROPS_DIR / f"{event_id}_scene.jpg"


def save_crop_jpeg(event_id: str, jpeg_bytes: bytes) -> str:
    """Lưu file; trả về đường dẫn tương đối (trong extra_json)."""
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    fp = crop_file_for_event(event_id)
    fp.write_bytes(jpeg_bytes)
    return f"event_crops/{event_id}.jpg"


def decode_crop_b64(crop_jpeg_b64: str) -> bytes:
    raw = crop_jpeg_b64.strip()
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    try:
        return base64.b64decode(raw, validate=True)
    except binascii.Error:
        return base64.b64decode(raw)


def load_crop_bytes(rel_path: Optional[str]) -> Optional[bytes]:
    if not rel_path:
        return None
    p = Path(s.IVM_DATA_DIR) / str(rel_path).replace("\\", "/").lstrip("/")
    try:
        p.resolve().relative_to(Path(s.IVM_DATA_DIR).resolve())
    except ValueError:
        return None
    if p.is_file():
        return p.read_bytes()
    return None


def should_replace_crop(
    old_det: Optional[float],
    new_det: Optional[float],
    *,
    merged: bool,
) -> bool:
    """merged=True khi đã có crop — chỉ thay nếu det_score cao hơn."""
    if not merged:
        return True
    if new_det is None:
        return False
    if old_det is None:
        return True
    return float(new_det) > float(old_det)
