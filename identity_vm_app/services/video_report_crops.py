"""Crop / vẽ box từ full frame — giống get_face_image_view VideoMaster."""

from __future__ import annotations

import ast
import io
import json
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from identity_vm_app.services import video_analyze_media as vmedia


def parse_box(box_raw: Any) -> Optional[List[int]]:
    if box_raw is None:
        return None
    if isinstance(box_raw, (list, tuple)) and len(box_raw) >= 4:
        return [int(box_raw[0]), int(box_raw[1]), int(box_raw[2]), int(box_raw[3])]
    s = str(box_raw).strip()
    if not s or s.lower() == "none":
        return None
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)) and len(val) >= 4:
            return [int(val[0]), int(val[1]), int(val[2]), int(val[3])]
    except (SyntaxError, ValueError, TypeError):
        pass
    return None


def load_frame_bgr(img_url: Optional[str]) -> Optional[np.ndarray]:
    if not img_url:
        return None
    fp = vmedia.resolve_media_path(str(img_url))
    if fp is None:
        return None
    frame = cv2.imread(str(fp))
    return frame if frame is not None and frame.size else None


def crop_from_frame(frame: np.ndarray, box_raw: Any, *, pad: float = 0.0) -> Optional[np.ndarray]:
    box = parse_box(box_raw)
    if box is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    if pad > 0:
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        x1 = max(0, int(x1 - bw * pad))
        y1 = max(0, int(y1 - bh * pad))
        x2 = min(w, int(x2 + bw * pad))
        y2 = min(h, int(y2 + bh * pad))
    else:
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def parse_weapon_boxes(raw: Any) -> List[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [w for w in raw if isinstance(w, dict)]
    s = str(raw).strip()
    if not s:
        return []
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [w for w in val if isinstance(w, dict)]
    except json.JSONDecodeError:
        pass
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return [w for w in val if isinstance(w, dict)]
    except (SyntaxError, ValueError, TypeError):
        pass
    return []


def _weapon_color(class_name: str) -> Tuple[int, int, int]:
    cls = str(class_name or "").lower()
    if cls == "gun":
        return (0, 255, 0)
    if cls == "knife":
        return (0, 165, 255)
    return (0, 0, 255)


def draw_weapon_boxes_on_frame(
    frame: np.ndarray,
    weapons: List[Dict[str, Any]],
) -> np.ndarray:
    out = frame.copy()
    for w in weapons or []:
        bb = parse_box(w.get("bbox"))
        if bb is None:
            continue
        cls = str(w.get("class") or "weapon")
        conf = float(w.get("conf") or w.get("fusion_score") or 0.0)
        color = _weapon_color(cls)
        cv2.rectangle(out, (bb[0], bb[1]), (bb[2], bb[3]), color, 2, cv2.LINE_AA)
        label = f"{cls} {conf:.2f}"
        cv2.putText(
            out,
            label,
            (bb[0], max(14, bb[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def draw_boxes_on_frame(
    frame: np.ndarray,
    *,
    box_person: Any = None,
    box_face: Any = None,
    weapon_boxes: Any = None,
) -> np.ndarray:
    out = frame.copy()
    pb = parse_box(box_person)
    if pb is not None:
        cv2.rectangle(out, (pb[0], pb[1]), (pb[2], pb[3]), (255, 180, 0), 2, cv2.LINE_AA)
    fb = parse_box(box_face)
    if fb is not None:
        cv2.rectangle(out, (fb[0], fb[1]), (fb[2], fb[3]), (0, 0, 255), 2, cv2.LINE_AA)
    weapons = weapon_boxes if isinstance(weapon_boxes, list) else parse_weapon_boxes(weapon_boxes)
    if weapons:
        out = draw_weapon_boxes_on_frame(out, weapons)
    return out


def draw_all_boxes_on_frame(
    frame: np.ndarray,
    rows: List[dict],
) -> np.ndarray:
    """Vẽ mọi box_person/box_face/weapon trên cùng một khung."""
    out = frame.copy()
    for row in rows:
        pb = parse_box(row.get("box_person"))
        if pb is not None:
            cv2.rectangle(out, (pb[0], pb[1]), (pb[2], pb[3]), (255, 180, 0), 2, cv2.LINE_AA)
        fb = parse_box(row.get("box_face"))
        if fb is not None:
            cv2.rectangle(out, (fb[0], fb[1]), (fb[2], fb[3]), (0, 0, 255), 2, cv2.LINE_AA)
        wlist = parse_weapon_boxes(row.get("weapon_boxes_json"))
        if wlist:
            out = draw_weapon_boxes_on_frame(out, wlist)
    return out


def encode_jpeg_bytes(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("imencode failed")
    return buf.tobytes()


def jpeg_response_bytes(img: np.ndarray, quality: int = 85) -> io.BytesIO:
    return io.BytesIO(encode_jpeg_bytes(img, quality=quality))
