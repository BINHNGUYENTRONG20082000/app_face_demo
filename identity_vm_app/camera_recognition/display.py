"""Vẽ bbox + HUD lên khung BGR (giống stream_camera / VisionMaster overlay)."""

from __future__ import annotations

import time
import unicodedata
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def _safe_cv_text(text: str) -> str:
    if not text:
        return "Unknown"
    normalized = unicodedata.normalize("NFKD", str(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = " ".join(ascii_text.split())
    return ascii_text if ascii_text else "Unknown"


def faces_xywh_from_payload(faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chuyển bbox xyxy từ API sang x,y,w,h cho vẽ."""
    out: List[Dict[str, Any]] = []
    for f in faces or []:
        bb = f.get("bbox")
        if not bb or len(bb) < 4:
            continue
        x1, y1, x2, y2 = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
        out.append(
            {
                "x": int(round(x1)),
                "y": int(round(y1)),
                "w": int(round(max(1.0, x2 - x1))),
                "h": int(round(max(1.0, y2 - y1))),
                "confidence": float(f.get("det_score") or 0.0),
                "matches": f.get("matches") or [],
            }
        )
    return out


def _weapon_visual(weapon: Dict[str, Any]) -> tuple[tuple[int, int, int], int, str]:
    """Màu bbox, độ dày, nhãn theo mức vũ khí."""
    dangerous = bool(weapon.get("dangerous"))
    armed = bool(weapon.get("armed"))
    weapon_alert = bool(weapon.get("weapon_alert"))
    if dangerous:
        return (0, 0, 140), 4, "DANGER"
    if weapon_alert:
        return (0, 100, 255), 3, "ALERT"
    if armed:
        return (0, 0, 220), 3, "ARMED"
    return (0, 255, 0), 2, ""


def draw_faces_payload_on_bgr(
    image_bgr: np.ndarray,
    faces_payload: List[Dict[str, Any]],
) -> None:
    """Vẽ từ payload API (bbox xyxy + matches + weapon)."""
    for idx, face in enumerate(faces_payload):
        bb = face.get("bbox")
        if not bb or len(bb) < 4:
            continue
        x = int(round(float(bb[0])))
        y = int(round(float(bb[1])))
        w = int(round(max(1.0, float(bb[2]) - float(bb[0]))))
        h = int(round(max(1.0, float(bb[3]) - float(bb[1]))))
        conf = float(face.get("det_score") or 0.0)
        weapon = face.get("weapon") or {}
        box_color, thickness, armed_tag = _weapon_visual(weapon)
        pb = face.get("person_bbox")
        if pb and len(pb) >= 4:
            px1, py1 = int(round(float(pb[0]))), int(round(float(pb[1])))
            px2, py2 = int(round(float(pb[2]))), int(round(float(pb[3])))
            cv2.rectangle(image_bgr, (px1, py1), (px2, py2), (255, 180, 0), 1)
            tid = face.get("track_id")
            if tid is not None:
                cv2.putText(
                    image_bgr,
                    f"P#{int(tid)}",
                    (px1, max(14, py1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 180, 0),
                    1,
                    cv2.LINE_AA,
                )
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), box_color, thickness)
        tag = armed_tag if armed_tag else f"Face {idx + 1}"
        cv2.putText(
            image_bgr,
            f"{tag} ({conf:.2%})",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2,
            cv2.LINE_AA,
        )
        matches = face.get("matches") or []
        line_y = y + h + 22
        if matches:
            m0 = matches[0]
            name = _safe_cv_text(m0.get("name") or m0.get("display_name") or "Unknown")
            dist = float(m0.get("distance", 0.0))
            cv2.putText(
                image_bgr,
                f"{name} ({dist:.2f})",
                (x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            line_y += 22
        if armed_tag:
            wlabel = _safe_cv_text(weapon.get("weapon_label") or armed_tag)
            cv2.putText(
                image_bgr,
                wlabel,
                (x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        _draw_weapon_boxes_on_bgr(image_bgr, list(weapon.get("weapons") or []))


def _draw_weapon_boxes_on_bgr(image_bgr: np.ndarray, weapons: List[Dict[str, Any]]) -> None:
    for w in weapons or []:
        bb = w.get("bbox")
        if not bb or len(bb) < 4:
            continue
        cls = str(w.get("class") or "weapon").lower()
        color = (0, 255, 0) if cls == "gun" else (0, 165, 255)
        x1, y1, x2, y2 = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        conf = float(w.get("conf") or w.get("fusion_score") or 0.0)
        cv2.putText(
            image_bgr,
            f"{cls} {conf:.2f}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_person_weapon_overlay(image_bgr: np.ndarray, entry: Dict[str, Any]) -> None:
    """Vẽ track chỉ có vũ khí (không mặt)."""
    pb = entry.get("person_bbox")
    weapon = entry.get("weapon") or {}
    box_color, thickness, armed_tag = _weapon_visual(weapon)
    if pb and len(pb) >= 4:
        px1, py1 = int(round(float(pb[0]))), int(round(float(pb[1])))
        px2, py2 = int(round(float(pb[2]))), int(round(float(pb[3])))
        cv2.rectangle(image_bgr, (px1, py1), (px2, py2), (255, 180, 0), thickness, cv2.LINE_AA)
        tid = entry.get("track_id")
        if tid is not None:
            tag = f"P#{int(tid)} {armed_tag}" if armed_tag else f"P#{int(tid)}"
            cv2.putText(
                image_bgr,
                tag,
                (px1, max(18, py1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2,
                cv2.LINE_AA,
            )
    if armed_tag:
        _draw_weapon_boxes_on_bgr(image_bgr, list(weapon.get("weapons") or []))
        wlabel = _safe_cv_text(weapon.get("weapon_label") or "Co vu khi")
        if pb and len(pb) >= 4:
            cv2.putText(
                image_bgr,
                wlabel,
                (int(pb[0]), int(pb[3]) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )


def draw_recognition_on_bgr(
    image_bgr: np.ndarray,
    faces: List[Dict[str, Any]],
) -> None:
    """Vẽ bbox + tên khớp; đỏ nếu có vũ khí."""
    for idx, face in enumerate(faces):
        x = int(face["x"])
        y = int(face["y"])
        w = int(face["w"])
        h = int(face["h"])
        conf = float(face.get("confidence", 0.0))
        weapon = face.get("weapon") or {}
        box_color, thickness, armed_tag = _weapon_visual(weapon)
        cv2.rectangle(image_bgr, (x, y), (x + w, y + h), box_color, thickness)
        tag = armed_tag if armed_tag else f"Face {idx + 1}"
        cv2.putText(
            image_bgr,
            f"{tag} ({conf:.2%})",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2,
            cv2.LINE_AA,
        )
        matches = face.get("matches") or []
        line_y = y + h + 22
        if matches:
            m0 = matches[0]
            name = _safe_cv_text(m0.get("name") or m0.get("display_name") or "Unknown")
            dist = float(m0.get("distance", 0.0))
            cv2.putText(
                image_bgr,
                f"{name} ({dist:.2f})",
                (x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            line_y += 22
        if armed_tag:
            wlabel = _safe_cv_text(weapon.get("weapon_label") or armed_tag)
            cv2.putText(
                image_bgr,
                wlabel,
                (x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )


def apply_analyze_hud(
    out_bgr: np.ndarray,
    *,
    camera_id: str,
    meta: Dict[str, Any],
    capture_fps: float = 0.0,
) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    h_frame, w_frame = out_bgr.shape[:2]
    bar_h = 56
    bar_w = min(w_frame, 820)
    cv2.rectangle(out_bgr, (0, 0), (bar_w, bar_h), (0, 0, 0), -1)
    fc_label = meta.get("frame_count")
    fc_str = str(int(fc_label)) if fc_label is not None else "?"
    line1 = f"{ts}  | {camera_id}  | Khung # {fc_str}  | Capture ~{capture_fps:.1f} fps"
    infer = float(meta.get("infer_ms", 0.0))
    nf = int(meta.get("n_faces", 0))
    enabled = bool(meta.get("recognition_enabled", False))
    if not enabled:
        line2 = "Nhận diện TẮT — POST /ivm/cameras/{id}/analyze {\"enabled\": true}"
    else:
        target = meta.get("analyze_target_fps")
        even_only = bool(meta.get("analyze_even_frames_only"))
        w_ms = float(meta.get("weapon_ms") or 0.0)
        armed_f = int(meta.get("armed_faces") or 0)
        w_scene = meta.get("weapon_scene") or {}
        w_status = str(w_scene.get("image_status") or "")
        if even_only:
            line2 = f"Nhận diện BẬT  |  frame chan  |  infer {infer:.0f} ms  |  Faces: {nf}"
        elif target:
            line2 = f"Nhận diện BẬT  |  infer {infer:.0f} ms  |  Faces: {nf}  |  ~{target} fps"
        else:
            line2 = f"Nhận diện BẬT  |  infer {infer:.0f} ms  |  Faces: {nf}"
        if w_ms > 0 or w_status:
            line2 += f"  |  Weapon {w_ms:.0f}ms armed={armed_f}"
            alert_n = int(w_scene.get("alert_persons") or meta.get("weapon_alert_tracks") or 0)
            if alert_n > 0:
                line2 += f" ALERT={alert_n}"
            if w_status == "DANGEROUS":
                line2 += " DANGER"
    err = meta.get("error")
    if err:
        line2 += f"  |  {str(err)[:48]}..."
    cv2.putText(out_bgr, line1, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
    hud_color = (40, 200, 255) if err else (200, 220, 255)
    cv2.putText(out_bgr, line2, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.44, hud_color, 1, cv2.LINE_AA)


def build_analyze_visual_bgr(
    frame_bgr: np.ndarray,
    *,
    camera_id: str,
    faces_payload: List[Dict[str, Any]],
    meta: Dict[str, Any],
    capture_fps: float,
    armed_persons: Optional[List[Dict[str, Any]]] = None,
) -> Optional[np.ndarray]:
    """Khung BGR có bbox + HUD — dùng ghi video phân tích và MJPEG."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    vis = frame_bgr.copy()
    if meta.get("recognition_enabled"):
        face_only = [f for f in (faces_payload or []) if f.get("bbox")]
        if face_only:
            draw_faces_payload_on_bgr(vis, face_only)
        for p in armed_persons or []:
            w = p.get("weapon") or {}
            if w.get("armed") or w.get("weapon_alert"):
                _draw_person_weapon_overlay(vis, p)
    apply_analyze_hud(vis, camera_id=camera_id, meta=meta, capture_fps=capture_fps)
    return vis


def encode_display_jpeg(
    frame_bgr: np.ndarray,
    *,
    camera_id: str,
    faces_payload: List[Dict[str, Any]],
    meta: Dict[str, Any],
    capture_fps: float,
    quality: int,
    armed_persons: Optional[List[Dict[str, Any]]] = None,
) -> Optional[bytes]:
    vis = build_analyze_visual_bgr(
        frame_bgr,
        camera_id=camera_id,
        faces_payload=faces_payload,
        meta=meta,
        capture_fps=capture_fps,
        armed_persons=armed_persons,
    )
    if vis is None:
        return None
    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None
