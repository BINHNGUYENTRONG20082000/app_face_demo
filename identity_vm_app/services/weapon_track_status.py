"""Quy tắc vũ khí theo số frame phát hiện trên một track / phiên."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from identity_vm_app import settings as s


def weapon_armed_min_frames() -> int:
    return max(1, int(s.IVM_WEAPON_ARMED_MIN_FRAMES))


def weapon_dangerous_min_frames() -> int:
    return max(1, int(s.IVM_WEAPON_DANGEROUS_MIN_FRAMES))


def weapon_alert_min_frames() -> int:
    return max(1, int(s.IVM_WEAPON_ALERT_MIN_FRAMES))


def weapon_should_alert(armed_frame_count: int) -> bool:
    """True khi số frame có det vũ khí trên track vượt ngưỡng cảnh báo (> alert_min)."""
    return int(armed_frame_count) > weapon_alert_min_frames()


def count_weapon_frames(rows: Sequence[Dict[str, Any]]) -> int:
    """Đếm khung mẫu có phát hiện vũ khí (cột armed=1 từng frame)."""
    n = 0
    for row in rows:
        if int(row.get("armed") or 0):
            n += 1
    return n


def weapon_types_from_rows(rows: Sequence[Dict[str, Any]]) -> List[str]:
    import json

    types: set[str] = set()
    for row in rows:
        raw = row.get("weapon_types_json")
        if not raw:
            continue
        try:
            parsed = json.loads(str(raw)) if isinstance(raw, str) else raw
            if isinstance(parsed, list):
                for t in parsed:
                    if t:
                        types.add(str(t))
        except (json.JSONDecodeError, TypeError):
            pass
    return sorted(types)


def classify_weapon_by_frame_count(
    armed_frame_count: int,
    weapon_types: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    > armed_min frame có vũ khí → coi là có vũ khí.
    > dangerous_min frame → cảnh báo nguy hiểm.
    """
    count = max(0, int(armed_frame_count))
    types = [str(t) for t in (weapon_types or []) if t]
    types_label = ", ".join(types) if types else ""
    armed_thr = weapon_armed_min_frames()
    danger_thr = weapon_dangerous_min_frames()

    if count > danger_thr:
        label = f"Cảnh báo nguy hiểm ({types_label})" if types_label else "Cảnh báo nguy hiểm"
        return {
            "armed": True,
            "dangerous": True,
            "weapon_status": "nguy_hiem",
            "weapon_label": label,
            "image_status": "DANGEROUS",
            "weapon_armed_frames": count,
        }
    if count > armed_thr:
        label = f"Có vũ khí ({types_label})" if types_label else "Có vũ khí"
        return {
            "armed": True,
            "dangerous": False,
            "weapon_status": "co_vu_khi",
            "weapon_label": label,
            "image_status": "ARMED",
            "weapon_armed_frames": count,
        }
    return {
        "armed": False,
        "dangerous": False,
        "weapon_status": "an_toan",
        "weapon_label": "Không vũ khí",
        "image_status": "SAFE",
        "weapon_armed_frames": count,
    }


def apply_track_weapon_summary(
    track_row: Dict[str, Any],
    frame_rows: Sequence[Dict[str, Any]],
) -> None:
    """Gán armed / weapon_status / weapon_label cho track từ danh sách khung mẫu."""
    armed_frames = count_weapon_frames(frame_rows)
    types = weapon_types_from_rows(frame_rows)
    cls = classify_weapon_by_frame_count(armed_frames, types)
    track_row["weapon_armed_frames"] = cls["weapon_armed_frames"]
    track_row["armed"] = 1 if cls["armed"] else 0
    track_row["dangerous"] = 1 if cls["dangerous"] else 0
    track_row["weapon_status"] = cls["weapon_status"]
    track_row["weapon_label"] = cls["weapon_label"]
    if types:
        import json

        track_row["weapon_types_json"] = json.dumps(types, ensure_ascii=False)
