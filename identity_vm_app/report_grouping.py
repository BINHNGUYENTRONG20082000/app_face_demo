"""Gom track nhận diện theo tên hiển thị — một thẻ mỗi đối tượng (cùng tên)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.text import remove_accents


def normalize_person_name(name: str) -> str:
    """Khóa gom nhóm: bỏ dấu, lower, gộp khoảng trắng."""
    s = remove_accents(str(name or "").strip().lower())
    return " ".join(s.split()) or "unknown"


def is_identified_track(track: Dict[str, Any]) -> bool:
    pref = str(track.get("person_ref") or "")
    return bool(track.get("known")) and pref != "unknown"


def group_key_for_track(track: Dict[str, Any]) -> Optional[str]:
    if not is_identified_track(track):
        return None
    dname = track.get("display_name") or track.get("identity")
    if dname and str(dname).strip() and str(dname).strip().lower() != "unknown":
        return normalize_person_name(str(dname))
    return str(track.get("person_ref") or "")


def weapon_summary_from_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Tổng hợp trạng thái vũ khí từ danh sách event/track (badge UI báo cáo)."""
    from identity_vm_app.services.weapon_track_status import classify_weapon_by_frame_count

    weapon_types = sorted(
        {str(t) for e in events for t in (e.get("weapon_types") or []) if t}
    )
    armed_frames = sum(int(e.get("weapon_armed_frames") or 0) for e in events)
    if armed_frames <= 0:
        armed_frames = sum(
            int(e.get("frame_hits") or 1) for e in events if e.get("armed")
        )
    cls = classify_weapon_by_frame_count(armed_frames, weapon_types)
    return {
        "has_weapon": bool(cls["armed"]),
        "dangerous": bool(cls["dangerous"]),
        "weapon_types": weapon_types,
        "weapon_label": str(cls["weapon_label"]),
        "weapon_status": str(cls["weapon_status"]),
        "weapon_armed_frames": armed_frames,
    }


def _pick_representative_crop(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    with_crop = [e for e in events if e.get("crop_url")]
    if not with_crop:
        return events[0] if events else None
    return max(
        with_crop,
        key=lambda e: (
            float(e.get("det_score") or 0.0),
            float(e.get("ts_utc") or 0.0),
        ),
    )


def group_persons_by_display_name(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Mỗi tên hiển thị (sau chuẩn hoá) = một đối tượng — gộp mọi person_ref / face_id cùng tên.
    """
    by_key: Dict[str, List[Dict[str, Any]]] = {}
    for tr in tracks:
        gk = group_key_for_track(tr)
        if gk is None:
            continue
        by_key.setdefault(gk, []).append(tr)

    persons: List[Dict[str, Any]] = []
    for gk, evts in by_key.items():
        evts.sort(key=lambda e: float(e.get("ts_utc") or 0), reverse=True)
        rep = _pick_representative_crop(evts)
        display_name = (
            (rep or {}).get("display_name")
            or (rep or {}).get("identity")
            or gk
        )
        person_refs = sorted({str(e.get("person_ref")) for e in evts if e.get("person_ref")})
        total_frames = sum(int(e.get("frame_hits") or 1) for e in evts)
        wsum = weapon_summary_from_events(evts)
        persons.append(
            {
                "group_key": gk,
                "person_ref": person_refs[0] if len(person_refs) == 1 else gk,
                "person_refs": person_refs,
                "display_name": str(display_name),
                "face_id": (rep or {}).get("face_id"),
                "total_frames": total_frames,
                "appearance_count": len(evts),
                "last_seen": float(evts[0].get("ts_utc") or 0),
                "first_seen": float(evts[-1].get("ts_utc") or 0),
                "has_weapon": wsum["has_weapon"],
                "dangerous": wsum["dangerous"],
                "weapon_types": wsum["weapon_types"],
                "weapon_label": wsum["weapon_label"],
                "representative": rep,
                "events": evts,
            }
        )
    persons.sort(key=lambda p: p["last_seen"], reverse=True)
    return persons


def filter_tracks_for_group(
    tracks: List[Dict[str, Any]],
    *,
    group_key: Optional[str] = None,
    display_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Lọc track thuộc một nhóm tên (cho API export)."""
    target = group_key
    if target is None and display_name:
        target = normalize_person_name(display_name)
    if not target:
        return []
    return [t for t in tracks if group_key_for_track(t) == target]
