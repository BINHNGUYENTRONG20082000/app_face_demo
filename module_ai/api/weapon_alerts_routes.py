"""API cảnh báo vũ khí live — Streamlit / dashboard poll."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from identity_vm_app import settings as s
from identity_vm_app.camera_analyze_control import snapshot_states
from module_ai.camera.weapon import weapon_detection_available
from module_ai.camera.weapon_alerts import (
    active_weapon_alerts_by_camera,
    get_alert_full_jpeg,
    recent_weapon_alerts,
    weapon_alert_history_by_camera,
)

router = APIRouter(prefix="/ivm", tags=["weapon-alerts"])


@router.get("/weapon-alerts/recent")
def weapon_alerts_recent(
    camera_id: Optional[str] = None,
    limit: int = Query(30, ge=1, le=200),
    since_ts: float = Query(0.0, ge=0.0),
) -> Dict[str, Any]:
    rows = recent_weapon_alerts(camera_id, limit=limit, since_ts=since_ts)
    return {
        "camera_id": camera_id,
        "count": len(rows),
        "alerts": rows,
        "weapon_detection_enabled": weapon_detection_available(),
    }


@router.get("/weapon-alerts/live")
def weapon_alerts_live(
    limit: int = Query(30, ge=1, le=200),
    since_ts: float = Query(0.0, ge=0.0),
) -> Dict[str, Any]:
    """
    Gói cho UI: cảnh báo gần đây + đang active theo camera + meta infer (track đang alert).
    """
    recent = recent_weapon_alerts(None, limit=limit, since_ts=since_ts)
    active = active_weapon_alerts_by_camera()
    states = snapshot_states()
    by_camera: Dict[str, Any] = {}

    try:
        from module_ai.camera.hub import get_recognition_hub

        hub = get_recognition_hub()
    except Exception:
        hub = None

    for cam, enabled in states.items():
        if not enabled:
            continue
        entry: Dict[str, Any] = {
            "recognition_enabled": True,
            "active_alerts": active.get(cam) or [],
            "alert_track_count": 0,
            "infer_meta": {},
        }
        if hub is not None:
            w = hub.get_worker(cam)
            if w is not None:
                meta = w.get_meta()
                entry["infer_meta"] = {
                    "weapon_alert_tracks": int(meta.get("weapon_alert_tracks") or 0),
                    "weapon_scene": meta.get("weapon_scene") or {},
                    "frame_count": meta.get("frame_count"),
                }
                entry["alert_track_count"] = int(meta.get("weapon_alert_tracks") or 0)
        by_camera[cam] = entry

    history = weapon_alert_history_by_camera()

    return {
        "recent": recent,
        "active_by_camera": active,
        "history_by_camera": history,
        "by_camera": by_camera,
        "analyzing_cameras": [c for c, on in states.items() if on],
        "history_per_camera": int(s.IVM_WEAPON_ALERT_HISTORY_PER_CAMERA),
    }


@router.get("/weapon-alerts/frame/{alert_id}.jpg")
def weapon_alert_frame_jpg(alert_id: str) -> Response:
    """Ảnh full khung lúc cảnh báo — dùng khi bấm Phóng to trên UI."""
    data = get_alert_full_jpeg(alert_id)
    if not data:
        raise HTTPException(status_code=404, detail="Không có ảnh cảnh báo (hết phiên hoặc alert_id sai)")
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})
