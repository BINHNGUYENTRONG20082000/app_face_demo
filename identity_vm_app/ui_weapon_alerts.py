"""Hiển thị cảnh báo vũ khí live trên Streamlit (poll API + ảnh thumb / phóng to)."""

from __future__ import annotations

import base64
import time
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore


def _fetch_live_alerts(
    get_json: Callable[..., Any],
    *,
    since_ts: float = 0.0,
    limit: int = 40,
) -> Optional[Dict[str, Any]]:
    try:
        r = get_json(
            "/ivm/weapon-alerts/live",
            params={"limit": limit, "since_ts": since_ts},
            timeout=8,
        )
        if getattr(r, "ok", False):
            return r.json() or {}
    except Exception:
        pass
    return None


def _image_display_width(jpeg_bytes: bytes, *, max_display_px: int = 480) -> int:
    """Không kéo ảnh nhỏ lên full cột — tránh nhìn mờ."""
    cap = max(120, int(max_display_px))
    if Image is None:
        return cap
    try:
        im = Image.open(BytesIO(jpeg_bytes))
        return min(cap, max(1, int(im.size[0])))
    except Exception:
        return cap


def _thumb_bytes(row: Dict[str, Any]) -> Optional[bytes]:
    b64 = row.get("thumb_jpeg_b64")
    if not b64:
        return None
    try:
        return base64.b64decode(str(b64))
    except Exception:
        return None


def _format_alert_time(row: Dict[str, Any]) -> str:
    ts = float(row.get("ts_utc") or 0)
    if ts <= 0:
        return "—"
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _render_alert_zoom_dialog(api_base: str) -> None:
    zoom_id = st.session_state.get("ivm_weapon_zoom_alert_id")
    if not zoom_id:
        return
    dialog = getattr(st, "dialog", None)
    if dialog is None:
        return

    @dialog("Khung cảnh báo — phóng to", width="large")
    def _show() -> None:
        base = (api_base or "").rstrip("/")
        url = f"{base}/ivm/weapon-alerts/frame/{zoom_id}.jpg"
        try:
            import requests

            r = requests.get(url, timeout=15)
            if r.ok and r.content:
                st.image(r.content, use_container_width=True)
            else:
                st.image(url, use_container_width=True)
        except Exception:
            st.image(url, use_container_width=True)
        if st.button("Đóng", key="ivm_weapon_zoom_close"):
            st.session_state.pop("ivm_weapon_zoom_alert_id", None)
            st.rerun()

    _show()


def _render_alert_history_gallery(
    history_by_camera: Dict[str, List[Dict[str, Any]]],
    analyze_states: Dict[str, bool],
    *,
    api_base: str = "",
) -> None:
    """Tối đa 3 ảnh / camera — thumb + nút phóng to."""
    any_img = False
    for cid, rows in sorted(history_by_camera.items()):
        if not analyze_states.get(cid):
            continue
        items = [r for r in (rows or []) if _thumb_bytes(r)][:3]
        if not items:
            continue
        any_img = True
        st.markdown(f"**`{cid}`** — lịch sử cảnh báo ({len(items)} gần nhất)")
        cols = st.columns(min(3, len(items)))
        for i, row in enumerate(items):
            with cols[i % len(cols)]:
                blob = _thumb_bytes(row)
                if blob:
                    st.image(blob, width=_image_display_width(blob, max_display_px=520))
                tid = row.get("track_id", "?")
                st.caption(
                    f"#{tid} · {_format_alert_time(row)}\n\n"
                    f"{row.get('message', '')[:80]}"
                )
                aid = str(row.get("alert_id") or "")
                if aid and st.button(
                    "Phóng to",
                    key=f"ivm_wpn_zoom_{cid}_{aid}",
                    use_container_width=True,
                ):
                    st.session_state["ivm_weapon_zoom_alert_id"] = aid
                    st.rerun()
    if not any_img:
        st.caption("Chưa có ảnh cảnh báo — sẽ hiện khi track vượt ngưỡng frame vũ khí.")


def render_live_weapon_alerts_panel(
    camera_ids: List[str],
    analyze_states: Dict[str, bool],
    *,
    get_json: Callable[..., Any],
    api_base: str = "",
    poll_interval_s: float = 2.0,
    enabled: bool = True,
) -> None:
    any_on = any(bool(analyze_states.get(cid)) for cid in camera_ids)
    if not enabled or not any_on:
        return

    if "ivm_weapon_alert_last_ts" not in st.session_state:
        st.session_state.ivm_weapon_alert_last_ts = 0.0
    if "ivm_weapon_alert_seen" not in st.session_state:
        st.session_state.ivm_weapon_alert_seen = set()

    since = float(st.session_state.ivm_weapon_alert_last_ts)
    data = _fetch_live_alerts(get_json, since_ts=0.0, limit=50)
    if not data:
        return

    recent = list(data.get("recent") or [])
    active = dict(data.get("active_by_camera") or {})
    by_cam = dict(data.get("by_camera") or {})
    history = dict(data.get("history_by_camera") or {})

    max_ts = since
    seen: set = set(st.session_state.ivm_weapon_alert_seen)
    new_toast: List[Dict[str, Any]] = []
    for row in recent:
        key = (
            str(row.get("camera_id")),
            int(row.get("track_id") or -1),
            str(row.get("job_id") or ""),
            float(row.get("ts_utc") or 0),
        )
        ts = float(row.get("ts_utc") or 0)
        if ts > max_ts:
            max_ts = ts
        if key not in seen and ts > since:
            new_toast.append(row)
            seen.add(key)
    st.session_state.ivm_weapon_alert_seen = seen
    if max_ts > since:
        st.session_state.ivm_weapon_alert_last_ts = max_ts

    for row in new_toast[:5]:
        cam = row.get("camera_id", "?")
        st.toast(f"⚠ [{cam}] {row.get('message', 'Cảnh báo vũ khí')}", icon="⚠️")

    banners: List[str] = []
    for cid in camera_ids:
        if not analyze_states.get(cid):
            continue
        acts = list(active.get(cid) or [])
        meta_n = int((by_cam.get(cid) or {}).get("alert_track_count") or 0)
        if acts:
            last = acts[-1]
            banners.append(
                f"**`{cid}`** — {last.get('message', 'Cảnh báo vũ khí')} "
                f"({len(acts)} track đã cảnh báo)"
            )
        elif meta_n > 0:
            banners.append(f"**`{cid}`** — đang có **{meta_n}** track vượt ngưỡng (overlay ALERT)")

    show_section = bool(banners) or bool(new_toast) or any(
        history.get(cid) for cid in camera_ids if analyze_states.get(cid)
    )
    if show_section:
        st.markdown("##### ⚠ Cảnh báo vũ khí (live)")
        for line in banners:
            st.warning(line)
        _render_alert_history_gallery(history, analyze_states, api_base=api_base)
        _render_alert_zoom_dialog(api_base)
    st.caption(
        f"Tự làm mới ~{poll_interval_s:.0f}s · tối đa "
        f"{int(data.get('history_per_camera') or 3)} ảnh / camera · bấm **Phóng to** xem khung gốc."
    )


def render_camera_weapon_alert_badge(
    camera_id: str,
    *,
    active_alerts: Optional[List[Dict[str, Any]]] = None,
    alert_track_count: int = 0,
) -> None:
    acts = list(active_alerts or [])
    if acts:
        last = acts[-1]
        st.error(f"⚠ {last.get('message', 'Cảnh báo vũ khí')}")
    elif int(alert_track_count) > 0:
        st.warning(f"⚠ {int(alert_track_count)} track — overlay ALERT trên video")


def weapon_alerts_auto_refresh_fragment(
    camera_ids: List[str],
    analyze_states: Dict[str, bool],
    *,
    get_json: Callable[..., Any],
    api_base: str = "",
    poll_interval_s: float = 2.0,
) -> None:
    frag = getattr(st, "fragment", None)
    if frag is None:
        render_live_weapon_alerts_panel(
            camera_ids,
            analyze_states,
            get_json=get_json,
            api_base=api_base,
            poll_interval_s=poll_interval_s,
        )
        return

    @frag(run_every=poll_interval_s)
    def _poll() -> None:
        render_live_weapon_alerts_panel(
            camera_ids,
            analyze_states,
            get_json=get_json,
            api_base=api_base,
            poll_interval_s=poll_interval_s,
        )

    _poll()
