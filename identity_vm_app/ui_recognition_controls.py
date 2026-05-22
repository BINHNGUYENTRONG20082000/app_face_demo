"""Panel Streamlit: bật/tắt nhận diện từng camera (API /ivm/cameras/{id}/analyze)."""

from __future__ import annotations

import html
from typing import Any, Callable, Dict, List, Optional

import streamlit as st


def render_per_camera_recognition_panel(
    camera_ids: List[str],
    states: Dict[str, bool],
    *,
    set_enabled: Callable[..., None],
    api_base: str = "",
    cols_per_row: int = 5,
    show_snapshots: bool = False,
    session_params: Optional[Dict[str, Dict[str, Any]]] = None,
    active_sessions: Optional[Dict[str, Dict[str, Any]]] = None,
    weapon_alerts_by_camera: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    weapon_meta_by_camera: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Hiển thị nút Bật/Tắt nhận diện cho từng camera."""
    if not camera_ids:
        st.info("Chưa có camera trong `camera_config.json`.")
        return

    st.markdown("##### Bật / tắt nhận diện theo camera")
    st.caption(
        "Chọn **FPS mẫu** trước khi bấm **Bật** (mặc định 5 FPS). Camera **BẬT** → phiên live + "
        "`session.mp4` + báo cáo DB. Overlay: `/ivm/cameras/{id}/infer/mjpeg`."
    )

    sessions = active_sessions or {}
    alerts_by_cam = weapon_alerts_by_camera or {}
    meta_by_cam = weapon_meta_by_camera or {}

    base = (api_base or "").rstrip("/")
    n = max(1, min(6, cols_per_row))
    for i in range(0, len(camera_ids), n):
        chunk = camera_ids[i : i + n]
        cols = st.columns(len(chunk))
        for col, cid in zip(cols, chunk):
            with col:
                en = bool(states.get(cid, False))
                sess = sessions.get(cid) or {}
                badge = "BẬT" if en else "TẮT"
                color = "#16a34a" if en else "#dc2626"
                st.markdown(
                    f"**`{cid}`** — "
                    f'<span style="color:{color};font-weight:700;">{badge}</span>',
                    unsafe_allow_html=True,
                )
                if en and sess.get("sample_fps_label"):
                    st.caption(f"Phiên đang chạy: **{sess['sample_fps_label']}** mẫu")
                if en:
                    from identity_vm_app.ui_weapon_alerts import render_camera_weapon_alert_badge

                    cam_meta = meta_by_cam.get(cid) or {}
                    render_camera_weapon_alert_badge(
                        cid,
                        active_alerts=alerts_by_cam.get(cid),
                        alert_track_count=int(cam_meta.get("alert_track_count") or 0),
                    )
                fps_opt = st.selectbox(
                    "FPS mẫu (khi bật)",
                    ["5", "10", "15", "0 (full frame)"],
                    index=0,
                    key=f"ivm_rec_fps_{cid}",
                    disabled=en,
                    help="Chỉ áp dụng lúc bấm Bật. Muốn đổi FPS → Tắt rồi Bật lại với FPS mới.",
                )
                sf_map = {"5": 5.0, "10": 10.0, "15": 15.0, "0 (full frame)": 0.0}
                with st.expander("Tham số phiên khác", expanded=False):
                    st.text_input("Tên phiên (tuỳ chọn)", key=f"ivm_rec_name_{cid}")
                    st.number_input(
                        "Ngưỡng distance",
                        min_value=0.0,
                        max_value=2.0,
                        value=0.45,
                        step=0.05,
                        key=f"ivm_rec_dist_{cid}",
                    )
                    st.checkbox("Lưu crop JPEG", value=False, key=f"ivm_rec_crops_{cid}")
                if show_snapshots and base:
                    snap_u = f"{base}/ivm/preview/{cid}/snapshot.jpg"
                    st.markdown(
                        f'<img src="{html.escape(snap_u)}" style="width:100%;max-height:96px;'
                        f'object-fit:contain;background:#111827;border-radius:6px;" />',
                        unsafe_allow_html=True,
                    )
                b_on, b_off = st.columns(2)
                with b_on:
                    if st.button(
                        "Bật",
                        key=f"ivm_rec_on_{cid}",
                        type="primary" if not en else "secondary",
                        disabled=en,
                        use_container_width=True,
                    ):
                        payload: Dict[str, Any] = {
                            "enabled": True,
                            "sample_fps": sf_map.get(str(st.session_state.get(f"ivm_rec_fps_{cid}", "5")), 5.0),
                        }
                        dn = str(st.session_state.get(f"ivm_rec_name_{cid}", "") or "").strip()
                        if dn:
                            payload["display_name"] = dn
                        payload["distance_threshold"] = float(
                            st.session_state.get(f"ivm_rec_dist_{cid}", 0.45)
                        )
                        payload["save_crops"] = bool(st.session_state.get(f"ivm_rec_crops_{cid}", False))
                        if session_params is not None:
                            session_params[cid] = payload
                        opts = {k: v for k, v in payload.items() if k != "enabled"}
                        set_enabled(cid, True, **opts)
                        st.rerun()
                with b_off:
                    if st.button(
                        "Tắt",
                        key=f"ivm_rec_off_{cid}",
                        disabled=not en,
                        use_container_width=True,
                    ):
                        set_enabled(cid, False)
                        st.rerun()
                if base and st.button("Log", key=f"ivm_rec_log_{cid}", use_container_width=True):
                    try:
                        import requests

                        r = requests.get(
                            f"{base}/ivm/cameras/{cid}/analyze/activity",
                            params={"limit": 15},
                            timeout=10,
                        )
                        if r.ok:
                            data = r.json()
                            prev_act = st.session_state.get(f"ivm_act_{cid}") or {}
                            st.session_state[f"ivm_act_{cid}"] = {
                                **prev_act,
                                "activity": data.get("activity") or [],
                                "last_meta": data.get("last_meta") or {},
                                "hub_worker_running": data.get("hub_worker_running"),
                                "reader_connected": data.get("reader_connected"),
                            }
                        else:
                            st.error(r.text[:200])
                    except Exception as ex:
                        st.error(str(ex))
                act = st.session_state.get(f"ivm_act_{cid}")
                if act and en:
                    meta = act.get("last_meta") or {}
                    avg100 = meta.get("infer_avg_ms")
                    cap = (
                        f"Hub: {'OK' if act.get('hub_worker_running') else '—'} | "
                        f"RTSP: {'OK' if act.get('reader_connected') else '—'} | "
                        f"infer {meta.get('infer_ms', '—')} ms"
                    )
                    if avg100 is not None:
                        cap += f" | TB100 {float(avg100):.0f} ms"
                    p_ms = meta.get("person_track_ms")
                    if p_ms is not None:
                        cap += f" | person {float(p_ms):.0f}"
                    pose_ms = meta.get("pose_refine_ms")
                    if pose_ms is not None:
                        cap += f" | pose {float(pose_ms):.0f}"
                    cap += f" | mặt {meta.get('n_faces', '—')}"
                    st.caption(cap)
                    rows = list(act.get("activity") or [])
                    stats_rows = [r for r in rows if r.get("event") == "infer_stats"]
                    other_rows = [r for r in rows if r.get("event") != "infer_stats"]
                    shown = (stats_rows[:1] + other_rows[:4])[:5]
                    for row in shown:
                        if row.get("event") == "infer_stats":
                            prefix = "📊 "
                        elif row.get("event") == "weapon_alert":
                            prefix = "⚠ "
                        else:
                            prefix = "• "
                        st.text(f"{prefix}{row.get('message', '')}")
