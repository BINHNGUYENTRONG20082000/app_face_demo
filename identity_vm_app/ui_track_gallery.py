"""UI Streamlit: báo cáo gom theo tên — một thẻ/đối tượng; chi tiết = mọi frame; xuất WebM."""

from __future__ import annotations

import html
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import streamlit as st

from identity_vm_app.report_grouping import group_persons_by_display_name, weapon_summary_from_events
from identity_vm_app.ui_face_weapon_stack import (
    SCENE_WIDTH,
    THUMB_PX,
    render_track_detail_three_images,
)
from identity_vm_app.services.visual_mp4 import (
    list_visual_sessions_from_disk,
    load_visual_mp4_bytes,
)


def fmt_ts(ts: Optional[float]) -> str:
    if ts is None:
        return "—"
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return str(ts)


def _fetch_visual_sessions(api_base: str, camera_id: str) -> List[Dict[str, Any]]:
    """Ưu tiên đọc từ đĩa; bổ sung qua API nếu cần."""
    sessions = list_visual_sessions_from_disk(camera_id, limit=20)
    if sessions:
        return sessions
    base = (api_base or "").rstrip("/")
    try:
        r = requests.get(
            f"{base}/ivm/cameras/{camera_id}/analyze/visual/sessions",
            params={"limit": 12},
            timeout=15,
        )
        if r.ok:
            return list(r.json().get("visual_sessions") or [])
    except requests.RequestException:
        pass
    return sessions


def render_analyze_visual_section(
    api_base: str,
    camera_id: str,
    *,
    key_prefix: str,
    expanded: bool = True,
) -> None:
    """Danh sách + phát video overlay (bbox + tên)."""
    if not camera_id:
        return

    with st.expander("Video phân tích (overlay bbox + tên)", expanded=expanded):
        st.caption(
            "Ghi khi **bật nhận diện** (~10 fps). **Tắt nhận diện** để đóng file. "
            "Base URL API: **`http://127.0.0.1:8010`** (không phải 8000)."
        )
        sessions = _fetch_visual_sessions(api_base, camera_id)
        if not sessions:
            st.warning(
                f"Chưa có video cho camera **{camera_id}**. "
                "Bật nhận diện trên đúng camera ≥ 20 giây, **tắt** nhận diện, rồi bấm **Tải lại danh sách**."
            )
            if st.button("Tải lại danh sách", key=f"{key_prefix}_vis_reload_empty"):
                st.rerun()
            return

        def _label(s: Dict[str, Any]) -> str:
            sid = str(s.get("session_id", "?"))
            sz = s.get("size_bytes")
            sz_mb = f"{int(sz) / 1e6:.1f} MB" if sz else "?"
            web_ok = " · H.264 ✓" if s.get("has_web") else ""
            rec = " · đang ghi" if s.get("recording") else ""
            t0 = fmt_ts(s.get("started_utc") or s.get("mtime_utc"))
            return f"{t0} · {sid} · {sz_mb}{web_ok}{rec}"

        labels = [_label(s) for s in sessions]
        pick_i = st.selectbox(
            "Chọn phiên",
            range(len(labels)),
            format_func=lambda i: labels[i],
            key=f"{key_prefix}_vis_sess",
        )
        sess = sessions[int(pick_i)]
        sid = str(sess.get("session_id", ""))
        is_rec = bool(sess.get("recording"))

        c_reload, c_play = st.columns(2)
        with c_reload:
            if st.button("Tải lại danh sách", key=f"{key_prefix}_vis_reload"):
                st.session_state.pop(f"{key_prefix}_vis_bytes_{sid}", None)
                st.rerun()
        with c_play:
            play_clicked = st.button(
                "▶ Phát video",
                type="primary",
                key=f"{key_prefix}_vis_play_{sid}",
                disabled=is_rec,
            )

        if is_rec:
            st.info("Phiên **đang ghi** — tắt nhận diện rồi bấm **Phát video**.")
            return

        cache_key = f"{key_prefix}_vis_bytes_{sid}"
        cache_path_key = f"{key_prefix}_vis_path_{sid}"

        if play_clicked or cache_key in st.session_state:
            if play_clicked:
                st.session_state.pop(cache_key, None)
            if cache_key not in st.session_state:
                blob: Optional[bytes] = None
                src_note = ""
                with st.spinner("Đang chuẩn bị video H.264 (có thể 1–2 phút lần đầu)…"):
                    try:
                        blob, src_note = load_visual_mp4_bytes(camera_id, sid)
                        src_note = f"đọc từ đĩa: {Path(src_note).name}"
                    except FileNotFoundError:
                        pass
                    except Exception as ex:
                        st.warning(f"Đọc đĩa/remux: {ex}")
                    if blob is None:
                        base = (api_base or "").rstrip("/")
                        vid_url = f"{base}/ivm/cameras/{camera_id}/analyze/visual/{sid}.mp4"
                        try:
                            vr = requests.get(vid_url, timeout=300)
                            if vr.ok and vr.content:
                                blob = vr.content
                                src_note = "tải qua API"
                        except requests.RequestException as ex:
                            st.error(f"Không tải được video: {ex}")
                            return
                    if not blob:
                        st.error("Không có dữ liệu video.")
                        return
                    st.session_state[cache_key] = blob
                    st.session_state[cache_path_key] = src_note

            blob = st.session_state.get(cache_key)
            if blob:
                st.video(blob, format="video/mp4")
                st.download_button(
                    "Tải MP4",
                    data=blob,
                    file_name=f"analyze_{camera_id}_{sid}.mp4",
                    mime="video/mp4",
                    key=f"{key_prefix}_vis_dl_{sid}",
                )
                note = st.session_state.get(cache_path_key, "")
                st.caption(f"{len(blob) / 1e6:.1f} MB · {note}")
            else:
                st.caption("Bấm **▶ Phát video** để xem.")
        else:
            st.caption("Chọn phiên rồi bấm **▶ Phát video**.")


def _fetch_crop_bytes(api_base: str, crop_url: str, timeout: float = 15.0) -> Optional[bytes]:
    base = (api_base or "").rstrip("/")
    path = str(crop_url)
    url = f"{base}{path}" if path.startswith("/") else path
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and r.content:
            return r.content
    except requests.RequestException:
        pass
    return None


def _group_unknown_tracks(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mỗi event unknown = một thẻ riêng (không gom tên)."""
    out: List[Dict[str, Any]] = []
    for tr in tracks:
        if str(tr.get("person_ref") or "") != "unknown":
            continue
        eid = str(tr.get("event_id") or tr.get("id") or "")
        evts = [tr]
        out.append(
            {
                "group_key": f"unknown:{eid[:8]}",
                "person_ref": f"unknown:{eid[:8]}",
                "person_refs": ["unknown"],
                "display_name": "Chưa định danh",
                "face_id": None,
                "total_frames": int(tr.get("frame_hits") or 1),
                "appearance_count": 1,
                "last_seen": float(tr.get("ts_utc") or 0),
                "first_seen": float(tr.get("ts_utc") or 0),
                **weapon_summary_from_events(evts),
                "representative": tr,
                "events": evts,
            }
        )
    out.sort(key=lambda p: p["last_seen"], reverse=True)
    return out


def _weapon_badge_html(armed: bool, label: str = "", *, dangerous: bool = False) -> str:
    if dangerous:
        text = html.escape(label or "Cảnh báo nguy hiểm")
        return (
            f'<span style="background:#7f1d1d;color:#fff;padding:2px 8px;'
            f'border-radius:4px;font-size:12px;font-weight:700;">{text}</span>'
        )
    if armed:
        text = html.escape(label or "Có vũ khí")
        return (
            f'<span style="background:#b91c1c;color:#fff;padding:2px 8px;'
            f'border-radius:4px;font-size:12px;font-weight:600;">{text}</span>'
        )
    text = html.escape(label or "Không vũ khí")
    return (
        f'<span style="background:#15803d;color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:12px;">{text}</span>'
    )


def expand_events_to_frame_slots(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Mỗi 'frame' = một đơn vị hiển thị (ảnh crop lặp theo frame_hits trong phiên debounce).
  Trả về list {event, frame_index, global_index, crop_url, ts_utc, ...}.
    """
    slots: List[Dict[str, Any]] = []
    events_sorted = sorted(events, key=lambda e: float(e.get("ts_utc") or 0))
    gidx = 0
    for ev in events_sorted:
        hits = max(1, int(ev.get("frame_hits") or 1))
        for fi in range(hits):
            gidx += 1
            slots.append(
                {
                    "event": ev,
                    "frame_index_in_session": fi + 1,
                    "session_frame_hits": hits,
                    "global_frame_index": gidx,
                    "crop_url": ev.get("crop_url"),
                    "ts_utc": ev.get("ts_utc"),
                    "distance": ev.get("distance"),
                    "det_score": ev.get("det_score"),
                    "armed": bool(ev.get("armed")),
                    "dangerous": bool(ev.get("dangerous")),
                    "weapon_types": list(ev.get("weapon_types") or []),
                    "weapon_label": ev.get("weapon_label") or "Không vũ khí",
                    "weapon_crop_url": ev.get("weapon_crop_url"),
                    "weapon_crop_urls": list(ev.get("weapon_crop_urls") or []),
                    "track_scene_url": ev.get("track_scene_url"),
                }
            )
    return slots


def _try_export_cut(
    api_base: str,
    event_id: str,
    *,
    key: str,
) -> None:
    """Gọi GET export-cut và hiện nút tải."""
    url = f"{api_base.rstrip('/')}/ivm/events/{event_id}/export-cut.mp4"
    try:
        r = requests.get(url, timeout=180)
        if r.status_code == 200 and r.content:
            st.session_state[f"cut_bytes_{key}"] = r.content
            st.session_state[f"cut_name_{key}"] = f"cut_{event_id[:8]}.mp4"
            st.success("Đã cắt đoạn video — bấm **Tải cut**.")
        else:
            detail = r.text[:400] if r.text else f"HTTP {r.status_code}"
            st.error(f"Không xuất cut: {detail}")
    except requests.RequestException as ex:
        st.error(f"Không xuất cut: {ex}")


def _weapon_crop_url_list(
    *,
    weapon_crop_urls: Optional[List[Dict[str, Any]]] = None,
    weapon_crop_url: Optional[str] = None,
    weapon_types: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    from identity_vm_app.services.weapon_crops import normalize_weapon_class

    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for item in weapon_crop_urls or []:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not url:
            continue
        cls = normalize_weapon_class(item.get("class"))
        if cls in seen:
            continue
        seen.add(cls)
        out.append({"class": cls, "url": str(url)})
    for raw_t in weapon_types or []:
        cls = normalize_weapon_class(raw_t)
        if cls in seen:
            continue
        seen.add(cls)
        if weapon_crop_url and len(out) == 0 and not weapon_crop_urls:
            out.append({"class": cls, "url": str(weapon_crop_url)})
        elif weapon_crop_urls is None and weapon_crop_url:
            base_url = str(weapon_crop_url).rsplit("/", 1)[0]
            out.append({"class": cls, "url": f"{base_url}/{cls}.jpg"})
    if not out and weapon_crop_url:
        out.append({"class": "weapon", "url": str(weapon_crop_url)})
    return out


def _render_frame_crops(
    *,
    api_base: str,
    face_crop_url: Optional[str] = None,
    weapon_crop_url: Optional[str] = None,
    weapon_crop_urls: Optional[List[Dict[str, Any]]] = None,
    weapon_types: Optional[List[str]] = None,
    track_scene_url: Optional[str] = None,
    armed: bool = False,
    caption: str = "",
) -> None:
    """Scene lớn + crop mặt + crop từng loại vũ khí."""
    wlist = _weapon_crop_url_list(
        weapon_crop_urls=weapon_crop_urls,
        weapon_crop_url=weapon_crop_url,
        weapon_types=weapon_types,
    )

    def _scene() -> None:
        _render_crop_box(
            None, api_base=api_base, crop_url=track_scene_url, width=SCENE_WIDTH
        )

    def _face() -> None:
        _render_crop_box(
            None, api_base=api_base, crop_url=face_crop_url, width=THUMB_PX
        )

    weapon_shows: List[Tuple[str, Callable[[], None]]] = []

    def _make_show(url: str) -> Callable[[], None]:
        def _show() -> None:
            _render_crop_box(None, api_base=api_base, crop_url=url, width=THUMB_PX)

        return _show

    for item in wlist:
        weapon_shows.append((item["class"], _make_show(item["url"])))

    render_track_detail_three_images(
        armed=armed,
        has_scene=bool(track_scene_url),
        has_face=bool(face_crop_url),
        show_scene=_scene,
        show_face=_face,
        weapon_shows=weapon_shows,
    )
    if caption:
        st.caption(caption)


def _render_crop_box(
    event: Optional[Dict[str, Any]],
    *,
    api_base: str,
    caption: str = "",
    crop_url: Optional[str] = None,
    width: Optional[int] = None,
) -> None:
    url = crop_url or (event.get("crop_url") if event else None)
    if not url:
        st.caption("Chưa có ảnh crop.")
        return
    blob = _fetch_crop_bytes(api_base, str(url))
    if blob:
        if width is not None:
            st.image(blob, width=int(width))
        else:
            st.image(blob, use_container_width=True)
    else:
        st.caption("Không tải được crop.")
    if caption:
        st.caption(caption)


def render_identified_person_grid(
    persons: List[Dict[str, Any]],
    *,
    api_base: str,
    cols: int,
    key_prefix: str,
) -> None:
    """Lưới tổng: mỗi tên = một thẻ."""
    if not persons:
        st.info(
            "Chưa có người **đã định danh** trong khoảng thời gian này "
            "(chỉ hiển thị người khớp thư viện khuôn mặt)."
        )
        return

    st.markdown("#### Đối tượng nhận diện (gom theo tên)")
    st.caption("Mỗi thẻ = **một tên** — bấm **Xem chi tiết** để xem toàn bộ frame xuất hiện.")

    n = max(1, min(6, int(cols)))
    for i in range(0, len(persons), n):
        chunk = persons[i : i + n]
        columns = st.columns(len(chunk))
        for col, person in zip(columns, chunk):
            with col:
                name = str(person.get("display_name") or person.get("group_key"))
                st.markdown(f"**{name}**")
                st.markdown(
                    _weapon_badge_html(
                        bool(person.get("has_weapon")),
                        str(person.get("weapon_label") or ""),
                        dangerous=bool(person.get("dangerous")),
                    ),
                    unsafe_allow_html=True,
                )
                rep = person.get("representative")
                _render_crop_box(rep, api_base=api_base)
                refs = person.get("person_refs") or []
                if len(refs) > 1:
                    st.caption(f"{len(refs)} mã trong DB: {', '.join(refs[:3])}")
                st.markdown(
                    f"**{person.get('total_frames', 0)}** frame · "
                    f"**{person.get('appearance_count', 0)}** phiên"
                )
                st.caption(f"Lần cuối: {fmt_ts(person.get('last_seen'))}")
                gk = str(person.get("group_key") or person.get("person_ref"))
                if st.button(
                    "Xem chi tiết",
                    key=f"{key_prefix}_detail_{gk}",
                    use_container_width=True,
                ):
                    st.session_state[f"{key_prefix}_detail_group"] = gk
                    st.rerun()


def render_person_appearance_detail(
    person: Dict[str, Any],
    *,
    api_base: str,
    key_prefix: str,
    camera_id: str = "",
    from_ts: float = 0.0,
    to_ts: float = 0.0,
) -> None:
    """Chi tiết: filmstrip toàn bộ frame + phiên + xuất WebM shortcut."""
    name = str(person.get("display_name") or person.get("group_key"))
    events: List[Dict[str, Any]] = list(person.get("events") or [])
    gk = str(person.get("group_key") or "")

    if st.button("← Quay lại danh sách", key=f"{key_prefix}_detail_back"):
        st.session_state.pop(f"{key_prefix}_detail_group", None)
        st.rerun()

    st.markdown(f"### Chi tiết — **{name}**")
    wsum = weapon_summary_from_events(events)
    has_weapon = bool(wsum["has_weapon"])
    dangerous = bool(wsum.get("dangerous"))
    weapon_label = str(wsum["weapon_label"])
    weapon_types = list(wsum.get("weapon_types") or [])
    st.markdown(
        _weapon_badge_html(has_weapon, weapon_label, dangerous=dangerous),
        unsafe_allow_html=True,
    )
    if has_weapon and weapon_types:
        st.caption(f"Loại phát hiện: **{', '.join(weapon_types)}**")
    frame_slots = expand_events_to_frame_slots(events)
    armed_frames = sum(1 for s in frame_slots if s.get("armed"))
    st.caption(
        f"Tổng **{len(frame_slots)}** frame (ước lượng từ tracking) · "
        f"**{armed_frames}** frame có vũ khí · "
        f"**{person.get('appearance_count', 0)}** phiên xuất hiện · "
        f"{fmt_ts(person.get('first_seen'))} → {fmt_ts(person.get('last_seen'))}"
    )

    if camera_id:
        render_analyze_visual_section(
            api_base,
            camera_id,
            key_prefix=f"{key_prefix}_detail",
            expanded=False,
        )

    if camera_id and gk:
        export_url = f"{api_base.rstrip('/')}/ivm/cameras/{camera_id}/reports/export-webm"
        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "Xuất video WebM (shortcut)",
                key=f"{key_prefix}_export_webm",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Đang ghép crop thành WebM…"):
                    try:
                        r = requests.get(
                            export_url,
                            params={
                                "group_key": gk,
                                "display_name": name,
                                "from_ts": from_ts,
                                "to_ts": to_ts,
                                "fps": 5,
                            },
                            timeout=180,
                        )
                        if r.status_code == 200:
                            st.session_state[f"{key_prefix}_webm_bytes"] = r.content
                            st.session_state[f"{key_prefix}_webm_name"] = (
                                f"bao_cao_{camera_id}_{gk[:24]}.webm"
                            )
                            st.success("Đã tạo video — bấm **Tải WebM** bên cạnh.")
                        else:
                            st.error(f"API {r.status_code}: {r.text[:400]}")
                    except Exception as ex:
                        st.error(str(ex))
        with c2:
            webm_b = st.session_state.get(f"{key_prefix}_webm_bytes")
            if webm_b:
                st.download_button(
                    "Tải WebM",
                    data=webm_b,
                    file_name=st.session_state.get(
                        f"{key_prefix}_webm_name", f"bao_cao_{camera_id}.webm"
                    ),
                    mime="video/webm",
                    key=f"{key_prefix}_dl_webm",
                    use_container_width=True,
                )

    if not events:
        st.info("Không có dữ liệu chi tiết trong khoảng thời gian.")
        return

    st.markdown("##### Toàn bộ frame (theo thời gian)")
    st.caption(
        "Mỗi ô = 1 frame trong luồng nhận diện. Cùng phiên debounce có thể dùng chung ảnh crop "
        "(frame_hits > 1)."
    )
    ncols = st.slider("Cột filmstrip", 3, 10, 6, key=f"{key_prefix}_film_cols")
    ncol = max(3, min(10, int(ncols)))
    for i in range(0, len(frame_slots), ncol):
        chunk = frame_slots[i : i + ncol]
        columns = st.columns(len(chunk))
        for col, slot in zip(columns, chunk):
            with col:
                gi = slot["global_frame_index"]
                si = slot["frame_index_in_session"]
                sh = slot["session_frame_hits"]
                dist = slot.get("distance")
                dist_s = f"{float(dist):.3f}" if dist is not None else "—"
                st.markdown(f"**#{gi}** · φ{si}/{sh}")
                st.markdown(
                    _weapon_badge_html(
                        bool(slot.get("armed")),
                        str(slot.get("weapon_label") or ""),
                        dangerous=bool(slot.get("dangerous")),
                    ),
                    unsafe_allow_html=True,
                )
                _render_frame_crops(
                    api_base=api_base,
                    face_crop_url=slot.get("crop_url"),
                    weapon_crop_url=slot.get("weapon_crop_url"),
                    weapon_crop_urls=slot.get("weapon_crop_urls"),
                    weapon_types=slot.get("weapon_types"),
                    track_scene_url=slot.get("track_scene_url"),
                    armed=bool(slot.get("armed")),
                    caption=f"{fmt_ts(slot.get('ts_utc'))} · d={dist_s}",
                )

    with st.expander("Theo phiên xuất hiện — cắt video archive", expanded=False):
        st.caption(
            "Mỗi phiên = một lần xuất hiện. **Cắt video** lấy từ file RTSP đã ghi khi nhận diện BẬT."
        )
        ncol2 = 3
        events_sorted = sorted(events, key=lambda e: float(e.get("ts_utc") or 0), reverse=True)
        for i in range(0, len(events_sorted), ncol2):
            chunk = events_sorted[i : i + ncol2]
            columns = st.columns(len(chunk))
            for col, ev in zip(columns, chunk):
                with col:
                    hits = int(ev.get("frame_hits") or 1)
                    eid = str(ev.get("event_id") or "")
                    st.markdown(f"**{hits}** frame · {fmt_ts(ev.get('ts_utc'))}")
                    st.markdown(
                        _weapon_badge_html(
                            bool(ev.get("armed")),
                            str(ev.get("weapon_label") or ""),
                            dangerous=bool(ev.get("dangerous")),
                        ),
                        unsafe_allow_html=True,
                    )
                    _render_frame_crops(
                        api_base=api_base,
                        face_crop_url=ev.get("crop_url"),
                        weapon_crop_url=ev.get("weapon_crop_url"),
                        weapon_crop_urls=ev.get("weapon_crop_urls"),
                        weapon_types=ev.get("weapon_types"),
                        track_scene_url=ev.get("track_scene_url"),
                        armed=bool(ev.get("armed")),
                    )
                    has_arch = bool(ev.get("recording_segment_id")) or ev.get("offset_start_s") is not None
                    if eid:
                        ck = f"{key_prefix}_cut_{eid[:8]}"
                        b1, b2 = st.columns(2)
                        with b1:
                            if st.button(
                                "Cắt video",
                                key=f"{ck}_btn",
                                disabled=not has_arch,
                                use_container_width=True,
                            ):
                                with st.spinner("FFmpeg đang cắt…"):
                                    _try_export_cut(api_base, eid, key=ck)
                        with b2:
                            cut_b = st.session_state.get(f"cut_bytes_{ck}")
                            if cut_b:
                                st.download_button(
                                    "Tải cut",
                                    data=cut_b,
                                    file_name=st.session_state.get(f"cut_name_{ck}", "cut.mp4"),
                                    mime="video/mp4",
                                    key=f"{ck}_dl",
                                    use_container_width=True,
                                )
                        if not has_arch:
                            st.caption("Chưa có archive — bật nhận diện (ghi RTSP) rồi thu thập lại.")


def render_camera_track_report(
    camera_ids: List[str],
    *,
    api_get: Callable[..., Dict[str, Any]],
    api_base: str,
    key_prefix: str = "rep",
) -> None:
    if not camera_ids:
        st.warning("Chưa có camera trong `camera_config.json`.")
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        rep_cam = st.selectbox("Camera", camera_ids, key=f"{key_prefix}_cam")
    with c2:
        hours = st.number_input(
            "Số giờ lùi",
            min_value=1,
            max_value=720,
            value=24,
            step=1,
            key=f"{key_prefix}_hours",
        )
    with c3:
        quick = st.selectbox(
            "Hoặc nhanh",
            ["Theo số giờ", "1 giờ", "6 giờ", "7 ngày"],
            key=f"{key_prefix}_quick",
        )

    now = time.time()
    if quick == "1 giờ":
        from_ts, to_ts = now - 3600, now
    elif quick == "6 giờ":
        from_ts, to_ts = now - 6 * 3600, now
    elif quick == "7 ngày":
        from_ts, to_ts = now - 7 * 86400, now
    else:
        to_ts = now
        from_ts = now - float(hours) * 3600.0

    st.caption(f"Khoảng: **{fmt_ts(from_ts)}** → **{fmt_ts(to_ts)}**")

    use_live = st.checkbox(
        "Báo cáo phiên live (DB video_person_reports)",
        value=True,
        key=f"{key_prefix}_use_live",
    )

    if use_live:
        try:
            sess_data = api_get(
                f"/ivm/cameras/{rep_cam}/analyze/sessions",
                params={"from_ts": from_ts, "to_ts": to_ts, "limit": 30},
                timeout=60.0,
            )
            sessions = list(sess_data.get("sessions") or [])
        except Exception as ex:
            sessions = []
            st.warning(f"Không tải phiên live: {ex}")
        if sessions:
            st.markdown("#### Phiên nhận diện gần đây")
            for srow in sessions[:10]:
                jid = srow.get("id")
                title = srow.get("title") or jid
                st.caption(
                    f"**{title}** · {fmt_ts(srow.get('session_start_utc'))} → "
                    f"{fmt_ts(srow.get('session_end_utc'))} · "
                    f"reports {((srow.get('report_counts') or {}).get('person_reports') or 0)}"
                )
                if jid and api_base:
                    st.markdown(
                        f"[Tải session.mp4]({api_base.rstrip('/')}/ivm/cameras/{rep_cam}/analyze/sessions/{jid}/session.mp4)"
                    )
            try:
                fp_data = api_get(
                    f"/ivm/cameras/{rep_cam}/reports/faces-person",
                    params={"from_ts": from_ts, "to_ts": to_ts},
                    timeout=90.0,
                )
                fp_rows = fp_data if isinstance(fp_data, list) else list(fp_data.get("items") or [])
            except Exception:
                fp_rows = []
            if fp_rows:
                st.markdown(f"**{len(fp_rows)}** track (faces-person merge)")
                st.dataframe(
                    [
                        {
                            "tên": r.get("display_name"),
                            "track": r.get("id_tracking"),
                            "clip": r.get("video_clip"),
                            "job": r.get("video_id"),
                        }
                        for r in fp_rows[:200]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                raw_reports = sum(
                    int((srow.get("report_counts") or {}).get("person_reports") or 0)
                    for srow in sessions
                )
                if raw_reports > 0:
                    st.info(
                        "Có báo cáo raw trong DB nhưng chưa có track khuôn mặt merge. "
                        "Cần `IVM_VIDEO_ANALYZE_PERSIST_EMBEDDINGS=1` và phiên nhận diện mới "
                        "(phiên cũ trước bản sửa có thể thiếu `features_face`)."
                    )

    render_analyze_visual_section(
        api_base,
        str(rep_cam),
        key_prefix=f"{key_prefix}_pick",
        expanded=True,
    )

    with st.expander("Xóa báo cáo", expanded=False):
        st.warning(
            "Xóa **toàn bộ sự kiện nhận diện**, ảnh crop, video overlay phân tích và cache export. "
            "**Không** xóa thư viện khuôn mặt đã đăng ký."
        )
        scope = st.radio(
            "Phạm vi",
            ["Tất cả camera", "Chỉ camera đang chọn"],
            horizontal=True,
            key=f"{key_prefix}_clear_scope",
        )
        wipe_arch = st.checkbox(
            "Xóa luôn archive RTSP (file .mkv đã ghi)",
            value=False,
            key=f"{key_prefix}_clear_arch",
        )
        confirm_txt = st.text_input(
            'Gõ **DELETE_REPORTS** để xác nhận',
            key=f"{key_prefix}_clear_confirm",
        )
        if st.button("Xóa toàn bộ báo cáo", type="primary", key=f"{key_prefix}_clear_btn"):
            if confirm_txt != "DELETE_REPORTS":
                st.error('Cần gõ đúng DELETE_REPORTS.')
            else:
                payload: Dict[str, Any] = {
                    "confirm": "DELETE_REPORTS",
                    "wipe_archive": wipe_arch,
                }
                if scope == "Chỉ camera đang chọn":
                    payload["camera_id"] = rep_cam
                try:
                    r = requests.post(
                        f"{api_base.rstrip('/')}/ivm/reports/clear",
                        json=payload,
                        timeout=120,
                    )
                    if r.ok:
                        st.success("Đã xóa báo cáo.")
                        st.json(r.json())
                        st.session_state.pop(f"{key_prefix}_go", None)
                        st.session_state.pop(f"{key_prefix}_detail_group", None)
                        st.rerun()
                    else:
                        st.error(f"{r.status_code}: {r.text[:500]}")
                except requests.RequestException as ex:
                    st.error(str(ex))

    if st.button("Tải báo cáo", type="primary", key=f"{key_prefix}_load"):
        st.session_state[f"{key_prefix}_go"] = rep_cam
        st.session_state[f"{key_prefix}_from"] = from_ts
        st.session_state[f"{key_prefix}_to"] = to_ts
        st.session_state.pop(f"{key_prefix}_detail_group", None)

    load_cam = st.session_state.get(f"{key_prefix}_go")
    if not load_cam:
        st.info("Chọn camera và bấm **Tải báo cáo**.")
        return

    f_from = float(st.session_state.get(f"{key_prefix}_from", from_ts))
    f_to = float(st.session_state.get(f"{key_prefix}_to", to_ts))

    st.markdown(f"### Camera `{load_cam}`")

    try:
        summ_data = api_get(
            f"/ivm/cameras/{load_cam}/reports/summary",
            params={"from_ts": f_from, "to_ts": f_to},
            timeout=60.0,
        )
        s = summ_data.get("summary") or {}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng sự kiện", s.get("total_events", 0))
        m2.metric("Đã định danh", s.get("known_events", 0))
        m3.metric("Chưa biết (unknown)", s.get("unknown_events", 0))
        m4.metric("Số người (đã biết)", s.get("distinct_known_persons", 0))
    except Exception as ex:
        st.error(f"Không tải tổng hợp: {ex}")
        summ_data = {}

    show_unknown = st.checkbox(
        "Hiển thị cả **chưa định danh** (mỗi lượt một thẻ)",
        value=False,
        key=f"{key_prefix}_show_unknown",
    )

    try:
        tr_data = api_get(
            f"/ivm/cameras/{load_cam}/reports/tracks",
            params={
                "from_ts": f_from,
                "to_ts": f_to,
                "limit": 2000,
                "known_only": not show_unknown,
            },
            timeout=90.0,
        )
        tracks = list(tr_data.get("tracks") or [])
    except Exception as ex:
        tracks = []
        st.error(f"Không tải tracks: {ex}. Restart `python main.py`.")

    persons = group_persons_by_display_name(tracks)
    if show_unknown and tracks:
        persons = persons + _group_unknown_tracks(tracks)

    detail_gk = st.session_state.get(f"{key_prefix}_detail_group")

    if detail_gk:
        person = next(
            (p for p in persons if str(p.get("group_key")) == str(detail_gk)),
            None,
        )
        if person:
            render_person_appearance_detail(
                person,
                api_base=api_base,
                key_prefix=key_prefix,
                camera_id=str(load_cam),
                from_ts=f_from,
                to_ts=f_to,
            )
        else:
            st.warning("Không tìm thấy đối tượng trong báo cáo hiện tại.")
            if st.button("Quay lại", key=f"{key_prefix}_detail_miss"):
                st.session_state.pop(f"{key_prefix}_detail_group", None)
                st.rerun()
    else:
        s_sum = (summ_data.get("summary") or {}) if summ_data else {}
        total_ev = int(s_sum.get("total_events") or 0)
        if not persons:
            if total_ev > 0 and int(s_sum.get("unknown_events") or 0) > 0 and not show_unknown:
                st.warning(
                    f"Có **{total_ev}** sự kiện nhưng **0 tên đã biết**. "
                    "Bật hiển thị chưa định danh hoặc đăng ký Face DB."
                )
            elif total_ev == 0:
                st.info("Chưa có sự kiện — bật nhận diện, đợi vài giây, tải lại báo cáo.")
        gal_cols = st.slider("Số cột lưới", 2, 6, 4, key=f"{key_prefix}_gal_cols")
        render_identified_person_grid(
            persons,
            api_base=api_base,
            cols=int(gal_cols),
            key_prefix=key_prefix,
        )

    with st.expander("Dữ liệu JSON (debug)"):
        st.json(
            {
                "summary": summ_data,
                "persons_count": len(persons),
                "tracks_count": len(tracks),
            }
        )
