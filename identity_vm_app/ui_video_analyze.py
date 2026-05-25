"""UI Streamlit: upload video + xem báo cáo người (job chạy ngầm trên backend)."""

from __future__ import annotations

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components

from identity_vm_app import settings as s
from identity_vm_app.services.video_analyze_fps import default_display_name, sample_fps_label
from identity_vm_app.services.video_analyze_media import FACE_THUMB_PX
from identity_vm_app.ui_face_weapon_stack import (
    SCENE_WIDTH,
    THUMB_PX,
    render_track_detail_three_images,
)
from identity_vm_app.services.video_report_merge import (
    filter_segments_by_track_total_frames,
    filter_tracks_min_frames,
    group_tracks_by_display_name,
    track_frame_count,
)
from identity_vm_app.services.video_report_vm import vm_draw_box_url, vm_root_image_url

_TIMEOUT_POST = int(os.getenv("IVM_UI_VIDEO_POST_TIMEOUT", os.getenv("APP_TIMEOUT_VIDEO_POST", "600")))
_POLL_INTERVAL_S = float(os.getenv("IVM_UI_VIDEO_POLL_INTERVAL_S", "3"))

_FPS_OPTIONS: List[Tuple[float, str]] = [
    (0.0, "0 — Full frame (mọi khung)"),
    (5.0, "5 FPS"),
    (10.0, "10 FPS"),
    (15.0, "15 FPS"),
]
_FPS_VALUES = [x[0] for x in _FPS_OPTIONS]
_FPS_LABEL_BY_VALUE = {v: lbl for v, lbl in _FPS_OPTIONS}
_GALLERY_PAGE_SIZE = max(4, min(96, int(os.getenv("IVM_UI_VA_GALLERY_PAGE_SIZE", "24"))))
_IMAGE_CACHE_TTL = max(60, int(os.getenv("IVM_UI_VA_IMAGE_CACHE_TTL", "3600")))


def _track_title_label(row: Dict[str, Any], *, fallback_tid: int = 0) -> str:
    """Tiêu đề track: top N tên nghi ngờ (track_names) hoặc tên đại diện."""
    names = row.get("track_names")
    if isinstance(names, list) and names:
        joined = " · ".join(str(x).strip() for x in names if str(x).strip())
        if joined:
            return joined
    label = row.get("track_name_label")
    if label and str(label).strip():
        return str(label).strip()
    tn = row.get("track_name")
    if isinstance(tn, list):
        joined = " · ".join(str(x).strip() for x in tn if str(x).strip())
        if joined:
            return joined
    if isinstance(tn, str) and tn.strip():
        return tn.strip()
    return str(
        row.get("identity_label") or row.get("display_name") or f"Track {fallback_tid}"
    )


@st.cache_data(ttl=_IMAGE_CACHE_TTL, show_spinner=False)
def _http_image_bytes(url: str) -> Optional[bytes]:
    """Cache bytes — tránh Streamlit gọi lại GET mỗi lần rerun (st.image(url) làm vậy)."""
    try:
        r = requests.get(url, timeout=45)
        if r.status_code == 200 and r.content:
            return r.content
    except requests.RequestException:
        pass
    return None


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_faces_person_cached(base: str, job_id: str) -> List[Dict[str, Any]]:
    r = requests.get(
        f"{base.rstrip('/')}/ivm/api/reports/faces-person",
        params={"video_ids": job_id},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def _st_image_cached(url: Optional[str], **kwargs: Any) -> None:
    if not url:
        return
    blob = _http_image_bytes(url)
    if blob:
        st.image(blob, **kwargs)
    else:
        st.caption("Không tải ảnh")


def _media_url(api_base: str, rel_path: Optional[str]) -> Optional[str]:
    if not rel_path:
        return None
    return f"{api_base.rstrip('/')}/ivm/video-analyze/media?path={requests.utils.quote(str(rel_path), safe='')}"


def _face_thumb_url(api_base: str, row: Dict[str, Any]) -> Optional[str]:
    """Ưu tiên crop file; không có thì API crop từ full frame (VideoMaster-style)."""
    rel = row.get("face_img")
    if rel:
        u = _media_url(api_base, rel)
        if u:
            return u
    rid = row.get("id")
    if rid:
        return f"{api_base.rstrip('/')}/ivm/video-analyze/reports/persons/{rid}/face-image"
    return _media_url(api_base, row.get("img_url"))


def _job_title(j: Dict[str, Any]) -> str:
    return str(j.get("title") or j.get("display_name") or j.get("original_name") or j.get("id") or "?")


def _fetch_jobs(base: str, limit: int = 50) -> List[Dict[str, Any]]:
    try:
        return requests.get(f"{base}/ivm/video-analyze/jobs", params={"limit": limit}, timeout=30).json().get(
            "jobs"
        ) or []
    except requests.RequestException:
        return []


def _job_is_active(j: Dict[str, Any]) -> bool:
    return str(j.get("status_name") or "") in ("pending", "queued", "running")


def render_video_analyze_panel(api_base: str) -> None:
    base = (api_base or "").rstrip("/")
    st.subheader("Phân tích video")
    max_jobs = int(s.IVM_VIDEO_ANALYZE_MAX_CONCURRENT)
    try:
        cfg = requests.get(f"{base}/ivm/video-analyze/config", timeout=10).json()
    except requests.RequestException:
        cfg = {}
    default_split = int(cfg.get("default_split_parts") or s.IVM_VIDEO_ANALYZE_SPLIT_PARTS)
    trk = cfg.get("tracker") or "bytetrack.yaml"
    wpn = "bật" if cfg.get("weapon_detection_enabled") else "tắt/thiếu model"
    vm_note = (
        f"Lưu ảnh: **{'crop file' if cfg.get('save_crops') else 'chỉ full frame (VM)'}** · "
        f"Vũ khí: **{wpn}** · "
        f"Person track: **{'ByteTrack' if cfg.get('yolo_tracking') and cfg.get('yolo_available') else 'IoU bbox'}** "
        f"({cfg.get('yolo_model') or 'yolo26m.pt'} + {trk})"
    )
    lim_parts: List[str] = []
    if cfg.get("max_upload_unlimited"):
        lim_parts.append("API: không giới hạn dung lượng upload")
    elif cfg.get("max_upload_mb"):
        lim_parts.append(f"API: tối đa **{cfg['max_upload_mb']} MB**/file")
    if cfg.get("max_duration_unlimited"):
        lim_parts.append("không giới hạn thời lượng")
    elif cfg.get("max_duration_s"):
        lim_parts.append(f"tối đa **{int(cfg['max_duration_s'])}s** video")
    lim_note = f" · {' · '.join(lim_parts)}" if lim_parts else ""
    st.caption(
        f"FPS: **0 = full frame** (mọi khung — video 2 phút ≈ **3000** lần infer, rất lâu). "
        f"Khuyên dùng **10–15 FPS** cho video dài. Tối đa **{max_jobs}** job song song. "
        f"Chọn **1–4 luồng** khi gửi job (mặc định API: **{default_split}**).{lim_note} {vm_note} · "
        "Tab **Xem track**: scene lớn + 2 crop mặt/vũ khí (`save_crops=1`, job mới)."
    )

    if "ivm_va_selected_job" not in st.session_state:
        st.session_state["ivm_va_selected_job"] = None
    if "ivm_va_watch_jobs" not in st.session_state:
        st.session_state["ivm_va_watch_jobs"] = []

    tab_up, tab_jobs, tab_rep = st.tabs(["Upload & phân tích", "Danh sách video", "Báo cáo người"])

    with tab_up:
        _render_upload_tab(base)

    with tab_jobs:
        _render_jobs_tab(base)

    with tab_rep:
        _render_reports_tab(base)


def _on_video_job_created(body: Dict[str, Any], *, title: str, sample_fps: float) -> None:
    job_id = body.get("job_id")
    if not job_id:
        st.error("Thiếu job_id")
        return
    st.session_state["ivm_va_selected_job"] = job_id
    watch: List[str] = list(st.session_state.get("ivm_va_watch_jobs") or [])
    if job_id not in watch:
        watch.append(job_id)
    st.session_state["ivm_va_watch_jobs"] = watch
    sp = int(body.get("split_parts") or 1)
    split_note = body.get("split_note") or (
        "1 luồng — không cắt file" if sp <= 1 else f"{sp} luồng song song"
    )
    st.success(
        f"Đã gửi **{body.get('display_name', title)}** (`{sample_fps_label(sample_fps)}`, {split_note}). "
        "Xem tiến trình ở tab **Danh sách video**."
    )


def _render_upload_tab(base: str) -> None:
    va_file = st.file_uploader("Tải video", type=["mp4", "avi", "mov", "mkv", "webm"], key="ivm_va_uploader")
    st.caption(
        "Upload trình duyệt do Streamlit giới hạn (~200 MB nếu không chạy `run_ui.bat`). "
        "File lớn: mở **Phân tích file trên máy chạy API** bên dưới."
    )

    fps_idx = _FPS_VALUES.index(0.0)
    va_sample_fps = st.selectbox(
        "Chế độ FPS lấy mẫu",
        options=_FPS_VALUES,
        index=fps_idx,
        format_func=lambda v: _FPS_LABEL_BY_VALUE.get(v, str(v)),
        key="ivm_va_sample_fps",
    )
    if float(va_sample_fps) <= 0:
        st.warning(
            "Full frame: ~25× số khung so với 10 FPS. Video 2 phút có thể **>3000** khung infer + ghi ảnh/DB."
        )

    split_choices = [1, 2, 3, 4]
    try:
        cfg_up = requests.get(f"{base}/ivm/video-analyze/config", timeout=10).json()
        def_split = int(cfg_up.get("default_split_parts") or 4)
    except requests.RequestException:
        def_split = int(s.IVM_VIDEO_ANALYZE_SPLIT_PARTS)
    def_split = max(1, min(4, def_split))
    split_idx = split_choices.index(def_split) if def_split in split_choices else 0
    va_split_parts = st.selectbox(
        "Số luồng phân tích",
        options=split_choices,
        index=split_idx,
        format_func=lambda n: (
            "1 — phân tích trực tiếp (không cắt file, phù hợp video ngắn)"
            if n == 1
            else f"{n} — cắt {n} đoạn, chạy song song"
        ),
        key="ivm_va_split_parts",
        help="Video ngắn: chọn 1 luồng để bỏ bước cắt ffmpeg và phân tích file gốc.",
    )

    fname = va_file.name if va_file else "upload.mp4"
    auto_name = default_display_name(fname, float(va_sample_fps))
    va_display_name = st.text_input(
        "Tên job (hiển thị danh sách)",
        value=auto_name,
        help="Ví dụ cùng file: «Camera A — Full frame» và «Camera A — 5 FPS»",
        key="ivm_va_display_name",
    )

    if st.button("Gửi phân tích", type="primary", key="ivm_va_run", disabled=va_file is None):
        fname = va_file.name or "upload.mp4"
        title = (va_display_name or "").strip() or default_display_name(fname, float(va_sample_fps))
        try:
            pr = requests.post(
                f"{base}/ivm/video-analyze/jobs",
                files={"file": (fname, va_file.getvalue(), "application/octet-stream")},
                data={
                    "sample_fps": str(int(va_sample_fps) if va_sample_fps > 0 else 0),
                    "display_name": title,
                    "split_parts": str(int(va_split_parts)),
                },
                timeout=_TIMEOUT_POST,
            )
        except requests.RequestException as ex:
            st.error(f"Không gửi được job: {ex}")
            return
        if pr.status_code >= 400:
            st.error(f"Lỗi ({pr.status_code}): {pr.text[:600]}")
            return
        _on_video_job_created(pr.json() or {}, title=title, sample_fps=float(va_sample_fps))

    with st.expander("Phân tích file trên máy chạy API (không qua upload Streamlit)"):
        local_path = st.text_input(
            "Đường dẫn file video",
            placeholder=r"E:\videos\camera01.mp4",
            key="ivm_va_local_path",
            help="Đường dẫn trên máy đang chạy API (python main.py), không phải máy trình duyệt từ xa.",
        )
        if st.button("Phân tích từ đường dẫn", key="ivm_va_run_local", disabled=not (local_path or "").strip()):
            path_str = (local_path or "").strip()
            path_name = Path(path_str).name or "video.mp4"
            title_local = (va_display_name or "").strip() or default_display_name(
                path_name, float(va_sample_fps)
            )
            try:
                pr = requests.post(
                    f"{base}/ivm/video-analyze/jobs/from-path",
                    data={
                        "video_path": path_str,
                        "sample_fps": str(int(va_sample_fps) if va_sample_fps > 0 else 0),
                        "display_name": title_local,
                        "split_parts": str(int(va_split_parts)),
                    },
                    timeout=_TIMEOUT_POST,
                )
            except requests.RequestException as ex:
                st.error(f"Không gửi được job: {ex}")
                return
            if pr.status_code >= 400:
                st.error(f"Lỗi ({pr.status_code}): {pr.text[:600]}")
                return
            _on_video_job_created(pr.json() or {}, title=title_local, sample_fps=float(va_sample_fps))


def _render_jobs_tab(base: str) -> None:
    auto = st.checkbox(
        f"Tự làm mới khi có job đang chạy (mỗi {_POLL_INTERVAL_S:.0f}s)",
        value=True,
        key="ivm_va_auto_refresh",
    )
    if st.button("Làm mới ngay", key="ivm_va_refresh_jobs"):
        st.rerun()

    pending_del_job = st.session_state.pop("ivm_va_pending_del_job", None)
    pending_del_rep = st.session_state.pop("ivm_va_pending_del_rep", None)
    if pending_del_job:
        _delete_job(base, str(pending_del_job))
        return
    if pending_del_rep:
        _delete_reports(base, str(pending_del_rep))
        return

    jobs = _fetch_jobs(base)
    watch_ids = set(st.session_state.get("ivm_va_watch_jobs") or [])
    completed_watch: List[str] = []

    for jid in list(watch_ids):
        match = next((j for j in jobs if str(j.get("id")) == jid), None)
        if match and str(match.get("status_name")) == "done":
            completed_watch.append(jid)
        elif match and str(match.get("status_name")) == "error":
            st.warning(f"**{_job_title(match)}** lỗi: {match.get('message') or match.get('error_code')}")
            completed_watch.append(jid)

    if completed_watch:
        for jid in completed_watch:
            watch_ids.discard(jid)
        st.session_state["ivm_va_watch_jobs"] = list(watch_ids)
        done_names = [_job_title(next((j for j in jobs if str(j.get("id")) == c), {"id": c})) for c in completed_watch]
        st.success(f"Hoàn tất: {', '.join(done_names)}")

    if not jobs:
        st.info("Chưa có video nào.")
    else:
        for j in jobs:
            jid = str(j.get("id"))
            st_name = str(j.get("status_name") or "?")
            idx = int(j.get("index_frame") or 0)
            tot = int(j.get("total_sample_frames") or 0)
            sf_lbl = j.get("sample_fps_label") or sample_fps_label(float(j.get("sample_fps") or 0))
            split_lbl = j.get("split_parts_label") or ""
            orig = j.get("original_name") or ""

            st.markdown(f"**{_job_title(j)}**")
            split_note = f" · {split_lbl}" if split_lbl else ""
            st.caption(f"File: `{orig}` · {sf_lbl}{split_note} · `{jid[:8]}…`")

            cols = st.columns([2, 2, 1, 1, 1])
            cols[0].write(f"Trạng thái: `{st_name}`")
            if st_name == "error":
                err_msg = (j.get("message") or j.get("error_code") or "").strip()
                if err_msg:
                    cols[0].caption(err_msg[:220] + ("…" if len(err_msg) > 220 else ""))
            prog_txt = f"{idx}/{tot} khung" if tot > 0 else "—"
            cols[1].write(prog_txt)
            if tot > 0 and st_name in ("running", "queued", "pending"):
                st.progress(min(1.0, idx / tot), text=f"{idx}/{tot}")

            if cols[2].button("Báo cáo", key=f"pick_job_{jid}"):
                st.session_state["ivm_va_selected_job"] = jid
                st.rerun()

            is_active = bool(j.get("is_active"))
            can_del = not is_active
            if cols[3].button(
                "Xóa BC",
                key=f"del_rep_{jid}",
                disabled=is_active,
                help="Xóa báo cáo + ảnh, giữ file video",
            ):
                st.session_state["ivm_va_pending_del_rep"] = jid
                st.rerun()

            if cols[4].button(
                "Xóa",
                key=f"del_job_{jid}",
                disabled=is_active,
                help="Xóa job, video và báo cáo",
            ):
                st.session_state["ivm_va_pending_del_job"] = jid
                st.rerun()

            with st.expander("Đổi tên job", expanded=False):
                new_name = st.text_input("Tên mới", value=_job_title(j), key=f"rename_{jid}")
                if st.button("Lưu tên", key=f"save_rename_{jid}"):
                    _rename_job(base, jid, new_name)

            st.divider()

    if auto and any(bool(j.get("is_active")) for j in jobs):
        time.sleep(_POLL_INTERVAL_S)
        st.rerun()


def _delete_job(base: str, job_id: str) -> None:
    try:
        r = requests.delete(f"{base}/ivm/video-analyze/jobs/{job_id}", timeout=60)
    except requests.RequestException as ex:
        st.error(str(ex))
        return
    if r.status_code == 409:
        st.error("Job vẫn đang phân tích trên server — đợi xong rồi xóa lại.")
        return
    if r.status_code >= 400:
        st.error(r.text[:400])
        return
    if st.session_state.get("ivm_va_selected_job") == job_id:
        st.session_state["ivm_va_selected_job"] = None
    _http_image_bytes.clear()
    _fetch_faces_person_cached.clear()
    _fetch_person_clips_cached.clear()
    _fetch_clip_subdata_cached.clear()
    _fetch_track_segments_cached.clear()
    _fetch_track_subdata_cached.clear()
    st.success("Đã xóa job và file video.")
    st.rerun()


def _delete_reports(base: str, job_id: str) -> None:
    try:
        r = requests.delete(f"{base}/ivm/video-analyze/jobs/{job_id}/reports", timeout=120)
    except requests.RequestException as ex:
        st.error(str(ex))
        return
    if r.status_code == 409:
        st.error("Job đang chạy — chưa xóa được báo cáo.")
        return
    if r.status_code >= 400:
        st.error(r.text[:400])
        return
    body = r.json() or {}
    _http_image_bytes.clear()
    _fetch_faces_person_cached.clear()
    _fetch_person_clips_cached.clear()
    _fetch_clip_subdata_cached.clear()
    _fetch_track_segments_cached.clear()
    _fetch_track_subdata_cached.clear()
    st.success(f"Đã xóa {body.get('reports_deleted', 0)} dòng báo cáo.")
    st.rerun()


def _rename_job(base: str, job_id: str, name: str) -> None:
    try:
        r = requests.patch(
            f"{base}/ivm/video-analyze/jobs/{job_id}",
            json={"display_name": (name or "").strip()},
            timeout=30,
        )
    except requests.RequestException as ex:
        st.error(str(ex))
        return
    if r.status_code >= 400:
        st.error(r.text[:400])
        return
    st.success("Đã đổi tên.")
    st.rerun()


def _render_reports_tab(base: str) -> None:
    job_id = st.session_state.get("ivm_va_selected_job")
    jobs = _fetch_jobs(base, limit=100)
    job_ids = [str(j["id"]) for j in jobs if j.get("id")]
    labels = {str(j["id"]): _job_title(j) for j in jobs}

    if job_ids:
        idx = job_ids.index(job_id) if job_id in job_ids else 0
        pick = st.selectbox(
            "Chọn job",
            job_ids,
            index=idx,
            format_func=lambda x: labels.get(x, x),
            key="ivm_va_report_job_pick",
        )
        st.session_state["ivm_va_selected_job"] = pick
        job_id = pick
        picked = next((j for j in jobs if str(j.get("id")) == pick), None)
        if picked:
            st.caption(
                f"File: `{picked.get('original_name', '')}` · "
                f"{picked.get('sample_fps_label') or sample_fps_label(float(picked.get('sample_fps') or 0))}"
            )
        if picked and str(picked.get("status_name")) not in ("done",):
            st.info(f"Job đang `{picked.get('status_name')}` — báo cáo có thể chưa đủ.")

        act = st.columns(3)
        is_active = bool(picked.get("is_active")) if picked else True
        if act[0].button("Xóa báo cáo job này", disabled=is_active, key="ivm_va_del_rep_tab"):
            st.session_state["ivm_va_pending_del_rep"] = job_id
            st.rerun()
        if act[1].button("Xóa toàn bộ job", disabled=is_active, key="ivm_va_del_job_tab"):
            st.session_state["ivm_va_pending_del_job"] = job_id
            st.rerun()
    elif job_id:
        st.caption(f"Job: `{job_id}`")
    else:
        st.info("Chưa chọn job — upload hoặc chọn từ danh sách.")
        return

    st.link_button(
        "Tải CSV",
        f"{base}/ivm/video-analyze/jobs/{job_id}/reports/export.csv?merged=true",
    )

    tab_track, tab_segments, tab_search = st.tabs(
        ["Theo tracking (định danh)", "Đoạn xuất hiện (track)", "Tìm kiếm mặt"]
    )
    with tab_track:
        _render_vm_face_tab(base, job_id)
    with tab_segments:
        _render_track_segments_tab(base, job_id)
    with tab_search:
        _render_face_search(base, job_id)


def _vm_frame_url(base: str, row: Dict[str, Any]) -> Optional[str]:
    u = row.get("img_url_full") or vm_root_image_url(base, row.get("img_url"))
    return u or _media_url(base, row.get("img_url"))


def _vm_person_url(base: str, row: Dict[str, Any]) -> Optional[str]:
    u = row.get("person_image_url")
    if u:
        return f"{base.rstrip('/')}{u}" if str(u).startswith("/") else str(u)
    rid = row.get("id")
    if rid:
        return f"{base.rstrip('/')}/ivm/reports/get-person-image/{rid}"
    return None


def _vm_face_url(base: str, row: Dict[str, Any]) -> Optional[str]:
    u = _face_thumb_url(base, row)
    if u:
        return u
    rid = row.get("first_frame_report_id") or row.get("id")
    if rid:
        return f"{base.rstrip('/')}/ivm/reports/get-face-image/{rid}"
    return None


def _suspect_face_url(base: str, suspect: Dict[str, Any]) -> Optional[str]:
    u = suspect.get("face_image_url")
    if u:
        return u
    rid = suspect.get("report_id")
    if rid:
        return f"{base.rstrip('/')}/ivm/reports/get-face-image/{rid}"
    return None


def _render_top_suspect_faces(
    base: str,
    suspects: List[Dict[str, Any]],
    *,
    key_prefix: str,
    limit: int = 5,
) -> None:
    """Top 5 khuôn mặt nghi ngờ (crop mặt) thay cho dòng % bỏ phiếu."""
    items = [s for s in (suspects or []) if isinstance(s, dict)][:limit]
    if not items:
        st.caption("_Chưa có khớp thư viện trong track._")
        return
    st.caption("**Top khuôn mặt nghi ngờ**")
    cols = st.columns(min(limit, len(items)))
    for i, col in enumerate(cols):
        with col:
            if i >= len(items):
                continue
            s = items[i]
            _st_image_cached(
                _suspect_face_url(base, s),
                width=FACE_THUMB_PX,
            )
            pct = float(s.get("vote_ratio") or 0) * 100
            sc = s.get("match_score")
            sc_txt = f" · {float(sc) * 100:.0f}%" if sc is not None else ""
            st.caption(
                f"#{s.get('rank', i + 1)} **{s.get('display_name') or '?'}**\n"
                f"{int(s.get('vote_count') or 0)} khung · {pct:.0f}%{sc_txt}"
            )


def _gallery_page_index(page_key: str, n_pages: int) -> int:
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    try:
        page = int(st.session_state[page_key])
    except (TypeError, ValueError):
        page = 1
    page = max(1, min(int(n_pages), page))
    st.session_state[page_key] = page
    return page


def _render_gallery_pagination(
    page_key: str,
    n_pages: int,
    *,
    total_hint: str = "",
) -> int:
    """Thanh chuyển trang ở cuối lưới — Trước / Sau."""
    page = _gallery_page_index(page_key, n_pages)
    if n_pages <= 1:
        if total_hint:
            st.caption(total_hint)
        return 1
    st.markdown("---")
    prev_c, mid_c, next_c = st.columns([1, 2, 1])
    with prev_c:
        if st.button(
            "← Trước",
            key=f"{page_key}_prev",
            disabled=page <= 1,
            use_container_width=True,
        ):
            st.session_state[page_key] = page - 1
            st.rerun()
    with mid_c:
        st.markdown(f"<p style='text-align:center;margin:0'><b>Trang {page} / {n_pages}</b></p>", unsafe_allow_html=True)
        if total_hint:
            st.caption(total_hint)
    with next_c:
        if st.button(
            "Sau →",
            key=f"{page_key}_next",
            disabled=page >= n_pages,
            use_container_width=True,
            type="primary",
        ):
            st.session_state[page_key] = page + 1
            st.rerun()
    return page


def _vm_gallery_thumb_url(base: str, row: Dict[str, Any]) -> Optional[str]:
    """Lưới tổng quan: khung đầu tiên xuất hiện + vẽ box (đồng đều)."""
    u = row.get("thumb_image_url") or row.get("draw_box_url")
    if u:
        return u
    rid = row.get("first_frame_report_id") or row.get("id")
    if rid:
        return vm_draw_box_url(base, str(rid))
    return _vm_track_thumb_url(base, row)


def _row_has_face_crop(row: Dict[str, Any]) -> bool:
    if row.get("face_img"):
        return True
    bf = row.get("box_face")
    if bf is None:
        return False
    s = str(bf).strip().lower()
    return bool(s and s not in ("none", "null", "[]"))


def _vm_track_scene_url(base: str, row: Dict[str, Any]) -> Optional[str]:
    u = row.get("track_scene_url")
    if u:
        return f"{base.rstrip('/')}{u}" if str(u).startswith("/") else str(u)
    rid = row.get("id")
    if rid:
        return f"{base.rstrip('/')}/ivm/reports/get-track-scene-image/{rid}"
    return None


def _vm_weapon_crop_urls(base: str, row: Dict[str, Any]) -> List[Dict[str, str]]:
    from identity_vm_app.services.video_report_vm import vm_weapon_crop_urls

    return vm_weapon_crop_urls(base, row)


def _weapon_badge_md(row: Dict[str, Any]) -> str:
    dangerous = bool(int(row.get("dangerous") or 0)) or str(
        row.get("weapon_status") or ""
    ).strip() == "nguy_hiem"
    armed = bool(int(row.get("armed") or 0))
    label = str(row.get("weapon_label") or ("Có vũ khí" if armed else "Không vũ khí"))
    if dangerous:
        bg = "#7f1d1d"
    elif armed:
        bg = "#b91c1c"
    else:
        bg = "#15803d"
    return (
        f'<span style="background:{bg};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:12px;">{label}</span>'
    )


def _render_face_weapon_crops_stacked(base: str, row: Dict[str, Any]) -> None:
    """Scene lớn + crop mặt + từng crop vũ khí (gun, knife, …)."""
    armed = bool(int(row.get("armed") or 0))
    scene_u = _vm_track_scene_url(base, row)
    face_u = _vm_face_url(base, row)
    has_face = bool(face_u and _row_has_face_crop(row))
    weapon_urls = _vm_weapon_crop_urls(base, row)

    def _scene() -> None:
        _st_image_cached(scene_u, width=SCENE_WIDTH)

    def _face() -> None:
        _st_image_cached(face_u, width=THUMB_PX)

    weapon_shows: List[Tuple[str, Callable[[], None]]] = []

    def _make_weapon_show(url: str) -> Callable[[], None]:
        def _show() -> None:
            _st_image_cached(url, width=THUMB_PX)

        return _show

    for item in weapon_urls:
        cls = str(item.get("class") or "weapon")
        url = item.get("url")
        if url:
            weapon_shows.append((cls, _make_weapon_show(url)))

    render_track_detail_three_images(
        armed=armed,
        has_scene=bool(scene_u),
        has_face=has_face,
        show_scene=_scene,
        show_face=_face,
        weapon_shows=weapon_shows,
        match_row=row,
    )


def _vm_track_thumb_url(base: str, row: Dict[str, Any]) -> Optional[str]:
    """Crop box người (person)."""
    u = row.get("track_image_url") or row.get("person_image_url")
    if u:
        return u
    rid = row.get("first_frame_report_id") or row.get("id")
    if rid:
        return f"{base.rstrip('/')}/ivm/reports/get-person-image/{rid}"
    return None


def _vm_track_identity_thumb_url(base: str, row: Dict[str, Any]) -> Optional[str]:
    """Ảnh đại diện track: crop mặt nếu có, không thì crop người."""
    if _row_has_face_crop(row):
        u = _vm_face_url(base, row)
        if u:
            return u
    return _vm_track_thumb_url(base, row)


def _prepare_face_report_cards(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lọc track ngắn + gom cùng tên thành một thẻ."""
    return group_tracks_by_display_name(filter_tracks_min_frames(reports))


def _segment_min_track_frames() -> int:
    """Ẩn track tổng < 10 khung (hoặc IVM_REPORT_MIN_TRACK_FRAMES nếu lớn hơn)."""
    return s.ivm_report_min_track_frames()


def _filter_segments_by_track_min_frames(
    segments: List[Dict[str, Any]],
    *,
    min_frames: int,
) -> List[Dict[str, Any]]:
    return filter_segments_by_track_total_frames(
        segments, min_track_frames=min_frames
    )


def _prepare_segment_cards(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lọc track tổng < ngưỡng + gom đoạn cùng tên thành một thẻ (giống tab định danh)."""
    min_f = _segment_min_track_frames()
    filtered = _filter_segments_by_track_min_frames(segments, min_frames=min_f)
    return group_tracks_by_display_name(filtered)


def _find_segment_group_card(
    cards: List[Dict[str, Any]],
    group_key: str,
) -> Optional[Dict[str, Any]]:
    return _find_face_group_card(cards, group_key)


def _vm_segment_thumb_url(base: str, seg: Dict[str, Any]) -> Optional[str]:
    u = seg.get("thumb_image_url") or seg.get("draw_box_url")
    if u:
        return f"{base.rstrip('/')}{u}" if str(u).startswith("/") else str(u)
    rid = seg.get("first_frame_report_id") or seg.get("report_id")
    if rid:
        return vm_draw_box_url(base, str(rid))
    return _vm_gallery_thumb_url(base, seg)


def _render_segment_group_members(
    base: str,
    job_id: str,
    card: Dict[str, Any],
    *,
    group_sel_key: str,
    seg_sel_key: str,
) -> None:
    """Danh sách đoạn trong một nhóm tên — bấm mới mở chi tiết từng đoạn."""
    members = card.get("member_tracks") or []
    if not members:
        st.warning("Nhóm không còn đoạn.")
        return
    title = _track_title_label(members[0])
    st.subheader(title)
    st.caption(
        f"{len(members)} đoạn · tổng {int(card.get('hit_count_total') or 0)} khung mẫu"
    )
    if st.button("← Quay lại lưới", key=f"ivm_va_seg_group_back_{job_id}"):
        st.session_state.pop(group_sel_key, None)
        st.rerun()
    ordered = sorted(
        members,
        key=lambda m: (
            float(m.get("time_analyze") or 0),
            int(m.get("id_tracking") or 0),
            int(m.get("segment_index") or 0),
        ),
    )
    for idx, m in enumerate(ordered):
        tid = int(m.get("id_tracking") or 0)
        seg_idx = int(m.get("segment_index") or 0)
        t0 = float(m.get("time_analyze") or 0)
        t1 = float(m.get("end_time") or t0)
        nf = track_frame_count(m)
        c1, c2 = st.columns([1, 3])
        with c1:
            _st_image_cached(_vm_segment_thumb_url(base, m), width=FACE_THUMB_PX)
        with c2:
            st.markdown(
                f"**Track {tid}** · đoạn {seg_idx + 1} · {nf} khung · {t0:.1f}s–{t1:.1f}s"
            )
            if int(m.get("armed") or 0):
                st.markdown(_weapon_badge_md(m), unsafe_allow_html=True)
            if st.button(
                "Xem đoạn",
                key=f"ivm_va_seg_open_member_{job_id}_{idx}_{tid}_{seg_idx}",
            ):
                st.session_state[seg_sel_key] = m
                st.rerun()


def _render_segment_detail(
    base: str,
    job_id: str,
    seg: Dict[str, Any],
    *,
    seg_sel_key: str,
    group_sel_key: str,
) -> None:
    """Chi tiết một đoạn track (video + khung)."""
    tid = int(seg.get("id_tracking") or 0)
    seg_idx = int(seg.get("segment_index") or 0)
    back_label = (
        "← Quay lại danh sách đoạn"
        if st.session_state.get(group_sel_key)
        else "← Quay lại lưới"
    )
    if st.button(back_label, key=f"ivm_va_seg_back_{job_id}"):
        st.session_state.pop(seg_sel_key, None)
        st.session_state.pop(f"ivm_va_seg_video_{job_id}", None)
        st.session_state.pop(f"ivm_va_seg_vid_bytes_{job_id}_{tid}_{seg_idx}", None)
        st.rerun()
    name = _track_title_label(seg, fallback_tid=tid)
    t0 = float(seg.get("time_analyze") or 0)
    t1 = float(seg.get("end_time") or t0)
    st.subheader(name)
    st.caption(
        f"track {tid} · đoạn #{seg_idx + 1} · {t0:.1f}s–{t1:.1f}s · "
        f"{int(seg.get('frame_count') or seg.get('hit_count') or 0)} khung mẫu"
    )
    if int(seg.get("armed") or 0):
        st.markdown(_weapon_badge_md(seg), unsafe_allow_html=True)
    _render_top_suspect_faces(
        base,
        seg.get("suspect_faces") or [],
        key_prefix=f"ivm_va_seg_suspects_{job_id}_{tid}_{seg_idx}",
    )

    try:
        sub = _fetch_track_subdata_cached(base, job_id, tid)
    except requests.RequestException as ex:
        st.error(str(ex))
        return
    subdata = [
        r
        for r in (sub.get("subdata") or [])
        if t0 <= float(r.get("time_analyze") or r.get("time_analyze_s") or 0) <= t1 + 0.001
    ]
    playable = [r for r in subdata if r.get("id")]

    st.markdown("##### Video đoạn (mượt như gốc + box)")
    st.caption(
        "Video gốc full FPS; box **giữ ổn định** giữa các khung mẫu (cập nhật khi tới mẫu mới). "
        "Bấm **Ghép lại** nếu vẫn thấy giật từ bản cache cũ."
    )
    video_key = f"ivm_va_seg_video_{job_id}"
    rebuild_key = f"{video_key}_rebuild"
    bytes_key = f"ivm_va_seg_vid_bytes_{job_id}_{tid}_{seg_idx}"
    c1, c2 = st.columns(2)
    if c1.button(
        "Tạo & xem video",
        key=f"ivm_va_seg_build_vid_{job_id}_{tid}_{seg_idx}",
        type="primary",
    ):
        st.session_state[video_key] = True
        st.session_state[rebuild_key] = False
        st.session_state.pop(bytes_key, None)
        st.rerun()
    if c2.button("Ghép lại", key=f"ivm_va_seg_rebuild_{job_id}_{tid}_{seg_idx}"):
        st.session_state[video_key] = True
        st.session_state[rebuild_key] = True
        st.session_state.pop(bytes_key, None)
        st.rerun()
    if st.session_state.get(video_key):
        rebuild_vid = bool(st.session_state.get(rebuild_key))
        if bytes_key not in st.session_state or rebuild_vid:
            try:
                with st.spinner(
                    "Đang cắt video gốc và vẽ box (lần đầu có thể 30–90 giây)…"
                ):
                    st.session_state[bytes_key] = _fetch_track_segment_video_bytes(
                        base,
                        job_id,
                        tid,
                        seg_idx,
                        rebuild=rebuild_vid,
                    )
                st.session_state[rebuild_key] = False
            except requests.RequestException as ex:
                st.error(f"Không tạo được video: {ex}")
            except ValueError as ex:
                st.error(str(ex))
        vid_bytes = st.session_state.get(bytes_key)
        if vid_bytes:
            st.video(vid_bytes, format="video/mp4")
            st.caption(
                f"MP4 H.264 · {len(vid_bytes) // 1024} KB · full FPS từ file gốc."
            )

    with st.expander("Phát ảnh từng khung (~5 FPS, giống VideoMaster)", expanded=False):
        _render_vm_style_slideshow(
            base,
            playable or subdata,
            state_prefix=f"ivm_va_seg_slide_{job_id}_{tid}_{seg_idx}",
        )

    with st.expander("Từng khung + crop", expanded=False):
        _render_frame_player(
            base,
            subdata,
            state_prefix=f"ivm_va_seg_frames_{job_id}_{tid}_{seg_idx}",
            show_draw_box=True,
            show_crops=True,
        )


def _find_face_group_card(
    cards: List[Dict[str, Any]],
    group_key: str,
) -> Optional[Dict[str, Any]]:
    for card in cards:
        if str(card.get("group_key") or "") == str(group_key):
            return card
    return None


def _render_face_group_members(
    base: str,
    job_id: str,
    card: Dict[str, Any],
    *,
    group_sel_key: str,
    track_sel_key: str,
) -> None:
    """Danh sách track trong một nhóm tên — bấm mới mở chi tiết từng track."""
    members = card.get("member_tracks") or []
    if not members:
        st.warning("Nhóm không còn track.")
        return
    title = _track_title_label(members[0])
    st.subheader(title)
    st.caption(
        f"{len(members)} track · tổng {int(card.get('hit_count_total') or 0)} khung mẫu"
    )
    if st.button("← Quay lại lưới", key=f"ivm_va_face_group_back_{job_id}"):
        st.session_state.pop(group_sel_key, None)
        st.rerun()
    ordered = sorted(
        members,
        key=lambda m: float(m.get("time_analyze") or m.get("time_analyze_s") or 0),
    )
    for idx, m in enumerate(ordered):
        tid = int(m.get("id_tracking") or 0)
        t0 = float(m.get("time_analyze") or 0)
        t1 = float(m.get("end_time") or t0)
        nf = track_frame_count(m)
        c1, c2 = st.columns([1, 3])
        with c1:
            _st_image_cached(_vm_track_identity_thumb_url(base, m), width=FACE_THUMB_PX)
        with c2:
            st.markdown(f"**Track {tid}** · {nf} khung · {t0:.1f}s–{t1:.1f}s")
            if st.button(
                "Xem track",
                key=f"ivm_va_face_open_member_{job_id}_{idx}_{tid}",
            ):
                st.session_state[track_sel_key] = tid
                st.rerun()


def _vm_clip_thumb_url(base: str, clip_row: Dict[str, Any]) -> Optional[str]:
    rid = clip_row.get("id")
    if not rid:
        return None
    return f"{base.rstrip('/')}/ivm/reports/draw-box-all-person/{rid}"


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_person_clips_cached(base: str, job_id: str) -> List[Dict[str, Any]]:
    r = requests.get(
        f"{base.rstrip('/')}/ivm/api/reports/persons",
        params={"video_ids": job_id},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_track_segments_cached(base: str, job_id: str) -> List[Dict[str, Any]]:
    r = requests.get(
        f"{base.rstrip('/')}/ivm/api/reports/track-segments",
        params={"video_ids": job_id},
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def _track_segment_video_url(
    base: str,
    job_id: str,
    id_tracking: int,
    segment_index: int,
    *,
    rebuild: bool = False,
) -> str:
    u = (
        f"{base.rstrip('/')}/ivm/api/reports/track-segment-video/"
        f"{job_id}/{int(id_tracking)}?segment_index={int(segment_index)}&draw_boxes=1"
    )
    if rebuild:
        u += "&rebuild=1"
    return u


def _fetch_track_segment_video_bytes(
    base: str,
    job_id: str,
    id_tracking: int,
    segment_index: int,
    *,
    rebuild: bool = False,
    timeout_s: int = 600,
) -> bytes:
    """Tải trọn file MP4 (tránh st.video(url) → 206 + đứt kết nối trên Windows)."""
    url = _track_segment_video_url(
        base, job_id, id_tracking, segment_index, rebuild=rebuild
    )
    r = requests.get(url, timeout=timeout_s)
    r.raise_for_status()
    if not r.content:
        raise ValueError("API trả video rỗng")
    return r.content


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_clip_subdata_cached(base: str, job_id: str, video_clip: int) -> Dict[str, Any]:
    r = requests.get(
        f"{base.rstrip('/')}/ivm/api/reports/get-sub-data-video-clip/{job_id}/{video_clip}",
        timeout=120,
    )
    r.raise_for_status()
    return r.json() if isinstance(r.json(), dict) else {}


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_track_subdata_cached(base: str, job_id: str, id_tracking: int) -> Dict[str, Any]:
    r = requests.get(
        f"{base.rstrip('/')}/ivm/reports/get-person-sub-data/{job_id}/{id_tracking}",
        timeout=120,
    )
    r.raise_for_status()
    return r.json() if isinstance(r.json(), dict) else {}


def _segment_slideshow_interval_ms() -> int:
    return int(s.IVM_TRACK_SEGMENT_SLIDESHOW_MS)


def _slideshow_frame_urls(base: str, frames: List[Dict[str, Any]]) -> List[str]:
    urls: List[str] = []
    for row in frames:
        rid = row.get("id")
        if rid:
            urls.append(vm_draw_box_url(base, str(rid)))
    return urls


def _render_vm_style_slideshow(
    base: str,
    frames: List[Dict[str, Any]],
    *,
    state_prefix: str,
    interval_ms: Optional[int] = None,
) -> None:
    """
    Phát đoạn giống VideoMaster SubDataVideoPlayer: lật ảnh draw-box ~200ms/khung
    (không dùng MP4 — mượt + box khớp từng khung mẫu).
    """
    if not frames:
        st.caption("Không có khung.")
        return
    urls = _slideshow_frame_urls(base, frames)
    if not urls:
        st.caption("Không có ảnh draw-box (thiếu report id).")
        return

    ms = int(interval_ms or _segment_slideshow_interval_ms())
    idx_key = f"{state_prefix}_slide_idx"
    auto_key = f"{state_prefix}_slide_auto"
    n = len(frames)
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0
    idx = int(st.session_state[idx_key]) % n

    def _show_at(i: int) -> None:
        row = frames[i]
        t = float(row.get("time_analyze") or row.get("time_analyze_s") or 0)
        st.markdown(_weapon_badge_md(row), unsafe_allow_html=True)
        u = urls[i] if i < len(urls) else None
        if u:
            _st_image_cached(u, width="stretch")
        st.caption(
            f"Khung **{i + 1} / {n}** · **{t:.1f}s** · "
            f"~{1000.0 / max(ms, 1):.0f} khung/giây (giống VideoMaster)"
        )

    use_fragment = hasattr(st, "fragment")
    if use_fragment and st.session_state.get(auto_key):

        @st.fragment(run_every=timedelta(milliseconds=ms))
        def _auto_tick() -> None:
            if not st.session_state.get(auto_key):
                return
            st.session_state[idx_key] = (int(st.session_state.get(idx_key, 0)) + 1) % n
            _show_at(int(st.session_state[idx_key]) % n)
            c1, c2, c3 = st.columns(3)
            if c1.button("⏸ Dừng", key=f"{state_prefix}_slide_pause_frag"):
                st.session_state[auto_key] = False
            if c2.button("⏮", key=f"{state_prefix}_slide_prev_frag"):
                st.session_state[idx_key] = (int(st.session_state[idx_key]) - 1) % n
                st.session_state[auto_key] = False
            if c3.button("⏭", key=f"{state_prefix}_slide_next_frag"):
                st.session_state[idx_key] = (int(st.session_state[idx_key]) + 1) % n
                st.session_state[auto_key] = False

        _auto_tick()
    elif st.session_state.get(auto_key):
        uid = "".join(c if c.isalnum() else "_" for c in state_prefix)
        caps = [
            f"Khung {i + 1}/{n} · {float(row.get('time_analyze') or row.get('time_analyze_s') or 0):.1f}s"
            for i, row in enumerate(frames)
        ]
        components.html(
            f"""
<div style="text-align:center;background:#1a1a1a;padding:6px;border-radius:8px;">
  <img id="ivm_slide_{uid}" style="max-width:100%;max-height:min(70vh,720px);height:auto;" />
  <p id="ivm_cap_{uid}" style="color:#ccc;font-size:13px;margin:8px 0 0;"></p>
</div>
<script>
(function() {{
  const frames = {json.dumps(urls)};
  const caps = {json.dumps(caps)};
  let i = {idx % max(len(urls), 1)};
  const img = document.getElementById("ivm_slide_{uid}");
  const cap = document.getElementById("ivm_cap_{uid}");
  function show() {{
    img.src = frames[i];
    cap.textContent = caps[i] || "";
    i = (i + 1) % frames.length;
  }}
  show();
  setInterval(show, {ms});
}})();
</script>
            """,
            height=560,
            scrolling=False,
        )
        if st.button("⏸ Dừng slideshow", key=f"{state_prefix}_slide_pause_html"):
            st.session_state[auto_key] = False
            st.rerun()
    else:
        _show_at(idx)
        if n > 1:
            picked = st.slider(
                "Khung",
                0,
                n - 1,
                idx,
                key=f"{state_prefix}_slide_slider",
            )
            if int(picked) != idx:
                st.session_state[idx_key] = int(picked)
                st.rerun()
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("⏮", key=f"{state_prefix}_slide_prev", disabled=n <= 1):
            st.session_state[idx_key] = (idx - 1) % n
            st.rerun()
        if c2.button("▶ Phát", key=f"{state_prefix}_slide_play", disabled=n <= 1):
            st.session_state[auto_key] = True
            st.rerun()
        if c3.button("⏭", key=f"{state_prefix}_slide_next", disabled=n <= 1):
            st.session_state[idx_key] = (idx + 1) % n
            st.rerun()
        if c4.button("⏹", key=f"{state_prefix}_slide_stop"):
            st.session_state[auto_key] = False
            st.rerun()


def _render_frame_player(
    base: str,
    frames: List[Dict[str, Any]],
    *,
    state_prefix: str,
    show_draw_box: bool = False,
    show_crops: bool = True,
) -> None:
    """Trình phát khung full-frame — giống SubDataVideoPeople / SubDataVideoPlayer."""
    if not frames:
        st.caption("Không có khung.")
        return
    n = len(frames)
    if n <= 1:
        idx = 0
    else:
        idx = st.slider(
            "Khung",
            0,
            n - 1,
            0,
            key=f"{state_prefix}_slider",
        )
    row = frames[int(idx)]
    t = float(row.get("time_analyze") or row.get("time_analyze_s") or 0)
    if show_draw_box and row.get("id"):
        img_u = vm_draw_box_url(base, str(row["id"]))
    else:
        img_u = _vm_frame_url(base, row)
    st.markdown(_weapon_badge_md(row), unsafe_allow_html=True)
    main, side = st.columns([3, 1])
    with main:
        _st_image_cached(img_u, width="stretch", caption=f"Khung {idx + 1}/{n} · {t:.1f}s")
    with side:
        st.caption(f"**{row.get('display_name') or '—'}**")
        st.caption(f"track {row.get('id_tracking')} · clip {row.get('video_clip')}")
    if show_crops and row.get("id"):
        st.markdown("##### Crop mặt & vũ khí (khung đang xem)")
        _render_face_weapon_crops_stacked(base, row)


def _render_track_segments_tab(base: str, job_id: str) -> None:
    """Đoạn xuất hiện — gom cùng tên; ẩn track tổng < ngưỡng khung (mặc định 10)."""
    seg_sel_key = f"ivm_va_track_seg_sel_{job_id}"
    group_sel_key = f"ivm_va_track_seg_group_sel_{job_id}"
    min_frames = _segment_min_track_frames()
    try:
        segments = _fetch_track_segments_cached(base, job_id)
    except requests.RequestException as ex:
        st.error(str(ex))
        return
    if not segments:
        st.info("Chưa có đoạn xuất hiện theo track.")
        return

    cards = _prepare_segment_cards(segments)
    if not cards:
        st.info(
            f"Không còn track nào (tổng khung track ≥ {min_frames}, giống tab định danh). "
            f"Có {len(segments)} đoạn thô nhưng không track nào đủ dài."
        )
        return

    selected = st.session_state.get(seg_sel_key)
    if selected is not None:
        _render_segment_detail(
            base,
            job_id,
            selected,
            seg_sel_key=seg_sel_key,
            group_sel_key=group_sel_key,
        )
        return

    selected_group = st.session_state.get(group_sel_key)
    if selected_group is not None:
        card = _find_segment_group_card(cards, str(selected_group))
        if card is None:
            st.session_state.pop(group_sel_key, None)
            st.rerun()
        _render_segment_group_members(
            base,
            job_id,
            card,
            group_sel_key=group_sel_key,
            seg_sel_key=seg_sel_key,
        )
        return

    page_key = f"ivm_va_seg_page_{job_id}"
    n_pages = max(1, (len(cards) + _GALLERY_PAGE_SIZE - 1) // _GALLERY_PAGE_SIZE)
    page = _gallery_page_index(page_key, n_pages)
    n_segments = sum(int(c.get("track_count") or 1) for c in cards)
    st.caption(
        f"{len(cards)} thẻ · {n_segments} đoạn (ẩn track tổng < {min_frames} khung) · "
        f"cùng tên gom một thẻ · bấm thẻ → danh sách đoạn"
    )
    start = (page - 1) * _GALLERY_PAGE_SIZE
    page_rows = cards[start : start + _GALLERY_PAGE_SIZE]
    cols_per = 4
    for i in range(0, len(page_rows), cols_per):
        cols = st.columns(cols_per)
        batch = page_rows[i : i + cols_per]
        for col_idx, (col, row) in enumerate(zip(cols, batch)):
            row_idx = start + i + col_idx
            tid = int(row.get("id_tracking") or 0)
            seg_n = int(row.get("track_count") or 1)
            nf = int(row.get("hit_count_total") or track_frame_count(row))
            with col:
                _st_image_cached(_vm_segment_thumb_url(base, row), width=FACE_THUMB_PX)
                t0 = row.get("time_analyze") or 0
                t1 = row.get("end_time") or t0
                name = _track_title_label(row, fallback_tid=tid)
                if seg_n > 1:
                    meta = f"**{name}** · {seg_n} đoạn · {nf} khung"
                else:
                    seg_idx = int(row.get("segment_index") or 0)
                    meta = (
                        f"**{name}** · track {tid} · đoạn {seg_idx + 1}\n"
                        f"{float(t0):.1f}s–{float(t1):.1f}s · {nf} khung"
                    )
                st.caption(meta)
                if int(row.get("armed") or 0):
                    st.markdown(_weapon_badge_md(row), unsafe_allow_html=True)
                btn_label = "Xem chi tiết" if seg_n > 1 else "Xem đoạn"
                if st.button(btn_label, key=f"ivm_va_seg_open_{job_id}_{row_idx}"):
                    if seg_n > 1:
                        st.session_state[group_sel_key] = row.get("group_key")
                    else:
                        st.session_state[seg_sel_key] = row
                    st.session_state.pop(f"ivm_va_seg_video_{job_id}", None)
                    st.rerun()
    _render_gallery_pagination(
        page_key,
        n_pages,
        total_hint=f"Hiển thị {start + 1}–{start + len(page_rows)} / {len(cards)} thẻ",
    )


def _render_vm_face_tab(base: str, job_id: str) -> None:
    """VideoMaster Face tab: faces-person → get-person-sub-data timeline."""
    track_sel_key = f"ivm_va_face_track_sel_{job_id}"
    group_sel_key = f"ivm_va_face_group_sel_{job_id}"
    min_frames = s.ivm_report_min_track_frames()
    try:
        reports = _fetch_faces_person_cached(base, job_id)
    except requests.RequestException as ex:
        st.error(str(ex))
        return
    if not reports:
        st.info(
            "Chưa có báo cáo khuôn mặt (job cần `IVM_VIDEO_ANALYZE_PERSIST_EMBEDDINGS=1`)."
        )
        return

    cards = _prepare_face_report_cards(reports)
    if not cards:
        st.info(
            f"Không còn track nào ≥ {min_frames} khung mẫu "
            f"(đã ẩn {len(reports)} track ngắn / nhiễu)."
        )
        return

    selected_tid = st.session_state.get(track_sel_key)
    if selected_tid is not None:
        back_label = (
            "← Quay lại danh sách track"
            if st.session_state.get(group_sel_key)
            else "← Quay lại lưới"
        )
        if st.button(back_label, key=f"ivm_va_face_back_{job_id}"):
            st.session_state.pop(track_sel_key, None)
            st.rerun()
        tid = int(selected_tid)
        track_row = next(
            (r for r in reports if int(r.get("id_tracking") or 0) == tid),
            None,
        )
        title = _track_title_label(track_row or {}, fallback_tid=tid)
        st.subheader(title)
        st.caption(f"id_tracking {tid} · {track_frame_count(track_row or {})} khung mẫu")
        if track_row:
            st.markdown(_weapon_badge_md(track_row), unsafe_allow_html=True)
            _render_top_suspect_faces(
                base,
                track_row.get("suspect_faces") or [],
                key_prefix=f"ivm_va_track_detail_{job_id}_{tid}",
            )
        try:
            sub = _fetch_track_subdata_cached(base, job_id, tid)
        except requests.RequestException as ex:
            st.error(str(ex))
            return
        subdata = sub.get("subdata") or []
        if not subdata:
            st.caption("Không có subdata.")
            return
        n = len(subdata)
        if n <= 1:
            idx = 0
        else:
            idx = st.slider(
                "Thời điểm trong track",
                0,
                n - 1,
                0,
                key=f"ivm_va_face_slider_{job_id}_{tid}",
            )
        row = subdata[int(idx)]
        st.markdown("##### Crop mặt & vũ khí")
        st.markdown(_weapon_badge_md(row), unsafe_allow_html=True)
        _render_face_weapon_crops_stacked(base, row)
        st.caption(
            f"Khung {int(idx) + 1}/{n} · {float(row.get('time_analyze') or 0):.1f}s · "
            f"{row.get('display_name') or '?'} · clip {row.get('video_clip')}"
        )
        st.markdown("##### Khung full + box (người / mặt / vũ khí)")
        _render_frame_player(
            base,
            subdata,
            state_prefix=f"ivm_va_face_frames_{job_id}_{tid}",
            show_draw_box=True,
            show_crops=False,
        )
        return

    selected_group = st.session_state.get(group_sel_key)
    if selected_group is not None:
        card = _find_face_group_card(cards, str(selected_group))
        if card is None:
            st.session_state.pop(group_sel_key, None)
            st.rerun()
        _render_face_group_members(
            base,
            job_id,
            card,
            group_sel_key=group_sel_key,
            track_sel_key=track_sel_key,
        )
        return

    page_key = f"ivm_va_face_page_{job_id}"
    n_pages = max(1, (len(cards) + _GALLERY_PAGE_SIZE - 1) // _GALLERY_PAGE_SIZE)
    page = _gallery_page_index(page_key, n_pages)
    n_tracks = sum(int(c.get("track_count") or 1) for c in cards)
    st.caption(
        f"{len(cards)} thẻ · {n_tracks} track (ẩn track < {min_frames} khung) · "
        f"cùng tên gom một thẻ · ảnh = crop mặt/người"
    )
    start = (page - 1) * _GALLERY_PAGE_SIZE
    page_rows = cards[start : start + _GALLERY_PAGE_SIZE]
    cols_per = 4
    for i in range(0, len(page_rows), cols_per):
        cols = st.columns(cols_per)
        batch = page_rows[i : i + cols_per]
        for col_idx, (col, row) in enumerate(zip(cols, batch)):
            row_idx = start + i + col_idx
            tid = int(row.get("id_tracking") or 0)
            track_n = int(row.get("track_count") or 1)
            nf = int(row.get("hit_count_total") or track_frame_count(row))
            with col:
                _st_image_cached(
                    _vm_track_identity_thumb_url(base, row),
                    width=FACE_THUMB_PX,
                )
                t0 = row.get("time_analyze") or 0
                t1 = row.get("end_time") or t0
                name = _track_title_label(row, fallback_tid=tid)
                if track_n > 1:
                    meta = f"**{name}** · {track_n} track · {nf} khung"
                else:
                    meta = (
                        f"**{name}** · track {tid} · "
                        f"{float(t0):.1f}s–{float(t1):.1f}s · {nf} khung"
                    )
                st.caption(meta)
                if track_n <= 1:
                    suspects = row.get("suspect_faces") or []
                    if suspects:
                        _render_top_suspect_faces(
                            base,
                            suspects,
                            key_prefix=f"ivm_va_face_card_{job_id}_{row_idx}",
                        )
                btn_label = "Xem chi tiết" if track_n > 1 else "Xem track"
                if st.button(btn_label, key=f"ivm_va_face_open_{job_id}_{row_idx}"):
                    if track_n > 1:
                        st.session_state[group_sel_key] = row.get("group_key")
                    else:
                        st.session_state[track_sel_key] = tid
                    st.rerun()
    _render_gallery_pagination(
        page_key,
        n_pages,
        total_hint=f"Hiển thị {start + 1}–{start + len(page_rows)} / {len(cards)} thẻ",
    )


def _render_face_search(base: str, job_id: str) -> None:
    st.caption("Tìm trong báo cáo job (giống VideoMaster search-faces-person)")
    up = st.file_uploader("Ảnh khuôn mặt", type=["jpg", "jpeg", "png"], key="ivm_va_search_face")
    pct = st.slider("Ngưỡng %", 0, 98, 50, key="ivm_va_search_pct")
    if st.button("Tìm", key="ivm_va_search_btn", disabled=up is None):
        try:
            pr = requests.post(
                f"{base}/ivm/api/reports/search-faces-person",
                data={"video_ids": job_id, "percent": str(int(pct)), "start_time": "0", "end_time": "0"},
                files={"img": (up.name, up.getvalue(), "image/jpeg")},
                timeout=120,
            )
        except requests.RequestException as ex:
            st.error(str(ex))
            return
        if pr.status_code >= 400:
            st.error(pr.text[:500])
            return
        body = pr.json() or {}
        persons = body.get("persons") or []
        st.success(f"Tìm thấy {len(persons)} đối tượng (gom track)")
        for row in persons[:24]:
            _st_image_cached(_vm_face_url(base, row), width=FACE_THUMB_PX)
            label = row.get("identity_label") or row.get("display_name") or f"Track {row.get('id_tracking')}"
            st.caption(
                f"{label} · {row.get('percent', row.get('match_score', ''))} "
                f"· track {row.get('id_tracking')}"
            )
