"""
Dashboard: xem camera trực tiếp, bật/tắt nhận diện (API), báo cáo & xuất CSV.

Chạy:
  streamlit run ui.py --server.port 8510

Cần backend: python backend.py (API + worker camera).
"""

from __future__ import annotations

import csv
import io
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from identity_vm_app.preview.ffmpeg_env import apply_ffmpeg_capture_env

apply_ffmpeg_capture_env()

import cv2
import requests
import streamlit as st

from identity_vm_app import settings as ivm_settings


def _page_config_once() -> None:
    if st.session_state.get("_ivm_page_cfg"):
        return
    st.set_page_config(
        page_title="Identity VM — Camera & Báo cáo",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state._ivm_page_cfg = True


API_DEFAULT = os.getenv("IVM_UI_API_URL", f"http://127.0.0.1:{ivm_settings.IVM_API_PORT}")


def _session_api() -> str:
    return st.session_state.get("api_base", API_DEFAULT).rstrip("/")


def _get(path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
    r = requests.get(f"{_session_api()}{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _post(path: str, json_body: Optional[Dict[str, Any]] = None, timeout: float = 30.0) -> Dict[str, Any]:
    r = requests.post(f"{_session_api()}{path}", json=json_body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _open_capture(source: Any) -> Optional[cv2.VideoCapture]:
    try:
        if isinstance(source, int):
            cap = cv2.VideoCapture(source)
        elif isinstance(source, str) and source.strip().isdigit():
            cap = cv2.VideoCapture(int(source.strip()))
        else:
            cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(ivm_settings.IVM_CAP_PROP_BUFFERSIZE))
        except Exception:
            pass
        if not cap.isOpened():
            cap.release()
            return None
        return cap
    except Exception:
        return None


@st.cache_resource(show_spinner="Đang mở luồng camera…")
def _rtsp_capture(cache_key: str, source: Any):
    """
    Giữ VideoCapture trong cache Streamlit (ổn qua mỗi lần rerun).
    cache_key phải đổi khi đổi camera/nguồn để tạo kết nối mới.
    """
    cap = _open_capture(source)
    if cap is None:
        raise RuntimeError("Không mở được nguồn (RTSP/webcam). Kiểm tra URL, mạng, FFmpeg.")
    # Xả vài frame đầu (RTSP thường đen vài frame)
    for _ in range(15):
        cap.read()
    return cap


def _stop_live_preview() -> None:
    st.session_state.live_on = False
    st.session_state.live_cache_key = None
    try:
        _rtsp_capture.clear()
    except Exception:
        pass


def _load_cameras() -> List[Dict[str, Any]]:
    data = _get("/ivm/cameras")
    return list(data.get("cameras") or [])


def _csv_report(summary: Dict[str, Any], subjects: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Khối", "Trường", "Giá trị"])
    for k, v in summary.items():
        w.writerow(["summary", k, v])
    w.writerow([])
    w.writerow(
        [
            "person_ref",
            "display_name",
            "face_id",
            "appearances_count",
            "first_seen",
            "last_seen",
            "avg_distance",
            "known",
        ]
    )
    for row in subjects:
        w.writerow(
            [
                row.get("person_ref"),
                row.get("display_name"),
                row.get("face_id"),
                row.get("appearances_count"),
                row.get("first_seen"),
                row.get("last_seen"),
                row.get("avg_distance"),
                row.get("known"),
            ]
        )
    return buf.getvalue()


def main() -> None:
    _page_config_once()
    if "api_base" not in st.session_state:
        st.session_state.api_base = API_DEFAULT
    if "live_on" not in st.session_state:
        st.session_state.live_on = False
    if "live_cache_key" not in st.session_state:
        st.session_state.live_cache_key = None
    if "live_grid_on" not in st.session_state:
        st.session_state.live_grid_on = False

    st.sidebar.title("Kết nối API")
    st.session_state.api_base = st.sidebar.text_input(
        "Base URL Identity VM",
        value=st.session_state.api_base,
        help="Phải trùng cổng backend (mặc định IVM_API_PORT)",
    )

    try:
        health = _get("/ivm/health", timeout=5.0)
        st.sidebar.success(f"API: **{health.get('status', 'ok')}** — model `{health.get('model_tag', '')}`")
    except Exception as e:
        st.sidebar.error(f"Không kết nối được API: {e}")
        st.info("Hãy chạy **`python backend.py`** (hoặc `python identity_vm_app/main.py`) rồi tải lại trang.")
        return

    tab_live, tab_ctl, tab_rep = st.tabs(["Xem camera trực tiếp", "Bật / tắt nhận diện", "Báo cáo & xuất"])

    # ——— Tab live ———
    with tab_live:
        st.subheader("Luồng xem trực tiếp (OpenCV)")
        st.caption("Chỉ hiển thị trên máy chạy Streamlit; không thay cho VMS. Nhấn **Dừng** trước khi đổi camera.")
        st.info(
            "**Camera hiển thị ở tab này** — tab **« Xem camera trực tiếp »** (tab **đầu tiên bên trái**). "
            "Cuộn xuống dưới dòng **Chọn camera** → nhấn **▶ Bắt đầu xem** — "
            "khung hình RTSP/webcam xuất hiện **ngay phía dưới** hai nút *Bắt đầu xem* / *Dừng xem*. "
            "Chế độ **Lưới**: chọn radio *Lưới* rồi **▶ Mở lưới (tất cả)**. "
            "Cần sidebar báo API **ok** và máy chạy Streamlit **truy cập được** URL trong `camera_config.json`."
        )

        try:
            cams = _load_cameras()
        except Exception as e:
            st.error(f"Không đọc danh sách camera: {e}")
            cams = []

        if not cams:
            st.warning("Chưa có camera trong `camera_config.json`.")
        else:
            ids = [str(c["id"]) for c in cams]
            view_mode = st.radio(
                "Cách xem",
                ["Một camera", "Lưới (tất cả camera trong config)"],
                horizontal=True,
                key="live_view_mode",
            )
            choice = st.selectbox("Chọn camera", ids, key="live_select")

            if view_mode.startswith("Một"):
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("▶ Bắt đầu xem", type="primary"):
                        st.session_state.live_grid_on = False
                        src = next(c["source"] for c in cams if str(c["id"]) == choice)
                        _stop_live_preview()
                        key = f"{choice}|{repr(src)}"
                        st.session_state.live_cache_key = key
                        st.session_state.live_on = True
                        try:
                            _ = _rtsp_capture(key, src)
                        except Exception as e:
                            st.error(str(e))
                            st.session_state.live_on = False
                            st.session_state.live_cache_key = None
                with col_b:
                    if st.button("⏹ Dừng xem"):
                        _stop_live_preview()

                if st.session_state.live_on and st.session_state.live_cache_key:
                    if choice != st.session_state.live_cache_key.split("|", 1)[0]:
                        st.warning("Bạn đã đổi camera trong danh sách — nhấn **Bắt đầu xem** lại cho đúng nguồn.")
                    else:
                        try:
                            src = next(c["source"] for c in cams if str(c["id"]) == choice)
                            cap = _rtsp_capture(st.session_state.live_cache_key, src)
                            ok, frame = cap.read()
                            if ok and frame is not None and frame.size > 0:
                                st.image(
                                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                    channels="RGB",
                                    use_container_width=True,
                                )
                            else:
                                st.warning(
                                    "Không đọc được frame — RTSP có thể gián đoạn. Thử **Dừng** rồi **Bắt đầu xem** lại."
                                )
                        except Exception as e:
                            st.error(f"Lỗi luồng: {e}")
                            _stop_live_preview()

                if st.session_state.live_on:
                    time.sleep(0.05)
                    st.rerun()
            else:
                st.caption(
                    "Mỗi ô là một kết nối RTSP riêng. NVR/camera có thể giới hạn số luồng — nếu lỗi, dùng chế độ **Một camera**."
                )
                g1, g2 = st.columns(2)
                with g1:
                    if st.button("▶ Mở lưới (tất cả)", type="primary", key="grid_open"):
                        _stop_live_preview()
                        st.session_state.live_grid_on = True
                with g2:
                    if st.button("⏹ Đóng lưới", key="grid_close"):
                        st.session_state.live_grid_on = False
                        try:
                            _rtsp_capture.clear()
                        except Exception:
                            pass

                if st.session_state.get("live_grid_on"):
                    for row_start in range(0, len(cams), 2):
                        chunk = cams[row_start : row_start + 2]
                        cols = st.columns(len(chunk))
                        for col_i, cam in enumerate(chunk):
                            cid = str(cam["id"])
                            src = cam["source"]
                            ck = f"grid|{cid}|{repr(src)}"
                            with cols[col_i]:
                                st.caption(f"`{cid}`")
                                try:
                                    cap = _rtsp_capture(ck, src)
                                    ok, frame = cap.read()
                                    if ok and frame is not None and frame.size > 0:
                                        st.image(
                                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                            channels="RGB",
                                            use_container_width=True,
                                        )
                                    else:
                                        st.warning("Không có frame")
                                except Exception as e:
                                    st.error(str(e))
                    time.sleep(0.08)
                    st.rerun()

        try:
            if (
                not st.session_state.get("live_on")
                and not st.session_state.get("live_grid_on")
            ) or (time.time() - float(st.session_state.get("_ivm_app_fetch_ts", 0))) >= 5.0:
                st.session_state._ivm_app_fetch_ts = time.time()
                recent = _get(
                    "/ivm/people/appearances",
                    params={"camera_id": st.session_state.get("live_select"), "limit": 8},
                    timeout=10.0,
                )
            else:
                recent = {"items": st.session_state.get("_ivm_cached_appearances", [])}
            items = recent.get("items") or []
            st.session_state._ivm_cached_appearances = items
            if items:
                st.subheader("Lượt xuất hiện gần đây (camera đang chọn)")
                st.dataframe(items, use_container_width=True, hide_index=True)
        except Exception:
            pass

    # ——— Tab điều khiển nhận diện ———
    with tab_ctl:
        st.subheader("Điều khiển nhận diện (worker `backend.py`)")
        st.caption(
            "Khi **Tắt**, worker vẫn giữ kết nối RTSP nhưng **không** gọi API nhận diện — không ghi sự kiện. "
            "Cần tiến trình **`python backend.py`** (có worker camera)."
        )

        try:
            states = _get("/ivm/cameras/analyze", timeout=10.0).get("states") or {}
        except Exception as e:
            st.error(str(e))
            states = {}

        for cid, en in sorted(states.items()):
            row1, row2 = st.columns([3, 1])
            with row1:
                st.write(f"**{cid}** — hiện: `{'BẬT' if en else 'TẮT'}`")
            with row2:
                if st.button("Đảo trạng thái", key=f"flip_{cid}"):
                    try:
                        _post(f"/ivm/cameras/{cid}/analyze", json_body={"enabled": not en})
                        st.success("Đã cập nhật.")
                        st.rerun()
                    except Exception as ex:
                        st.error(str(ex))

        if not states:
            st.info("Chưa có camera trong cấu hình.")

    # ——— Tab báo cáo ———
    with tab_rep:
        st.subheader("Báo cáo nhận diện theo camera (tương tự tổng hợp VisionMaster)")
        try:
            cams2 = _load_cameras()
        except Exception:
            cams2 = []
        cam_ids = [str(c["id"]) for c in cams2] or ["cam0"]

        c1, _ = st.columns(2)
        with c1:
            rep_cam = st.selectbox("Camera", cam_ids, key="rep_cam")

        if st.button("Tải báo cáo nhanh (24 giờ)", type="primary"):
            st.session_state["_rep_from"] = time.time() - 86400
            st.session_state["_rep_to"] = time.time()
            st.session_state["_rep_go"] = True

        c3, c4 = st.columns(2)
        d_to = datetime.now().date()
        d_from = d_to - timedelta(days=1)
        with c3:
            df = st.date_input("Từ ngày", value=d_from, key="d_from")
        with c4:
            dt = st.date_input("Đến ngày", value=d_to, key="d_to")

        if st.button("Tải báo cáo theo khoảng ngày"):
            t0 = datetime.combine(df, datetime.min.time()).timestamp()
            t1 = datetime.combine(dt, datetime.max.time()).timestamp()
            st.session_state["_rep_from"] = t0
            st.session_state["_rep_to"] = t1
            st.session_state["_rep_go"] = True

        if st.session_state.get("_rep_go"):
            f_from = float(st.session_state.get("_rep_from", time.time() - 86400))
            f_to = float(st.session_state.get("_rep_to", time.time()))
            try:
                data = _get(
                    f"/ivm/cameras/{rep_cam}/reports/summary",
                    params={"from_ts": f_from, "to_ts": f_to},
                    timeout=60.0,
                )
                summ = data.get("summary") or {}
                subs = data.get("subjects") or []

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Tổng sự kiện", summ.get("total_events", 0))
                m2.metric("Đã nhận diện tên", summ.get("known_events", 0))
                m3.metric("Chưa nhận diện", summ.get("unknown_events", 0))
                m4.metric("Số người (khớp)", summ.get("distinct_known_persons", 0))

                st.json(
                    {
                        "khoảng_UTC": {
                            "from_ts": summ.get("from_ts"),
                            "to_ts": summ.get("to_ts"),
                        },
                        "avg_distance": summ.get("avg_distance"),
                    }
                )

                st.subheader("Chi tiết theo người / unknown")
                if subs:
                    st.dataframe(subs, use_container_width=True, hide_index=True)
                else:
                    st.info("Không có bản ghi trong khoảng thời gian này.")

                csv_text = _csv_report(summ, subs)
                st.download_button(
                    "Tải CSV báo cáo",
                    data=csv_text.encode("utf-8-sig"),
                    file_name=f"bao_cao_{rep_cam}_{int(time.time())}.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Lỗi báo cáo: {e}")
            st.session_state["_rep_go"] = False


if __name__ == "__main__":
    main()
