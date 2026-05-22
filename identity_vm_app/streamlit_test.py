"""
Giao diện Streamlit — test & vận hành Identity VM.

Chạy từ identity_vm_app (2 terminal):

  cd E:\\app_face\\identity_vm_app
  python main.py
  streamlit run streamlit_test.py --server.port 8510
  (hoặc run_ui.bat)

API: http://127.0.0.1:8010 — sidebar / IVM_UI_API_URL
Tab **Phân tích video**: upload file → job /ivm/video-analyze/jobs
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import html
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from identity_vm_app.preview.ffmpeg_env import apply_ffmpeg_capture_env

apply_ffmpeg_capture_env()

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw, ImageFont

from identity_vm_app import settings as ivm_settings
from identity_vm_app.ui_recognition_controls import render_per_camera_recognition_panel
from identity_vm_app.ui_weapon_alerts import weapon_alerts_auto_refresh_fragment
from identity_vm_app.ui_track_gallery import render_camera_track_report
from identity_vm_app.ui_video_analyze import render_video_analyze_panel

st.set_page_config(page_title="Identity VM Test", layout="wide")

API_DEFAULT = os.getenv("IVM_UI_API_URL", f"http://127.0.0.1:{ivm_settings.IVM_API_PORT}")
PEOPLE_GALLERY_PAGE_SIZE = 20
_REG_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _mime_from_name(name: str) -> str:
    e = Path(name).suffix.lower()
    if e == ".png":
        return "image/png"
    if e in (".jpg", ".jpeg"):
        return "image/jpeg"
    if e == ".webp":
        return "image/webp"
    if e == ".bmp":
        return "image/bmp"
    return "image/jpeg"


def _images_from_zip_bytes(zbuf: bytes) -> List[tuple[str, bytes]]:
    out: List[tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(zbuf), "r") as zf:
        for m in zf.infolist():
            if m.is_dir():
                continue
            raw_name = m.filename
            if not raw_name or raw_name.startswith("__MACOSX/") or "/__MACOSX/" in raw_name.replace("\\", "/"):
                continue
            if Path(raw_name).name.startswith("."):
                continue
            suf = Path(raw_name).suffix.lower()
            if suf not in _REG_IMAGE_EXT:
                continue
            try:
                data = zf.read(m.filename)
            except Exception:
                continue
            if not data:
                continue
            safe_name = Path(raw_name).as_posix().replace("/", "_").replace("\\", "_")
            if not safe_name:
                continue
            out.append((safe_name, data))
    return out


def _list_local_image_paths(root: Path, recursive: bool) -> List[Path]:
    """Chỉ quét đường dẫn (chưa đọc bytes) — để hiển thị tiến trình khi đọc."""
    root = root.expanduser().resolve()
    if not root.is_dir():
        return []
    it = root.rglob("*") if recursive else root.glob("*")
    paths: List[Path] = []
    for p in it:
        if not p.is_file():
            continue
        if p.suffix.lower() not in _REG_IMAGE_EXT:
            continue
        if p.name.startswith("."):
            continue
        try:
            p.resolve().relative_to(root)
        except ValueError:
            continue
        paths.append(p)
    paths.sort(key=lambda x: str(x).lower())
    return paths


def _bytes_from_path_list(
    root: Path,
    paths: List[Path],
    *,
    progress: Optional[Any] = None,
    status: Optional[Any] = None,
) -> List[tuple[str, bytes]]:
    """Đọc bytes từ danh sách path đã quét (root dùng cho tên giả lập)."""
    out: List[tuple[str, bytes]] = []
    root = root.expanduser().resolve()
    n = len(paths)
    for i, p in enumerate(paths):
        if progress is not None and n:
            progress.progress((i + 1) / n)
        if status is not None:
            status.markdown(f"**Đọc từ đĩa** ({i + 1}/{n}): `{p.name}`")
        try:
            rel = p.relative_to(root)
            pseudo = str(rel).replace(os.sep, "_").replace("/", "_")
            out.append((pseudo, p.read_bytes()))
        except OSError:
            continue
    return out


def _images_from_local_dir(
    root: Path,
    recursive: bool,
    *,
    progress: Optional[Any] = None,
    status: Optional[Any] = None,
) -> List[tuple[str, bytes]]:
    paths = _list_local_image_paths(root, recursive)
    if not paths:
        return []
    return _bytes_from_path_list(root, paths, progress=progress, status=status)


def _run_register_items_progress(
    items: List[tuple[str, bytes]],
    data0: Dict[str, str],
    *,
    phase_label: str = "Đăng ký qua API",
) -> tuple[int, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Gửi lần lượt từng ảnh lên /ivm/register. items: (tên file gửi API, bytes)."""
    n = len(items)
    ok_n = 0
    err_n = 0
    cards: List[Dict[str, Any]] = []
    batch_errors: List[Dict[str, Any]] = []
    prog = st.progress(0)
    status = st.empty()
    t0 = time.perf_counter()
    for i, (fname, blob) in enumerate(items):
        elapsed = time.perf_counter() - t0
        eta_s = (elapsed / (i + 1)) * (n - i - 1) if i + 1 > 0 and n else 0.0
        status.markdown(
            f"**{phase_label}** ({i + 1}/{n}): `{fname}`  \n"
            f"⏱ Đã chạy **{elapsed:.1f}s**"
            + (f" · ước lượng còn **~{eta_s:.0f}s**" if n > 1 and i + 1 < n else "")
        )
        prog.progress((i + 1) / n if n else 1.0)
        mime = _mime_from_name(fname)
        multipart = [("files", (fname, blob, mime))]
        t_req = time.perf_counter()
        r = _post("/ivm/register", files=multipart, data=data0.copy())
        req_ms = (time.perf_counter() - t_req) * 1000
        if not r.ok:
            err_n += 1
            try:
                batch_errors.append(
                    {"filename": fname, "error": r.json(), "http": r.status_code, "request_ms": round(req_ms, 1)}
                )
            except Exception:
                batch_errors.append(
                    {"filename": fname, "error": r.text, "http": r.status_code, "request_ms": round(req_ms, 1)}
                )
            continue
        payload = r.json()
        ok_n += int(payload.get("count_success") or 0)
        err_n += int(payload.get("count_error") or 0)
        for it in payload.get("registered") or []:
            cards.append(
                {
                    "id": it.get("face_id"),
                    "name": it.get("person_name"),
                    "image_path": it.get("image_path"),
                }
            )
        for er in payload.get("errors") or []:
            batch_errors.append(er)
    prog.progress(1.0)
    total_s = time.perf_counter() - t0
    status.markdown(f"✅ **{phase_label} xong** — {n} ảnh trong **{total_s:.1f}s** (trung bình {total_s/n:.2f}s/ảnh)." if n else f"✅ **{phase_label} xong**.")
    return ok_n, err_n, cards, batch_errors


def _reg_upload_fingerprint(files_list: List[Any]) -> tuple:
    return tuple((u.name, len(u.getvalue())) for u in files_list)


def _b64_jpeg_to_image(b64_text: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_text))).convert("RGB")


def _register_preview_api_available() -> bool:
    try:
        base = st.session_state.get("api_base", API_DEFAULT).rstrip("/")
        r = requests.get(f"{base}/openapi.json", timeout=5)
        if not r.ok:
            return False
        paths = r.json().get("paths") or {}
        return "/ivm/register/preview" in paths
    except Exception:
        return False


def _run_register_preview(items: List[tuple[str, bytes]]) -> Dict[str, Any]:
    if not _register_preview_api_available():
        raise RuntimeError(
            "API trả 404 cho /ivm/register/preview — backend đang chạy **bản cũ**. "
            "Trong terminal API: **Ctrl+C**, rồi chạy lại `python main.py` "
            "(hoặc `python main.py --reload` khi dev). "
            "Kiểm tra OpenAPI: {base}/docs → phải có POST /ivm/register/preview.".format(
                base=st.session_state.get("api_base", API_DEFAULT).rstrip("/")
            )
        )
    multipart = [("files", (fname, blob, _mime_from_name(fname))) for fname, blob in items]
    r = _post("/ivm/register/preview", files=multipart, timeout=120)
    if r.status_code == 404:
        raise RuntimeError(
            "404 Not Found — `/ivm/register/preview` chưa có trên server. "
            "Khởi động lại `python main.py` trong thư mục identity_vm_app."
        )
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


REG_MODE_SINGLE = "Đơn (1 mặt / ảnh)"
REG_MODE_MULTI = "Đa mặt trong ảnh"


def _preview_face_issues(preview: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Trả (ảnh có >1 mặt, ảnh không có mặt)."""
    multi: List[str] = []
    no_face: List[str] = []
    for img in preview.get("images") or []:
        fn = str(img.get("filename") or "?")
        if img.get("error"):
            no_face.append(f"`{fn}` ({img['error']})")
            continue
        n = int(img.get("faces_count") or len(img.get("faces") or []))
        if n == 0:
            no_face.append(f"`{fn}`")
        elif n > 1:
            multi.append(f"`{fn}` ({n} mặt)")
    return multi, no_face


def _auto_analyze_register_upload(files_list: List[Any], mode: str) -> None:
    items_u = [(u.name, u.getvalue()) for u in files_list]
    st.session_state["reg_preview"] = _run_register_preview(items_u)
    st.session_state["reg_preview_mode"] = mode
    st.session_state["reg_preview_fp"] = _reg_upload_fingerprint(files_list)


def _commit_register_faces(commit_faces: List[Dict[str, Any]]) -> None:
    ok_api, health_msg = _ivm_health_check()
    if not ok_api:
        st.error(health_msg)
        return
    with st.spinner("Đang lưu vào face DB…"):
        try:
            result = _run_register_commit(commit_faces)
            st.session_state.pop("reg_preview", None)
            st.session_state.pop("reg_preview_fp", None)
            st.session_state.pop("reg_preview_mode", None)
            ok_n = int(result.get("registered_count") or result.get("count_success") or 0)
            fail_n = int(result.get("failed_count") or 0)
            st.success(
                f"**Hoàn thành:** đã đăng ký **{ok_n}** khuôn mặt."
                + (f" — **{fail_n}** lỗi." if fail_n else "")
            )
            cards = [
                {
                    "id": it.get("face_id") or it.get("person_id"),
                    "name": it.get("name"),
                    "image_path": it.get("image_path"),
                }
                for it in (result.get("registered") or [])
            ]
            if cards:
                _render_people_cards(cards, "Vừa đăng ký thành công")
            if result.get("errors"):
                with st.expander("Lỗi khi lưu", expanded=False):
                    st.json(result["errors"])
        except Exception as ex:
            st.error(str(ex))


def _run_register_commit(faces_payload: List[Dict[str, Any]]) -> Dict[str, Any]:
    r = _post("/ivm/register/commit", json={"faces": faces_payload}, timeout=120)
    if r.status_code == 404:
        raise RuntimeError(
            "404 Not Found — `/ivm/register/commit` chưa có. Khởi động lại `python main.py`."
        )
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def _ivm_health_check() -> Tuple[bool, str]:
    try:
        r = _get("/ivm/health", timeout=5)
        if r.ok:
            return True, "API sẵn sàng."
        return False, f"/ivm/health → {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"Không kết nối được API: {e}"


def _get(path: str, **kw) -> requests.Response:
    base = st.session_state.get("api_base", API_DEFAULT).rstrip("/")
    kw.setdefault("timeout", 60)
    return requests.get(f"{base}{path}", **kw)


def _post(path: str, **kw) -> requests.Response:
    base = st.session_state.get("api_base", API_DEFAULT).rstrip("/")
    kw.setdefault("timeout", 120)
    return requests.post(f"{base}{path}", **kw)


def _load_ivm_cameras_json() -> Tuple[List[Dict[str, Any]], str]:
    """
    Trả (danh sách [{id, source}, ...], nguồn).
    nguồn: \"api\" — GET /ivm/cameras OK;
           \"file\" — đọc camera_config (API không gọi được hoặc trả rỗng);
           \"none\" — không có dữ liệu.
    """
    try:
        r = _get("/ivm/cameras", timeout=60)
        if r.ok:
            cams = list((r.json() or {}).get("cameras") or [])
            if cams:
                return cams, "api"
    except Exception:
        pass
    try:
        from camera_channel_config import load_camera_channel_specs

        specs = load_camera_channel_specs(str(ivm_settings.IVM_CAMERA_CONFIG))
        if specs:
            return specs, "file"
    except Exception:
        pass
    return [], "none"



def choose_folder_dialog(initial_dir: str = "") -> Tuple[Optional[str], Optional[str]]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        return None, f"Không mở được hộp thoại chọn folder: {exc}"

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        folder = filedialog.askdirectory(initialdir=initial_dir or os.getcwd(), mustexist=True)
    finally:
        root.destroy()

    if not folder:
        return None, None
    return folder, None


def _ui_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates = [
        os.path.join(windir, "Fonts", "segoeui.ttf"),
        os.path.join(windir, "Fonts", "arial.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _annotate_identify_image(img_bytes: bytes, payload: Dict[str, Any]) -> Image.Image:
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(im)
    font = _ui_font(18)
    faces = payload.get("faces") or []
    for face in faces:
        bbox = face.get("bbox") or []
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = (int(float(bbox[i])) for i in range(4))
        draw.rectangle([x1, y1, x2, y2], outline="#00FF88", width=3)
        matches = face.get("matches") or []
        if matches:
            m0 = matches[0]
            name = str(m0.get("name") or "?")
            dist = m0.get("distance")
            try:
                d = float(dist) if dist is not None else 1.0
            except (TypeError, ValueError):
                d = 1.0
            sim_pct = max(0.0, min(100.0, (1.0 - d) * 100.0))
            label = f"{name}  ·  {sim_pct:.1f}%"
        else:
            label = "Unknown (không khớp DB)"
        bbox_txt = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox_txt[2] - bbox_txt[0], bbox_txt[3] - bbox_txt[1]
        ty = max(0, y1 - th - 10)
        pad_x, pad_y = 4, 2
        draw.rectangle(
            [x1, ty, x1 + tw + pad_x * 2, ty + th + pad_y * 2],
            fill=(20, 20, 20),
        )
        draw.text((x1 + pad_x, ty + pad_y), label, fill="#00FF88", font=font)
    return im


def _identify_result_table(faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, face in enumerate(faces):
        det = face.get("det_score")
        matches = face.get("matches") or []
        if matches:
            m0 = matches[0]
            d = m0.get("distance")
            try:
                dist_f = float(d) if d is not None else 1.0
            except (TypeError, ValueError):
                dist_f = 1.0
            sim = max(0.0, min(100.0, (1.0 - dist_f) * 100.0))
            top_names = ", ".join(str(m.get("name") or "?") for m in matches[:5])
            rows.append(
                {
                    "STT": i + 1,
                    "Tên (khớp nhất)": m0.get("name"),
                    "Độ tương đồng %": round(sim, 2),
                    "Khoảng cách cosine": round(dist_f, 4),
                    "face_id": m0.get("face_id"),
                    "det_score": round(float(det), 4) if det is not None else None,
                    "Top khớp": top_names,
                }
            )
        else:
            rows.append(
                {
                    "STT": i + 1,
                    "Tên (khớp nhất)": "—",
                    "Độ tương đồng %": None,
                    "Khoảng cách cosine": None,
                    "face_id": None,
                    "det_score": round(float(det), 4) if det is not None else None,
                    "Top khớp": "—",
                }
            )
    return rows


def _filter_faces_by_search(faces: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    q = (query or "").strip()
    if not q:
        return list(faces)
    q_lower = q.lower()
    if q.isdigit():
        fid = int(q)
        exact = [f for f in faces if int(f.get("id", -1)) == fid]
        if exact:
            return exact
    return [f for f in faces if q_lower in str(f.get("name") or "").lower()]


def _render_people_cards(
    faces: List[Dict[str, Any]],
    title: str = "Danh sách đã đăng ký",
    *,
    detailed: bool = False,
) -> None:
    if not faces:
        st.info("Chưa có khuôn mặt nào trong database.")
        return
    st.subheader(title)
    ncols = 4
    for row_start in range(0, len(faces), ncols):
        chunk = faces[row_start : row_start + ncols]
        cols = st.columns(ncols)
        for j, f in enumerate(chunk):
            with cols[j]:
                name = str(f.get("name") or "(no name)")
                fid = f.get("id")
                path_raw = f.get("image_path")
                st.markdown(f"**{name}**")
                st.caption(f"face_id: `{fid}`")
                p = Path(str(path_raw)) if path_raw else None
                if p and p.is_file():
                    st.image(str(p), use_container_width=True)
                elif path_raw:
                    st.warning("File ảnh không tồn tại:\n" + str(path_raw))
                else:
                    st.caption("Không có image_path")
                if detailed:
                    with st.expander("Thông tin chi tiết", expanded=True):
                        st.markdown(f"- **Tên:** {name}")
                        st.markdown(f"- **face_id:** `{fid}`")
                        if path_raw:
                            exists = bool(p and p.is_file())
                            st.markdown(
                                f"- **Ảnh đăng ký:** `{path_raw}` "
                                f"({'có trên đĩa' if exists else 'thiếu file'})"
                            )
                        else:
                            st.markdown("- **Ảnh đăng ký:** *(chưa có image_path)*")


with st.sidebar:
    st.header("Kết nối API")
    st.session_state["api_base"] = st.text_input("Base URL", value=API_DEFAULT)
    if st.button("Kiểm tra /ivm/health"):
        try:
            r = _get("/ivm/health")
            st.json(r.json() if r.ok else {"error": r.text})
        except Exception as e:
            st.error(str(e))
    base_u = st.session_state.get("api_base", API_DEFAULT).rstrip("/")
    st.markdown(f"[OpenAPI docs]({base_u}/docs)")

    st.divider()
    with st.expander("Xóa toàn bộ dữ liệu (thật trên DB + đĩa)", expanded=False):
        st.caption(
            "Gọi **POST /ivm/admin/reset-all-data**: xóa **face embeddings + metadata + ảnh đăng ký**, "
            "xóa hết bảng SQLite (sự kiện, segment, gallery registry, ảnh lỗi đăng ký), "
            "làm sạch thư mục **registration_errors**, **gallery**, **export_cache**. "
            "Tuỳ chọn xóa cả **archive** (video đã ghi)."
        )
        wipe_arch = st.checkbox("Xóa cả thư mục archive (video)", value=False, key="reset_wipe_archive")
        reset_token = st.text_input(
            "Token (chỉ khi server đặt IVM_RESET_SECRET)",
            type="password",
            key="reset_secret_token",
        )
        confirm_reset = st.text_input('Gõ DELETE_ALL để xác nhận', key="reset_confirm_text")
        if st.button("Thực hiện xóa toàn bộ", type="primary", key="reset_do_btn"):
            if confirm_reset != "DELETE_ALL":
                st.error('Phải gõ đúng DELETE_ALL.')
            else:
                payload: Dict[str, Any] = {"confirm": "DELETE_ALL", "wipe_archive": wipe_arch}
                if str(reset_token).strip():
                    payload["token"] = str(reset_token).strip()
                try:
                    rr = requests.post(
                        f'{st.session_state.get("api_base", API_DEFAULT).rstrip("/")}/ivm/admin/reset-all-data',
                        json=payload,
                        timeout=120,
                    )
                    if rr.ok:
                        st.success("Đã xóa xong.")
                        st.json(rr.json())
                        for k in ("people_cache", "reg_failures"):
                            st.session_state.pop(k, None)
                        st.session_state["people_gallery_page"] = 0
                    else:
                        try:
                            st.error(rr.json())
                        except Exception:
                            st.error(f"{rr.status_code}: {rr.text}")
                except Exception as ex:
                    st.error(str(ex))

st.info(
    "**Backend:** `python main.py` (thư mục `identity_vm_app`, cổng mặc định sidebar) · "
    "**Frontend:** `streamlit run streamlit_test.py --server.port 8510` · "
    "**Báo cáo camera / video:** **Xem chi tiết** = 3 ảnh: scene lớn (box người+mặt+vũ khí), dưới crop mặt | crop vũ khí (cùng cỡ). "
    "Cần bật nhận diện camera + `IVM_WEAPON_ENABLED=1`; crop vũ khí lưu khi có event `armed`."
)

tab_a, tab_b, tab_c, tab_d, tab_e, tab_f = st.tabs(
    [
        "Đăng ký & nhận diện",
        "Camera & Recorder",
        "Người & Gallery",
        "Báo cáo & sự kiện",
        "Export cut",
        "Phân tích video",
    ]
)

with tab_a:
    st.subheader("Đăng ký khuôn mặt (upload ảnh)")
    reg_mode = st.radio(
        "Chế độ đăng ký",
        [REG_MODE_SINGLE, REG_MODE_MULTI],
        horizontal=True,
        key="reg_mode",
        help="Đơn: nhiều ảnh, mỗi ảnh đúng 1 mặt (như trước). Đa: một ảnh có thể nhiều mặt, đặt tên từng crop.",
    )
    if reg_mode == REG_MODE_SINGLE:
        st.caption(
            "Tải **nhiều ảnh** (mỗi ảnh **một** người). Tên mặc định từ **tên file** (bỏ dấu, `_` → khoảng). "
            "Có thể **ghi đè tên** để mọi ảnh trong lô dùng chung một tên. "
            "Hệ thống **tự phát hiện** khi chọn ảnh; ảnh có **≥2 mặt** sẽ bị **cảnh báo** và không cho đăng ký ở chế độ này."
        )
    else:
        st.caption(
            "Một hoặc nhiều ảnh, mỗi ảnh có thể **nhiều khuôn mặt**. Tự phát hiện → nhập tên từng crop → **Hoàn thành đăng ký**."
        )

    up = st.file_uploader(
        "Ảnh (có thể chọn nhiều file)",
        type=["jpg", "jpeg", "png"],
        key="reg_img",
        accept_multiple_files=True,
    )

    if up:
        files_list = up if isinstance(up, list) else [up]
        fp = _reg_upload_fingerprint(files_list)
        mode_now = str(st.session_state.get("reg_mode") or reg_mode)
        need_analyze = (
            st.session_state.get("reg_preview_fp") != fp
            or st.session_state.get("reg_preview_mode") != mode_now
            or "reg_preview" not in st.session_state
        )
        if need_analyze:
            ok_api, health_msg = _ivm_health_check()
            analyzed_ok = False
            if not ok_api:
                st.error(health_msg)
            elif not _register_preview_api_available():
                st.warning(
                    "API chưa có `/ivm/register/preview` — khởi động lại `python main.py`. "
                    "Chế độ đơn vẫn có thể dùng **Đăng ký (không kiểm tra đa mặt)** bên dưới."
                )
                st.session_state.pop("reg_preview", None)
            else:
                try:
                    with st.spinner("Đang tự động phát hiện khuôn mặt…"):
                        _auto_analyze_register_upload(files_list, mode_now)
                    analyzed_ok = True
                except Exception as ex:
                    st.session_state.pop("reg_preview", None)
                    st.error(str(ex))
            if analyzed_ok:
                st.rerun()

        if st.button("🔄 Phát hiện lại", key="reg_reanalyze_btn"):
            st.session_state.pop("reg_preview", None)
            st.rerun()

    preview = st.session_state.get("reg_preview")
    mode_now = str(st.session_state.get("reg_mode") or reg_mode)

    if preview and preview.get("images") and mode_now == REG_MODE_SINGLE:
        multi, no_face = _preview_face_issues(preview)
        n_ok = sum(
            1
            for img in preview["images"]
            if not img.get("error") and int(img.get("faces_count") or 0) == 1
        )
        st.markdown(f"**Đã quét {len(preview['images'])} ảnh** — **{n_ok}** ảnh hợp lệ (đúng 1 mặt). Chỉ hiển thị **crop mặt**, không xem ảnh gốc.")

        if multi:
            st.error(
                "**Không thể đăng ký (chế độ đơn):** các ảnh sau có **nhiều hơn 1 khuôn mặt**:\n"
                + "\n".join(f"- {m}" for m in multi)
                + "\n\n→ Chuyển sang **Đa mặt trong ảnh**, hoặc tách ảnh từng người."
            )
        if no_face:
            st.warning(
                "**Không thấy mặt** trong:\n" + "\n".join(f"- {m}" for m in no_face)
            )

        can_register = bool(n_ok) and not multi

        if can_register:
            name_override = st.text_input(
                "Ghi đè tên (tuỳ chọn — cùng tên cho mọi ảnh hợp lệ)",
                key="reg_name_single",
                placeholder="Để trống = mỗi ảnh một tên theo tên file",
            )
            ov = str(name_override).strip()
            commit_faces: List[Dict[str, Any]] = []
            slot = 0
            for img_entry in preview["images"]:
                if img_entry.get("error"):
                    continue
                faces = img_entry.get("faces") or []
                if len(faces) != 1:
                    continue
                face = faces[0]
                filename = img_entry.get("filename", "unknown")
                default_name = face.get("default_name") or Path(filename).stem
                crop_b64 = face.get("crop_jpeg_b64") or ""
                row_l, row_r = st.columns([1, 2])
                with row_l:
                    if crop_b64:
                        st.image(_b64_jpeg_to_image(crop_b64), caption=filename, width=120)
                    st.caption(f"Độ tin cậy: {float(face.get('confidence', 0)):.1%}")
                with row_r:
                    init_name = ov if ov else default_name
                    person_name = st.text_input(
                        "Tên đối tượng",
                        value=init_name,
                        key=f"reg_single_name_{slot}",
                    )
                commit_faces.append({
                    "name": person_name,
                    "embedding": face.get("embedding"),
                    "crop_jpeg_b64": crop_b64,
                    "source_filename": filename,
                })
                slot += 1

            if commit_faces and st.button("✅ Đăng ký", type="primary", key="reg_single_commit_btn"):
                if any(not str(f.get("name") or "").strip() for f in commit_faces):
                    st.error("Vui lòng nhập tên cho tất cả ảnh.")
                else:
                    _commit_register_faces(commit_faces)

    elif preview and preview.get("images") and mode_now == REG_MODE_MULTI:
        total_faces = int(preview.get("total_faces_count") or 0)
        multi, no_face = _preview_face_issues(preview)
        if no_face:
            st.warning(
                "**Không thấy mặt:**\n" + "\n".join(f"- {m}" for m in no_face)
            )
        st.markdown(f"**Đã phát hiện {total_faces} khuôn mặt** — nhập tên cho từng crop (không hiển thị ảnh gốc):")
        commit_faces: List[Dict[str, Any]] = []
        face_slot = 0
        for img_entry in preview["images"]:
            filename = img_entry.get("filename", "unknown")
            if img_entry.get("error"):
                continue
            faces = img_entry.get("faces") or []
            if len(faces) > 1:
                st.caption(f"📷 `{filename}` — {len(faces)} khuôn mặt")
            for face in faces:
                crop_b64 = face.get("crop_jpeg_b64") or ""
                row_l, row_r = st.columns([1, 2])
                with row_l:
                    if crop_b64:
                        st.image(
                            _b64_jpeg_to_image(crop_b64),
                            caption=f"{filename} · mặt {int(face.get('face_index', 0)) + 1}",
                            width=140,
                        )
                    st.caption(f"Độ tin cậy: {float(face.get('confidence', 0)):.1%}")
                with row_r:
                    person_name = st.text_input(
                        "Tên đối tượng",
                        value=face.get("default_name", ""),
                        key=f"reg_face_name_{face_slot}",
                    )
                commit_faces.append({
                    "name": person_name,
                    "embedding": face.get("embedding"),
                    "crop_jpeg_b64": crop_b64,
                    "source_filename": filename,
                })
                face_slot += 1
                st.divider()

        if commit_faces and st.button("✅ Hoàn thành đăng ký", type="primary", key="reg_commit_btn"):
            if any(not str(f.get("name") or "").strip() for f in commit_faces):
                st.error("Vui lòng nhập tên cho tất cả khuôn mặt.")
            else:
                _commit_register_faces(commit_faces)
        elif up and not commit_faces:
            st.info("Không có mặt nào để đăng ký — thử ảnh khác.")

    elif up and _register_preview_api_available():
        st.info("Đang phát hiện… Nếu không thấy kết quả, bấm **Phát hiện lại**.")

    elif up and not _register_preview_api_available() and mode_now == REG_MODE_SINGLE:
        st.warning("Khởi động lại `python main.py` để bật phát hiện tự động và cảnh báo ảnh nhiều mặt.")
        name_legacy = st.text_input(
            "Ghi đè tên (tuỳ chọn)",
            key="reg_name_legacy",
            placeholder="Để trống = tên theo file",
        )
        if st.button("Đăng ký (không kiểm tra đa mặt)", key="reg_legacy_btn"):
            files_list = up if isinstance(up, list) else [up]
            items_u = [(u.name, u.getvalue()) for u in files_list]
            data0: Dict[str, str] = {}
            if str(name_legacy).strip():
                data0["name"] = str(name_legacy).strip()
            ok_n, err_n, cards, batch_errors = _run_register_items_progress(items_u, data0)
            st.success(f"**Hoàn thành:** {ok_n} thành công — **{err_n}** lỗi.")
            if cards:
                _render_people_cards(cards, "Vừa đăng ký")
            if batch_errors:
                with st.expander("Lỗi", expanded=False):
                    st.json(batch_errors)

    st.divider()
    st.subheader("Đăng ký cả thư mục / file ZIP")
    name = st.text_input(
        "Ghi đè tên (tuỳ chọn — bulk: cùng tên cho mọi ảnh)",
        key="reg_name",
        placeholder="Để trống = mỗi ảnh một tên theo tên file",
    )
    st.caption(
        "**ZIP:** nén folder ảnh thành `.zip` rồi tải lên — ảnh trong thư mục con vẫn đăng ký; "
        "tên gửi API là đường dẫn gốc trong zip với `/` thay bằng `_` (để tên đối tượng từ **stem** file không bị trùng). "
        "**Đường dẫn thư mục** chỉ dùng khi Streamlit chạy **trên máy bạn** (đọc trực tiếp ổ đĩa). "
        "Bulk vẫn gửi từng ảnh lên `/ivm/register` (một mặt lớn nhất/ảnh). "
        "Thư mục **rất lớn**: mục **Thư mục lớn** bên dưới hoặc CLI `python -m identity_vm_app.cli_bulk_register`. "
        "Ô **ghi đè tên** áp dụng cho bulk nếu điền."
    )
    zip_up = st.file_uploader("Chọn file .zip chứa ảnh", type=["zip"], key="reg_zip")
    folder_txt = st.text_input(
        "Hoặc đường dẫn thư mục trên máy",
        placeholder=r"E:\data\faces hoặc /home/user/faces",
        key="reg_folder",
    )
    recurse = st.checkbox("Quét cả thư mục con (recursive)", value=True, key="reg_recurse")
    if st.button("Đăng ký từ ZIP hoặc thư mục", key="reg_bulk_btn"):
        ok_api, health_msg = _ivm_health_check()
        if not ok_api:
            st.error(health_msg)
        else:
            st.caption(f"🔗 {health_msg} (`{st.session_state.get('api_base', API_DEFAULT)}`)")
        items_bulk: List[tuple[str, bytes]] = []
        if ok_api and zip_up is not None:
            with st.spinner("Đang giải nén và đọc ảnh từ ZIP…"):
                items_bulk = _images_from_zip_bytes(zip_up.getvalue())
            if folder_txt.strip():
                st.info("Đang dùng **ZIP** — bỏ qua ô đường dẫn thư mục.")
        elif ok_api and folder_txt.strip():
            root_bulk = Path(folder_txt.strip()).expanduser()
            resolved = root_bulk.resolve()
            if not resolved.is_dir():
                st.error(f"Không phải thư mục hợp lệ: `{folder_txt.strip()}`")
            else:
                st.info(f"📂 Thư mục: `{resolved}`")
                paths_bulk = _list_local_image_paths(root_bulk, recurse)
                n_prev = len(paths_bulk)
                st.info(f"Đã quét **{n_prev}** file ảnh hợp lệ — đang đọc từ đĩa vào RAM (có thể lâu với thư mục lớn)…")
                prog_read = st.progress(0)
                lbl_read = st.empty()
                items_bulk = _bytes_from_path_list(
                    root_bulk,
                    paths_bulk,
                    progress=prog_read,
                    status=lbl_read,
                )
                lbl_read.markdown(f"✅ Đọc xong **{len(items_bulk)}** ảnh vào RAM.")
                prog_read.progress(1.0)
        elif ok_api:
            st.warning("Tải lên file **.zip** hoặc nhập **đường dẫn thư mục**.")
        if ok_api and items_bulk:
            st.info(f"Bắt đầu **đăng ký qua API**: **{len(items_bulk)}** ảnh (mỗi ảnh một request)…")
            data0b: Dict[str, str] = {}
            if str(name).strip():
                data0b["name"] = str(name).strip()
            ok_nb, err_nb, cards_b, batch_err_b = _run_register_items_progress(
                items_bulk, data0b, phase_label="Gửi /ivm/register"
            )
            ov2 = str(name).strip() or None
            st.success(
                f"**Hoàn thành (bulk):** {ok_nb} thành công — **{err_nb}** lỗi."
                + (f"  \nGhi đè tên: `{ov2}`." if ov2 else "  \nTên theo từng file (stem).")
            )
            if cards_b:
                _render_people_cards(cards_b, "Vừa đăng ký (bulk)")
            if batch_err_b:
                with st.expander(f"Lỗi bulk ({len(batch_err_b)})", expanded=False):
                    st.json(batch_err_b)
        elif ok_api and (zip_up is not None or folder_txt.strip()):
            st.warning("Không tìm thấy ảnh hợp lệ (.jpg/.jpeg/.png/.webp/.bmp) trong ZIP hoặc thư mục.")

    st.divider()
    st.subheader("Thư mục lớn — đăng ký từ đường dẫn đĩa")
    # Gán vào key widget *trước* khi tạo text_input — Streamlit cấm sửa sau khi widget đã khởi tạo.
    pending_pick = st.session_state.pop("_bulk_local_folder_pick", None)
    if pending_pick:
        st.session_state["bulk_local_folder"] = str(pending_pick)

    st.caption(
        "Máy chạy API: model InsightFace nạp **một lần lúc khởi động server** và **giữ** cho nhận diện; bulk có thể chỉnh **số luồng infer** (mỗi luồng load lại ONNX — VRAM ↑) hoặc để máy chủ dùng env **IVM_BULK_INFER_WORKERS**. "
        "Server chạy job nền và ghi file tiến trình; không bị chặn một worker HTTP. "
        "Luôn **batch + checkpoint** — nhiều worker chia shard + prefetch. "
        "Tên đối tượng lấy từ **đường dẫn file**. Nếu có **`IVM_BULK_ALLOWED_ROOTS`**, root phải thuộc danh sách. "
        "Giao diện này chỉ **gọi API** (`POST …/admin/register-folder` + poll progress). Đăng ký bulk trực tiếp không HTTP: `python -m identity_vm_app.cli_bulk_register --root …`."
    )
    c_pf, c_pick = st.columns([5, 1])
    with c_pf:
        bulk_local_path = st.text_input(
            "Đường dẫn thư mục ảnh",
            key="bulk_local_folder",
            placeholder=r"E:\datasets\faces",
        )
    with c_pick:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("Chọn…", key="bulk_pick_folder", help="Hộp thoại hệ thống (Windows)"):
            init = str(bulk_local_path).strip() or os.getcwd()
            picked, dlg_err = choose_folder_dialog(initial_dir=init)
            if dlg_err:
                st.error(dlg_err)
            elif picked:
                st.session_state["_bulk_local_folder_pick"] = picked
                st.rerun()

    ivm_bulk_token = st.text_input(
        "Token admin (bắt buộc nếu server đặt IVM_RESET_SECRET)",
        type="password",
        key="ivm_bulk_token",
        autocomplete="off",
    )
    c_bl1, c_bl2, c_bl3 = st.columns(3)
    with c_bl1:
        bl_recurse = st.checkbox("Quét cả thư mục con", value=True, key="bulk_local_recurse")
    with c_bl2:
        bl_resume = st.checkbox("Resume (bỏ qua ảnh đã thành công)", value=True, key="bulk_local_resume")
    with c_bl3:
        bl_skip_fail = st.checkbox("Resume: bỏ qua cả ảnh đã lỗi trước", value=True, key="bulk_local_skip_fail")
    bl_clear_ckpt = st.checkbox(
        "Xóa checkpoint SQLite trước khi chạy (chạy lại tất cả path; không xóa face DB)",
        value=False,
        key="bulk_local_clear_ckpt",
    )
    bl_infer_override = st.checkbox(
        "Chỉ định số luồng infer bulk (ghi đè env máy API)",
        value=False,
        key="bulk_infer_override",
        help="Không tick: API dùng IVM_BULK_INFER_WORKERS trên máy chạy server. Tick + chọn N: mỗi luồng một phiên ONNX (VRAM lớn hơn, có thể nhanh hơn).",
    )
    bl_infer_n = st.number_input(
        "Số luồng infer bulk (FaceAnalysis)",
        min_value=1,
        max_value=16,
        value=4,
        step=1,
        key="bulk_infer_n",
        disabled=not bool(bl_infer_override),
    )
    bl_batch = st.number_input("Ghi DB mỗi N ảnh", min_value=1, max_value=2048, value=64, key="bulk_local_batch")
    bl_prog = st.number_input(
        "Cập nhật tiến trình mỗi N ảnh",
        min_value=1,
        max_value=2000,
        value=25,
        key="bulk_local_prog_every",
    )
    bl_max_files_api = st.number_input(
        "Giới hạn số ảnh mỗi lần gọi API (0 = không giới hạn — quét cả thư mục; server vẫn có thể áp IVM_BULK_API_MAX_FILES)",
        min_value=0,
        max_value=50_000_000,
        value=0,
        step=1000,
        key="bulk_api_max_files",
        help="Trước đây API mặc định chỉ 5000 ảnh/request. Đặt 0 để chạy hết ~500k ảnh trong một job.",
    )
    if st.button("Bắt đầu đăng ký (thư mục lớn)", type="primary", key="bulk_local_go"):
        root_s = str(bulk_local_path).strip()
        if not root_s:
            st.warning("Nhập đường dẫn thư mục.")
        else:
            log_box = st.empty()
            _bulk_log_lines: List[str] = []

            def _bulk_log_ui(msg: str) -> None:
                line = f"{time.strftime('%H:%M:%S')} — {msg}"
                _bulk_log_lines.append(line)
                log_box.markdown(
                    "### Nhật ký tiến trình\n\n" + "\n\n".join(f"- `{ln}`" for ln in _bulk_log_lines)
                )
                print(f"[ivm-bulk] {line}", flush=True)

            try:
                st.toast("Đã nhận lệnh — xem nhật ký bên dưới.", icon="📂")
            except TypeError:
                st.toast("Đã nhận lệnh — xem nhật ký bên dưới.")

            _bulk_log_ui(f"Bắt đầu — qua API, thư mục `{root_s}`")

            ok_api, api_msg = _ivm_health_check()
            if not ok_api:
                st.warning(f"{api_msg} — vẫn có thể thử gọi register-folder nếu bạn chắc API đang mở.")
                _bulk_log_ui(f"Cảnh báo health API: {api_msg}")
            body: Dict[str, Any] = {
                "root_path": root_s,
                "recursive": bool(bl_recurse),
                "resume": bool(bl_resume),
                "resume_skip_failed": bool(bl_skip_fail),
                "clear_checkpoint": bool(bl_clear_ckpt),
                "db_batch_size": int(bl_batch),
                "progress_every": int(bl_prog),
            }
            if bool(bl_infer_override):
                body["infer_workers"] = int(bl_infer_n)
            if int(bl_max_files_api) > 0:
                body["max_files"] = int(bl_max_files_api)
            tok = str(ivm_bulk_token or "").strip()
            if tok:
                body["token"] = tok
            headers = {}
            if tok:
                headers["X-IVM-Reset-Token"] = tok
            api_register_url = (
                f'{st.session_state.get("api_base", API_DEFAULT).rstrip("/")}/ivm/admin/register-folder'
            )
            _bulk_log_ui(f"Đang POST `{api_register_url}` (timeout 120s)…")
            r = requests.post(api_register_url, json=body, timeout=120, headers=headers)
            _bulk_log_ui(f"Phản hồi register-folder: HTTP {r.status_code}")
            if r.status_code == 409:
                st.error("Server báo đang có job register-folder khác (409). Đợi xong hoặc kiểm tra progress.")
            elif not r.ok:
                try:
                    st.error(r.json())
                except Exception:
                    st.error(f"{r.status_code}: {r.text[:500]}")
            else:
                start_payload = r.json()
                _bulk_log_ui(
                    f"Máy API dùng infer_workers≈ **{start_payload.get('infer_workers', '?')}** "
                    "(mỗi luồng ~một ONNX; xem chi tiết `parallel_workers` trong file progress)."
                )
                _bulk_log_ui(
                    f"Job đã queue — poll `{start_payload.get('progress_url', '/ivm/admin/register-folder/progress')}`; "
                    f"log worker: `{start_payload.get('bulk_log_file', '')}` (terminal máy API)"
                )
                st.success(
                    f"Job đã bắt đầu — tiến trình: `{start_payload.get('progress_url', '/ivm/admin/register-folder/progress')}`  \n"
                    f"Log throughput worker (terminal **máy chạy API** + file): `{start_payload.get('bulk_log_file', '')}`"
                )
                prog_bar = st.progress(0.0)
                st_status = st.empty()
                prog: Dict[str, Any] = {}
                max_wait = 86400.0
                t_poll0 = time.perf_counter()
                _bulk_log_ui("Đang poll GET …/register-folder/progress (0.5s/lần)…")
                while (time.perf_counter() - t_poll0) < max_wait:
                    try:
                        pr = _get("/ivm/admin/register-folder/progress", timeout=30)
                        if pr.ok:
                            prog = pr.json()
                            pct = float(prog.get("progress_pct") or 0) / 100.0
                            prog_bar.progress(min(1.0, pct))
                            ph = str(prog.get("phase") or "")
                            reg = prog.get("registered", prog.get("success", 0))
                            st_status.markdown(
                                f"**Tiến độ** (`{ph}`): **{float(prog.get('progress_pct') or 0):.1f}%** — "
                                f"xử lý **{prog.get('processed', 0)}** / **{prog.get('total', 0)}** · "
                                f"ghi DB **{reg}** · lỗi **{prog.get('failed', 0)}** · checkpoint **{prog.get('skipped_checkpoint', 0)}**"
                            )
                            if not prog.get("running"):
                                break
                            if ph == "error":
                                break
                    except Exception as ex:
                        st_status.markdown(f"(Lỗi khi đọc progress: `{ex}` — thử lại…)")
                    time.sleep(0.5)
                err_txt = prog.get("error")
                if str(prog.get("phase") or "") == "error" or err_txt:
                    _bulk_log_ui(f"Kết thúc lỗi: {err_txt or prog.get('phase')}")
                    st.error(str(err_txt or "Job kết thúc lỗi — xem JSON bên dưới."))
                elif not prog:
                    _bulk_log_ui("Không đọc được JSON tiến trình từ API.")
                    st.warning("Không đọc được trạng thái tiến trình từ API.")
                else:
                    dt = float(prog.get("elapsed_s") or (time.perf_counter() - t_poll0))
                    _bulk_log_ui(
                        f"Hoàn thành ~{dt:.1f}s — thành công {prog.get('success', 0)}, lỗi {prog.get('failed', 0)}"
                    )
                    st.success(
                        f"**Hoàn thành** (~{dt:.1f}s) — thành công **{prog.get('success', 0)}**, "
                        f"lỗi **{prog.get('failed', 0)}**, bỏ qua (checkpoint) **{prog.get('skipped_checkpoint', 0)}**."
                    )
                with st.expander("Chi tiết JSON tiến trình"):
                    st.json(prog if prog else {})

    st.subheader("Ảnh đăng ký lỗi (đã lưu để xem lại)")
    st.caption("Dữ liệu trong SQLite + file tại `identity_vm_data/registration_errors/`.")
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        if st.button("Tải danh sách lỗi từ API", key="load_reg_fails"):
            fr = _get("/ivm/register/failures", params={"limit": 80})
            if fr.ok:
                st.session_state["reg_failures"] = fr.json().get("items") or []
            else:
                st.error(fr.text)
    with col_f2:
        if st.button("Xóa cache hiển thị lỗi", key="clear_reg_fails"):
            st.session_state["reg_failures"] = []

    fails = st.session_state.get("reg_failures") or []
    if fails:
        base = st.session_state.get("api_base", API_DEFAULT).rstrip("/")
        for it in fails[:40]:
            fid = it.get("id")
            fn = it.get("original_filename") or ""
            em = it.get("error_message") or ""
            ip = it.get("image_path")
            with st.container():
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.markdown(f"**{fn}**")
                    st.caption(f"`{fid}`")
                    st.caption(em[:200] + ("…" if len(str(em)) > 200 else ""))
                with c2:
                    if fid and ip:
                        img_url = f"{base}/ivm/register/failures/{fid}/file"
                        try:
                            ir = requests.get(img_url, timeout=60)
                            if ir.ok and ir.content:
                                st.image(io.BytesIO(ir.content), use_container_width=True)
                            else:
                                st.caption("(Không tải được ảnh)")
                        except Exception as ex:
                            st.caption(str(ex))
                with c3:
                    if fid and st.button("Xóa bản ghi", key=f"del_fail_{fid}"):
                        dr = requests.delete(f"{base}/ivm/register/failures/{fid}", timeout=30)
                        if dr.ok:
                            st.session_state["reg_failures"] = [
                                x for x in fails if x.get("id") != fid
                            ]
                            st.rerun()
                        else:
                            st.error(dr.text)
                st.divider()

    st.divider()
    if st.button("Tải danh sách đã đăng ký (ảnh + tên)", key="reload_people_tab_a"):
        rr = _get("/ivm/people")
        if rr.ok:
            _render_people_cards(rr.json().get("faces") or [])
            with st.expander("JSON /ivm/people"):
                st.json(rr.json())
        else:
            st.error(rr.text)

    st.subheader("Nhận diện ảnh")
    st.caption(
        "Chọn một hoặc nhiều ảnh — API `/ivm/identify_images` gom embedding và tra cứu DB một lần. "
        "(Giới hạn số ảnh/request chỉ áp khi đặt `IVM_IDENTIFY_BATCH_MAX_FILES` > 0 trên máy API.)"
    )
    infer_max = max(1, min(16, int(ivm_settings.IVM_IDENTIFY_BATCH_API_MAX_INFER_WORKERS)))
    infer_n = st.number_input(
        "Số worker infer (song song)",
        min_value=1,
        max_value=infer_max,
        value=1,
        step=1,
        key="id_infer_workers",
        help="Decode + infer nhiều ảnh song song khi >1; mỗi worker một engine.",
    )
    st.caption("GPU có thể không ổn khi >1 worker.")
    up2_list = st.file_uploader(
        "Ảnh cần nhận diện",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="id_img",
    )
    thr = st.number_input("distance_threshold", value=0.5, min_value=0.0, max_value=1.0, step=0.05, key="id_thr")
    if st.button("Nhận diện & hiển thị", key="id_go") and up2_list:
        uploads = list(up2_list)
        multipart = [
            ("files", (up.name or f"upload_{i}", up.getvalue(), up.type or "image/jpeg"))
            for i, up in enumerate(uploads)
        ]
        r = _post(
            "/ivm/identify_images",
            files=multipart,
            params={"distance_threshold": thr, "infer_workers": int(infer_n)},
        )
        if not r.ok:
            st.error(f"{r.status_code}: {r.text}")
        else:
            data = r.json()
            timing = data.get("timing") or {}
            results = data.get("results") or []

            if timing:
                img_timing = timing.get("images") or []
                eligible_t = [x for x in img_timing if not x.get("error")]
                ad = timing.get("avg_detect_ms_per_image")
                ae = timing.get("avg_embedding_ms_per_image")
                if ad is not None and ae is not None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "TB detect (ms/ảnh)",
                        f"{float(ad):.2f}",
                        help="SCRFD một lần / ảnh — ảnh không lỗi.",
                    )
                    c2.metric(
                        "TB embedding (ms/ảnh)",
                        f"{float(ae):.2f}",
                        help="norm_crop + ONNX recognition batch — ảnh không lỗi.",
                    )
                    if eligible_t:
                        avg_inf = sum(float(x.get("infer_ms") or 0) for x in eligible_t) / len(eligible_t)
                        c3.metric(
                            "TB detect+embed (infer_ms)",
                            f"{avg_inf:.2f}",
                            help="detect_ms + embedding_ms (không gồm decode/search).",
                        )
                elif eligible_t:
                    sums = [
                        float(x.get("decode_ms") or 0)
                        + float(x.get("infer_ms") or 0)
                        + float(x.get("search_ms") or 0)
                        for x in eligible_t
                    ]
                    avg_wall = sum(sums) / len(eligible_t)
                    st.metric(
                        "Trung bình ms/ảnh (decode + infer + search phân bổ)",
                        f"{avg_wall:.2f}",
                        help="Chỉ tính ảnh không lỗi (empty/decode error bị loại). Ảnh không có mặt vẫn gồm decode+infer.",
                    )
                if eligible_t:
                    pipe_avg = sum(
                        float(x.get("decode_ms") or 0)
                        + float(x.get("infer_ms") or 0)
                        + float(x.get("search_ms") or 0)
                        for x in eligible_t
                    ) / len(eligible_t)
                    st.caption(
                        f"TB decode + detect + embed + search (chia search): **{pipe_avg:.2f}** ms/ảnh "
                        f"(cùng tập ảnh không lỗi)."
                    )
                tot = timing.get("total_ms")
                if tot is not None:
                    st.caption(f"Tổng request (server): **{float(tot):.2f}** ms — `search_batch_ms` là **một** lần gọi DB, "
                               f"chia theo số mặt cho từng dòng ảnh.")
                iw = timing.get("infer_workers")
                piw = timing.get("parallel_infer_wall_ms")
                if iw is not None or piw is not None:
                    extra = []
                    if iw is not None:
                        extra.append(f"`infer_workers` (resolved): **{iw}**")
                    if piw is not None:
                        extra.append(f"`parallel_infer_wall_ms`: **{float(piw):.2f}**")
                    st.caption(" — ".join(extra))
                note = timing.get("search_batch_amortization_note")
                if note:
                    st.caption(str(note))
                tbl: List[Dict[str, Any]] = []
                for x in img_timing:
                    tbl.append(
                        {
                            "filename": x.get("filename", ""),
                            "decode_ms": x.get("decode_ms"),
                            "detect_ms": x.get("detect_ms"),
                            "embedding_ms": x.get("embedding_ms"),
                            "infer_ms": x.get("infer_ms"),
                            "search_ms": x.get("search_ms"),
                            "face_count": x.get("face_count"),
                            "lỗi": x.get("error") or "",
                        }
                    )
                tbl.append(
                    {
                        "filename": "— search_batch (tổng) —",
                        "decode_ms": None,
                        "detect_ms": None,
                        "embedding_ms": None,
                        "infer_ms": None,
                        "search_ms": timing.get("search_batch_ms"),
                        "face_count": timing.get("search_batch_face_count"),
                        "lỗi": "một lần cho cả batch",
                    }
                )
                with st.expander("Timing chi tiết (từng ảnh + search_batch)", expanded=False):
                    st.dataframe(tbl, use_container_width=True, hide_index=True)

            if not results:
                st.warning("API trả về danh sách ảnh rỗng.")
            else:
                for i, item in enumerate(results):
                    fname = str(item.get("filename") or "").strip()
                    if not fname and i < len(uploads):
                        fname = uploads[i].name or f"upload_{i}"
                    if not fname:
                        fname = f"#{i}"
                    err = item.get("error")
                    faces = item.get("faces") or []
                    raw_bytes = uploads[i].getvalue() if i < len(uploads) else b""
                    st.markdown(f"#### `{fname}`")
                    if err:
                        st.warning(f"Lỗi file: {err}")
                        continue
                    if not faces:
                        st.warning("Không phát hiện khuôn mặt.")
                        continue
                    annotated = _annotate_identify_image(raw_bytes, {"faces": faces})
                    st.image(
                        annotated,
                        caption="Khung + tên + độ tương đồng (ước lượng từ khoảng cách cosine)",
                        use_container_width=True,
                    )
                    rows = _identify_result_table(faces)
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                    names_line = []
                    for row in rows:
                        n = row.get("Tên (khớp nhất)")
                        sim_v = row.get("Độ tương đồng %")
                        if n and n != "—" and sim_v is not None:
                            names_line.append(f"**{n}** ({sim_v}%)")
                    if names_line:
                        st.markdown("**Tóm tắt tên:** " + " · ".join(names_line))
                    else:
                        st.markdown("**Tóm tắt:** không có khớp trong DB (Unknown).")
                    st.divider()
            with st.expander("JSON đầy đủ từ API"):
                st.json(data)

with tab_b:
    st.subheader("Xem trực tiếp — theo camera_config.json")
    st.caption(
        "Mỗi dòng config = một ô trên lưới. **Lưới** dùng **ảnh snapshot** (polling tuần tự — tất cả camera đều có hình, không bị kẹt ~2 luồng MJPEG). "
        "**Phóng to** dùng MJPEG mượt một camera. Cần `ffmpeg` + `ffprobe` và **`python main.py`**."
    )
    if "ivm_zoom_cam" not in st.session_state:
        st.session_state.ivm_zoom_cam = None

    cams_live, cam_src = _load_ivm_cameras_json()
    st.session_state.ivm_cams_cache = cams_live
    st.session_state.ivm_cams_source = cam_src
    cams_live = st.session_state.get("ivm_cams_cache") or []
    api_base = str(st.session_state.get("api_base", API_DEFAULT)).rstrip("/")

    if cam_src == "file" and cams_live:
        st.info(
            "Đang hiển thị camera từ **file cấu hình** (cùng logic API), vì **GET /ivm/cameras** không lấy được "
            "(API chưa chạy, sai **Base URL**, hoặc lỗi mạng). "
            "Luồng MJPEG vẫn cần API — chạy **`python main.py`** và đặt Base URL đúng cổng (vd. `http://127.0.0.1:8010`, không phải cổng Streamlit)."
        )

    if not cams_live:
        st.warning(
            "Không có camera nào để hiển thị. Kiểm tra: (1) Sidebar **Base URL** trùng API "
            r"(vd. `http://127.0.0.1:8010`) — bấm **Kiểm tra /ivm/health**; (2) Đã chạy **`python main.py`**; "
            "(3) File **`camera_config.json`** có `cameras` hoặc `camera_source0`, …"
        )
    else:
        valid_ids = {str(c["id"]) for c in cams_live}
        if st.session_state.ivm_zoom_cam and st.session_state.ivm_zoom_cam not in valid_ids:
            st.session_state.ivm_zoom_cam = None

        use_native_preview = st.checkbox(
            "Dùng module StableCameraReader (test ổn định kết nối)",
            value=False,
            key="st_ivm_native_preview",
            help="Endpoint /ivm/preview_native/... — cùng kiểu MJPEG, khác class đọc camera. Tắt khi so sánh với preview OpenCV cũ.",
        )
        pv_base = "/ivm/preview_native" if use_native_preview else "/ivm/preview"
        if use_native_preview:
            st.caption("Xem FPS / lỗi: **GET /ivm/preview_native/status** (nút bên dưới).")
            if st.button("GET /ivm/preview_native/status", key="st_ivm_preview_native_status"):
                r = _get("/ivm/preview_native/status", timeout=20)
                st.json(r.json() if r.ok else {"error": r.text})

        if "ivm_preview_armed" not in st.session_state:
            st.session_state.ivm_preview_armed = False

        prev_cols = st.columns([1, 1, 2])
        with prev_cols[0]:
            if st.button(
                "Bật stream xem trước (tất cả camera)",
                key="st_ivm_preview_warm_btn",
                type="primary",
            ):
                try:
                    wr = _post("/ivm/preview/warm", timeout=30)
                    if wr.ok:
                        st.session_state.ivm_preview_armed = True
                        st.success(f"Đã bật preview cho {wr.json().get('count', '?')} camera.")
                    else:
                        st.error(wr.text[:300])
                except Exception as ex:
                    st.error(str(ex))
        with prev_cols[1]:
            if st.button("Ngắt stream xem trước", key="st_ivm_preview_disarm_btn"):
                ok_n, fail_n = 0, 0
                for c in cams_live:
                    cid = str(c["id"])
                    try:
                        rr = _post(f"{pv_base}/{cid}/stop", timeout=20)
                        if rr.ok:
                            ok_n += 1
                        else:
                            fail_n += 1
                    except Exception:
                        fail_n += 1
                st.session_state.ivm_preview_armed = False
                st.success(f"Đã dừng preview: {ok_n} OK, {fail_n} lỗi.")
        with prev_cols[2]:
            if st.session_state.ivm_preview_armed:
                st.caption("Preview **đang bật** — reload trang không tự mở lại stream.")
            else:
                st.caption("Preview **tắt** — bấm nút bật stream trước khi xem lưới.")

        show_streams = st.checkbox(
            "Hiển thị lưới camera",
            value=False,
            key="st_ivm_mjpeg_on",
            disabled=not st.session_state.ivm_preview_armed,
        )

        st.caption(f"**{len(cams_live)}** camera — lưới hiển thị đủ theo config.")

        cam_ids_preview = [str(c["id"]) for c in cams_live]

        try:
            _analyze_payload = (_get("/ivm/cameras/analyze", timeout=10).json() or {})
            _analyze_states = dict(_analyze_payload.get("states") or {})
            _analyze_sessions = dict(_analyze_payload.get("sessions") or {})
        except Exception as ex:
            _analyze_states = {cid: False for cid in cam_ids_preview}
            _analyze_sessions = {}
            st.warning(f"Không đọc trạng thái nhận diện: {ex}")

        def _set_camera_recognition(cid: str, enabled: bool, **kwargs: Any) -> None:
            payload: Dict[str, Any] = {"enabled": enabled, **kwargs}
            timeout_s = 15 if enabled else 30
            try:
                r = _post(f"/ivm/cameras/{cid}/analyze", json=payload, timeout=timeout_s)
            except requests.exceptions.ReadTimeout:
                st.warning(
                    f"`{cid}`: API không phản hồi kịp — có thể server đang bận. "
                    "Thử **Tải lại trang** hoặc kiểm tra log `main.py`."
                )
                return
            if r.ok:
                data = r.json() or {}
                if data.get("draining"):
                    pending = data.get("infer_queue_pending")
                    st.info(
                        f"`{cid}`: đang xử lý **{pending}** khung còn lại trong hàng đợi "
                        "(phiên đóng sau khi model xử lý xong)."
                    )
                st.session_state[f"ivm_act_{cid}"] = {
                    "hub_worker_running": data.get("hub_worker_running"),
                    "reader_connected": data.get("reader_connected"),
                    "activity": data.get("recent_activity") or [],
                    "last_meta": {},
                }
                if not data.get("hub_worker_running"):
                    st.warning(
                        f"`{cid}`: hub chưa chạy — cần **`python main.py`** (không `--no-camera`). "
                        "Xem log trong terminal đó."
                    )
            else:
                st.error(f"Bật/tắt nhận diện lỗi: {r.text[:300]}")

        weapon_live: Dict[str, Any] = {}
        try:
            wr = _get("/ivm/weapon-alerts/live", params={"limit": 40}, timeout=8)
            if wr.ok:
                weapon_live = wr.json() or {}
        except Exception:
            weapon_live = {}

        weapon_alerts_auto_refresh_fragment(
            cam_ids_preview,
            _analyze_states,
            get_json=_get,
            api_base=api_base,
            poll_interval_s=2.0,
        )

        _by_cam = dict(weapon_live.get("by_camera") or {})
        render_per_camera_recognition_panel(
            cam_ids_preview,
            _analyze_states,
            set_enabled=_set_camera_recognition,
            api_base=api_base,
            cols_per_row=5,
            show_snapshots=False,
            active_sessions=_analyze_sessions,
            weapon_alerts_by_camera=dict(weapon_live.get("active_by_camera") or {}),
            weapon_meta_by_camera={
                cid: {
                    "alert_track_count": int((_by_cam.get(cid) or {}).get("alert_track_count") or 0),
                }
                for cid in cam_ids_preview
            },
        )
        st.divider()

        row_a, row_b = st.columns([2, 1])
        with row_a:
            n_grid = st.selectbox("Số cột lưới", [2, 3, 4], index=0, key="st_ivm_grid_cols")
        with row_b:
            if st.button("Ngắt tất cả preview (RTSP xem trước)", key="st_ivm_preview_stop_all"):
                ok_n, fail_n = 0, 0
                for c in cams_live:
                    cid = str(c["id"])
                    try:
                        rr = _post(f"{pv_base}/{cid}/stop", timeout=20)
                        if rr.ok:
                            ok_n += 1
                        else:
                            fail_n += 1
                    except Exception:
                        fail_n += 1
                st.success(f"Đã gửi dừng ({pv_base}): {ok_n} OK, {fail_n} lỗi.")

        zoom_id = st.session_state.ivm_zoom_cam
        if zoom_id and zoom_id in valid_ids:
            zu = f"{api_base}{pv_base}/{zoom_id}/mjpeg"
            st.markdown(f"### Phóng to — `{zoom_id}`")
            if show_streams:
                st.markdown(
                    f'<img src="{html.escape(zu)}" loading="lazy" decoding="async" '
                    f'style="width:100%;max-height:88vh;object-fit:contain;'
                    f'background:#0b1220;border-radius:10px;border:1px solid #374151;" alt="zoom" />',
                    unsafe_allow_html=True,
                )
                with st.expander("URL luồng", expanded=False):
                    st.code(zu, language="text")
            else:
                st.info("Bật **Bật hiển thị hình camera** để xem hình.")
            if st.button("← Thu nhỏ — quay lại lưới", type="primary", key="st_ivm_zoom_back"):
                st.session_state.ivm_zoom_cam = None
                st.rerun()
            st.divider()

        if not zoom_id or zoom_id not in valid_ids:
            st.markdown(f"##### Lưới camera — **{len(cams_live)}** kênh")
            n_cols = int(n_grid)
            cam_ids = cam_ids_preview

            if show_streams and not st.session_state.ivm_preview_armed:
                st.warning("Bấm **Bật stream xem trước** trước khi mở lưới.")
            elif show_streams:
                if use_native_preview:
                    st.caption(
                        "Lưới dùng **snapshot** (`/ivm/preview/...`) — tránh lặp hình do giới hạn MJPEG trình duyệt. "
                        "Phóng to dùng **StableCameraReader**."
                    )
                n_rows = (len(cam_ids) + n_cols - 1) // n_cols
                grid_h = min(2400, 56 + n_rows * 300)
                grid_url = f"{api_base}/ivm/preview/grid?cols={n_cols}"
                components.iframe(grid_url, height=grid_h, scrolling=True)
                zopts = ["— chọn camera —"] + cam_ids
                zpick = st.selectbox("Phóng to camera", zopts, key="ivm_zoom_pick")
                if zpick != zopts[0] and st.button("Mở phóng to", key="ivm_zoom_open"):
                    st.session_state.ivm_zoom_cam = zpick
                    st.rerun()
            else:
                for i in range(0, len(cams_live), n_cols):
                    chunk = cams_live[i : i + n_cols]
                    cols = st.columns(len(chunk))
                    for j, cam in enumerate(chunk):
                        with cols[j]:
                            st.caption(f"`{cam['id']}` (tắt hiển thị)")

    st.divider()
    st.subheader("Danh sách camera & Recorder (API)")
    if st.button("GET /ivm/cameras"):
        r = _get("/ivm/cameras")
        st.json(r.json() if r.ok else {"error": r.text})
    cam = st.text_input("camera_id (vd cam0)", value="cam0")
    src_override = st.text_input("source_url override (tuỳ chọn, RTSP/HTTP)", value="")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Start recorder"):
            body: Optional[Dict[str, Any]] = None
            if src_override.strip():
                body = {"source_url": src_override.strip()}
            r = _post(f"/ivm/cameras/{cam}/recorder/start", json=body)
            st.json(r.json() if r.ok else {"error": r.text})
    with col2:
        if st.button("Stop recorder"):
            r = _post(f"/ivm/cameras/{cam}/recorder/stop")
            st.json(r.json() if r.ok else {"error": r.text})
    with col3:
        if st.button("Recorder status"):
            r = _get(f"/ivm/cameras/{cam}/recorder/status")
            st.json(r.json() if r.ok else {"error": r.text})

    st.info(
        "Nhận diện từ camera: chạy **`python main.py`** (worker) — hoặc CLI "
        "`python -m identity_vm_app.worker.camera_worker --camera cam0`."
    )

with tab_c:
    st.subheader("Danh sách đã đăng ký (xem ảnh & tên)")
    if "people_gallery_page" not in st.session_state:
        st.session_state["people_gallery_page"] = 0
    if st.button("Tải & hiển thị từ API", key="people_load"):
        r = _get("/ivm/people")
        if r.ok:
            st.session_state["people_cache"] = r.json()
            st.session_state["people_gallery_page"] = 0
        else:
            st.session_state["people_cache"] = None
            st.error(r.text)

    cached = st.session_state.get("people_cache")
    if cached:
        faces_all = list(cached.get("faces") or [])
        grouped = cached.get("grouped_by_name") or {}

        if "people_search_prev" not in st.session_state:
            st.session_state["people_search_prev"] = ""

        search_col, clear_col = st.columns([5, 1])
        with search_col:
            search_q = st.text_input(
                "Tìm theo tên hoặc face_id",
                placeholder="Ví dụ: Nguyễn, Minh, 12 …",
                key="people_search_query",
            )
        with clear_col:
            st.write("")
            if st.button("Xóa lọc", key="people_search_clear"):
                st.session_state["people_search_query"] = ""
                st.session_state["people_search_prev"] = ""
                st.session_state["people_gallery_page"] = 0
                st.rerun()

        search_q = str(search_q or "")
        if search_q != st.session_state["people_search_prev"]:
            st.session_state["people_search_prev"] = search_q
            st.session_state["people_gallery_page"] = 0

        faces_filtered = _filter_faces_by_search(faces_all, search_q)
        searching = bool((search_q or "").strip())
        total_all = len(faces_all)
        total = len(faces_filtered)

        if searching:
            unique_names = len({str(f.get("name") or "") for f in faces_filtered})
            st.caption(
                f"Tìm **「{search_q.strip()}」**: **{total}** khuôn mặt "
                f"(**{unique_names}** tên khác nhau) / **{total_all}** trong DB"
            )
            if total == 0:
                st.warning("Không tìm thấy khuôn mặt phù hợp. Thử từ khóa ngắn hơn hoặc face_id.")
            else:
                _render_people_cards(
                    faces_filtered,
                    f"Kết quả tìm kiếm ({total})",
                    detailed=True,
                )
                with st.expander("Gom theo tên (thống kê nhanh)"):
                    name_counts: Dict[str, int] = {}
                    for f in faces_filtered:
                        n = str(f.get("name") or "(no name)")
                        name_counts[n] = name_counts.get(n, 0) + 1
                    for n, cnt in sorted(name_counts.items(), key=lambda x: (-x[1], x[0].lower())):
                        ids = [
                            int(x.get("id"))
                            for x in faces_filtered
                            if str(x.get("name") or "(no name)") == n
                        ]
                        st.markdown(f"**{n}** — {cnt} ảnh · face_id: `{', '.join(map(str, ids))}`")
        else:
            n_pages = max(1, (total + PEOPLE_GALLERY_PAGE_SIZE - 1) // PEOPLE_GALLERY_PAGE_SIZE)
            p = int(st.session_state["people_gallery_page"])
            p = max(0, min(p, n_pages - 1))
            st.session_state["people_gallery_page"] = p

            start = p * PEOPLE_GALLERY_PAGE_SIZE
            slice_faces = faces_filtered[start : start + PEOPLE_GALLERY_PAGE_SIZE]
            st.caption(
                f"**{total}** khuôn mặt — hiển thị **{start + 1}–{start + len(slice_faces)}** "
                f"— trang **{p + 1}/{n_pages}** ({PEOPLE_GALLERY_PAGE_SIZE} ảnh/trang) · "
                f"**{len(grouped)}** tên khác nhau"
            )
            col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 4])
            with col_nav1:
                if st.button("← Trước", key="people_pg_prev", disabled=p <= 0):
                    st.session_state["people_gallery_page"] = p - 1
                    st.rerun()
            with col_nav2:
                if st.button("Sau →", key="people_pg_next", disabled=p >= n_pages - 1):
                    st.session_state["people_gallery_page"] = p + 1
                    st.rerun()

            _render_people_cards(slice_faces, "Đã đăng ký (trang hiện tại)")

        with st.expander("JSON thô / thống kê"):
            st.json(cached)

    st.divider()
    st.caption("Thêm ảnh gallery cho một face_id")
    fid = st.number_input("face_id cho gallery", min_value=0, value=0, step=1)
    gup = st.file_uploader("Ảnh gallery", type=["jpg", "jpeg", "png"], key="gal")
    also = st.checkbox("also_embed (thêm vector vào DB)", value=False)
    if st.button("POST gallery") and gup:
        files = {"file": (gup.name, gup.getvalue(), gup.type or "image/jpeg")}
        r = _post(
            f"/ivm/people/{int(fid)}/media",
            files=files,
            params={"also_embed": also},
        )
        st.json(r.json() if r.ok else {"error": r.text})
    if st.button("List media"):
        r = _get(f"/ivm/people/{int(fid)}/media")
        st.json(r.json() if r.ok else {"error": r.text})

with tab_d:
    st.subheader("Báo cáo camera — người đã định danh")
    st.caption(
        "Chỉ hiển thị người khớp thư viện. Bấm **Tải báo cáo** → chọn tên → **Xem chi tiết**: "
        "mỗi frame: **ảnh lớn** (box người, mặt, vũ khí) + **2 crop nhỏ** mặt và vũ khí bên dưới. "
        "Cần API `main.py` + nhận diện camera BẬT để có `weapon_crop` trong sự kiện."
    )

    def _api_get_json(path: str, **kw) -> Dict[str, Any]:
        r = _get(path, **kw)
        if not r.ok:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        return r.json()

    _cams_rep, _ = _load_ivm_cameras_json()
    _rep_ids = [str(c["id"]) for c in (_cams_rep or [])] or ["cam0"]
    render_camera_track_report(
        _rep_ids,
        api_get=_api_get_json,
        api_base=str(st.session_state.get("api_base", API_DEFAULT)).rstrip("/"),
        key_prefix="st_rep",
    )

    st.divider()
    with st.expander("API thô (JSON)"):
        cam2 = st.text_input("Camera", value="cam0", key="rep_json_cam")
        hours2 = st.number_input("Giờ lùi", value=24, min_value=1, key="rep_json_h")
        if st.button("GET summary JSON", key="rep_json_btn"):
            to_ts = time.time()
            from_ts = to_ts - float(hours2) * 3600.0
            r = _get(
                f"/ivm/cameras/{cam2}/reports/summary",
                params={"from_ts": from_ts, "to_ts": to_ts},
            )
            st.json(r.json() if r.ok else {"error": r.text})
        pref = st.text_input("person_ref (appearances)", value="", key="rep_json_pref")
        if st.button("GET appearances", key="rep_json_app"):
            params: Dict[str, Any] = {}
            if pref.strip():
                params["person_ref"] = pref.strip()
            r = _get("/ivm/people/appearances", params=params)
            st.json(r.json() if r.ok else {"error": r.text})

with tab_e:
    ev = st.text_input("event_id (UUID)", value="")
    if st.button("Export cut (JSON path)") and ev.strip():
        r = _post("/ivm/export/cut", json={"event_id": ev.strip()})
        st.json(r.json() if r.ok else {"error": r.text})
    if st.button("Gợi ý tải file (curl)") and ev.strip():
        base = st.session_state.get("api_base", API_DEFAULT).rstrip("/")
        url = f"{base}/ivm/export/cut?download=true"
        st.code(
            f'curl -X POST "{url}" -H "Content-Type: application/json" '
            f'-d "{{\\"event_id\\": \\"{ev.strip()}\\"}}" --output cut.mkv',
            language="bash",
        )

with tab_f:
    st.caption(
        "Upload/chạy job qua API `main.py`. Sau khi job **done**, mở **Báo cáo người → Theo tracking → Xem track** "
        "để xem crop mặt và vũ khí cạnh nhau (cần `IVM_VIDEO_ANALYZE_SAVE_CROPS=1` và weapon bật)."
    )
    render_video_analyze_panel(str(st.session_state.get("api_base", API_DEFAULT)))
