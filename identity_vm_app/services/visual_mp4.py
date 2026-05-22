"""MP4 overlay phân tích — chuẩn hóa H.264 để trình duyệt / Streamlit phát được."""

from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from identity_vm_app import settings as s


def visual_root() -> Path:
    return Path(
        os.getenv("IVM_ANALYZE_VISUAL_DIR", str(s.IVM_DATA_DIR / "analyze_visual"))
    ).resolve()


def list_visual_sessions_from_disk(camera_id: str, *, limit: int = 20) -> List[Dict[str, Any]]:
    """Liệt kê phiên video từ đĩa (không cần API)."""
    cam_dir = visual_root() / camera_id
    if not cam_dir.is_dir():
        return []
    items: List[Dict[str, Any]] = []
    for mp4 in sorted(cam_dir.glob("session_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        if mp4.stem.endswith("_web"):
            continue
        if mp4.stat().st_size <= 0:
            continue
        sid = mp4.stem.replace("session_", "", 1)
        web = browser_mp4_path(mp4)
        items.append(
            {
                "session_id": sid,
                "path": str(mp4),
                "browser_path": str(web) if web.is_file() else None,
                "size_bytes": mp4.stat().st_size,
                "mtime_utc": mp4.stat().st_mtime,
                "has_web": web.is_file() and web.stat().st_size > 0,
            }
        )
        if len(items) >= limit:
            break
    return items


def load_visual_mp4_bytes(camera_id: str, session_id: str) -> tuple[bytes, str]:
    """
    Đọc MP4 phát được trên trình duyệt (ưu tiên _web, remux nếu cần).
    Trả về (bytes, đường dẫn nguồn).
    """
    raw = visual_root() / camera_id / f"session_{session_id}.mp4"
    if not raw.is_file() or raw.stat().st_size <= 0:
        raise FileNotFoundError(f"Không có file: {raw}")
    play = resolve_browser_visual_mp4(raw)
    return play.read_bytes(), str(play)


def browser_mp4_path(src: Path) -> Path:
    """File H.264 tương ứng (session_123_web.mp4)."""
    p = Path(src).resolve()
    return p.with_name(f"{p.stem}_web.mp4")


def remux_visual_mp4_for_browser(
    src: Path,
    *,
    ffmpeg_bin: Optional[str] = None,
) -> Path:
    """
    Tạo bản H.264 + faststart từ OpenCV mp4v.
    Trả về đường dẫn file phát được trên trình duyệt.
    """
    src = Path(src).resolve()
    if not src.is_file() or src.stat().st_size == 0:
        raise FileNotFoundError(f"Video rỗng hoặc không tồn tại: {src}")

    out = browser_mp4_path(src)
    if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime and out.stat().st_size > 0:
        return out

    ffmpeg_bin = ffmpeg_bin or s.IVM_FFMPEG_BIN
    if not shutil.which(ffmpeg_bin) and not Path(ffmpeg_bin).is_file():
        return src

    tmp = out.with_suffix(".tmp.mp4")
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    proc = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace")[-400:]
        raise RuntimeError(err or f"ffmpeg remux failed ({proc.returncode})")
    if out.is_file():
        out.unlink(missing_ok=True)
    tmp.replace(out)
    return out


def resolve_browser_visual_mp4(src: Path) -> Path:
    """Ưu tiên bản _web.mp4; nếu chưa có thì remux."""
    src = Path(src).resolve()
    web = browser_mp4_path(src)
    if web.is_file() and web.stat().st_size > 0:
        return web
    try:
        return remux_visual_mp4_for_browser(src)
    except Exception:
        return src
