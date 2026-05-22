"""FFprobe / FFmpeg CLI helpers — dùng chung preview và reader nhận diện."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Optional, Tuple

from identity_vm_app import settings as s

_probe_cache: dict[str, Tuple[int, int, float]] = {}
_probe_lock = threading.Lock()
_PROBE_TTL_S = 120.0


def ffmpeg_bin() -> str:
    return (s.IVM_FFMPEG_BIN or "ffmpeg").strip() or "ffmpeg"


def ffprobe_bin(ffmpeg_exe: Optional[str] = None) -> str:
    fb = ffmpeg_exe or ffmpeg_bin()
    env_bin = os.getenv("IVM_FFPROBE_BIN", "").strip()
    if env_bin and os.path.isfile(env_bin):
        return env_bin
    root = os.path.dirname(os.path.abspath(fb))
    for name in ("ffprobe.exe", "ffprobe"):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return p
    w = shutil.which("ffprobe")
    return w or "ffprobe"


def ffmpeg_cli_available() -> bool:
    fb = ffmpeg_bin()
    return bool(shutil.which(fb) or (os.path.isfile(fb) if fb else False))


def ffprobe_stream_size(url: str, *, timeout: float = 20.0) -> Optional[Tuple[int, int]]:
    ffprobe = ffprobe_bin()
    try:
        kwargs: dict = {
            "capture_output": True,
            "text": True,
            "timeout": timeout,
        }
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        cp = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                url,
            ],
            **kwargs,
        )
        if cp.returncode != 0 or not (cp.stdout or "").strip():
            return None
        parts = (cp.stdout.strip().splitlines()[0]).split(",")
        if len(parts) < 2:
            return None
        w, h = int(parts[0]), int(parts[1])
        if w <= 0 or h <= 0:
            return None
        return w, h
    except Exception:
        return None


def cached_ffprobe_stream_size(url: str, *, timeout: float = 20.0) -> Optional[Tuple[int, int]]:
    now = time.monotonic()
    with _probe_lock:
        hit = _probe_cache.get(url)
        if hit is not None:
            w, h, ts = hit
            if now - ts < _PROBE_TTL_S:
                return w, h
    wh = ffprobe_stream_size(url, timeout=timeout)
    if wh:
        with _probe_lock:
            _probe_cache[url] = (wh[0], wh[1], now)
    return wh


def even_dims(width: int, height: int) -> Tuple[int, int]:
    w = max(2, int(width))
    h = max(2, int(height))
    return (w // 2) * 2, (h // 2) * 2


def scale_dims_keep_aspect(src_w: int, src_h: int, max_h: int) -> Tuple[int, int]:
    """max_h <= 0 → giữ nguyên kích thước nguồn."""
    if max_h <= 0 or src_h <= max_h:
        return even_dims(src_w, src_h)
    mh = max(64, min(4096, int(max_h)))
    h = mh
    w = max(2, int(round(src_w * (mh / float(src_h)))))
    return even_dims(w, h)


def rtsp_url(source: object) -> Optional[str]:
    if not isinstance(source, str):
        return None
    u = source.strip()
    low = u.lower()
    if low.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return u
    return None


def popen_kwargs(*, bufsize: int = 0) -> dict:
    kw: dict = {}
    if bufsize > 0:
        kw["bufsize"] = bufsize
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kw


def terminate_process(proc: subprocess.Popen) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=4.0)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
