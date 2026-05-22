"""
Đọc RTSP/HTTP bằng tiến trình FFmpeg → rawvideo BGR24 trên stdout.

OpenCV không giải HEVC nữa (tránh lỗi POC / Duplicate POC). Chỉ dùng cv2.imencode cho MJPEG preview.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np

from identity_vm_app import settings as s


def _resolve_ffprobe(ffmpeg_bin: str) -> str:
    env_bin = os.getenv("IVM_FFPROBE_BIN", "").strip()
    if env_bin and os.path.isfile(env_bin):
        return env_bin
    root = os.path.dirname(os.path.abspath(ffmpeg_bin))
    for name in ("ffprobe.exe", "ffprobe"):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return p
    w = shutil.which("ffprobe")
    if w:
        return w
    return "ffprobe"


def ffprobe_stream_size(url: str, ffprobe_exe: str, *, timeout: float = 20.0) -> Optional[tuple[int, int]]:
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
                ffprobe_exe,
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


_ffprobe_size_cache: dict[str, tuple[int, int, float]] = {}
_ffprobe_cache_lock = threading.Lock()
_FFPROBE_CACHE_TTL_S = 120.0


def _cached_probe_size(url: str, ffprobe_exe: str, *, timeout: float = 20.0) -> Optional[tuple[int, int]]:
    now = time.monotonic()
    with _ffprobe_cache_lock:
        hit = _ffprobe_size_cache.get(url)
        if hit is not None:
            w, h, ts = hit
            if now - ts < _FFPROBE_CACHE_TTL_S:
                return w, h
    wh = ffprobe_stream_size(url, ffprobe_exe, timeout=timeout)
    if wh:
        with _ffprobe_cache_lock:
            _ffprobe_size_cache[url] = (wh[0], wh[1], now)
    return wh


def _scaled_size(src_w: int, src_h: int, max_h: int) -> tuple[int, int]:
    """Giữ tỉ lệ; chiều cao tối đa max_h (chẵn pixel)."""
    mh = max(64, min(2160, int(max_h)))
    if src_h <= mh:
        w, h = src_w, src_h
    else:
        h = mh
        w = max(2, int(round(src_w * (mh / float(src_h)))))
    w = (w // 2) * 2
    h = (h // 2) * 2
    return max(w, 2), max(h, 2)


def run_ffmpeg_preview_loop(
    url: str,
    *,
    jpeg_quality: int,
    capture_period_s: float,
    reconnect_delay_s: float,
    open_backoff_cap_s: float,
    max_height: int,
    stop_event: threading.Event,
    lock: threading.Lock,
    last_jpeg_out: list,
    error_out: list,
    camera_id: str,
) -> None:
    """
    Gán last_jpeg_out[0] = bytes JPEG; error_out[0] = str | None.
    """
    ffmpeg_bin = s.IVM_FFMPEG_BIN
    if not shutil.which(ffmpeg_bin) and not os.path.isfile(ffmpeg_bin):
        error_out[0] = "Không tìm thấy ffmpeg (cài FFmpeg hoặc đặt IVM_FFMPEG_BIN)"
        return

    ffprobe = _resolve_ffprobe(ffmpeg_bin)
    open_failures = 0

    while not stop_event.is_set():
        wh = _cached_probe_size(url, ffprobe)
        if not wh:
            open_failures += 1
            error_out[0] = f"ffprobe không đọc được kích thước (lần {open_failures})"
            exp = min(
                open_backoff_cap_s,
                reconnect_delay_s * (1.45 ** min(open_failures, 12)),
            )
            time.sleep(min(open_backoff_cap_s, exp + random.uniform(0, 0.75)))
            continue

        sw, sh = wh
        w, h = _scaled_size(sw, sh, max_height)
        frame_bytes = w * h * 3
        # CLI FFmpeg dùng -timeout (µs); -stimeout chỉ có trong OPENCV_FFMPEG_CAPTURE_OPTIONS.
        timeout_us = str(int(os.getenv("IVM_FFMPEG_STIMEOUT_US", "8000000")))

        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-rtsp_transport",
            "tcp",
            "-timeout",
            timeout_us,
            "-fflags",
            "+discardcorrupt+genpts",
            "-i",
            url,
            "-an",
            "-map",
            "0:v:0",
            "-vf",
            f"scale={w}:{h}:flags=fast_bilinear,format=bgr24",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]

        proc: Any = None
        try:
            kwargs: dict = {"bufsize": frame_bytes * 2}
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **kwargs,
            )

            def _drain_stderr() -> None:
                if proc and proc.stderr:
                    try:
                        proc.stderr.read()
                    except Exception:
                        pass

            threading.Thread(target=_drain_stderr, daemon=True).start()

            error_out[0] = None
            open_failures = 0
            assert proc.stdout is not None

            while not stop_event.is_set():
                buf = b""
                while len(buf) < frame_bytes and not stop_event.is_set():
                    need = frame_bytes - len(buf)
                    chunk = proc.stdout.read(need)
                    if not chunk:
                        break
                    buf += chunk
                if len(buf) != frame_bytes:
                    error_out[0] = "Thiếu byte khung rawvideo — đóng FFmpeg và mở lại"
                    break

                frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
                enc_ok, enc = cv2.imencode(
                    ".jpg",
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
                if enc_ok:
                    with lock:
                        last_jpeg_out[0] = enc.tobytes()
                time.sleep(capture_period_s)

        except Exception as e:
            error_out[0] = str(e)
            open_failures += 1
        finally:
            if proc is not None:
                try:
                    proc.terminate()
                    proc.wait(timeout=4.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        if stop_event.is_set():
            break
        time.sleep(reconnect_delay_s + random.uniform(0, 1.0))


def preview_should_use_ffmpeg(source: object) -> bool:
    if not s.IVM_PREVIEW_DECODE_VIA_FFMPEG:
        return False
    if not isinstance(source, str):
        return False
    u = source.strip().lower()
    return u.startswith(("rtsp://", "rtsps://", "http://", "https://"))
