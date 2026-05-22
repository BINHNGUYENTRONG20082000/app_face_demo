from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from identity_vm_app import settings as s


def export_segment_cut(
    *,
    src_path: str,
    offset_start_s: float,
    offset_end_s: float,
    out_path: Optional[Path] = None,
    ffmpeg_bin: Optional[str] = None,
    accurate_seek: bool = False,
) -> Path:
    """Cắt đoạn [offset_start_s, offset_end_s] từ file archive bằng ffmpeg.

    accurate_seek=True: -ss sau -i (khớp khung hơn khi vẽ box lên đoạn cắt).
    """
    src = Path(src_path)
    if not src.is_file():
        raise FileNotFoundError(f"Không tìm thấy file archive: {src_path}")

    ffmpeg_bin = ffmpeg_bin or s.IVM_FFMPEG_BIN
    s.IVM_EXPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = max(0.0, float(offset_start_s))
    t1 = max(t0 + 0.05, float(offset_end_s))
    dur = max(0.5, t1 - t0)
  # Tối thiểu 0.5s để player đọc được
    out_path = out_path or s.IVM_EXPORT_CACHE_DIR / f"cut_{int(t0)}_{int(t1)}.mp4"

    def _run(cmd: list[str]) -> None:
        proc = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", errors="replace")[-800:]
            raise RuntimeError(err or f"ffmpeg exit {proc.returncode}")

    if not accurate_seek:
        copy_cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{t0:.3f}",
            "-i",
            str(src),
            "-t",
            f"{dur:.3f}",
            "-c",
            "copy",
            str(out_path),
        ]
        try:
            _run(copy_cmd)
            if out_path.is_file() and out_path.stat().st_size > 0:
                return out_path
        except Exception:
            pass

    out_re = out_path.with_suffix(".reencode.mp4") if not accurate_seek else out_path
    if accurate_seek:
        re_cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(src),
            "-ss",
            f"{t0:.3f}",
            "-t",
            f"{dur:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            "-an",
            str(out_re),
        ]
    else:
        re_cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{t0:.3f}",
            "-i",
            str(src),
            "-t",
            f"{dur:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            "-an",
            str(out_re),
        ]
    _run(re_cmd)
    if not out_re.is_file() or out_re.stat().st_size == 0:
        raise RuntimeError("Xuất cut thất bại (file rỗng)")
    if not accurate_seek and out_re.resolve() != out_path.resolve():
        if out_path.is_file():
            out_path.unlink(missing_ok=True)
        out_re.replace(out_path)
        return out_path
    return out_re
