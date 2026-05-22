"""Cắt video thành N đoạn (ffmpeg) — giống VideoMaster `__split_video`."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2

from identity_vm_app import settings as s


@dataclass(frozen=True)
class VideoSegmentSpec:
    cut_session: int
    path: Path
    start_time_s: float


def split_dir_for_job(job_id: str) -> Path:
    return Path(s.IVM_VIDEO_ANALYZE_DIR) / "jobs" / job_id / "split"


def _ffmpeg_exe() -> str:
    return (s.IVM_FFMPEG_BIN or "ffmpeg").strip() or "ffmpeg"


def probe_duration_s(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return 0.0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps > 0 and n > 0:
            return n / fps
    finally:
        cap.release()
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
        return max(0.0, float(out))
    except Exception:
        return 0.0


def _cut_segment(
    source: Path,
    output: Path,
    start_s: float,
    duration_s: float,
    *,
    use_gpu_encode: bool,
) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        try:
            output.unlink()
        except OSError:
            pass
    encoders = ("h264_nvenc", "libx264") if use_gpu_encode else ("libx264",)
    for enc in encoders:
        cmd = [
            _ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(max(0.0, start_s)),
            "-i",
            str(source),
            "-t",
            str(max(0.01, duration_s)),
            "-c:v",
            enc,
            "-an",
            "-y",
            str(output),
        ]
        try:
            subprocess.run(cmd, check=True)
            if output.is_file() and output.stat().st_size > 0:
                return True
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            continue
    return False


def split_video_for_job(source: Path, job_id: str, n_parts: int) -> List[VideoSegmentSpec]:
    """
    Chia `source` thành `n_parts` file trong jobs/{id}/split/.
    n_parts <= 1 → một segment trỏ thẳng source (không cắt).
    """
    n = max(1, int(n_parts))
    if n <= 1:
        return [VideoSegmentSpec(cut_session=1, path=source, start_time_s=0.0)]

    duration = probe_duration_s(source)
    if duration <= 0.1:
        return [VideoSegmentSpec(cut_session=1, path=source, start_time_s=0.0)]

    part_dur = duration / float(n)
    split_dir = split_dir_for_job(job_id)
    split_dir.mkdir(parents=True, exist_ok=True)
    ext = source.suffix.lower() if source.suffix else ".mp4"
    use_gpu = bool(s.IVM_VIDEO_ANALYZE_SPLIT_GPU_ENCODE)

    specs: List[VideoSegmentSpec] = []
    tasks: List[tuple[int, Path, float, float]] = []
    for i in range(n):
        start = i * part_dur
        out = split_dir / f"video_split_{i + 1}{ext}"
        specs.append(VideoSegmentSpec(cut_session=i + 1, path=out, start_time_s=start))
        tasks.append((i, out, start, part_dur))

    ok_flags: List[bool] = [False] * n
    with ThreadPoolExecutor(max_workers=min(n, 8)) as ex:
        futs = {
            ex.submit(_cut_segment, source, out, start, part_dur, use_gpu_encode=use_gpu): i
            for i, out, start, part_dur in tasks
        }
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                ok_flags[idx] = bool(fut.result())
            except Exception:
                ok_flags[idx] = False

    if not all(ok_flags):
        return [VideoSegmentSpec(cut_session=1, path=source, start_time_s=0.0)]

    return specs


def remove_split_dir(job_id: str) -> None:
    d = split_dir_for_job(job_id)
    if d.is_dir():
        import shutil

        shutil.rmtree(d, ignore_errors=True)
