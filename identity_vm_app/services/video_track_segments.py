"""Đoạn xuất hiện theo id_tracking + ghép video on-demand.

Video có box (ưu tiên): cắt video gốc full FPS + giữ box theo frame_index mẫu (mượt + không lệch nội suy).
Fallback: ghép khung mẫu img_url. Video không box: chỉ cắt gốc.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from identity_vm_app import settings as s
from identity_vm_app.services.export_cut import export_segment_cut
from identity_vm_app.services.video_report_crops import (
    draw_boxes_on_frame,
    load_frame_bgr,
    parse_box,
    parse_weapon_boxes,
)
from identity_vm_app.services.video_report_merge import (
    pick_first_appearance_row,
    resolve_track_identity,
)
from identity_vm_app.services.visual_mp4 import browser_mp4_path, resolve_browser_visual_mp4
from identity_vm_app.store.video_analyze_store import get_video_analyze_store

_build_locks: Dict[str, threading.Lock] = {}

# FPS ghép MP4: ưu tiên fps file gốc (mượt), không chỉ sample_fps*2 (dễ giật).
_DEFAULT_ENCODE_FPS = 25.0
_MAX_HOLD_S = 8.0
_MIN_HOLD_FRAMES_AT_FPS = 2  # tối thiểu ~2 khung / mẫu


def _encode_fps_bounds() -> Tuple[float, float]:
    lo = float(s.IVM_TRACK_SEGMENT_ENCODE_FPS_MIN)
    hi = float(s.IVM_TRACK_SEGMENT_ENCODE_FPS_MAX)
    if lo > hi:
        lo, hi = hi, lo
    return max(1.0, lo), max(lo, hi)


def _encode_fps_for_job(job: Optional[Dict[str, Any]], override: Optional[float] = None) -> float:
    fps_min, fps_max = _encode_fps_bounds()
    if override is not None and float(override) > 0:
        return max(fps_min, min(fps_max, float(override)))
    try:
        vf = float((job or {}).get("fps") or 0)
        if vf > 0:
            return max(fps_min, min(fps_max, vf))
    except (TypeError, ValueError):
        pass
    try:
        af = float((job or {}).get("analyze_fps") or 0)
        if af > 0:
            return max(fps_min, min(fps_max, af))
    except (TypeError, ValueError):
        pass
    try:
        sf = float((job or {}).get("sample_fps") or 0)
        if sf > 0:
            return max(fps_min, min(fps_max, max(sf * 4.0, 10.0)))
    except (TypeError, ValueError):
        pass
    return max(fps_min, min(fps_max, _DEFAULT_ENCODE_FPS))


def frame_hold_counts(times_s: List[float], encode_fps: float) -> List[int]:
    """
    Số lần ghi lại cùng một khung mẫu để khớp timeline time_analyze_s (tránh phát quá nhanh/giật).
    """
    if not times_s:
        return []
    fps_min, fps_max = _encode_fps_bounds()
    fps = max(fps_min, min(fps_max, float(encode_fps)))
    min_dt = 1.0 / fps
    max_hold = max(1, int(round(fps * _MAX_HOLD_S)))
    min_hold = max(1, _MIN_HOLD_FRAMES_AT_FPS)
    if len(times_s) == 1:
        return [max(min_hold, min(max_hold, int(round(fps))))]

    holds: List[int] = []
    for i in range(len(times_s) - 1):
        dt = max(min_dt, float(times_s[i + 1]) - float(times_s[i]))
        holds.append(
            max(min_hold, min(max_hold, int(round(dt * fps))))
        )
    dts = [float(times_s[i + 1]) - float(times_s[i]) for i in range(len(times_s) - 1)]
    med = sorted(dts)[len(dts) // 2] if dts else min_dt
    holds.append(max(min_hold, min(max_hold, int(round(med * fps)))))
    return holds


def gap_s_for_job(job: Optional[Dict[str, Any]]) -> float:
    """Khoảng cách tối đa giữa 2 khung mẫu liên tiếp trong cùng một đoạn."""
    sf = 0.0
    if job:
        try:
            sf = float(job.get("sample_fps") or 0)
        except (TypeError, ValueError):
            sf = 0.0
    if sf > 0:
        return max(1.0, 2.5 / sf)
    return 3.0


def split_time_segments(
    rows: List[Dict[str, Any]],
    *,
    gap_s: float,
) -> List[List[Dict[str, Any]]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: float(r.get("time_analyze_s") or r.get("time_analyze") or 0))
    segments: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    last_t: Optional[float] = None
    for row in ordered:
        t = float(row.get("time_analyze_s") or row.get("time_analyze") or 0)
        if last_t is not None and current and (t - last_t) > float(gap_s):
            segments.append(current)
            current = []
        current.append(row)
        last_t = t
    if current:
        segments.append(current)
    return segments


def _segment_summary(
    job_id: str,
    id_tracking: int,
    segment_index: int,
    seg_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ordered = _track_timeline(seg_rows)
    first = pick_first_appearance_row(ordered) or ordered[0]
    t0 = float(first.get("time_analyze_s") or first.get("time_analyze") or 0)
    t1 = float(ordered[-1].get("time_analyze_s") or ordered[-1].get("time_analyze") or t0)
    identity = resolve_track_identity(ordered)
    armed_frames = sum(1 for r in ordered if int(r.get("armed") or 0))
    weapon_types: List[str] = []
    for r in ordered:
        raw = r.get("weapon_types_json")
        if not raw:
            continue
        try:
            parsed = json.loads(str(raw))
            if isinstance(parsed, list):
                weapon_types.extend(str(t) for t in parsed if t)
        except json.JSONDecodeError:
            pass
    weapon_types = sorted(set(weapon_types))
    from identity_vm_app.services.weapon_track_status import classify_weapon_by_frame_count

    wcls = classify_weapon_by_frame_count(armed_frames, weapon_types)
    armed_any = bool(wcls["armed"])
    weapon_label = str(wcls["weapon_label"])
    return {
        "segment_id": f"{job_id}:{id_tracking}:{segment_index}",
        "job_id": job_id,
        "id_tracking": int(id_tracking),
        "segment_index": int(segment_index),
        "time_analyze": t0,
        "end_time": t1,
        "frame_count": len(seg_rows),
        "hit_count": len(seg_rows),
        "display_name": identity.get("display_name"),
        "track_name": identity.get("display_name") or "unknown",
        "face_id": identity.get("face_id"),
        "match_score": identity.get("match_score"),
        "suspect_faces": identity.get("suspect_faces") or [],
        "report_id": first.get("id"),
        "first_frame_report_id": first.get("id"),
        "video_clip_min": min(int(r.get("video_clip") or 1) for r in ordered),
        "video_clip_max": max(int(r.get("video_clip") or 1) for r in ordered),
        "armed": 1 if armed_any else 0,
        "dangerous": 1 if wcls["dangerous"] else 0,
        "weapon_armed_frames": armed_frames,
        "weapon_label": weapon_label,
        "weapon_types_json": json.dumps(weapon_types, ensure_ascii=False) if weapon_types else None,
    }


def list_track_appearance_segments(
    job_id: str,
    *,
    gap_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    store = get_video_analyze_store()
    job = store.get_job(job_id) or {}
    gap = float(gap_s) if gap_s is not None else gap_s_for_job(job)
    raw = store.list_person_reports(job_id, limit=50000)
    if not raw:
        return []

    by_track: Dict[int, List[Dict[str, Any]]] = {}
    for row in raw:
        tid = int(row.get("id_tracking") or 0)
        by_track.setdefault(tid, []).append(row)

    out: List[Dict[str, Any]] = []
    for tid in sorted(by_track.keys()):
        for seg_idx, seg_rows in enumerate(split_time_segments(by_track[tid], gap_s=gap)):
            out.append(_segment_summary(job_id, tid, seg_idx, seg_rows))
    out.sort(key=lambda x: (int(x["id_tracking"]), float(x["time_analyze"])))
    return out


def _frames_for_segment(
    job_id: str,
    id_tracking: int,
    *,
    segment_index: Optional[int] = None,
    start_time_s: Optional[float] = None,
    end_time_s: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Trả về (segment_rows, unique_frames_by_img_url) cho track."""
    store = get_video_analyze_store()
    rows = store.list_person_reports_by_tracking(job_id, int(id_tracking), limit=50000)
    if not rows:
        return [], []

    job = store.get_job(job_id) or {}
    gap = gap_s_for_job(job)

    if segment_index is not None:
        segments = split_time_segments(rows, gap_s=gap)
        idx = int(segment_index)
        if idx < 0 or idx >= len(segments):
            return [], []
        seg_rows = segments[idx]
    else:
        seg_rows = rows
        if start_time_s is not None:
            seg_rows = [
                r
                for r in seg_rows
                if float(r.get("time_analyze_s") or 0) >= float(start_time_s)
            ]
        if end_time_s is not None and float(end_time_s) > 0:
            seg_rows = [
                r
                for r in seg_rows
                if float(r.get("time_analyze_s") or 0) <= float(end_time_s)
            ]

    if not seg_rows:
        return [], []

    by_url: Dict[str, Dict[str, Any]] = {}
    for r in sorted(seg_rows, key=lambda x: float(x.get("time_analyze_s") or 0)):
        url = str(r.get("img_url") or "")
        if not url:
            continue
        if url not in by_url:
            by_url[url] = {
                "img_url": url,
                "time_analyze_s": float(r.get("time_analyze_s") or 0),
                "rows": [],
            }
        by_url[url]["rows"].append(r)

    frames = sorted(by_url.values(), key=lambda x: float(x["time_analyze_s"]))
    return seg_rows, frames


def resolve_job_source_video(job_id: str) -> Optional[Path]:
    """File video gốc của job (đường dẫn staged khi upload/phân tích)."""
    job = get_video_analyze_store().get_job(job_id) or {}
    vp = str(job.get("video_path") or "").strip()
    if vp:
        p = Path(vp)
        if p.is_file():
            return p.resolve()
    job_dir = Path(s.IVM_VIDEO_ANALYZE_DIR) / "jobs" / job_id
    if job_dir.is_dir():
        for child in sorted(job_dir.glob("source.*")):
            if child.is_file():
                return child.resolve()
    return None


def segment_time_bounds(
    seg_rows: List[Dict[str, Any]],
    *,
    align_first_sample: bool = False,
) -> Tuple[float, float]:
    """Khoảng thời gian cắt từ video gốc [t0, t1].

    align_first_sample=True: bắt đầu đúng khung mẫu đầu (không lùi 0.15s) — box khớp draw-box-person.
    """
    times = sorted(float(r.get("time_analyze_s") or r.get("time_analyze") or 0) for r in seg_rows)
    if not times:
        return 0.0, 0.5
    t0, t1 = times[0], times[-1]
    pad_end = 0.5
    if len(times) >= 2:
        dts = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        pad_end = max(0.25, sorted(dts)[len(dts) // 2] * 1.5)
    if align_first_sample:
        return max(0.0, t0), t1 + pad_end
    return max(0.0, t0 - 0.15), t1 + pad_end


def _cache_path(
    job_id: str,
    id_tracking: int,
    segment_index: int,
    *,
    mode: str,
) -> Path:
    s.IVM_EXPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{job_id}:{id_tracking}:{segment_index}:{mode}:hybridv4"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return s.IVM_EXPORT_CACHE_DIR / f"track_seg_{digest}.mp4"


def _track_timeline(seg_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        seg_rows,
        key=lambda r: float(r.get("time_analyze_s") or r.get("time_analyze") or 0),
    )


def _timeline_has_frame_index(timeline: List[Dict[str, Any]]) -> bool:
    return any(int(r.get("frame_index") or -1) >= 0 for r in timeline)


def _track_timeline_for_overlay(seg_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = _track_timeline(seg_rows)
    if _timeline_has_frame_index(rows):
        return sorted(rows, key=lambda r: int(r.get("frame_index") or 0))
    return rows


def _boxes_from_row(row: Dict[str, Any]) -> Tuple[
    Optional[List[int]], Optional[List[int]], List[Dict[str, Any]]
]:
    return (
        parse_box(row.get("box_person")),
        parse_box(row.get("box_face")),
        parse_weapon_boxes(row.get("weapon_boxes_json")),
    )


def _frame_index_to_row(timeline: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for r in timeline:
        fi = int(r.get("frame_index") or -1)
        if fi >= 0:
            out[fi] = r
    return out


def row_for_video_frame_index(
    timeline: List[Dict[str, Any]],
    frame_index: int,
    *,
    fi_map: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Box ổn định: cập nhật tại đúng frame_index mẫu, giữa hai mẫu dùng mẫu trước (không tắt/bật)."""
    if not timeline:
        raise ValueError("timeline rỗng")
    fi_map = fi_map or _frame_index_to_row(timeline)
    fi = int(frame_index)
    if fi in fi_map:
        return fi_map[fi]
    if _timeline_has_frame_index(timeline):
        best: Optional[Dict[str, Any]] = None
        for r in timeline:
            rfi = int(r.get("frame_index") or -1)
            if rfi < 0:
                continue
            if rfi <= fi:
                best = r
            elif best is not None:
                break
        if best is not None:
            return best
    return timeline[0]


def boxes_at_playhead(
    timeline: List[Dict[str, Any]],
    *,
    frame_index: Optional[int] = None,
    time_s: Optional[float] = None,
    fi_map: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Tuple[Optional[List[int]], Optional[List[int]], List[Dict[str, Any]]]:
    """Ưu tiên khớp đúng frame_index mẫu; giữa hai mẫu dùng box mẫu trước đó."""
    if not timeline:
        return None, None, []
    fi_map = fi_map or _frame_index_to_row(timeline)
    if frame_index is not None:
        fi = int(frame_index)
        if fi in fi_map:
            return _boxes_from_row(fi_map[fi])
        if _timeline_has_frame_index(timeline):
            best: Optional[Dict[str, Any]] = None
            for r in timeline:
                rfi = int(r.get("frame_index") or -1)
                if rfi < 0:
                    continue
                if rfi <= fi:
                    best = r
                elif best is not None:
                    break
            if best is not None:
                return _boxes_from_row(best)
    if time_s is not None:
        best = None
        for r in timeline:
            t = float(r.get("time_analyze_s") or r.get("time_analyze") or 0)
            if t <= float(time_s):
                best = r
        if best is not None:
            return _boxes_from_row(best)
    return _boxes_from_row(timeline[0])


def _box_scale_xy(
    box: Optional[List[int]],
    sx: float,
    sy: float,
) -> Optional[List[int]]:
    if box is None or (abs(sx - 1.0) < 1e-4 and abs(sy - 1.0) < 1e-4):
        return box
    return [
        int(round(box[0] * sx)),
        int(round(box[1] * sy)),
        int(round(box[2] * sx)),
        int(round(box[3] * sy)),
    ]


def _scale_weapon_boxes(
    weapons: List[Dict[str, Any]],
    sx: float,
    sy: float,
) -> List[Dict[str, Any]]:
    if abs(sx - 1.0) < 1e-4 and abs(sy - 1.0) < 1e-4:
        return weapons
    out: List[Dict[str, Any]] = []
    for w in weapons:
        bb = parse_box(w.get("bbox"))
        if bb is None:
            continue
        scaled = _box_scale_xy(bb, sx, sy)
        if scaled is None:
            continue
        out.append({**w, "bbox": scaled})
    return out


def _overlay_box_scale(timeline: List[Dict[str, Any]], vid_w: int, vid_h: int) -> Tuple[float, float]:
    """Box lưu theo ảnh root_imgs; scale nếu video cắt khác kích thước."""
    for r in timeline:
        fr = load_frame_bgr(r.get("img_url"))
        if fr is None:
            continue
        ih, iw = fr.shape[:2]
        if iw > 0 and ih > 0 and (iw != vid_w or ih != vid_h):
            return vid_w / float(iw), vid_h / float(ih)
        break
    return 1.0, 1.0


def overlay_track_boxes_on_video(
    cut_path: Path,
    seg_rows: List[Dict[str, Any]],
    *,
    cut_start_s: float,
    out_path: Path,
    source_fps: float,
) -> Path:
    """Vẽ box lên mọi khung video cắt (full FPS) — box giữ ổn định giữa các mẫu."""
    timeline = _track_timeline_for_overlay(seg_rows)
    fi_map = _frame_index_to_row(timeline)
    cap = cv2.VideoCapture(str(cut_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {cut_path}")

    cap_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if cap_fps <= 0:
        cap_fps = 25.0
    src_fps = max(1.0, float(source_fps or cap_fps))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("Video cắt không có kích thước hợp lệ")

    first = timeline[0]
    fi_anchor = int(first.get("frame_index") or round(
        float(first.get("time_analyze_s") or cut_start_s) * src_fps
    ))
    sx, sy = _overlay_box_scale(timeline, w, h)

    tmp = out_path.with_suffix(f".ov_{uuid.uuid4().hex[:8]}.mp4")
    writer = cv2.VideoWriter(
        str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), cap_fps, (w, h)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Không mở được VideoWriter")

    written = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fi_now = fi_anchor + int(round(written * src_fps / cap_fps))
            row = row_for_video_frame_index(timeline, fi_now, fi_map=fi_map)
            bp, bf, weapons = _boxes_from_row(row)
            bp = _box_scale_xy(bp, sx, sy)
            bf = _box_scale_xy(bf, sx, sy)
            weapons = _scale_weapon_boxes(weapons, sx, sy)
            if bp is not None or bf is not None or weapons:
                frame = draw_boxes_on_frame(
                    frame, box_person=bp, box_face=bf, weapon_boxes=weapons
                )
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()

    if written == 0 or not tmp.is_file() or tmp.stat().st_size == 0:
        raise RuntimeError("Không ghi được khung video có box")
    if out_path.is_file():
        out_path.unlink(missing_ok=True)
    tmp.replace(out_path)
    return out_path


def _check_cached_playable(out_path: Path, *, force_rebuild: bool) -> Optional[Path]:
    web_path = browser_mp4_path(out_path)
    if force_rebuild:
        for p in (web_path, out_path):
            if p.is_file():
                try:
                    p.unlink()
                except OSError:
                    pass
        return None
    if web_path.is_file() and web_path.stat().st_size > 0:
        return web_path
    if out_path.is_file() and out_path.stat().st_size > 0:
        try:
            return resolve_browser_visual_mp4(out_path)
        except Exception:
            return out_path
    return None


def _build_track_segment_video_from_source(
    job_id: str,
    id_tracking: int,
    segment_index: int,
    seg_rows: List[Dict[str, Any]],
    *,
    draw_boxes: bool = True,
    force_rebuild: bool = False,
) -> Path:
    """Cắt video gốc (ffmpeg) + tùy chọn vẽ box theo frame_index mẫu."""
    src = resolve_job_source_video(job_id)
    if src is None:
        raise FileNotFoundError("Không tìm thấy file video gốc của job")

    mode = "hybrid" if draw_boxes else "src"
    out_path = _cache_path(job_id, id_tracking, segment_index, mode=mode)
    cached = _check_cached_playable(out_path, force_rebuild=force_rebuild)
    if cached is not None:
        return cached

    lock_key = f"{mode}:{job_id}:{id_tracking}:{segment_index}"
    lock = _build_locks.setdefault(lock_key, threading.Lock())
    with lock:
        cached = _check_cached_playable(out_path, force_rebuild=False)
        if cached is not None:
            return cached

        t0, t1 = segment_time_bounds(
            seg_rows, align_first_sample=bool(draw_boxes)
        )
        job = get_video_analyze_store().get_job(job_id) or {}
        try:
            dur = float(job.get("duration_s") or 0)
            if dur > 0:
                t1 = min(t1, dur)
        except (TypeError, ValueError):
            pass
        if t1 <= t0:
            t1 = t0 + 0.5

        cut_tmp = out_path.with_suffix(f".cut_{uuid.uuid4().hex[:8]}.mp4")
        try:
            job_fps = float(job.get("fps") or 0)
            raw_cut = export_segment_cut(
                src_path=str(src),
                offset_start_s=t0,
                offset_end_s=t1,
                out_path=cut_tmp,
                accurate_seek=bool(draw_boxes),
            )
            if raw_cut.resolve() != cut_tmp.resolve():
                if cut_tmp.is_file():
                    cut_tmp.unlink(missing_ok=True)
                raw_cut.replace(cut_tmp)

            if draw_boxes:
                overlay_track_boxes_on_video(
                    cut_tmp,
                    seg_rows,
                    cut_start_s=t0,
                    out_path=out_path,
                    source_fps=job_fps if job_fps > 0 else 25.0,
                )
            elif cut_tmp.resolve() != out_path.resolve():
                if out_path.is_file():
                    out_path.unlink(missing_ok=True)
                cut_tmp.replace(out_path)
        finally:
            if cut_tmp.is_file() and cut_tmp.resolve() != out_path.resolve():
                try:
                    cut_tmp.unlink()
                except OSError:
                    pass

        return resolve_browser_visual_mp4(out_path)


def _build_track_segment_video_from_frames(
    job_id: str,
    id_tracking: int,
    segment_index: int,
    frames: List[Dict[str, Any]],
    *,
    draw_boxes: bool,
    fps: Optional[float],
    force_rebuild: bool,
) -> Path:
    """Ghép khung mẫu đã lưu — mỗi ảnh vẽ box đúng report (giống slider Khung trong đoạn)."""
    out_path = _cache_path(
        job_id, id_tracking, segment_index, mode=f"frames_{int(draw_boxes)}"
    )
    cached = _check_cached_playable(out_path, force_rebuild=force_rebuild)
    if cached is not None:
        return cached

    lock_key = f"frames:{job_id}:{id_tracking}:{segment_index}:{int(draw_boxes)}"
    lock = _build_locks.setdefault(lock_key, threading.Lock())
    with lock:
        cached = _check_cached_playable(out_path, force_rebuild=False)
        if cached is not None:
            return cached

        job = get_video_analyze_store().get_job(job_id) or {}
        encode_fps = _encode_fps_for_job(job, fps)
        times_s = [float(fr["time_analyze_s"]) for fr in frames]
        holds = frame_hold_counts(times_s, encode_fps)

        first = load_frame_bgr(frames[0]["img_url"])
        if first is None:
            raise ValueError("Không đọc được khung đầu tiên")
        h, w = first.shape[:2]
        web_path = browser_mp4_path(out_path)
        tmp = out_path.with_suffix(f".{uuid.uuid4().hex[:8]}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp), fourcc, encode_fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Không mở được VideoWriter MP4")

        try:
            for fr, hold in zip(frames, holds):
                frame = load_frame_bgr(fr["img_url"])
                if frame is None:
                    continue
                if frame.shape[0] != h or frame.shape[1] != w:
                    frame = cv2.resize(frame, (w, h))
                if draw_boxes:
                    track_row = next(
                        (
                            r
                            for r in fr["rows"]
                            if int(r.get("id_tracking") or 0) == int(id_tracking)
                        ),
                        None,
                    )
                    if track_row is not None:
                        frame = draw_boxes_on_frame(
                            frame,
                            box_person=track_row.get("box_person"),
                            box_face=track_row.get("box_face"),
                            weapon_boxes=track_row.get("weapon_boxes_json"),
                        )
                for _ in range(max(1, int(hold))):
                    writer.write(frame)
        finally:
            writer.release()

        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError("Xuất video track thất bại (file rỗng)")
        for p in (out_path, web_path):
            if p.is_file():
                try:
                    p.unlink()
                except OSError:
                    pass
        tmp.replace(out_path)
        try:
            return resolve_browser_visual_mp4(out_path)
        except Exception:
            return out_path


def build_track_segment_video(
    job_id: str,
    id_tracking: int,
    *,
    segment_index: int = 0,
    draw_boxes: bool = True,
    fps: Optional[float] = None,
    force_rebuild: bool = False,
    prefer_source: bool = True,
) -> Path:
    """
    draw_boxes=True: video gốc full FPS + box theo frame_index mẫu; fallback ghép img_url.
    draw_boxes=False: chỉ cắt video gốc.
    """
    seg_rows, frames = _frames_for_segment(
        job_id, id_tracking, segment_index=int(segment_index)
    )
    if not seg_rows:
        raise ValueError("Không có khung cho đoạn track này")

    if draw_boxes and prefer_source and resolve_job_source_video(job_id) is not None:
        try:
            return _build_track_segment_video_from_source(
                job_id,
                id_tracking,
                int(segment_index),
                seg_rows,
                draw_boxes=True,
                force_rebuild=force_rebuild,
            )
        except Exception:
            pass

    if draw_boxes and frames:
        return _build_track_segment_video_from_frames(
            job_id,
            id_tracking,
            int(segment_index),
            frames,
            draw_boxes=True,
            fps=fps,
            force_rebuild=force_rebuild,
        )

    if prefer_source and resolve_job_source_video(job_id) is not None:
        try:
            return _build_track_segment_video_from_source(
                job_id,
                id_tracking,
                int(segment_index),
                seg_rows,
                draw_boxes=draw_boxes,
                force_rebuild=force_rebuild,
            )
        except Exception:
            pass

    if not frames:
        raise ValueError("Không có khung cho đoạn track này")
    return _build_track_segment_video_from_frames(
        job_id,
        id_tracking,
        int(segment_index),
        frames,
        draw_boxes=draw_boxes,
        fps=fps,
        force_rebuild=force_rebuild,
    )
