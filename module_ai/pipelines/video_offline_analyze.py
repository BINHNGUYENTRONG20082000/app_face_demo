"""Job phân tích video offline: lấy mẫu khung → DB + ảnh từng detection (giống VideoMaster)."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from module_ai.config import settings as s
from module_ai.engine.insightface_engine import InsightFaceEngine
from module_ai.pipelines.video_person_face_pipeline import (
    run_person_track_weapon_face_pipeline,
)
from identity_vm_app.services import video_analyze_media as vmedia
from identity_vm_app.services.video_analyze_fps import frame_skip_for_sample
from identity_vm_app.services.camera_report_writer import FRAME_MARKER_TRACKING_ID
from identity_vm_app.services.video_match_candidates import serialize_match_candidates
from identity_vm_app.services.video_analyze_split import (
    VideoSegmentSpec,
    remove_split_dir,
    split_video_for_job,
)
from identity_vm_app.services.video_analyze_tracking import VideoClipCounter
from identity_vm_app.store.video_analyze_store import (
    VA_STATUS_COMPLETED,
    VA_STATUS_ERROR,
    VA_STATUS_PROCESSING,
    get_video_analyze_store,
)

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}

_job_slot = threading.Semaphore(max(1, int(s.IVM_VIDEO_ANALYZE_MAX_CONCURRENT)))


def upload_suffix_from_name(name: str) -> str:
    p = Path(name)
    suf = p.suffix.lower()
    if suf in s.IVM_VIDEO_ANALYZE_ALLOWED_SUFFIXES:
        return suf
    return ""


def _sanitize_stem(name: str, max_len: int = 80) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-]+", "_", stem, flags=re.UNICODE).strip("_") or "video"
    return stem[:max_len]


def try_acquire_job_slot() -> bool:
    return bool(_job_slot.acquire(blocking=False))


def release_job_slot() -> None:
    try:
        _job_slot.release()
    except ValueError:
        pass


def _normalize_split_parts(n: Any) -> int:
    try:
        v = int(n)
    except (TypeError, ValueError):
        v = int(s.IVM_VIDEO_ANALYZE_SPLIT_PARTS)
    return max(1, min(4, v))


def _feature_analyze_dict(*, split_parts: int) -> Dict[str, Any]:
    sp = _normalize_split_parts(split_parts)
    return {
        "detect_person": True,
        "analyze_face": True,
        "pipeline": "yolo_person_track_then_face_recognition",
        "split_parts": sp,
    }


def _split_parts_for_job(job_id: str) -> int:
    row = get_video_analyze_store().get_job(job_id) or {}
    fa = row.get("feature_analyze") or {}
    if isinstance(fa, dict) and fa.get("split_parts") is not None:
        return _normalize_split_parts(fa.get("split_parts"))
    with _jobs_lock:
        mem = _jobs.get(job_id) or {}
    if mem.get("split_parts") is not None:
        return _normalize_split_parts(mem.get("split_parts"))
    return _normalize_split_parts(s.IVM_VIDEO_ANALYZE_SPLIT_PARTS)


def register_job(
    job_id: str,
    path: Path,
    *,
    original_name: str,
    display_name: str,
    sample_fps: float,
    split_parts: Optional[int] = None,
) -> None:
    sp = _normalize_split_parts(
        split_parts if split_parts is not None else s.IVM_VIDEO_ANALYZE_SPLIT_PARTS
    )
    get_video_analyze_store().insert_job(
        job_id,
        original_name=original_name,
        display_name=display_name,
        video_path=str(path),
        thumb_path=None,
        feature_analyze=_feature_analyze_dict(split_parts=sp),
        sample_fps=float(sample_fps),
    )
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": {"current": 0, "total": 0},
            "message": "",
            "result": None,
            "error": None,
            "path": path,
            "original_name": original_name,
            "display_name": display_name,
            "split_parts": sp,
            "created_utc": time.time(),
        }


def job_is_active(job_id: str) -> bool:
    """
    Chỉ coi job đang chạy khi còn bản ghi in-memory và status queued/running.
    DB pending/processing nhưng không còn worker (restart API / job treo) → cho phép xóa.
    """
    db = get_video_analyze_store().get_job(job_id)
    if db is None:
        with _jobs_lock:
            _jobs.pop(job_id, None)
        return False
    st = int(db.get("status") or -1)
    if st in (VA_STATUS_COMPLETED, VA_STATUS_ERROR):
        with _jobs_lock:
            _jobs.pop(job_id, None)
        return False
    with _jobs_lock:
        mem = _jobs.get(job_id)
    if not mem:
        return False
    return str(mem.get("status") or "") in ("queued", "running")


def new_job_id() -> str:
    return str(uuid.uuid4())


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    db = get_video_analyze_store().get_job(job_id)
    with _jobs_lock:
        mem = _jobs.get(job_id)
    if mem is None and db is None:
        return None
    out: Dict[str, Any] = {}
    if mem:
        out.update(_public_job_view(dict(mem)))
    if db:
        st_name = db.get("status_name") or "unknown"
        if not mem:
            out["status"] = st_name
        counts = get_video_analyze_store().count_reports(job_id)
        total_s = int(db.get("total_sample_frames") or 0)
        idx = int(db.get("index_frame") or 0)
        out.setdefault("progress", {"current": idx, "total": total_s})
        out.setdefault("message", db.get("message") or "")
        out.setdefault("original_name", db.get("original_name"))
        out.setdefault("display_name", db.get("display_name"))
        out["db"] = {
            "duration_s": db.get("duration_s"),
            "person_reports": counts["person_reports"],
        }
        if st_name == "done" and out.get("result") is None:
            out["result"] = _summary_from_db(job_id, db, counts)
    return out


def _summary_from_db(job_id: str, db: Dict[str, Any], counts: Dict[str, int]) -> Dict[str, Any]:
    return {
        "video": {
            "job_id": job_id,
            "duration_s": db.get("duration_s"),
            "fps": db.get("fps"),
            "sampled_frames": db.get("total_sample_frames"),
        },
        "person": {"total_rows": counts["person_reports"]},
        "persisted": True,
    }


def _public_job_view(j: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": j.get("status"),
        "progress": dict(j.get("progress") or {}),
        "message": j.get("message") or "",
        "result": j.get("result"),
        "error": j.get("error"),
        "original_name": j.get("original_name"),
        "display_name": j.get("display_name"),
    }


def rename_job(job_id: str, display_name: str) -> bool:
    name = (display_name or "").strip()
    if not name:
        return False
    ok = get_video_analyze_store().update_job_display_name(job_id, name)
    if not ok:
        return False
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["display_name"] = name
    return True


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def delete_job_reports(job_id: str) -> Dict[str, Any]:
    """Xóa báo cáo + ảnh analyze; giữ file video gốc."""
    if get_video_analyze_store().get_job(job_id) is None:
        return {"ok": False, "reason": "not_found"}
    if job_is_active(job_id):
        return {"ok": False, "reason": "job_running"}
    n = get_video_analyze_store().clear_job_reports(job_id)
    files = vmedia.clear_analyze_artifacts(job_id)
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["progress"] = {"current": 0, "total": 0}
            _jobs[job_id]["result"] = None
    return {"ok": True, "reports_deleted": n, "image_files_deleted": files}


def delete_job(job_id: str) -> bool:
    if job_is_active(job_id):
        return False
    with _jobs_lock:
        j = _jobs.pop(job_id, None)
    db = get_video_analyze_store().get_job(job_id)
    if j is None and db is None:
        return False
    get_video_analyze_store().delete_job_data(job_id)
    remove_split_dir(job_id)
    job_dir = Path(s.IVM_VIDEO_ANALYZE_DIR) / "jobs" / job_id
    if job_dir.is_dir():
        try:
            shutil.rmtree(job_dir, ignore_errors=True)
        except OSError:
            pass
    if j:
        p = j.get("path")
        if isinstance(p, Path) and p.is_file():
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
    return True


def _tracking_id_for_session(local_id: int, cut_session: int) -> int:
    return int(local_id) + int(cut_session) * 1_000_000_000


def _video_clip_base(cut_session: int) -> int:
    return 1 + int(cut_session) * 1_000_000


class _JobProgress:
    def __init__(self, job_id: str, total: int) -> None:
        self._job_id = job_id
        self.total = max(1, int(total))
        self._lock = threading.Lock()
        self._done = 0
        self._every = max(1, int(s.IVM_VIDEO_ANALYZE_PROGRESS_EVERY))

    def bump(self) -> int:
        with self._lock:
            self._done += 1
            cur = self._done
        if cur >= self.total or cur % self._every == 0:
            get_video_analyze_store().update_job_progress(self._job_id, cur)
            _update_job(self._job_id, progress={"current": cur, "total": self.total})
        return cur


def _features_face_str(embedding: Any) -> Optional[str]:
    if embedding is None:
        return None
    try:
        return np.array_str(np.asarray(embedding, dtype=np.float32).reshape(-1))
    except Exception:
        return None


def _person_report_row(
    job_id: str,
    *,
    t_s: float,
    frame_index: int,
    sample_index: int,
    video_clip: int,
    img_url: str,
    id_tracking: int,
    box_person: str,
    face: Optional[Dict[str, Any]] = None,
    weapon: Optional[Dict[str, Any]] = None,
    persist_embeddings: bool = False,
    save_crops: bool = False,
    frame: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Một dòng báo cáo: box_person từ YOLO; nhánh mặt / vũ khí tùy chọn."""
    bb: List[float] = []
    face_img = None
    face_id = None
    display_name = None
    distance = None
    match_score = None
    match_candidates_json = None
    det_score = None
    gender = None
    age = None
    features_face = None
    box_face = None

    person_img = None
    pb_list = parse_box_literal(box_person)

    if face is not None:
        bb = list((face.get("bbox") or [])[:4])
        if len(bb) >= 4 and save_crops and frame is not None:
            face_img = vmedia.save_face_crop(job_id, frame, bb)
        ms = face.get("matches") or []
        if ms:
            m0 = ms[0]
            face_id = m0.get("face_id")
            display_name = m0.get("name") or m0.get("display_name")
            distance = m0.get("distance")
            if distance is not None:
                match_score = 1.0 - float(distance)
        match_candidates_json = serialize_match_candidates(ms, limit=int(s.IVM_SEARCH_K))
        det_score = face.get("det_score")
        gender = face.get("gender")
        age = face.get("age")
        if persist_embeddings:
            features_face = _features_face_str(face.get("embedding"))
        box_face = f"{bb}" if len(bb) >= 4 else None

    armed = 0
    weapon_status = None
    weapon_label = None
    weapon_types_json = None
    weapon_score = None
    weapon_img = None
    weapon_crops_json = None
    weapon_boxes_json = None
    if weapon:
        frame_armed = bool(weapon.get("frame_armed")) or bool(weapon.get("weapons"))
        armed = 1 if frame_armed else 0
        weapon_status = weapon.get("weapon_status")
        weapon_label = weapon.get("weapon_label")
        types = weapon.get("weapon_types") or []
        if types:
            weapon_types_json = json.dumps(list(types), ensure_ascii=False)
        weapon_score = weapon.get("weapon_score")
        weapon_boxes_json = _weapon_boxes_json(weapon)
        if save_crops and frame is not None and pb_list and len(pb_list) >= 4:
            person_img = vmedia.save_person_crop(job_id, frame, pb_list)
            if frame_armed:
                weapon_img, weapon_crops_json = vmedia.save_weapon_bbox_crops(
                    job_id,
                    frame,
                    list(weapon.get("weapons") or []),
                )

    return {
        "job_id": job_id,
        "time_analyze_s": float(t_s),
        "frame_index": frame_index,
        "sample_index": sample_index,
        "img_url": img_url,
        "id_tracking": id_tracking,
        "video_clip": video_clip,
        "face_id": face_id,
        "display_name": display_name,
        "distance": distance,
        "match_score": match_score,
        "match_candidates_json": match_candidates_json,
        "det_score": det_score,
        "gender": gender,
        "age": age,
        "box_face": box_face,
        "face_img": face_img,
        "box_person": box_person,
        "person_img": person_img,
        "features_face": features_face,
        "armed": armed,
        "weapon_status": weapon_status,
        "weapon_label": weapon_label,
        "weapon_types_json": weapon_types_json,
        "weapon_score": weapon_score,
        "weapon_img": weapon_img,
        "weapon_crops_json": weapon_crops_json,
        "weapon_boxes_json": weapon_boxes_json,
    }


def _weapon_boxes_json(weapon: Optional[Dict[str, Any]]) -> Optional[str]:
    if not weapon:
        return None
    items = weapon.get("weapons") or []
    if not items:
        return None
    payload = []
    for w in items:
        bb = w.get("bbox")
        if not bb or len(bb) < 4:
            continue
        payload.append(
            {
                "bbox": [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
                "class": str(w.get("class") or "weapon"),
                "conf": float(w.get("conf") or w.get("fusion_score") or 0.0),
            }
        )
    return json.dumps(payload, ensure_ascii=False) if payload else None


def parse_box_literal(box_raw: Any) -> Optional[List[int]]:
    if box_raw is None:
        return None
    if isinstance(box_raw, (list, tuple)) and len(box_raw) >= 4:
        return [int(box_raw[0]), int(box_raw[1]), int(box_raw[2]), int(box_raw[3])]
    s = str(box_raw).strip()
    if not s:
        return None
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)) and len(val) >= 4:
            return [int(val[0]), int(val[1]), int(val[2]), int(val[3])]
    except (SyntaxError, ValueError, TypeError):
        pass
    return None


def _build_person_report_rows_from_assignments(
    job_id: str,
    frame: np.ndarray,
    *,
    t_s: float,
    frame_index: int,
    sample_index: int,
    video_clip: int,
    assignments: List[Dict[str, Any]],
    faces_with_matches: List[Dict[str, Any]],
    cut_session: int,
    img_url: Optional[str],
    persist_embeddings: bool,
    save_crops: bool,
    log_all_infer_frames: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Ghi báo cáo DB cho mỗi frame infer (tùy IVM_REPORT_ALL_INFER_FRAMES)."""
    log_all = (
        bool(s.IVM_REPORT_ALL_INFER_FRAMES)
        if log_all_infer_frames is None
        else bool(log_all_infer_frames)
    )
    if not assignments:
        if not log_all:
            return []
        if img_url is None:
            img_url = vmedia.save_root_frame(job_id, frame)
        return [
            _person_report_row(
                job_id,
                t_s=t_s,
                frame_index=frame_index,
                sample_index=sample_index,
                video_clip=video_clip,
                img_url=img_url,
                id_tracking=FRAME_MARKER_TRACKING_ID,
                box_person="[]",
                frame=frame,
            )
        ]
    if img_url is None:
        img_url = vmedia.save_root_frame(job_id, frame)
    rows: List[Dict[str, Any]] = []
    fi = 0
    for asn in assignments:
        pt = asn.get("person_track") or []
        if len(pt) < 5:
            continue
        pb = list(pt[:4])
        local_tid = int(pt[4])
        tid = _tracking_id_for_session(local_tid, cut_session)
        weapon = asn.get("weapon")
        face_entry = None
        if asn.get("has_face"):
            if fi < len(faces_with_matches):
                face_entry = faces_with_matches[fi]
            fi += 1
            if not log_all:
                if not face_entry:
                    continue
        elif not log_all and not (weapon or {}).get("armed"):
            continue
        rows.append(
            _person_report_row(
                job_id,
                t_s=t_s,
                frame_index=frame_index,
                sample_index=sample_index,
                video_clip=video_clip,
                img_url=img_url,
                id_tracking=tid,
                box_person=f"{pb}",
                face=face_entry,
                weapon=weapon,
                persist_embeddings=persist_embeddings,
                save_crops=save_crops,
                frame=frame,
            )
        )
    return rows


def _downsample_timeline(rows: List[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
    if len(rows) <= cap:
        return rows
    step = max(1, len(rows) // cap)
    return rows[::step][:cap]


def _count_samples(total_frames: int, skip: int) -> int:
    if total_frames <= 0:
        return 0
    return len(list(range(0, total_frames, max(1, skip))))


def _analyze_segment(
    *,
    job_id: str,
    segment: VideoSegmentSpec,
    engine: InsightFaceEngine,
    face_db: Any,
    thr: float,
    max_w: int,
    fps: float,
    skip: int,
    progress: _JobProgress,
    sample_index_base: int = 0,
) -> Dict[str, Any]:
    """Phân tích một đoạn: YOLO person+track → detect mặt → embed/recognition theo lô."""
    cut_session = int(segment.cut_session)
    path = segment.path
    t_offset = float(segment.start_time_s)
    clip_counter = VideoClipCounter(max_miss=10, initial_clip=_video_clip_base(cut_session))
    vstore = get_video_analyze_store()
    persist_emb = bool(s.IVM_VIDEO_ANALYZE_PERSIST_EMBEDDINGS)
    save_crops = bool(s.IVM_VIDEO_ANALYZE_SAVE_CROPS)

    from module_ai.engine.yolo_person_tracker import (
        create_person_tracker,
        tracker_unavailable_reason,
        vm_tracking_available,
    )

    if not vm_tracking_available():
        reason = tracker_unavailable_reason() or "YOLO person tracker không khả dụng"
        raise RuntimeError(
            f"{reason}. Cài ultralytics và đặt model (IVM_VIDEO_ANALYZE_YOLO_MODEL)."
        )
    yolo_tracker = create_person_tracker()

    emb_batch_sz = max(1, int(s.IVM_VIDEO_ANALYZE_EMBED_BATCH))
    db_batch_sz = max(1, int(s.IVM_VIDEO_ANALYZE_DB_BATCH))

    name_counter: Counter[str] = Counter()
    frames_with_any_person = 0
    frames_with_any_face = 0
    unknown_frame_hits = 0
    max_faces = 0
    max_persons = 0
    person_timeline: List[Dict[str, Any]] = []
    sum_infer = 0.0

    crop_buf: List[np.ndarray] = []
    frame_pack_buf: List[Dict[str, Any]] = []
    db_rows: List[Dict[str, Any]] = []

    def _flush_db() -> None:
        if not db_rows:
            return
        vstore.insert_person_reports_batch(db_rows)
        db_rows.clear()

    def _flush_embed() -> None:
        nonlocal sum_infer, frames_with_any_face, max_faces, unknown_frame_hits
        if not frame_pack_buf:
            return
        all_matches: List[List[Dict[str, Any]]] = []
        if crop_buf:
            t0 = time.perf_counter()
            feats, _emb_ms = engine.embed_aligned_crops(crop_buf, max_batch=emb_batch_sz)
            sum_infer += (time.perf_counter() - t0) * 1000.0
            mat = np.stack(
                [np.asarray(feats[j], dtype=np.float32).reshape(-1) for j in range(feats.shape[0])],
                axis=0,
            )
            all_matches = face_db.search_batch(mat, k=s.IVM_SEARCH_K, distance_threshold=thr)

        embed_i = 0
        for pack in frame_pack_buf:
            assignments = pack["assignments"]
            faces_out: List[Dict[str, Any]] = []
            for asn in assignments:
                if not asn.get("has_face"):
                    faces_out.append({})
                    continue
                fm = asn.get("face_meta") or {}
                face_entry: Dict[str, Any] = {
                    "bbox": fm.get("bbox"),
                    "det_score": fm.get("det_score"),
                    "matches": all_matches[embed_i],
                }
                if persist_emb:
                    face_entry["embedding"] = feats[embed_i]
                faces_out.append(face_entry)
                embed_i += 1

            nf = sum(1 for a in assignments if a.get("has_face"))
            if nf > 0:
                frames_with_any_face += 1
                max_faces = max(max_faces, nf)
            names_row: List[str] = []
            has_unknown = False
            for f in faces_out:
                if not f:
                    names_row.append("(no face)")
                    continue
                ms = f.get("matches") or []
                label = "unknown"
                if ms:
                    m0 = ms[0]
                    label = str(m0.get("name") or m0.get("display_name") or "?")
                names_row.append(label)
                name_counter[label] += 1
                if label in ("unknown", "?"):
                    has_unknown = True
            if has_unknown and nf > 0:
                unknown_frame_hits += 1
            person_timeline.append(
                {
                    "t_s": round(float(pack["t_s"]), 3),
                    "n_persons": len(assignments),
                    "n_faces": nf,
                    "names_sample": ", ".join(names_row[:6])
                    + ("…" if len(names_row) > 6 else ""),
                }
            )
            face_rows_for_report = [
                faces_out[i] for i, a in enumerate(assignments) if a.get("has_face")
            ]
            if bool(s.IVM_REPORT_ALL_INFER_FRAMES):
                db_rows.extend(
                    _build_person_report_rows_from_assignments(
                        job_id,
                        pack["frame"],
                        t_s=float(pack["t_s"]),
                        frame_index=int(pack["frame_index"]),
                        sample_index=int(pack["sample_index"]),
                        video_clip=int(pack["video_clip"]),
                        assignments=assignments,
                        faces_with_matches=face_rows_for_report,
                        cut_session=cut_session,
                        img_url=None,
                        persist_embeddings=persist_emb,
                        save_crops=save_crops,
                    )
                )
            else:
                has_report = any(
                    a.get("has_face") or (a.get("weapon") or {}).get("armed")
                    for a in assignments
                )
                face_asn = [a for a in assignments if a.get("has_face")]
                face_rows_out = [f for f in face_rows_for_report if f]
                if has_report and (
                    face_asn
                    or any((a.get("weapon") or {}).get("armed") for a in assignments)
                ):
                    db_rows.extend(
                        _build_person_report_rows_from_assignments(
                            job_id,
                            pack["frame"],
                            t_s=float(pack["t_s"]),
                            frame_index=int(pack["frame_index"]),
                            sample_index=int(pack["sample_index"]),
                            video_clip=int(pack["video_clip"]),
                            assignments=face_asn,
                            faces_with_matches=face_rows_out,
                            cut_session=cut_session,
                            img_url=None,
                            persist_embeddings=persist_emb,
                            save_crops=save_crops,
                            log_all_infer_frames=False,
                        )
                    )

        crop_buf.clear()
        frame_pack_buf.clear()
        if len(db_rows) >= db_batch_sz:
            _flush_db()

    cap_v = cv2.VideoCapture(str(path))
    if not cap_v.isOpened():
        raise RuntimeError(f"Không mở được đoạn video: {path}")

    try:
        seg_frames = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if seg_frames <= 0:
            return {
                "name_counter": name_counter,
                "frames_with_any_person": 0,
                "frames_with_any_face": 0,
                "unknown_frame_hits": 0,
                "max_faces": 0,
                "max_persons": 0,
                "person_timeline": [],
                "sum_infer": 0.0,
            }
        indices = list(range(0, seg_frames, max(1, skip)))
        frame_i = 0
        global_frame_base = int(round(t_offset * fps))

        for si, target_fi in enumerate(indices):
            while frame_i < target_fi:
                if not cap_v.grab():
                    frame_i = seg_frames
                    break
                frame_i += 1
            if frame_i >= seg_frames:
                break

            ret, frame = cap_v.read()
            if not ret:
                break
            frame_i += 1

            t_s = t_offset + target_fi / fps
            g_frame_index = global_frame_base + target_fi
            sample_idx = sample_index_base + si
            vclip = clip_counter.clip

            person_tracks, _weapon_by_track, assignments, detect_ms, _ptiming = (
                run_person_track_weapon_face_pipeline(
                    frame,
                    person_tracker=yolo_tracker,
                    engine=engine,
                    camera_id=f"video:{job_id}",
                    max_w=max_w,
                    run_weapon=True,
                )
            )
            sum_infer += float(detect_ms) + float(_ptiming.get("weapon_ms", 0.0))

            if not person_tracks:
                clip_counter.on_frame(False)
                progress.bump()
                continue

            frames_with_any_person += 1
            max_persons = max(max_persons, len(person_tracks))
            clip_counter.on_frame(True)

            for asn in assignments:
                if not asn.get("has_face"):
                    continue
                fm = asn.get("face_meta") or {}
                crop = fm.get("aligned_crop")
                if crop is None:
                    asn["has_face"] = False
                    continue
                crop_buf.append(crop)

            frame_pack_buf.append(
                {
                    "frame": frame,
                    "t_s": t_s,
                    "frame_index": g_frame_index,
                    "sample_index": sample_idx,
                    "video_clip": vclip,
                    "assignments": assignments,
                }
            )

            if len(crop_buf) >= emb_batch_sz:
                _flush_embed()

            progress.bump()

        _flush_embed()
        _flush_db()
    finally:
        cap_v.release()
        if yolo_tracker is not None:
            try:
                yolo_tracker.dispose()
            except Exception:
                pass

    return {
        "name_counter": name_counter,
        "frames_with_any_person": frames_with_any_person,
        "frames_with_any_face": frames_with_any_face,
        "unknown_frame_hits": unknown_frame_hits,
        "max_faces": max_faces,
        "max_persons": max_persons,
        "person_timeline": person_timeline,
        "sum_infer": sum_infer,
    }


def _run_job(
    job_id: str,
    *,
    sample_fps: float,
    distance_threshold: float,
) -> None:
    path: Optional[Path] = None
    try:
        with _jobs_lock:
            rec = _jobs.get(job_id)
            if not rec:
                return
            path = rec.get("path")
            if not isinstance(path, Path):
                _update_job(job_id, status="error", error="missing_path", message="Thiếu đường dẫn file")
                return

        vstore = get_video_analyze_store()
        vstore.update_job_status(job_id, VA_STATUS_PROCESSING, message="Đang phân tích…")
        t_job0 = time.time()

        cap_v = cv2.VideoCapture(str(path))
        try:
            if not cap_v.isOpened():
                get_video_analyze_store().update_job_status(
                    job_id, VA_STATUS_ERROR, message="Không mở được video", error_code="open_failed"
                )
                _update_job(job_id, status="error", error="open_failed", message="Không mở được video (OpenCV)")
                return

            fps = float(cap_v.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps < 1.0:
                fps = 25.0
            total_frames = int(cap_v.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w = int(cap_v.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap_v.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            if total_frames <= 0:
                get_video_analyze_store().update_job_status(
                    job_id,
                    VA_STATUS_ERROR,
                    message="Video không có FRAME_COUNT",
                    error_code="no_frames",
                )
                _update_job(
                    job_id,
                    status="error",
                    error="no_frames",
                    message="Video không có FRAME_COUNT — thử định dạng khác (mp4 H.264)",
                )
                return

            duration_s = total_frames / fps
            max_dur = float(s.IVM_VIDEO_ANALYZE_MAX_DURATION_S)
            if max_dur > 0 and duration_s > max_dur:
                get_video_analyze_store().update_job_status(
                    job_id,
                    VA_STATUS_ERROR,
                    message=f"Video quá dài ({duration_s:.0f}s)",
                    error_code="duration_exceeded",
                )
                _update_job(
                    job_id,
                    status="error",
                    error="duration_exceeded",
                    message=f"Video dài {duration_s:.0f}s vượt giới hạn {s.IVM_VIDEO_ANALYZE_MAX_DURATION_S:.0f}s",
                )
                return

            skip = frame_skip_for_sample(fps, sample_fps)
            indices: List[int] = list(range(0, total_frames, skip))
            total_samples = len(indices)
            if total_samples == 0:
                vstore.update_job_status(
                    job_id, VA_STATUS_ERROR, message="Không có khung mẫu", error_code="empty_sample"
                )
                _update_job(job_id, status="error", error="empty_sample", message="Không có khung mẫu")
                return

            analyze_fps_eff = float(sample_fps) if sample_fps > 0 else float(fps)
            _update_job(job_id, message=f"Chế độ: {'full frame' if sample_fps <= 0 else f'{int(sample_fps)} FPS'}")
            vstore.update_job_meta(
                job_id,
                duration_s=duration_s,
                fps=fps,
                width=w,
                height=h,
                total_frames=total_frames,
                analyze_fps=analyze_fps_eff,
                total_sample_frames=total_samples,
            )

            ret0, frame0 = cap_v.read()
            if ret0 and frame0 is not None:
                thumb_rel = vmedia.write_thumb(job_id, frame0)
                vstore.update_job_thumb(job_id, thumb_rel)
                cap_v.set(cv2.CAP_PROP_POS_FRAMES, 0)

            thr = float(distance_threshold)
            max_w = int(s.IVM_ANALYZE_MAX_FRAME_WIDTH)
            n_parts = _split_parts_for_job(job_id)

            from identity_vm_app.api.deps import get_face_db

            face_db = get_face_db()
            ctx_ids = s.ivm_bulk_worker_ctx_ids()

            if n_parts <= 1:
                _update_job(
                    job_id,
                    status="running",
                    message="Đang phân tích (1 luồng, không cắt file)…",
                    progress={"current": 0, "total": total_samples},
                )
                vstore.update_job_status(
                    job_id, VA_STATUS_PROCESSING, message="Đang phân tích (1 luồng)…"
                )
            else:
                _update_job(
                    job_id,
                    status="running",
                    message=f"Đang cắt video ({n_parts} luồng)…",
                    progress={"current": 0, "total": total_samples},
                )
                vstore.update_job_status(
                    job_id, VA_STATUS_PROCESSING, message=f"Đang cắt video ({n_parts} luồng)…"
                )

            segments = split_video_for_job(path, job_id, n_parts)
            split_used = len(segments) > 1 or (
                len(segments) == 1 and segments[0].path.resolve() != path.resolve()
            )

            seg_sample_totals: List[int] = []
            for seg in segments:
                cap_seg = cv2.VideoCapture(str(seg.path))
                seg_n = int(cap_seg.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if cap_seg.isOpened() else 0
                cap_seg.release()
                seg_sample_totals.append(_count_samples(seg_n, skip))
            progress_total = max(1, sum(seg_sample_totals))
            progress = _JobProgress(job_id, progress_total)
            vstore.update_job_total_sample_frames(job_id, progress_total)
            _update_job(
                job_id,
                status="running",
                message=f"Đang phân tích ({len(segments)} luồng)…",
                progress={"current": 0, "total": progress_total},
            )

            from module_ai.jobs.bulk_folder_register import _dispose_worker_engine, _release_bulk_infer_resources

            seg_errors: List[str] = []
            partials: List[Dict[str, Any]] = []
            partials_lock = threading.Lock()
            threads: List[threading.Thread] = []
            engines: List[InsightFaceEngine] = []

            sample_bases: List[int] = []
            acc = 0
            for n in seg_sample_totals:
                sample_bases.append(acc)
                acc += n

            def _segment_worker(seg: VideoSegmentSpec, eng: InsightFaceEngine, s_base: int) -> None:
                try:
                    part = _analyze_segment(
                        job_id=job_id,
                        segment=seg,
                        engine=eng,
                        face_db=face_db,
                        thr=thr,
                        max_w=max_w,
                        fps=fps,
                        skip=skip,
                        progress=progress,
                        sample_index_base=s_base,
                    )
                    with partials_lock:
                        partials.append(part)
                except Exception as ex:
                    seg_errors.append(f"cut_session={seg.cut_session}: {ex}")

            try:
                for i, seg in enumerate(segments):
                    eng = InsightFaceEngine(ctx_id=ctx_ids[i % len(ctx_ids)])
                    engines.append(eng)
                    t = threading.Thread(
                        target=_segment_worker,
                        args=(seg, eng, sample_bases[i]),
                        name=f"ivm-video-{job_id[:8]}-s{seg.cut_session}",
                        daemon=True,
                    )
                    threads.append(t)
                    t.start()

                for t in threads:
                    t.join()
            finally:
                for eng in engines:
                    _dispose_worker_engine(eng)
                engines.clear()
                _release_bulk_infer_resources()

            if seg_errors:
                raise RuntimeError("; ".join(seg_errors[:3]))

            name_counter: Counter[str] = Counter()
            frames_with_any_person = 0
            frames_with_any_face = 0
            unknown_frame_hits = 0
            max_faces = 0
            max_persons = 0
            person_timeline: List[Dict[str, Any]] = []
            sum_infer = 0.0
            for p in partials:
                name_counter.update(p.get("name_counter") or Counter())
                frames_with_any_person += int(p.get("frames_with_any_person") or 0)
                frames_with_any_face += int(p.get("frames_with_any_face") or 0)
                unknown_frame_hits += int(p.get("unknown_frame_hits") or 0)
                max_faces = max(max_faces, int(p.get("max_faces") or 0))
                max_persons = max(max_persons, int(p.get("max_persons") or 0))
                person_timeline.extend(p.get("person_timeline") or [])
                sum_infer += float(p.get("sum_infer") or 0.0)
            person_timeline.sort(key=lambda x: float(x.get("t_s") or 0.0))

            tmax = int(s.IVM_VIDEO_ANALYZE_TIMELINE_MAX)
            result: Dict[str, Any] = {
                "video": {
                    "path": str(path),
                    "fps": round(fps, 3),
                    "width": w,
                    "height": h,
                    "total_frames": total_frames,
                    "duration_s": round(duration_s, 3),
                    "sampled_frames": progress_total,
                    "sample_fps_requested": round(sample_fps, 3),
                    "frame_skip": skip,
                    "split_parts": len(segments),
                    "split_ffmpeg": split_used,
                },
                "options": {
                    "analyze_person": True,
                    "pipeline": "yolo_person_track_then_face_recognition",
                    "distance_threshold": thr,
                    "split_parts": len(segments),
                    "save_crops": bool(s.IVM_VIDEO_ANALYZE_SAVE_CROPS),
                    "yolo_tracking": True,
                    "persist_embeddings": bool(s.IVM_VIDEO_ANALYZE_PERSIST_EMBEDDINGS),
                },
                "timing": {
                    "total_infer_ms": round(sum_infer, 2),
                    "avg_infer_ms": round(sum_infer / max(1, progress_total), 3),
                },
            }

            uniq = {k: v for k, v in name_counter.items()}
            result["person"] = {
                "frames_with_any_person": frames_with_any_person,
                "frames_with_any_face": frames_with_any_face,
                "max_persons_per_frame": max_persons,
                "max_faces_per_frame": max_faces,
                "unique_display_names": uniq,
                "unknown_frame_hits": unknown_frame_hits,
                "timeline": _downsample_timeline(person_timeline, tmax),
            }

            counts = vstore.count_reports(job_id)
            result["persisted"] = True
            result["person"]["total_rows"] = counts["person_reports"]

            if split_used:
                remove_split_dir(job_id)

            vstore.update_job_status(
                job_id,
                VA_STATUS_COMPLETED,
                message="Hoàn tất",
                total_time_analyze_s=time.time() - t_job0,
            )
            _update_job(job_id, status="done", message="Hoàn tất", result=result, error=None)
        finally:
            try:
                cap_v.release()
            except Exception:
                pass
    except Exception as ex:
        get_video_analyze_store().update_job_status(
            job_id, VA_STATUS_ERROR, message=str(ex), error_code="exception"
        )
        _update_job(job_id, status="error", error="exception", message=str(ex))
    finally:
        try:
            from module_ai.engine.gpu_cleanup import maybe_gpu_soft_cleanup

            maybe_gpu_soft_cleanup(log_label=f"video_analyze:{job_id}")
        except Exception:
            pass


def start_job_thread(
    job_id: str,
    *,
    sample_fps: float,
    distance_threshold: float,
) -> None:
    def wrapper() -> None:
        try:
            _run_job(
                job_id,
                sample_fps=sample_fps,
                distance_threshold=distance_threshold,
            )
        finally:
            release_job_slot()

    t = threading.Thread(target=wrapper, name=f"ivm-video-{job_id[:8]}", daemon=True)
    t.start()


def local_video_path_allowed(path: Path) -> bool:
    roots = s.IVM_VIDEO_ANALYZE_LOCAL_PATH_ROOTS
    if not roots:
        return True
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def stage_local_video(source: Path, job_id: str) -> Path:
    """Copy/hardlink video từ đĩa máy chủ vào thư mục job (bỏ qua upload HTTP/Streamlit)."""
    src = source.resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {src}")
    if not local_video_path_allowed(src):
        raise PermissionError("Đường dẫn nằm ngoài IVM_VIDEO_ANALYZE_LOCAL_PATH_ROOTS")
    suf = upload_suffix_from_name(src.name)
    if not suf:
        raise ValueError(
            f"Định dạng không hỗ trợ (chấp nhận: {', '.join(s.IVM_VIDEO_ANALYZE_ALLOWED_SUFFIXES)})"
        )
    job_dir = Path(s.IVM_VIDEO_ANALYZE_DIR) / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / f"source{suf}"
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)
    return dest


def save_upload_bytes(data: bytes, original_filename: str, job_id: str) -> Path:
    suf = upload_suffix_from_name(original_filename)
    job_dir = Path(s.IVM_VIDEO_ANALYZE_DIR) / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / f"source{suf}"
    path.write_bytes(data)
    return path
