"""Gom báo cáo theo track — giống merge_reports VideoMaster."""

from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Optional

from identity_vm_app import settings as s
from identity_vm_app.services.video_match_candidates import aggregate_track_suspects

_UNKNOWN_NAMES = frozenset({"", "unknown", "?", "none", "null"})


def _is_known_name(name: Any) -> bool:
    if name is None:
        return False
    return str(name).strip().lower() not in _UNKNOWN_NAMES


def _identity_bucket_key(row: Dict[str, Any]) -> Optional[str]:
    """Gom khớp theo face_id; không có face_id thì theo display_name."""
    name = row.get("display_name")
    if not _is_known_name(name):
        return None
    fid = row.get("face_id")
    if fid not in (None, "", "None"):
        return f"id:{fid}"
    return f"name:{str(name).strip().lower()}"


def _row_match_score(row: Dict[str, Any]) -> float:
    sc = row.get("match_score")
    try:
        return float(sc) if sc is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def top_suspect_faces(rows: List[Dict[str, Any]], *, limit: int = 5) -> List[Dict[str, Any]]:
    """Top nghi ngờ track = gom top-K search_batch từ mọi khung (xem aggregate_track_suspects)."""
    suspects, _, _ = aggregate_track_suspects(rows, limit=limit)
    return suspects


def names_from_suspect_faces(
    suspects: List[Dict[str, Any]],
    *,
    limit: Optional[int] = None,
) -> List[str]:
    """Danh sách tên (không trùng) từ suspect_faces, tối đa limit."""
    cap = int(limit if limit is not None else s.IVM_TRACK_SUSPECT_NAMES_LIMIT)
    cap = max(1, min(10, cap))
    names: List[str] = []
    seen: set[str] = set()
    for item in suspects or []:
        if not isinstance(item, dict):
            continue
        name = item.get("display_name")
        if not _is_known_name(name):
            continue
        key = str(name).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        names.append(str(name).strip())
        if len(names) >= cap:
            break
    return names


def apply_track_identity_fields(
    row: Dict[str, Any],
    group_rows: List[Dict[str, Any]],
    *,
    limit: Optional[int] = None,
) -> None:
    """Gắn display_name, suspect_faces, track_names (top N) cho một dòng track đã gom."""
    identity = resolve_track_identity(group_rows)
    suspects = identity.get("suspect_faces") or []
    names = names_from_suspect_faces(suspects, limit=limit)
    row["suspect_faces"] = suspects
    row["suspect_names"] = names
    row["track_names"] = names
    row["track_name"] = names
    if identity.get("display_name"):
        row["display_name"] = identity.get("display_name")
    if identity.get("face_id") not in (None, "", "None"):
        row["face_id"] = identity.get("face_id")
    if identity.get("match_score") is not None:
        row["match_score"] = identity.get("match_score")
    row["identity_vote_count"] = identity.get("identity_vote_count", 0)
    row["identity_vote_total"] = identity.get("identity_vote_total", 0)
    row["identity_vote_ratio"] = identity.get("identity_vote_ratio", 0.0)
    n_frames = len(group_rows)
    row["hit_count"] = n_frames
    row["frame_count"] = n_frames
    if not names:
        row["track_name_label"] = str(row.get("display_name") or "unknown")
    else:
        row["track_name_label"] = " · ".join(names)


def resolve_track_identity(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Tên đại diện = #1 sau khi gom toàn bộ top-K từng khung trong track.
    Job cũ (chỉ lưu top-1/khung) vẫn chạy được qua fallback trong row_match_candidates.
    """
    total_frames = len(rows)
    suspects, frames_with_candidates, _used_topk = aggregate_track_suspects(rows, limit=5)
    if not suspects:
        fallback_name: Optional[str] = None
        for row in rows:
            name = row.get("display_name")
            if name is not None and str(name).strip():
                fallback_name = str(name).strip()
                break
        return {
            "display_name": fallback_name,
            "face_id": None,
            "match_score": None,
            "identity_vote_count": 0,
            "identity_vote_total": total_frames,
            "identity_vote_ratio": 0.0,
            "suspect_faces": [],
        }

    winner = suspects[0]
    return {
        "display_name": winner.get("display_name"),
        "face_id": winner.get("face_id"),
        "match_score": winner.get("match_score"),
        "identity_vote_count": winner.get("vote_count", 0),
        "identity_vote_total": frames_with_candidates or total_frames,
        "identity_vote_ratio": winner.get("vote_ratio", 0.0),
        "suspect_faces": suspects,
    }


def pick_track_display_name(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Tên đại diện track sau bỏ phiếu đa khung (xem resolve_track_identity)."""
    return resolve_track_identity(rows).get("display_name")


def pick_first_appearance_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Ảnh lưới tổng quan = khung đầu tiên track xuất hiện."""
    if not rows:
        return None
    ordered = sorted(
        rows,
        key=lambda r: float(r.get("time_analyze_s") or r.get("time_analyze") or 0),
    )
    for r in ordered:
        if r.get("id") and (r.get("img_url") or r.get("box_person")):
            return r
    return ordered[0]


def pick_track_representative_row(
    rows: List[Dict[str, Any]],
    *,
    face_id: Any = None,
    display_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Khung crop người đại diện: thuộc danh tính thắng, match_score cao nhất trong nhóm đó."""
    if not rows:
        return None
    identity_rows = rows
    if face_id not in (None, "", "None"):
        identity_rows = [r for r in rows if r.get("face_id") == face_id]
    elif _is_known_name(display_name):
        dn = str(display_name).strip().lower()
        identity_rows = [
            r
            for r in rows
            if _is_known_name(r.get("display_name"))
            and str(r.get("display_name")).strip().lower() == dn
        ]
    with_person = [r for r in identity_rows if r.get("box_person")]
    pool = with_person or identity_rows or rows
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for row in pool:
        if not _is_known_name(row.get("display_name")):
            continue
        score = _row_match_score(row)
        if score >= best_score:
            best_score = score
            best = row
    if best is not None:
        return best
    return pool[0]


def _apply_row_to_track(accum: Dict[str, Any], row: Dict[str, Any]) -> None:
    """Cập nhật dòng gom track: tên, hit_count, ảnh đại diện."""
    accum["hit_count"] = int(accum.get("hit_count") or 1) + 1
    name = row.get("display_name")
    sc = row.get("match_score")
    try:
        score = float(sc) if sc is not None else 0.0
    except (TypeError, ValueError):
        score = 0.0
    cur_name = accum.get("display_name")
    cur_score_raw = accum.get("match_score")
    try:
        cur_score = float(cur_score_raw) if cur_score_raw is not None else -1.0
    except (TypeError, ValueError):
        cur_score = -1.0
    replace_name = False
    if _is_known_name(name):
        if not _is_known_name(cur_name) or score >= cur_score:
            replace_name = True
    elif not _is_known_name(cur_name) and name is not None and str(name).strip():
        replace_name = True
    if replace_name:
        accum["display_name"] = name
        accum["match_score"] = sc
        accum["face_id"] = row.get("face_id")
        accum["distance"] = row.get("distance")
        if row.get("id"):
            accum["id"] = row.get("id")
        if row.get("person_img") or row.get("img_url") or row.get("box_person"):
            accum["person_img"] = row.get("person_img")
            accum["img_url"] = row.get("img_url")
            accum["box_face"] = row.get("box_face")
            accum["box_person"] = row.get("box_person")
    accum["video_clip_min"] = min(
        int(accum.get("video_clip_min") or row.get("video_clip") or 1),
        int(row.get("video_clip") or 1),
    )
    accum["video_clip_max"] = max(
        int(accum.get("video_clip_max") or row.get("video_clip") or 1),
        int(row.get("video_clip") or 1),
    )


def tracking_key(
    row: MutableMapping[str, Any],
    *,
    job_id_key: str = "job_id",
    include_clip: bool = True,
) -> str:
    jid = row.get(job_id_key) or row.get("video_id") or ""
    tid = row.get("id_tracking")
    if include_clip:
        clip = row.get("video_clip", 1)
        return f"{jid}:{tid}:{clip}"
    return f"{jid}:{tid}"


def merge_reports_vm(
    rows: List[Dict[str, Any]],
    *,
    job_id_key: str = "job_id",
    include_clip: bool = False,
) -> List[Dict[str, Any]]:
    """
    Gom báo cáo giống VideoMaster_BE merge_reports (reports.py):
    key = video_id:id_tracking; chỉ cập nhật end_time.
    """
    results: List[Dict[str, Any]] = []
    index: Dict[str, int] = {}
    rows_by_key: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        data = dict(row)
        vid = data.get(job_id_key) or data.get("video_id") or ""
        tid = data.get("id_tracking")
        if include_clip:
            clip = int(data.get("video_clip") or 1)
            key = f"{vid}:{tid}:{clip}"
        else:
            key = f"{vid}:{tid}"
        rows_by_key.setdefault(key, []).append(data)

        t = data.get("time_analyze_s")
        if t is None:
            t = data.get("time_analyze")
        if t is not None:
            t = float(t)
            if data.get("time_analyze") is None:
                data["time_analyze"] = int(t) if t == int(t) else t

        dn = data.get("display_name")
        if dn is not None and str(dn).strip():
            data["identity_label"] = str(dn).strip()

        if key not in index:
            if t is not None:
                data["end_time"] = int(t) if t == int(t) else t
            if data.get("id"):
                data["first_frame_report_id"] = data.get("id")
            results.append(data)
            index[key] = len(results) - 1
        else:
            i = index[key]
            if t is not None:
                results[i]["end_time"] = int(t) if t == int(t) else t
            if dn is not None and str(dn).strip() and not results[i].get("identity_label"):
                results[i]["identity_label"] = str(dn).strip()
            # Đại diện track: giữ khung có match_score cao hơn (nhiều mặt cùng track).
            try:
                new_sc = float(data.get("match_score") or 0)
                cur_sc = float(results[i].get("match_score") or 0)
            except (TypeError, ValueError):
                new_sc, cur_sc = 0.0, 0.0
            if new_sc >= cur_sc:
                for fld in (
                    "id",
                    "img_url",
                    "box_person",
                    "box_face",
                    "face_img",
                    "person_img",
                    "face_id",
                    "display_name",
                    "match_score",
                    "distance",
                ):
                    v = data.get(fld)
                    if v not in (None, "", "None"):
                        results[i][fld] = v

    for r in results:
        vid = r.get(job_id_key) or r.get("video_id") or ""
        tid = r.get("id_tracking")
        if include_clip:
            clip = int(r.get("video_clip") or 1)
            key = f"{vid}:{tid}:{clip}"
        else:
            key = f"{vid}:{tid}"
        group = rows_by_key.get(key, [])
        apply_track_identity_fields(r, group)
        from identity_vm_app.services.weapon_track_status import apply_track_weapon_summary

        apply_track_weapon_summary(r, group)

    return results


def track_frame_count(row: Dict[str, Any]) -> int:
    try:
        return int(row.get("hit_count") or row.get("frame_count") or 0)
    except (TypeError, ValueError):
        return 0


def filter_tracks_min_frames(
    rows: List[Dict[str, Any]],
    *,
    min_frames: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Bỏ track có ít hơn min_frames khung mẫu."""
    n = int(s.ivm_report_min_track_frames() if min_frames is None else min_frames)
    if n <= 0:
        return rows
    return [r for r in rows if track_frame_count(r) >= n]


def filter_segments_min_frames(
    segments: List[Dict[str, Any]],
    *,
    min_frames: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Bỏ từng đoạn xuất hiện có ít hơn min_frames khung (không gom theo track_id)."""
    n = int(s.ivm_report_min_track_frames() if min_frames is None else min_frames)
    if n <= 0:
        return segments
    return [seg for seg in segments if track_frame_count(seg) >= n]


def filter_segments_by_track_total_frames(
    segments: List[Dict[str, Any]],
    *,
    min_track_frames: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Giữ mọi đoạn của track nếu **tổng** khung mẫu track >= ngưỡng (giống tab định danh).

    Tránh trường hợp track 15 khung nhưng tách 3 đoạn × 5 khung → tab đoạn trống
    trong khi tab định danh vẫn có track.
    """
    n = int(s.ivm_report_min_track_frames() if min_track_frames is None else min_track_frames)
    if n <= 0:
        return segments
    by_track: Dict[int, List[Dict[str, Any]]] = {}
    for seg in segments:
        tid = int(seg.get("id_tracking") or 0)
        by_track.setdefault(tid, []).append(seg)
    out: List[Dict[str, Any]] = []
    for segs in by_track.values():
        if sum(track_frame_count(s) for s in segs) >= n:
            out.extend(segs)
    return out


def _track_group_name_key(row: Dict[str, Any]) -> str:
    """Khóa gom thẻ: tên định danh; track unknown mỗi id_tracking một thẻ."""
    label = row.get("identity_label") or row.get("display_name")
    if _is_known_name(label):
        return str(label).strip().lower()
    tid = int(row.get("id_tracking") or 0)
    return f"__track_{tid}__"


def group_tracks_by_display_name(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Gom các track cùng tên thành một thẻ (member_tracks).
    Thẻ đại diện = track có nhiều khung nhất (rồi match_score).
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for row in rows:
        key = _track_group_name_key(row)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(row)

    cards: List[Dict[str, Any]] = []
    for key in order:
        members = buckets[key]
        rep = max(
            members,
            key=lambda m: (
                track_frame_count(m),
                float(m.get("match_score") or 0),
            ),
        )
        card = dict(rep)
        card["group_key"] = key
        card["member_tracks"] = members
        card["track_count"] = len(members)
        card["hit_count_total"] = sum(track_frame_count(m) for m in members)
        cards.append(card)
    return cards


def merge_reports(
    rows: List[Dict[str, Any]],
    *,
    job_id_key: str = "job_id",
    include_clip: bool = True,
) -> List[Dict[str, Any]]:
    """
    Gom các dòng theo track; cập nhật end_time + bỏ phiếu định danh (IVM nội bộ).
    VideoMaster API dùng merge_reports_vm.
    """
    results: List[Dict[str, Any]] = []
    index: Dict[str, int] = {}
    rows_by_key: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        data = dict(row)
        key = tracking_key(data, job_id_key=job_id_key, include_clip=include_clip)
        rows_by_key.setdefault(key, []).append(data)
        t = data.get("time_analyze_s")
        if t is None:
            t = data.get("time_analyze")
        if t is not None:
            t = float(t)

        if key not in index:
            if t is not None:
                data["end_time"] = t
            if "time_analyze" not in data and t is not None:
                data["time_analyze"] = t
            data["hit_count"] = 1
            vc = int(data.get("video_clip") or 1)
            data["video_clip_min"] = vc
            data["video_clip_max"] = vc
            results.append(data)
            index[key] = len(results) - 1
        else:
            i = index[key]
            if t is not None:
                results[i]["end_time"] = t
            _apply_row_to_track(results[i], data)

    for r in results:
        key = tracking_key(r, job_id_key=job_id_key, include_clip=include_clip)
        group = rows_by_key.get(key, [])
        apply_track_identity_fields(r, group)
        r["count_persons"] = 1
        rep = pick_first_appearance_row(group)
        if rep:
            for fld in ("id", "img_url", "box_person", "box_face", "person_img", "face_img", "video_clip"):
                v = rep.get(fld)
                if v not in (None, "", "None"):
                    r[fld] = v
            r["first_frame_report_id"] = rep.get("id")

    return results


def dedupe_reports_by_img_url(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Lọc trùng img_url (sub-data VideoMaster)."""
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        url = str(r.get("img_url") or "")
        if url and url in seen:
            continue
        if url:
            seen.add(url)
        out.append(r)
    return out
