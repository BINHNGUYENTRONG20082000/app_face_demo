"""Chuẩn hóa báo cáo video IVM → format VideoMaster PersonReportSchema + API routes."""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import quote

from identity_vm_app.services.video_report_merge import (
    filter_tracks_min_frames,
    merge_reports_vm,
    track_frame_count,
)

# Trùng PersonReportSchema (VideoMaster_BE/src/ma_schemas.py)
VM_PERSON_REPORT_FIELDS: Sequence[str] = (
    "id",
    "video_id",
    "video_name",
    "time_video",
    "time_analyze",
    "img_url",
    "id_tracking",
    "age",
    "gender",
    "mask",
    "box_face",
    "face_img",
    "box_person",
    "person_img",
    "video_clip",
    "sleeve_length",
    "type_of_lower_body_clothing",
    "length_of_lower_body_clothing",
    "carrying_handbag",
    "wearing_hat",
    "color",
    "color_tag",
    "end_time",
    "count_persons",
)


def vm_face_image_url(api_base: str, report_id: str) -> str:
    return f"{api_base.rstrip('/')}/ivm/reports/get-face-image/{report_id}"


def vm_person_image_url(api_base: str, report_id: str) -> str:
    return f"{api_base.rstrip('/')}/ivm/reports/get-person-image/{report_id}"


def vm_draw_box_url(api_base: str, report_id: str) -> str:
    return f"{api_base.rstrip('/')}/ivm/reports/draw-box-person/{report_id}"


def vm_weapon_image_url(api_base: str, report_id: str, weapon_class: Optional[str] = None) -> str:
    base = f"{api_base.rstrip('/')}/ivm/reports/get-weapon-image/{report_id}"
    if weapon_class:
        from urllib.parse import quote

        return f"{base}?weapon_class={quote(str(weapon_class))}"
    return base


def _weapon_types_from_row(row: Dict[str, Any]) -> List[str]:
    import json

    from identity_vm_app.services.weapon_crops import normalize_weapon_class

    raw = row.get("weapon_types_json")
    if not raw:
        return []
    try:
        val = json.loads(str(raw)) if isinstance(raw, str) else raw
        if isinstance(val, list):
            return [normalize_weapon_class(t) for t in val if t]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def vm_weapon_crop_urls(api_base: str, row: Dict[str, Any]) -> List[Dict[str, str]]:
    """URL crop từng loại vũ khí (gun + knife đồng thời nếu có trong báo cáo)."""
    from identity_vm_app.services.video_report_crops import parse_weapon_boxes
    from identity_vm_app.services.weapon_crops import (
        normalize_weapon_class,
        parse_weapon_crops_json,
        weapons_best_per_class,
    )

    rid = str(row.get("id") or "")
    if not rid:
        return []
    base = api_base.rstrip("/")
    saved_path: Dict[str, str] = {}
    for item in parse_weapon_crops_json(row.get("weapon_crops_json")):
        cls = normalize_weapon_class(item.get("class"))
        path = item.get("path")
        if path and cls not in saved_path:
            saved_path[cls] = str(path)

    classes_order: List[str] = []
    for wb in weapons_best_per_class(parse_weapon_boxes(row.get("weapon_boxes_json"))):
        cls = normalize_weapon_class(wb.get("class"))
        if cls not in classes_order:
            classes_order.append(cls)
    for cls in _weapon_types_from_row(row):
        if cls not in classes_order:
            classes_order.append(cls)
    for cls in saved_path:
        if cls not in classes_order:
            classes_order.append(cls)

    out: List[Dict[str, str]] = []
    for cls in classes_order:
        path = saved_path.get(cls)
        if path:
            url = f"{base}/ivm/video-analyze/media?path={quote(path, safe='')}"
        else:
            url = vm_weapon_image_url(api_base, rid, cls)
        out.append({"class": cls, "url": url})

    if not out and (row.get("weapon_img") or int(row.get("armed") or 0)):
        out.append({"class": "weapon", "url": vm_weapon_image_url(api_base, rid)})
    return out


def vm_track_scene_image_url(api_base: str, report_id: str) -> str:
    return f"{api_base.rstrip('/')}/ivm/reports/get-track-scene-image/{report_id}"


def vm_root_image_url(api_base: str, img_url: Optional[str]) -> Optional[str]:
    if not img_url:
        return None
    s = str(img_url)
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return f"{api_base.rstrip('/')}/ivm/video-analyze/media?path={quote(s, safe='')}"


def job_time_video(job: Optional[Dict[str, Any]]) -> float:
    """VideoMaster: Video.time_video — dùng time_upload_utc của job IVM."""
    if not job:
        return 0.0
    try:
        return float(job.get("time_video") or job.get("time_upload_utc") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def effective_analyze_time(row: Dict[str, Any], job: Optional[Dict[str, Any]] = None) -> float:
    """time_analyze + time_video (giống filter VM)."""
    try:
        t = float(row.get("time_analyze_s") if row.get("time_analyze_s") is not None else row.get("time_analyze") or 0)
    except (TypeError, ValueError):
        t = 0.0
    return t + job_time_video(job)


def _as_time_analyze_value(row: Dict[str, Any]) -> Any:
    t = row.get("time_analyze_s")
    if t is None:
        t = row.get("time_analyze")
    if t is None:
        return 0
    try:
        fv = float(t)
        if fv == int(fv):
            return int(fv)
        return fv
    except (TypeError, ValueError):
        return 0


def _normalize_box_str(val: Any) -> Any:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() == "none":
        return None
    return s


def extract_colors_from_reports(rows: List[Dict[str, Any]]) -> List[str]:
    """Giống get_person_sub_data_view — gom màu từ cột color (chuỗi list)."""
    colors: List[str] = []
    for r in rows:
        raw = r.get("color")
        if raw is None:
            continue
        s = str(raw).strip()
        if not s or s.lower() == "none":
            continue
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                colors.extend(str(x) for x in parsed if x is not None)
            else:
                colors.append(str(parsed))
        except (ValueError, SyntaxError):
            colors.append(s)
    return sorted(set(colors))


def to_vm_person_row(
    row: Dict[str, Any],
    *,
    job: Optional[Dict[str, Any]] = None,
    api_base: str = "",
    include_media_urls: bool = False,
) -> Dict[str, Any]:
    """Map SQLite → field names VideoMaster FE (PersonReportSchema)."""
    job_id = str(row.get("job_id") or row.get("video_id") or "")
    job_meta = job or {}
    video_name = str(job_meta.get("display_name") or job_meta.get("original_name") or job_id)
    rid = str(row.get("id") or "")
    tv = job_time_video(job_meta)

    out: Dict[str, Any] = {
        "id": rid,
        "video_id": job_id,
        "video_name": video_name,
        "time_video": tv,
        "time_analyze": _as_time_analyze_value(row),
        "img_url": row.get("img_url"),
        "id_tracking": int(row.get("id_tracking") or 0),
        "face_id": row.get("face_id"),
        "display_name": row.get("display_name"),
        "distance": row.get("distance"),
        "match_score": row.get("match_score"),
        "match_candidates_json": row.get("match_candidates_json"),
        "age": row.get("age"),
        "gender": row.get("gender"),
        "mask": row.get("mask"),
        "box_face": _normalize_box_str(row.get("box_face")),
        "face_img": row.get("face_img"),
        "box_person": _normalize_box_str(row.get("box_person")),
        "person_img": row.get("person_img"),
        "video_clip": int(row.get("video_clip") or 1),
        "sleeve_length": row.get("sleeve_length"),
        "type_of_lower_body_clothing": row.get("type_of_lower_body_clothing"),
        "length_of_lower_body_clothing": row.get("length_of_lower_body_clothing"),
        "carrying_handbag": row.get("carrying_handbag"),
        "wearing_hat": row.get("wearing_hat"),
        "color": row.get("color"),
        "color_tag": row.get("color_tag"),
        "end_time": row.get("end_time"),
        "count_persons": row.get("count_persons"),
        "armed": int(row.get("armed") or 0),
        "weapon_status": row.get("weapon_status"),
        "weapon_label": row.get("weapon_label"),
        "weapon_types_json": row.get("weapon_types_json"),
        "weapon_score": row.get("weapon_score"),
        "weapon_boxes_json": row.get("weapon_boxes_json"),
        "weapon_img": row.get("weapon_img"),
        "weapon_crops_json": row.get("weapon_crops_json"),
    }
    if include_media_urls and api_base and rid:
        out["face_image_url"] = vm_face_image_url(api_base, rid)
        out["person_image_url"] = vm_person_image_url(api_base, rid)
        out["draw_box_url"] = vm_draw_box_url(api_base, rid)
        out["thumb_image_url"] = out["draw_box_url"]
        out["img_url_full"] = vm_root_image_url(api_base, row.get("img_url"))
        out["track_scene_url"] = vm_track_scene_image_url(api_base, rid)
        wurls = vm_weapon_crop_urls(api_base, row)
        out["weapon_crop_urls"] = wurls
        if wurls:
            out["weapon_image_url"] = wurls[0]["url"]
            out["weapon_crop_url"] = wurls[0]["url"]
        elif row.get("weapon_img") or int(row.get("armed") or 0):
            wurl = vm_weapon_image_url(api_base, rid)
            out["weapon_image_url"] = wurl
            out["weapon_crop_url"] = wurl
    track_names = row.get("track_names")
    if isinstance(track_names, list) and track_names:
        out["track_names"] = track_names
        out["identity_labels"] = track_names
    elif row.get("track_name_label"):
        out["track_name_label"] = row.get("track_name_label")
    return out


def dump_vm_person_reports(
    rows: List[Dict[str, Any]],
    *,
    job: Optional[Dict[str, Any]] = None,
    api_base: str = "",
    include_media_urls: bool = False,
) -> List[Dict[str, Any]]:
    return [
        to_vm_person_row(r, job=job, api_base=api_base, include_media_urls=include_media_urls)
        for r in rows
    ]


def merge_and_dump_vm_faces_person(
    rows: List[Dict[str, Any]],
    *,
    job: Optional[Dict[str, Any]] = None,
    api_base: str = "",
    min_track_frames: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """GET /api/reports/faces-person — merge video_id:id_tracking rồi dump schema VM."""
    merged = filter_tracks_min_frames(merge_reports_vm(rows), min_frames=min_track_frames)
    out: List[Dict[str, Any]] = []
    for r in merged:
        vm = to_vm_person_row(r, job=job, api_base=api_base, include_media_urls=True)
        label = r.get("identity_label") or r.get("display_name")
        if label is not None and str(label).strip():
            vm["identity_label"] = str(label).strip()
        tn = r.get("track_names")
        if isinstance(tn, list) and tn:
            vm["track_names"] = tn
            vm["identity_labels"] = tn
        ff = r.get("first_frame_report_id") or r.get("id")
        if ff:
            vm["first_frame_report_id"] = str(ff)
        hc = track_frame_count(r)
        if hc:
            vm["hit_count"] = hc
            vm["frame_count"] = hc
        out.append(vm)
    return out


def merge_vm_person_rows(
    rows: List[Dict[str, Any]],
    *,
    job: Optional[Dict[str, Any]] = None,
    api_base: str = "",
    include_clip: bool = False,
    vm_simple_merge: bool = False,
) -> List[Dict[str, Any]]:
    """
    Gom track. vm_simple_merge=True → giống VideoMaster merge_reports (mặc định cho /api/reports/*).
    vm_simple_merge=False → merge IVM (bỏ phiếu đa khung) cho endpoint nội bộ.
    """
    from identity_vm_app.services.video_report_merge import merge_reports

    if vm_simple_merge:
        merged = merge_reports_vm(rows, include_clip=include_clip)
    else:
        merged = merge_reports(rows, include_clip=include_clip)
    return dump_vm_person_reports(merged, job=job, api_base=api_base, include_media_urls=True)
