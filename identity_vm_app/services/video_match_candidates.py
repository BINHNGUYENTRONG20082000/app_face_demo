"""Lưu & gom top-K ứng viên so khớp từ search_batch (mỗi khung) → top nghi ngờ theo track."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

_UNKNOWN_NAMES = frozenset({"", "unknown", "?", "none", "null"})


def _is_known_name(name: Any) -> bool:
    if name is None:
        return False
    return str(name).strip().lower() not in _UNKNOWN_NAMES


def serialize_match_candidates(
    matches: List[Dict[str, Any]],
    *,
    limit: int = 5,
) -> Optional[str]:
    """Chuẩn hóa top-K từ face_db.search_batch để lưu SQLite."""
    if not matches:
        return None
    out: List[Dict[str, Any]] = []
    for rank, m in enumerate(matches[: max(1, int(limit))]):
        if not isinstance(m, dict):
            continue
        name = m.get("name") or m.get("display_name")
        fid = m.get("face_id")
        if fid is None:
            fid = m.get("id")
        out.append(
            {
                "rank": rank,
                "face_id": fid,
                "display_name": name,
                "distance": m.get("distance"),
            }
        )
    return json.dumps(out, ensure_ascii=False) if out else None


def row_match_candidates(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Top-K đã lưu trên dòng báo cáo; job cũ chỉ có top-1 trên cột display_name."""
    raw = row.get("match_candidates_json")
    if raw:
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(data, list):
                return [c for c in data if isinstance(c, dict)]
        except (json.JSONDecodeError, TypeError):
            pass
    name = row.get("display_name")
    if not _is_known_name(name):
        return []
    dist = row.get("distance")
    return [
        {
            "rank": 0,
            "face_id": row.get("face_id"),
            "display_name": name,
            "distance": dist,
        }
    ]


def _candidate_key(c: Dict[str, Any]) -> Optional[str]:
    fid = c.get("face_id")
    if fid not in (None, "", "None"):
        return f"id:{fid}"
    name = c.get("display_name")
    if _is_known_name(name):
        return f"name:{str(name).strip().lower()}"
    return None


def _candidate_score(c: Dict[str, Any]) -> float:
    dist = c.get("distance")
    try:
        return 1.0 - float(dist) if dist is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def candidate_match_percent(c: Dict[str, Any]) -> Optional[float]:
    """Phần trăm khớp định danh (0–100) từ distance hoặc match_score."""
    dist = c.get("distance")
    if dist is not None:
        try:
            return max(0.0, min(100.0, (1.0 - float(dist)) * 100.0))
        except (TypeError, ValueError):
            pass
    ms = c.get("match_score")
    if ms is not None:
        try:
            return max(0.0, min(100.0, float(ms) * 100.0))
        except (TypeError, ValueError):
            pass
    return None


def aggregate_track_suspects(
    rows: List[Dict[str, Any]],
    *,
    limit: int = 5,
) -> Tuple[List[Dict[str, Any]], int, bool]:
    """
    Gom top-K từ mọi khung trong track (không chỉ top-1 đã ghi vào display_name).
    Trả về (suspects, frames_with_candidates, used_full_topk).
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    frames_with_candidates = 0
    used_full_topk = False

    for row in rows:
        cands = row_match_candidates(row)
        if not cands:
            continue
        if len(cands) > 1 or row.get("match_candidates_json"):
            used_full_topk = True
        frames_with_candidates += 1
        rid = row.get("id")
        for c in cands:
            key = _candidate_key(c)
            if not key:
                continue
            rank = int(c.get("rank") or 0)
            score = _candidate_score(c)
            weight = 1.0 / (rank + 1)
            name = str(c.get("display_name") or "").strip()
            if key not in buckets:
                buckets[key] = {
                    "display_name": name,
                    "face_id": c.get("face_id"),
                    "frame_hits": 0,
                    "weighted_votes": 0.0,
                    "score_sum": 0.0,
                    "max_score": -1.0,
                    "best_report_id": None,
                    "best_rank": 999,
                    "best_score": -1.0,
                }
            b = buckets[key]
            b["frame_hits"] += 1
            b["weighted_votes"] += weight
            b["score_sum"] += score * weight
            b["max_score"] = max(b["max_score"], score)
            if rank < b["best_rank"] or (rank == b["best_rank"] and score > b["best_score"]):
                b["best_rank"] = rank
                b["best_score"] = score
                b["best_report_id"] = rid

    if not buckets:
        return [], frames_with_candidates, used_full_topk

    def _rank_item(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, float, float, float]:
        b = item[1]
        return (
            int(b["frame_hits"]),
            float(b["weighted_votes"]),
            float(b["score_sum"]),
            float(b["max_score"]),
        )

    ranked = sorted(buckets.items(), key=_rank_item, reverse=True)[: max(1, int(limit))]
    suspects: List[Dict[str, Any]] = []
    denom = max(frames_with_candidates, 1)
    for i, (_key, b) in enumerate(ranked, start=1):
        suspects.append(
            {
                "rank": i,
                "display_name": b["display_name"],
                "face_id": b.get("face_id"),
                "vote_count": int(b["frame_hits"]),
                "vote_ratio": round(int(b["frame_hits"]) / denom, 4),
                "match_score": b["max_score"] if b["max_score"] >= 0 else None,
                "weighted_votes": round(float(b["weighted_votes"]), 4),
                "report_id": b.get("best_report_id"),
            }
        )
    return suspects, frames_with_candidates, used_full_topk
