"""Tìm khuôn mặt trong báo cáo video qua features_face (format VideoMaster np.array_str)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def parse_features_face(raw: Optional[str]) -> Optional[np.ndarray]:
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        if s.startswith("[") and s.endswith("]"):
            return np.fromstring(s.strip("[]"), sep=" ", dtype=np.float32)
        # np.array_str: "[ 0.1  0.2 ...]"
        inner = s.strip()
        if inner.startswith("[") and inner.endswith("]"):
            return np.fromstring(inner[1:-1], sep=" ", dtype=np.float32)
        return np.fromstring(s, sep=" ", dtype=np.float32)
    except Exception:
        return None


def compare_features_vm(f1: np.ndarray, f2: np.ndarray) -> Tuple[float, float]:
    """Cosine sim → percent (giống CompareFeaturesFace VideoMaster)."""
    a = f1.reshape(-1).astype(np.float32)
    b = f2.reshape(-1).astype(np.float32)
    na = float(np.linalg.norm(a)) or 1.0
    nb = float(np.linalg.norm(b)) or 1.0
    similarity = float(np.dot(a, b) / (na * nb))
    if similarity > 0.45:
        percent = 0.98
    else:
        percent = similarity * 2.5
    if percent > 0.98:
        percent = 0.98
    return similarity, percent * 100.0


def search_reports_by_embedding(
    query: np.ndarray,
    rows: List[Dict[str, Any]],
    *,
    min_percent: float = 0.0,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for row in rows:
        feat = parse_features_face(row.get("features_face"))
        if feat is None:
            continue
        sim, pct = compare_features_vm(query, feat)
        if pct < min_percent:
            continue
        out = dict(row)
        out["similarity"] = round(sim, 4)
        out["percent"] = round(pct, 2)
        hits.append(out)
    hits.sort(key=lambda x: float(x.get("percent") or 0), reverse=True)
    return hits[: max(1, int(limit))]
