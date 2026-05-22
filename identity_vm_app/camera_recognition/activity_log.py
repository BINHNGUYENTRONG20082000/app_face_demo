"""Nhật ký hoạt động nhận diện theo camera (RAM) — xem qua API hoặc console."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("camera_recognition.activity")

_lock = threading.Lock()
_MAX_PER_CAMERA = 80
_entries: Dict[str, Deque[Dict[str, Any]]] = {}


def _deque_for(camera_id: str) -> Deque[Dict[str, Any]]:
    cid = str(camera_id)
    if cid not in _entries:
        _entries[cid] = deque(maxlen=_MAX_PER_CAMERA)
    return _entries[cid]


def record(
    camera_id: str,
    event: str,
    message: str,
    *,
    level: str = "info",
    extra: Optional[Dict[str, Any]] = None,
    also_console: bool = True,
) -> None:
    row = {
        "ts_utc": time.time(),
        "camera_id": str(camera_id),
        "event": event,
        "level": level,
        "message": message,
        "extra": extra or {},
    }
    with _lock:
        _deque_for(camera_id).append(row)
    if also_console:
        line = f"[{camera_id}] {message}"
        if level == "error":
            logger.error(line)
        elif level == "warning":
            logger.warning(line)
        else:
            logger.info(line)


def recent(camera_id: Optional[str] = None, *, limit: int = 40) -> List[Dict[str, Any]]:
    lim = max(1, min(200, int(limit)))
    with _lock:
        if camera_id is not None:
            items = list(_deque_for(str(camera_id)))[-lim:]
            return list(reversed(items))
        merged: List[Dict[str, Any]] = []
        for dq in _entries.values():
            merged.extend(dq)
        merged.sort(key=lambda r: float(r.get("ts_utc") or 0), reverse=True)
        return merged[:lim]
