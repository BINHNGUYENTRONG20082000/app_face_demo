# -*- coding: utf-8 -*-
"""Đọc camera_config.json — dùng chung camera pipeline và Identity VM app."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_APP_ROOT, "camera_config.json")

# camera_source1, camera_source2, ... (sắp xếp theo số)
_NUMBERED_SOURCE_KEY = re.compile(r"^camera_source(\d+)$", re.IGNORECASE)


def _parse_json_loose(raw: str) -> Any | None:
    """Parse JSON; nếu lỗi thì thử bỏ dấu phẩy thừa trước } hoặc ] (lỗi phổ biến khi sửa tay file)."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        fixed = re.sub(r",(\s*[}\]])", r"\1", raw)
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


def _specs_from_numbered_keys(config: Dict[str, Any]) -> List[Dict[str, Any]] | None:
    items: list[tuple[int, Any]] = []
    for k, v in config.items():
        if not isinstance(k, str):
            continue
        m = _NUMBERED_SOURCE_KEY.match(k.strip())
        if m:
            items.append((int(m.group(1)), v))
    if not items:
        return None
    items.sort(key=lambda x: x[0])
    # id = số trên khóa camera_sourceN (vd. source2 bị thiếu vẫn là cam0, cam1, cam3)
    return [{"id": f"cam{num}", "source": src} for num, src in items]


def load_camera_channel_specs(config_path: str | None = None) -> List[Dict[str, Any]]:
    """
    Trả về [{"id", "source"}, ...].

    Thứ tự ưu tiên:
    1. `cameras`: [{ "id"?, "source"|"camera_source" }, ...]
    2. Các khóa `camera_source1`, `camera_source2`, ... (theo số)
    3. Một kênh: `camera_source` + `camera_id` (legacy)
    4. Mặc định: webcam index 0
    """
    default: List[Dict[str, Any]] = [{"id": "cam0", "source": 0}]
    path = config_path or _CONFIG_PATH
    if not os.path.isfile(path):
        return default
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            raw = f.read()
        config = _parse_json_loose(raw)
        if config is None:
            return default
    except Exception:
        return default
    if not isinstance(config, dict):
        return default

    cams = config.get("cameras")
    if isinstance(cams, list) and len(cams) > 0:
        out: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for i, c in enumerate(cams):
            if not isinstance(c, dict):
                continue
            cid = str(c.get("id", f"cam{i}")).strip() or f"cam{i}"
            if cid in seen:
                continue
            seen.add(cid)
            if "source" in c:
                src = c["source"]
            elif "camera_source" in c:
                src = c["camera_source"]
            else:
                src = 0
            out.append({"id": cid, "source": src})
        return out if out else default

    numbered = _specs_from_numbered_keys(config)
    if numbered is not None:
        return numbered

    src = config.get("camera_source", 0)
    cid = str(config.get("camera_id", "cam0")).strip() or "cam0"
    return [{"id": cid, "source": src}]


def stream_camera_ids(config_path: str | None = None) -> List[str]:
    """Danh sách `id` kênh camera từ cấu hình."""
    return [str(s["id"]) for s in load_camera_channel_specs(config_path)]
