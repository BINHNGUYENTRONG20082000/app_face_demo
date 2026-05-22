"""Xóa toàn bộ dữ liệu nghiệp vụ (face DB, SQLite, thư mục phụ)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from identity_vm_app import settings as s
from identity_vm_app.services.event_crops import crop_file_for_event, crop_weapon_file_for_event
from identity_vm_app.state import state


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def wipe_disk_folders(*, wipe_archive: bool) -> List[str]:
    """Xóa và tạo lại thư mục dưới IVM_DATA_DIR (trừ face_db — do FaceDatabase.reset xử lý images)."""
    done: List[str] = []
    pairs = [
        (s.IVM_DATA_DIR / "registration_errors", True),
        (s.IVM_DATA_DIR / "gallery", True),
        (s.IVM_DATA_DIR / "event_crops", True),
        (s.IVM_EXPORT_CACHE_DIR.resolve(), True),
        (s.IVM_ARCHIVE_ROOT.resolve(), wipe_archive),
    ]
    for path, do in pairs:
        if not do:
            continue
        p = Path(path)
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
        _ensure_dir(p)
        try:
            rel = p.relative_to(s.IVM_DATA_DIR.resolve())
            done.append(str(rel))
        except ValueError:
            done.append(str(p))
    return done


def execute_clear_reports(
    *,
    camera_id: Optional[str] = None,
    wipe_archive: bool = False,
    wipe_visual: bool = True,
) -> Dict[str, Any]:
    """Xóa báo cáo nhận diện (events, crop, video overlay, export cache) — giữ Face DB."""
    import os

    if state.store is None:
        raise RuntimeError("Store chưa khởi tạo — không thể xóa báo cáo.")

    store = state.store
    sql_counts = store.clear_recognition_reports(
        camera_id=camera_id,
        clear_segments=bool(wipe_archive),
    )
    event_ids: List[str] = list(sql_counts.get("event_ids") or [])

    dirs_cleared: List[str] = []
    visual_root = Path(
        os.getenv("IVM_ANALYZE_VISUAL_DIR", str(s.IVM_DATA_DIR / "analyze_visual"))
    ).resolve()

    if wipe_visual:
        if camera_id:
            p = visual_root / camera_id
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
            _ensure_dir(p)
            try:
                dirs_cleared.append(str(p.relative_to(s.IVM_DATA_DIR.resolve())))
            except ValueError:
                dirs_cleared.append(str(p))
        else:
            if visual_root.exists():
                shutil.rmtree(visual_root, ignore_errors=True)
            _ensure_dir(visual_root)
            dirs_cleared.append("analyze_visual")

    crops = Path(s.IVM_DATA_DIR) / "event_crops"
    if camera_id:
        removed = 0
        for eid in event_ids:
            for fp in (crop_file_for_event(eid), crop_weapon_file_for_event(eid)):
                if fp.is_file():
                    fp.unlink(missing_ok=True)
                    removed += 1
        dirs_cleared.append(f"event_crops ({removed} files)")
    else:
        if crops.exists():
            shutil.rmtree(crops, ignore_errors=True)
        _ensure_dir(crops)
        dirs_cleared.append("event_crops")

    if not camera_id:
        cache = s.IVM_EXPORT_CACHE_DIR.resolve()
        if cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
        _ensure_dir(cache)
        dirs_cleared.append("export_cache")

    if wipe_archive:
        arch = s.IVM_ARCHIVE_ROOT.resolve()
        if camera_id:
            p = arch / camera_id
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
            _ensure_dir(p)
            try:
                dirs_cleared.append(str(p.relative_to(s.IVM_DATA_DIR.resolve())))
            except ValueError:
                dirs_cleared.append(str(p))
        else:
            if arch.exists():
                shutil.rmtree(arch, ignore_errors=True)
            _ensure_dir(arch)
            dirs_cleared.append("archive")

    sqlite_out = {k: v for k, v in sql_counts.items() if k != "event_ids"}
    return {
        "ok": True,
        "camera_id": camera_id,
        "sqlite": sqlite_out,
        "cleared_dirs": dirs_cleared,
        "wipe_archive": wipe_archive,
        "face_database": "unchanged",
    }


def execute_full_reset(*, wipe_archive: bool = False) -> Dict[str, Any]:
    if state.recorders is not None:
        state.recorders.stop_all()
    if state.face_db is None or state.store is None:
        raise RuntimeError("FaceDatabase hoặc Store chưa khởi tạo — không thể reset.")
    face = state.face_db
    store = state.store
    face.reset()
    sql_counts = store.clear_all_tables()
    dirs = wipe_disk_folders(wipe_archive=wipe_archive)
    return {
        "ok": True,
        "face_database": "reset (embeddings, metadata, images/)",
        "sqlite": sql_counts,
        "recreated_dirs": dirs,
        "wipe_archive": wipe_archive,
    }
