from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

from identity_vm_app import settings as s
from identity_vm_app.services.event_crops import (
    decode_crop_b64,
    save_crop_jpeg,
    should_replace_crop,
)
from identity_vm_app.services.weapon_crops import save_track_scene_crop_jpeg, save_weapon_crop_jpeg


@dataclass
class RecordingSegmentRow:
    id: int
    camera_id: str
    path: str
    started_at_utc: float
    ended_at_utc: Optional[float]
    format: str


@dataclass
class RecognitionEventRow:
    id: str
    ts_utc: float
    camera_id: str
    source: str
    person_ref: str
    face_id: Optional[int]
    display_name: Optional[str]
    match_score: Optional[float]
    distance: Optional[float]
    det_score: Optional[float]
    model_tag: str
    recording_segment_id: Optional[int]
    offset_start_s: Optional[float]
    offset_end_s: Optional[float]
    gender: Optional[int]
    age: Optional[int]


class IdentityVmStore:
    """SQLite: recording_segments + recognition_events (+ optional person_note)."""

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path or s.IVM_SQLITE_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS recording_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    started_at_utc REAL NOT NULL,
                    ended_at_utc REAL,
                    format TEXT NOT NULL DEFAULT 'mkv',
                    UNIQUE(camera_id, path)
                );
                CREATE INDEX IF NOT EXISTS idx_seg_camera ON recording_segments(camera_id);
                CREATE INDEX IF NOT EXISTS idx_seg_started ON recording_segments(started_at_utc);

                CREATE TABLE IF NOT EXISTS recognition_events (
                    id TEXT PRIMARY KEY,
                    ts_utc REAL NOT NULL,
                    camera_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    person_ref TEXT NOT NULL,
                    face_id INTEGER,
                    display_name TEXT,
                    match_score REAL,
                    distance REAL,
                    det_score REAL,
                    model_tag TEXT NOT NULL,
                    recording_segment_id INTEGER,
                    offset_start_s REAL,
                    offset_end_s REAL,
                    gender INTEGER,
                    age INTEGER,
                    extra_json TEXT,
                    FOREIGN KEY(recording_segment_id) REFERENCES recording_segments(id)
                );
                CREATE INDEX IF NOT EXISTS idx_evt_cam_ts ON recognition_events(camera_id, ts_utc);
                CREATE INDEX IF NOT EXISTS idx_evt_person ON recognition_events(person_ref);

                CREATE TABLE IF NOT EXISTS person_media (
                    id TEXT PRIMARY KEY,
                    face_id INTEGER NOT NULL,
                    display_name TEXT,
                    media_type TEXT NOT NULL,
                    path TEXT NOT NULL,
                    created_at_utc REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_pm_face ON person_media(face_id);

                CREATE TABLE IF NOT EXISTS registration_failures (
                    id TEXT PRIMARY KEY,
                    ts_utc REAL NOT NULL,
                    original_filename TEXT,
                    error_message TEXT NOT NULL,
                    image_path TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_regfail_ts ON registration_failures(ts_utc);

                CREATE TABLE IF NOT EXISTS bulk_register_checkpoint (
                    path_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    ts_utc REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_bulk_ckpt_status ON bulk_register_checkpoint(status);
                """
            )

    def insert_segment(
        self,
        camera_id: str,
        path: str,
        started_at_utc: float,
        ended_at_utc: Optional[float],
        fmt: str = "mkv",
    ) -> int:
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT INTO recording_segments (camera_id, path, started_at_utc, ended_at_utc, format)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(camera_id, path) DO UPDATE SET
                  ended_at_utc=excluded.ended_at_utc
                """,
                (camera_id, path, started_at_utc, ended_at_utc, fmt),
            )
            cur = c.execute("SELECT id FROM recording_segments WHERE camera_id=? AND path=?", (camera_id, path))
            row = cur.fetchone()
            return int(row[0])

    def finalize_segment(self, segment_id: int, ended_at_utc: float) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE recording_segments SET ended_at_utc=? WHERE id=? AND (ended_at_utc IS NULL OR ended_at_utc <?)",
                (ended_at_utc, segment_id, ended_at_utc),
            )

    def find_segment_for_timestamp(
        self, camera_id: str, ts_utc: float
    ) -> Optional[RecordingSegmentRow]:
        """Tìm file archive chứa thời điểm ts (khi event chưa gắn segment_id)."""
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT * FROM recording_segments
                WHERE camera_id=?
                  AND started_at_utc <= ?
                  AND (ended_at_utc IS NULL OR ended_at_utc >= ?)
                ORDER BY started_at_utc DESC
                LIMIT 1
                """,
                (camera_id, ts_utc, ts_utc),
            )
            r = cur.fetchone()
            if not r:
                cur2 = c.execute(
                    """
                    SELECT * FROM recording_segments
                    WHERE camera_id=?
                    ORDER BY ABS(started_at_utc - ?) ASC
                    LIMIT 1
                    """,
                    (camera_id, ts_utc),
                )
                r = cur2.fetchone()
            if not r:
                return None
            return RecordingSegmentRow(
                id=int(r["id"]),
                camera_id=str(r["camera_id"]),
                path=str(r["path"]),
                started_at_utc=float(r["started_at_utc"]),
                ended_at_utc=float(r["ended_at_utc"]) if r["ended_at_utc"] is not None else None,
                format=str(r["format"]),
            )

    def get_segment(self, segment_id: int) -> Optional[RecordingSegmentRow]:
        with self._conn() as c:
            cur = c.execute("SELECT * FROM recording_segments WHERE id=?", (segment_id,))
            r = cur.fetchone()
            if not r:
                return None
            return RecordingSegmentRow(
                id=int(r["id"]),
                camera_id=str(r["camera_id"]),
                path=str(r["path"]),
                started_at_utc=float(r["started_at_utc"]),
                ended_at_utc=float(r["ended_at_utc"]) if r["ended_at_utc"] is not None else None,
                format=str(r["format"]),
            )

    def insert_event(
        self,
        *,
        ts_utc: float,
        camera_id: str,
        source: str,
        person_ref: str,
        face_id: Optional[int],
        display_name: Optional[str],
        match_score: Optional[float],
        distance: Optional[float],
        det_score: Optional[float],
        model_tag: str,
        recording_segment_id: Optional[int],
        offset_start_s: Optional[float],
        offset_end_s: Optional[float],
        gender: Optional[int] = None,
        age: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        eid = str(uuid.uuid4())
        extra_json = json.dumps(extra) if extra else None
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT INTO recognition_events (
                  id, ts_utc, camera_id, source, person_ref, face_id, display_name,
                  match_score, distance, det_score, model_tag,
                  recording_segment_id, offset_start_s, offset_end_s,
                  gender, age, extra_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    eid,
                    ts_utc,
                    camera_id,
                    source,
                    person_ref,
                    face_id,
                    display_name,
                    match_score,
                    distance,
                    det_score,
                    model_tag,
                    recording_segment_id,
                    offset_start_s,
                    offset_end_s,
                    gender,
                    age,
                    extra_json,
                ),
            )
        return eid

    def merge_or_insert_event(
        self,
        *,
        debounce_s: float,
        ts_utc: float,
        camera_id: str,
        source: str,
        person_ref: str,
        face_id: Optional[int],
        display_name: Optional[str],
        match_score: Optional[float],
        distance: Optional[float],
        det_score: Optional[float],
        model_tag: str,
        recording_segment_id: Optional[int],
        offset_start_s: Optional[float],
        offset_end_s: Optional[float],
        gender: Optional[int] = None,
        age: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        """
        Nếu person_ref != unknown và có bản ghi gần đây (cùng camera, cùng segment): cập nhật offset_end, ts_utc.
        Trả về (event_id, merged).
        """
        if debounce_s <= 0:
            eid = self.insert_event(
                ts_utc=ts_utc,
                camera_id=camera_id,
                source=source,
                person_ref=person_ref,
                face_id=face_id,
                display_name=display_name,
                match_score=match_score,
                distance=distance,
                det_score=det_score,
                model_tag=model_tag,
                recording_segment_id=recording_segment_id,
                offset_start_s=offset_start_s,
                offset_end_s=offset_end_s,
                gender=gender,
                age=age,
                extra=extra,
            )
            return eid, False

        with self._lock, self._conn() as c:
            cur = c.execute(
                """
                SELECT id, recording_segment_id, offset_start_s, offset_end_s, det_score, distance
                FROM recognition_events
                WHERE camera_id=? AND person_ref=? AND ts_utc >= ? AND ts_utc <= ?
                ORDER BY ts_utc DESC
                LIMIT 1
                """,
                (camera_id, person_ref, ts_utc - debounce_s, ts_utc + 0.001),
            )
            row = cur.fetchone()
            can_merge = False
            if row is not None:
                old_seg = row["recording_segment_id"]
                new_seg = recording_segment_id
                if old_seg is None and new_seg is None:
                    can_merge = True
                elif old_seg is not None and new_seg is not None and int(old_seg) == int(new_seg):
                    can_merge = True

            if row is not None and can_merge:
                eid = str(row["id"])
                old_end = row["offset_end_s"]
                old_det = row["det_score"]
                old_dist = row["distance"]
                new_end = offset_end_s if offset_end_s is not None else old_end
                if old_end is not None and new_end is not None:
                    new_end = max(float(old_end), float(new_end))
                elif new_end is None:
                    new_end = old_end

                best_det = old_det
                if det_score is not None:
                    if best_det is None or float(det_score) > float(best_det):
                        best_det = det_score

                best_dist = old_dist
                if distance is not None:
                    if best_dist is None or float(distance) < float(best_dist):
                        best_dist = distance

                best_gender = gender
                best_age = age
                extra_json = json.dumps(extra) if extra else None

                old_start = row["offset_start_s"]
                new_start = old_start
                if old_start is None and offset_start_s is not None:
                    new_start = offset_start_s
                seg_to_set = row["recording_segment_id"]
                if seg_to_set is None and recording_segment_id is not None:
                    seg_to_set = recording_segment_id

                c.execute(
                    """
                    UPDATE recognition_events SET
                      ts_utc=?,
                      offset_start_s=COALESCE(offset_start_s, ?),
                      offset_end_s=?,
                      recording_segment_id=COALESCE(recording_segment_id, ?),
                      det_score=?,
                      distance=?,
                      match_score=COALESCE(?, match_score),
                      display_name=COALESCE(?, display_name),
                      face_id=COALESCE(?, face_id),
                      gender=COALESCE(?, gender),
                      age=COALESCE(?, age),
                      extra_json=COALESCE(?, extra_json)
                    WHERE id=?
                    """,
                    (
                        ts_utc,
                        new_start,
                        new_end,
                        seg_to_set,
                        best_det,
                        best_dist,
                        match_score,
                        display_name,
                        face_id,
                        best_gender,
                        best_age,
                        extra_json,
                        eid,
                    ),
                )
                return eid, True

            eid = str(uuid.uuid4())
            extra_json = json.dumps(extra) if extra else None
            c.execute(
                """
                INSERT INTO recognition_events (
                  id, ts_utc, camera_id, source, person_ref, face_id, display_name,
                  match_score, distance, det_score, model_tag,
                  recording_segment_id, offset_start_s, offset_end_s,
                  gender, age, extra_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    eid,
                    ts_utc,
                    camera_id,
                    source,
                    person_ref,
                    face_id,
                    display_name,
                    match_score,
                    distance,
                    det_score,
                    model_tag,
                    recording_segment_id,
                    offset_start_s,
                    offset_end_s,
                    gender,
                    age,
                    extra_json,
                ),
            )
            return eid, False

    @staticmethod
    def _merge_weapon_extra(extra: Dict[str, Any], weapon: Optional[Dict[str, Any]]) -> None:
        if not weapon:
            return
        from identity_vm_app.services.weapon_track_status import classify_weapon_by_frame_count

        if bool(weapon.get("frame_armed")):
            extra["weapon_armed_frames"] = int(extra.get("weapon_armed_frames") or 0) + 1
        types = {str(t) for t in (extra.get("weapon_types") or []) if t}
        for t in weapon.get("weapon_types") or []:
            if t:
                types.add(str(t))
        extra["weapon_types"] = sorted(types)
        armed_frames = int(extra.get("weapon_armed_frames") or 0)
        cls = classify_weapon_by_frame_count(armed_frames, extra["weapon_types"])
        extra["armed"] = bool(cls["armed"])
        extra["dangerous"] = bool(cls["dangerous"])
        extra["weapon_status"] = cls["weapon_status"]
        extra["weapon_label"] = cls["weapon_label"]
        extra["weapon_frame_armed_last"] = bool(weapon.get("frame_armed"))
        score = float(weapon.get("weapon_score") or 0.0)
        extra["weapon_score_max"] = max(float(extra.get("weapon_score_max") or 0.0), score)
        pb = weapon.get("person_bbox")
        if pb and len(pb) >= 4:
            extra["person_bbox"] = [float(x) for x in pb[:4]]
        wlist = weapon.get("weapons") or []
        if wlist:
            extra["weapon_boxes"] = [
                {
                    "bbox": [int(w["bbox"][0]), int(w["bbox"][1]), int(w["bbox"][2]), int(w["bbox"][3])],
                    "class": str(w.get("class") or "weapon"),
                    "conf": float(w.get("conf") or w.get("fusion_score") or 0.0),
                }
                for w in wlist
                if w.get("bbox") and len(w.get("bbox")) >= 4
            ]

    def apply_tracking_update(
        self,
        event_id: str,
        *,
        merged: bool,
        det_score: Optional[float] = None,
        crop_jpeg_b64: Optional[str] = None,
        bbox: Optional[List[float]] = None,
        weapon: Optional[Dict[str, Any]] = None,
        weapon_crop_jpeg_b64: Optional[str] = None,
        weapon_crops_jpeg_b64: Optional[List[Dict[str, Any]]] = None,
        track_scene_crop_jpeg_b64: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Cập nhật frame_hits và ảnh crop (chọn ảnh det_score cao nhất khi đã có crop)."""
        with self._lock, self._conn() as c:
            cur = c.execute(
                "SELECT det_score, extra_json FROM recognition_events WHERE id=?",
                (event_id,),
            )
            row = cur.fetchone()
            if not row:
                return {"frame_hits": 0, "crop_path": None}
            extra: Dict[str, Any] = {}
            if row["extra_json"]:
                try:
                    extra = json.loads(str(row["extra_json"]))
                except json.JSONDecodeError:
                    extra = {}
            hits = int(extra.get("frame_hits") or 0)
            extra["frame_hits"] = hits + 1 if merged and hits > 0 else 1
            if bbox is not None:
                extra["bbox"] = bbox
            old_weapon_max = float(extra.get("weapon_score_max") or 0.0)
            self._merge_weapon_extra(extra, weapon)
            old_det = float(row["det_score"]) if row["det_score"] is not None else None
            crop_path = extra.get("crop_path")
            weapon_crop_path = extra.get("weapon_crop_path")
            track_scene_path = extra.get("track_scene_path")
            new_w_score = float((weapon or {}).get("weapon_score") or 0.0)
            replace_weapon_crop = bool(weapon and weapon.get("frame_armed")) and (
                not weapon_crop_path
                or (
                    bool(weapon.get("frame_armed"))
                    and new_w_score >= old_weapon_max
                )
            )
            crops_payload = list(weapon_crops_jpeg_b64 or [])
            if weapon_crop_jpeg_b64 and not crops_payload:
                crops_payload = [{"class": "weapon", "jpeg_b64": weapon_crop_jpeg_b64}]
            if crops_payload and replace_weapon_crop:
                saved_crops: List[Dict[str, str]] = []
                for item in crops_payload:
                    if not isinstance(item, dict):
                        continue
                    b64 = item.get("jpeg_b64") or item.get("crop_jpeg_b64")
                    if not b64:
                        continue
                    try:
                        from identity_vm_app.services.weapon_crops import normalize_weapon_class

                        cls = normalize_weapon_class(item.get("class"))
                        jpeg_w = decode_crop_b64(str(b64))
                        rel = save_weapon_crop_jpeg(event_id, jpeg_w, cls)
                        saved_crops.append({"class": cls, "path": rel})
                    except Exception:
                        continue
                if saved_crops:
                    extra["weapon_crops"] = saved_crops
                    weapon_crop_path = saved_crops[0]["path"]
                    extra["weapon_crop_path"] = weapon_crop_path
            replace_scene = bool(track_scene_crop_jpeg_b64) and (
                not track_scene_path
                or should_replace_crop(old_det, det_score, merged=bool(track_scene_path))
                or replace_weapon_crop
            )
            if track_scene_crop_jpeg_b64 and replace_scene:
                try:
                    jpeg_s = decode_crop_b64(track_scene_crop_jpeg_b64)
                    track_scene_path = save_track_scene_crop_jpeg(event_id, jpeg_s)
                    extra["track_scene_path"] = track_scene_path
                except Exception:
                    pass
            crop_path = extra.get("crop_path")
            if crop_jpeg_b64 and bool(s.IVM_EVENT_SAVE_CROPS):
                if should_replace_crop(old_det, det_score, merged=bool(crop_path)):
                    try:
                        jpeg = decode_crop_b64(crop_jpeg_b64)
                        crop_path = save_crop_jpeg(event_id, jpeg)
                        extra["crop_path"] = crop_path
                    except Exception:
                        pass
            c.execute(
                "UPDATE recognition_events SET extra_json=? WHERE id=?",
                (json.dumps(extra, ensure_ascii=False), event_id),
            )
        return {
            "frame_hits": int(extra.get("frame_hits", 1)),
            "crop_path": crop_path,
            "weapon_crop_path": weapon_crop_path,
            "track_scene_path": track_scene_path,
        }

    def list_camera_track_events_for_display_names(
        self,
        camera_id: str,
        from_ts: float,
        to_ts: float,
        display_names: Sequence[str],
        *,
        limit: int = 2000,
        order_asc: bool = True,
    ) -> List[Dict[str, Any]]:
        """Mọi event có display_name khớp (so sánh không phân biệt hoa thường)."""
        names = [str(n).strip() for n in display_names if str(n).strip()]
        if not names:
            return []
        all_tracks = self.list_camera_track_events(
            camera_id,
            from_ts,
            to_ts,
            limit=limit,
            known_only=True,
            order_asc=order_asc,
        )
        norm_targets = {n.casefold() for n in names}
        return [
            t
            for t in all_tracks
            if (t.get("display_name") or t.get("identity") or "").strip().casefold() in norm_targets
        ]

    def list_camera_track_events(
        self,
        camera_id: str,
        from_ts: float,
        to_ts: float,
        *,
        limit: int = 300,
        known_only: bool = False,
        person_ref: Optional[str] = None,
        order_asc: bool = False,
    ) -> List[Dict[str, Any]]:
        """Danh sách track (mỗi event = một lần xuất hiện / phiên debounce)."""
        lim = max(1, min(int(limit), 2000))
        clauses = ["camera_id=?", "ts_utc >= ?", "ts_utc <= ?"]
        args: List[Any] = [camera_id, from_ts, to_ts]
        if known_only:
            clauses.append("person_ref != 'unknown'")
        if person_ref is not None:
            clauses.append("person_ref=?")
            args.append(person_ref)
        order = "ASC" if order_asc else "DESC"
        where = " AND ".join(clauses)
        q = f"""
            SELECT id, ts_utc, person_ref, face_id, display_name, distance, det_score,
                   gender, age, extra_json, recording_segment_id, offset_start_s, offset_end_s
            FROM recognition_events
            WHERE {where}
            ORDER BY ts_utc {order}
            LIMIT ?
        """
        args.append(lim)
        with self._conn() as c:
            cur = c.execute(q, tuple(args))
            out: List[Dict[str, Any]] = []
            for r in cur.fetchall():
                extra: Dict[str, Any] = {}
                if r["extra_json"]:
                    try:
                        extra = json.loads(str(r["extra_json"]))
                    except json.JSONDecodeError:
                        extra = {}
                pref = str(r["person_ref"])
                dname = r["display_name"]
                if pref == "unknown" or not dname:
                    identity = "Unknown"
                else:
                    identity = str(dname)
                out.append(
                    {
                        "event_id": str(r["id"]),
                        "ts_utc": float(r["ts_utc"]),
                        "person_ref": pref,
                        "face_id": int(r["face_id"]) if r["face_id"] is not None else None,
                        "display_name": dname,
                        "identity": identity,
                        "known": pref != "unknown",
                        "distance": float(r["distance"]) if r["distance"] is not None else None,
                        "det_score": float(r["det_score"]) if r["det_score"] is not None else None,
                        "frame_hits": int(extra.get("frame_hits") or 1),
                        "crop_path": extra.get("crop_path"),
                        "bbox": extra.get("bbox"),
                        "armed": bool(extra.get("armed")),
                        "dangerous": bool(extra.get("dangerous")),
                        "weapon_armed_frames": int(extra.get("weapon_armed_frames") or 0),
                        "weapon_types": list(extra.get("weapon_types") or []),
                        "weapon_status": extra.get("weapon_status") or "an_toan",
                        "weapon_label": extra.get("weapon_label") or "Không vũ khí",
                        "weapon_score_max": float(extra.get("weapon_score_max") or 0.0),
                        "weapon_crop_path": extra.get("weapon_crop_path"),
                        "weapon_crops": list(extra.get("weapon_crops") or []),
                        "track_scene_path": extra.get("track_scene_path"),
                        "gender": int(r["gender"]) if r["gender"] is not None else None,
                        "age": int(r["age"]) if r["age"] is not None else None,
                        "recording_segment_id": int(r["recording_segment_id"])
                        if r["recording_segment_id"] is not None
                        else None,
                        "offset_start_s": float(r["offset_start_s"])
                        if r["offset_start_s"] is not None
                        else None,
                        "offset_end_s": float(r["offset_end_s"])
                        if r["offset_end_s"] is not None
                        else None,
                        "can_export_cut": bool(
                            r["recording_segment_id"] is not None
                            or r["offset_start_s"] is not None
                        ),
                    }
                )
        return out

    def get_event_extra(self, event_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            cur = c.execute("SELECT extra_json FROM recognition_events WHERE id=?", (event_id,))
            r = cur.fetchone()
            if not r or not r["extra_json"]:
                return None
            try:
                return json.loads(str(r["extra_json"]))
            except json.JSONDecodeError:
                return None

    def insert_person_media(
        self,
        *,
        face_id: int,
        display_name: Optional[str],
        media_type: str,
        path: str,
        created_at_utc: float,
    ) -> str:
        mid = str(uuid.uuid4())
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT INTO person_media (id, face_id, display_name, media_type, path, created_at_utc)
                VALUES (?,?,?,?,?,?)
                """,
                (mid, face_id, display_name, media_type, path, created_at_utc),
            )
        return mid

    def list_person_media(self, face_id: int) -> List[Dict[str, Any]]:
        with self._conn() as c:
            cur = c.execute(
                "SELECT * FROM person_media WHERE face_id=? ORDER BY created_at_utc DESC",
                (face_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_person_media(self, media_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            cur = c.execute("SELECT * FROM person_media WHERE id=?", (media_id,))
            r = cur.fetchone()
            return dict(r) if r else None

    def delete_person_media(self, media_id: str) -> bool:
        with self._lock, self._conn() as c:
            cur = c.execute("DELETE FROM person_media WHERE id=?", (media_id,))
            return cur.rowcount > 0

    def insert_registration_failure(
        self,
        *,
        original_filename: Optional[str],
        error_message: str,
        raw_bytes: Optional[bytes],
    ) -> str:
        eid = str(uuid.uuid4())
        ts = time.time()
        img_path: Optional[str] = None
        if raw_bytes:
            d = Path(s.IVM_DATA_DIR) / "registration_errors"
            d.mkdir(parents=True, exist_ok=True)
            suf = Path(original_filename or "").suffix.lower()
            if suf not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                suf = ".jpg"
            p = d / f"{eid}{suf}"
            p.write_bytes(raw_bytes)
            img_path = str(p.resolve())
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT INTO registration_failures (id, ts_utc, original_filename, error_message, image_path)
                VALUES (?,?,?,?,?)
                """,
                (eid, ts, original_filename, error_message, img_path),
            )
        return eid

    def bulk_checkpoint_get(self, path_key: str) -> Optional[str]:
        with self._conn() as c:
            r = c.execute(
                "SELECT status FROM bulk_register_checkpoint WHERE path_key=?",
                (path_key,),
            ).fetchone()
            return str(r["status"]) if r else None

    def bulk_checkpoint_lookup_many(
        self, path_keys: List[str], *, chunk_size: Optional[int] = None
    ) -> Dict[str, str]:
        """path_key→status chỉ cho key đã có bản ghi; key không có = chưa checkpoint."""
        out: Dict[str, str] = {}
        if not path_keys:
            return out
        k = chunk_size if chunk_size is not None else int(getattr(s, "IVM_BULK_CHECKPOINT_LOOKUP_CHUNK", 900))
        k = max(1, min(int(k), 900))
        with self._conn() as c:
            for i in range(0, len(path_keys), k):
                chunk = path_keys[i : i + k]
                placeholders = ",".join("?" * len(chunk))
                cur = c.execute(
                    f"""
                    SELECT path_key, status
                    FROM bulk_register_checkpoint
                    WHERE path_key IN ({placeholders})
                    """,
                    chunk,
                )
                for row in cur:
                    out[str(row["path_key"])] = str(row["status"])
        return out

    def bulk_checkpoint_set(self, path_key: str, status: str) -> None:
        ts = time.time()
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT INTO bulk_register_checkpoint (path_key, status, ts_utc)
                VALUES (?,?,?)
                ON CONFLICT(path_key) DO UPDATE SET
                  status = excluded.status,
                  ts_utc = excluded.ts_utc
                """,
                (path_key, status, ts),
            )

    def bulk_checkpoint_clear(self) -> int:
        with self._lock, self._conn() as c:
            return int(c.execute("DELETE FROM bulk_register_checkpoint").rowcount or 0)

    def list_registration_failures(self, limit: int = 100) -> List[Dict[str, Any]]:
        lim = max(1, min(int(limit), 500))
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT id, ts_utc, original_filename, error_message, image_path
                FROM registration_failures
                ORDER BY ts_utc DESC
                LIMIT ?
                """,
                (lim,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_registration_failure(self, failure_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            cur = c.execute("SELECT * FROM registration_failures WHERE id=?", (failure_id,))
            r = cur.fetchone()
            return dict(r) if r else None

    def delete_registration_failure(self, failure_id: str) -> bool:
        row = self.get_registration_failure(failure_id)
        if not row:
            return False
        p = row.get("image_path")
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM registration_failures WHERE id=?", (failure_id,))
        try:
            if p:
                fp = Path(str(p)).resolve()
                root = Path(s.IVM_DATA_DIR).resolve()
                try:
                    fp.relative_to(root)
                except ValueError:
                    pass
                else:
                    if fp.is_file():
                        fp.unlink(missing_ok=True)
        except OSError:
            pass
        return True

    def clear_recognition_reports(
        self,
        *,
        camera_id: Optional[str] = None,
        clear_segments: bool = False,
    ) -> Dict[str, Any]:
        """Xóa sự kiện nhận diện (báo cáo). Tuỳ chọn xóa recording_segments."""
        with self._lock, self._conn() as c:
            if camera_id:
                event_ids = [
                    str(r[0])
                    for r in c.execute(
                        "SELECT id FROM recognition_events WHERE camera_id=?",
                        (camera_id,),
                    ).fetchall()
                ]
                ce = c.execute(
                    "DELETE FROM recognition_events WHERE camera_id=?",
                    (camera_id,),
                ).rowcount
                cs = 0
                if clear_segments:
                    cs = c.execute(
                        "DELETE FROM recording_segments WHERE camera_id=?",
                        (camera_id,),
                    ).rowcount
            else:
                event_ids = [
                    str(r[0])
                    for r in c.execute("SELECT id FROM recognition_events").fetchall()
                ]
                ce = c.execute("DELETE FROM recognition_events").rowcount
                cs = 0
                if clear_segments:
                    cs = c.execute("DELETE FROM recording_segments").rowcount
        return {
            "recognition_events_deleted": int(ce),
            "recording_segments_deleted": int(cs),
            "event_ids": event_ids,
        }

    def clear_all_tables(self) -> Dict[str, int]:
        """Xóa hết bản ghi ứng dụng (SQLite). Thứ tự FK: events trước segments."""
        with self._lock, self._conn() as c:
            ce = c.execute("DELETE FROM recognition_events").rowcount
            cs = c.execute("DELETE FROM recording_segments").rowcount
            cp = c.execute("DELETE FROM person_media").rowcount
            cr = c.execute("DELETE FROM registration_failures").rowcount
            cb = c.execute("DELETE FROM bulk_register_checkpoint").rowcount
        return {
            "recognition_events_deleted": int(ce),
            "recording_segments_deleted": int(cs),
            "person_media_deleted": int(cp),
            "registration_failures_deleted": int(cr),
            "bulk_register_checkpoint_deleted": int(cb),
        }

    def camera_report_summary(
        self, camera_id: str, from_ts: float, to_ts: float
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT
                  COUNT(*) AS total_events,
                  COALESCE(SUM(CASE WHEN person_ref != 'unknown' THEN 1 ELSE 0 END), 0) AS known_events,
                  COALESCE(SUM(CASE WHEN person_ref = 'unknown' THEN 1 ELSE 0 END), 0) AS unknown_events,
                  COUNT(DISTINCT CASE WHEN person_ref != 'unknown' THEN person_ref END) AS distinct_known,
                  AVG(CASE WHEN distance IS NOT NULL THEN distance END) AS avg_distance
                FROM recognition_events
                WHERE camera_id=? AND ts_utc >= ? AND ts_utc <= ?
                """,
                (camera_id, from_ts, to_ts),
            )
            agg = cur.fetchone()
            summary = {
                "camera_id": camera_id,
                "from_ts": from_ts,
                "to_ts": to_ts,
                "total_events": int(agg["total_events"] or 0),
                "known_events": int(agg["known_events"] or 0),
                "unknown_events": int(agg["unknown_events"] or 0),
                "distinct_known_persons": int(agg["distinct_known"] or 0),
                "avg_distance": float(agg["avg_distance"]) if agg["avg_distance"] is not None else None,
            }

            cur2 = c.execute(
                """
                SELECT
                  person_ref,
                  MAX(display_name) AS display_name,
                  MAX(face_id) AS face_id,
                  COUNT(*) AS appearances_count,
                  MIN(ts_utc) AS first_seen,
                  MAX(ts_utc) AS last_seen,
                  AVG(distance) AS avg_distance,
                  MAX(gender) AS last_gender,
                  MAX(age) AS last_age
                FROM recognition_events
                WHERE camera_id=? AND ts_utc >= ? AND ts_utc <= ?
                GROUP BY person_ref
                ORDER BY appearances_count DESC
                """,
                (camera_id, from_ts, to_ts),
            )
            subjects: List[Dict[str, Any]] = []
            for r in cur2.fetchall():
                subjects.append(
                    {
                        "person_ref": str(r["person_ref"]),
                        "display_name": r["display_name"],
                        "face_id": int(r["face_id"]) if r["face_id"] is not None else None,
                        "appearances_count": int(r["appearances_count"]),
                        "first_seen": float(r["first_seen"]),
                        "last_seen": float(r["last_seen"]),
                        "avg_distance": float(r["avg_distance"]) if r["avg_distance"] is not None else None,
                        "last_gender": int(r["last_gender"]) if r["last_gender"] is not None else None,
                        "last_age": int(r["last_age"]) if r["last_age"] is not None else None,
                        "known": str(r["person_ref"]) != "unknown",
                    }
                )
        return summary, subjects

    def get_event(self, event_id: str) -> Optional[RecognitionEventRow]:
        with self._conn() as c:
            cur = c.execute("SELECT * FROM recognition_events WHERE id=?", (event_id,))
            r = cur.fetchone()
            if not r:
                return None
            return self._row_to_event(r)

    def list_appearances(
        self,
        *,
        person_ref: Optional[str] = None,
        camera_id: Optional[str] = None,
        from_ts: Optional[float] = None,
        to_ts: Optional[float] = None,
        limit: int = 500,
    ) -> List[RecognitionEventRow]:
        clauses: List[str] = ["1=1"]
        args: List[Any] = []
        if person_ref is not None:
            clauses.append("person_ref=?")
            args.append(person_ref)
        if camera_id is not None:
            clauses.append("camera_id=?")
            args.append(camera_id)
        if from_ts is not None:
            clauses.append("ts_utc >= ?")
            args.append(from_ts)
        if to_ts is not None:
            clauses.append("ts_utc <= ?")
            args.append(to_ts)
        where = " AND ".join(clauses)
        q = f"SELECT * FROM recognition_events WHERE {where} ORDER BY ts_utc DESC LIMIT ?"
        args.append(int(limit))
        with self._conn() as c:
            cur = c.execute(q, tuple(args))
            return [self._row_to_event(r) for r in cur.fetchall()]

    @staticmethod
    def _row_to_event(r: sqlite3.Row) -> RecognitionEventRow:
        return RecognitionEventRow(
            id=str(r["id"]),
            ts_utc=float(r["ts_utc"]),
            camera_id=str(r["camera_id"]),
            source=str(r["source"]),
            person_ref=str(r["person_ref"]),
            face_id=int(r["face_id"]) if r["face_id"] is not None else None,
            display_name=str(r["display_name"]) if r["display_name"] is not None else None,
            match_score=float(r["match_score"]) if r["match_score"] is not None else None,
            distance=float(r["distance"]) if r["distance"] is not None else None,
            det_score=float(r["det_score"]) if r["det_score"] is not None else None,
            model_tag=str(r["model_tag"]),
            recording_segment_id=int(r["recording_segment_id"]) if r["recording_segment_id"] is not None else None,
            offset_start_s=float(r["offset_start_s"]) if r["offset_start_s"] is not None else None,
            offset_end_s=float(r["offset_end_s"]) if r["offset_end_s"] is not None else None,
            gender=int(r["gender"]) if r["gender"] is not None else None,
            age=int(r["age"]) if r["age"] is not None else None,
        )
