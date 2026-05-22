"""SQLite: video upload jobs + báo cáo từng khung (giống VideoMaster person_reports)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from identity_vm_app import settings as s

# Trạng thái job (giống VIDEO_ANALYZE_STATUS VideoMaster)
VA_STATUS_PENDING = 0
VA_STATUS_PROCESSING = 1
VA_STATUS_COMPLETED = 2
VA_STATUS_ERROR = 3


class VideoAnalyzeStore:
    def __init__(self, path: Optional[Path] = None) -> None:
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
                CREATE TABLE IF NOT EXISTS video_analyze_jobs (
                    id TEXT PRIMARY KEY,
                    original_name TEXT NOT NULL,
                    video_path TEXT NOT NULL,
                    thumb_path TEXT,
                    duration_s REAL,
                    fps REAL,
                    width INTEGER,
                    height INTEGER,
                    total_frames INTEGER,
                    analyze_fps REAL,
                    sample_fps REAL,
                    feature_analyze_json TEXT,
                    status INTEGER NOT NULL DEFAULT 0,
                    index_frame INTEGER DEFAULT 0,
                    total_sample_frames INTEGER DEFAULT 0,
                    time_upload_utc REAL NOT NULL,
                    total_time_analyze_s REAL,
                    message TEXT,
                    error_code TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_vaj_status ON video_analyze_jobs(status);
                CREATE INDEX IF NOT EXISTS idx_vaj_upload ON video_analyze_jobs(time_upload_utc);

                CREATE TABLE IF NOT EXISTS video_person_reports (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    time_analyze_s REAL NOT NULL,
                    frame_index INTEGER,
                    sample_index INTEGER,
                    img_url TEXT,
                    id_tracking INTEGER NOT NULL,
                    video_clip INTEGER NOT NULL DEFAULT 1,
                    face_id INTEGER,
                    display_name TEXT,
                    distance REAL,
                    match_score REAL,
                    det_score REAL,
                    gender INTEGER,
                    age INTEGER,
                    box_face TEXT,
                    face_img TEXT,
                    box_person TEXT,
                    person_img TEXT,
                    features_face TEXT,
                    armed INTEGER DEFAULT 0,
                    weapon_status TEXT,
                    weapon_label TEXT,
                    weapon_types_json TEXT,
                    weapon_score REAL,
                    FOREIGN KEY(job_id) REFERENCES video_analyze_jobs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_vpr_job ON video_person_reports(job_id);
                CREATE INDEX IF NOT EXISTS idx_vpr_job_time ON video_person_reports(job_id, time_analyze_s);
                CREATE INDEX IF NOT EXISTS idx_vpr_job_track ON video_person_reports(job_id, id_tracking);

                CREATE TABLE IF NOT EXISTS video_weapon_reports (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    time_analyze_s REAL NOT NULL,
                    frame_index INTEGER,
                    sample_index INTEGER,
                    img_url TEXT,
                    id_tracking INTEGER NOT NULL,
                    video_clip INTEGER NOT NULL DEFAULT 1,
                    person_bbox TEXT,
                    image_status TEXT,
                    armed INTEGER DEFAULT 0,
                    weapon_types_json TEXT,
                    weapon_score REAL,
                    scene_crop_path TEXT,
                    FOREIGN KEY(job_id) REFERENCES video_analyze_jobs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_vwr_job ON video_weapon_reports(job_id);
                CREATE INDEX IF NOT EXISTS idx_vwr_job_track ON video_weapon_reports(job_id, id_tracking);
                """
            )
            cols = {row[1] for row in c.execute("PRAGMA table_info(video_analyze_jobs)").fetchall()}
            if "display_name" not in cols:
                c.execute("ALTER TABLE video_analyze_jobs ADD COLUMN display_name TEXT")
            cols = {row[1] for row in c.execute("PRAGMA table_info(video_analyze_jobs)").fetchall()}
            for col, typedef in (
                ("camera_id", "TEXT"),
                ("source_type", "TEXT"),
                ("session_start_utc", "REAL"),
                ("session_end_utc", "REAL"),
            ):
                if col not in cols:
                    c.execute(f"ALTER TABLE video_analyze_jobs ADD COLUMN {col} {typedef}")
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_vaj_camera_time "
                "ON video_analyze_jobs(camera_id, session_start_utc)"
            )
            c.execute(
                "CREATE INDEX IF NOT EXISTS idx_vaj_source "
                "ON video_analyze_jobs(source_type, status)"
            )
            c.execute(
                "UPDATE video_analyze_jobs SET source_type='upload' "
                "WHERE source_type IS NULL OR TRIM(source_type)=''"
            )
            pr_cols = {
                row[1] for row in c.execute("PRAGMA table_info(video_person_reports)").fetchall()
            }
            if "match_candidates_json" not in pr_cols:
                c.execute(
                    "ALTER TABLE video_person_reports ADD COLUMN match_candidates_json TEXT"
                )
            for col, typedef in (
                ("mask", "INTEGER"),
                ("sleeve_length", "TEXT"),
                ("type_of_lower_body_clothing", "TEXT"),
                ("length_of_lower_body_clothing", "TEXT"),
                ("carrying_handbag", "TEXT"),
                ("wearing_hat", "TEXT"),
                ("color", "TEXT"),
                ("color_tag", "TEXT"),
                ("features_person", "TEXT"),
            ):
                if col not in pr_cols:
                    c.execute(f"ALTER TABLE video_person_reports ADD COLUMN {col} {typedef}")
            for col, typedef in (
                ("weapon_img", "TEXT"),
                ("weapon_boxes_json", "TEXT"),
                ("weapon_crops_json", "TEXT"),
            ):
                if col not in pr_cols:
                    c.execute(f"ALTER TABLE video_person_reports ADD COLUMN {col} {typedef}")

    @staticmethod
    def job_title(job: Dict[str, Any]) -> str:
        dn = (job.get("display_name") or "").strip()
        if dn:
            return dn
        return str(job.get("original_name") or job.get("id") or "?")

    def insert_camera_live_job(
        self,
        job_id: str,
        *,
        camera_id: str,
        video_path: str,
        display_name: str,
        sample_fps: float,
        feature_analyze: Dict[str, Any],
        session_start_utc: float,
    ) -> None:
        original_name = f"live-{camera_id}"
        now = float(session_start_utc)
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT INTO video_analyze_jobs (
                    id, original_name, display_name, video_path, thumb_path,
                    feature_analyze_json, status, sample_fps, time_upload_utc,
                    camera_id, source_type, session_start_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    original_name,
                    display_name,
                    video_path,
                    None,
                    json.dumps(feature_analyze, ensure_ascii=False),
                    VA_STATUS_PROCESSING,
                    float(sample_fps),
                    now,
                    str(camera_id),
                    "camera_live",
                    now,
                ),
            )

    def finalize_camera_live_job(
        self,
        job_id: str,
        *,
        video_path: str,
        thumb_path: Optional[str],
        duration_s: float,
        analyze_fps: float,
        total_sample_frames: int,
        session_end_utc: float,
        width: int = 0,
        height: int = 0,
    ) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """
                UPDATE video_analyze_jobs SET
                    video_path=?, thumb_path=?, duration_s=?, analyze_fps=?,
                    total_sample_frames=?, session_end_utc=?, status=?,
                    width=COALESCE(NULLIF(?, 0), width),
                    height=COALESCE(NULLIF(?, 0), height),
                    total_frames=?
                WHERE id=?
                """,
                (
                    video_path,
                    thumb_path,
                    float(duration_s),
                    float(analyze_fps),
                    int(total_sample_frames),
                    float(session_end_utc),
                    VA_STATUS_COMPLETED,
                    int(width),
                    int(height),
                    int(total_sample_frames),
                    job_id,
                ),
            )

    def list_camera_live_sessions(
        self,
        camera_id: str,
        *,
        from_ts: float = 0.0,
        to_ts: float = 0.0,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses = ["camera_id=?", "source_type='camera_live'"]
        params: List[Any] = [str(camera_id)]
        if from_ts > 0:
            clauses.append("session_start_utc >= ?")
            params.append(float(from_ts))
        if to_ts > 0:
            clauses.append("session_start_utc <= ?")
            params.append(float(to_ts))
        params.append(max(1, min(500, int(limit))))
        sql = f"""
            SELECT * FROM video_analyze_jobs
            WHERE {' AND '.join(clauses)}
            ORDER BY session_start_utc DESC
            LIMIT ?
        """
        with self._conn() as c:
            cur = c.execute(sql, params)
            return [self._job_row_to_dict(r) for r in cur.fetchall()]

    def cleanup_stuck_camera_live_jobs(self) -> int:
        """Job camera_live còn PROCESSING sau restart — đánh dấu ERROR."""
        with self._lock, self._conn() as c:
            cur = c.execute(
                """
                UPDATE video_analyze_jobs SET
                    status=?, message=?, error_code=?, session_end_utc=COALESCE(session_end_utc, ?)
                WHERE source_type='camera_live' AND status=?
                """,
                (
                    VA_STATUS_ERROR,
                    "Phiên bị gián đoạn (restart API)",
                    "session_interrupted",
                    time.time(),
                    VA_STATUS_PROCESSING,
                ),
            )
            return int(cur.rowcount)

    def insert_job(
        self,
        job_id: str,
        *,
        original_name: str,
        display_name: str,
        video_path: str,
        thumb_path: Optional[str],
        feature_analyze: Dict[str, Any],
        sample_fps: float,
    ) -> None:
        now = time.time()
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT INTO video_analyze_jobs (
                    id, original_name, display_name, video_path, thumb_path,
                    feature_analyze_json, status, sample_fps, time_upload_utc,
                    source_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    original_name,
                    display_name,
                    video_path,
                    thumb_path,
                    json.dumps(feature_analyze, ensure_ascii=False),
                    VA_STATUS_PENDING,
                    float(sample_fps),
                    now,
                    "upload",
                ),
            )

    def update_job_display_name(self, job_id: str, display_name: str) -> bool:
        name = (display_name or "").strip()
        if not name:
            return False
        with self._lock, self._conn() as c:
            cur = c.execute(
                "UPDATE video_analyze_jobs SET display_name=? WHERE id=?",
                (name, job_id),
            )
            return cur.rowcount > 0

    def update_job_meta(
        self,
        job_id: str,
        *,
        duration_s: float,
        fps: float,
        width: int,
        height: int,
        total_frames: int,
        analyze_fps: float,
        total_sample_frames: int,
    ) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """
                UPDATE video_analyze_jobs SET
                    duration_s=?, fps=?, width=?, height=?,
                    total_frames=?, analyze_fps=?, total_sample_frames=?
                WHERE id=?
                """,
                (duration_s, fps, width, height, total_frames, analyze_fps, total_sample_frames, job_id),
            )

    def update_job_total_sample_frames(self, job_id: str, total_sample_frames: int) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE video_analyze_jobs SET total_sample_frames=? WHERE id=?",
                (int(total_sample_frames), job_id),
            )

    def update_job_thumb(self, job_id: str, thumb_path: str) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE video_analyze_jobs SET thumb_path=? WHERE id=?",
                (thumb_path, job_id),
            )

    def update_job_progress(self, job_id: str, index_frame: int) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE video_analyze_jobs SET index_frame=? WHERE id=?",
                (int(index_frame), job_id),
            )

    def update_job_status(
        self,
        job_id: str,
        status: int,
        *,
        message: Optional[str] = None,
        error_code: Optional[str] = None,
        total_time_analyze_s: Optional[float] = None,
    ) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                """
                UPDATE video_analyze_jobs SET
                    status=?, message=?, error_code=?, total_time_analyze_s=COALESCE(?, total_time_analyze_s)
                WHERE id=?
                """,
                (status, message, error_code, total_time_analyze_s, job_id),
            )

    def insert_person_report(self, row: Dict[str, Any]) -> str:
        rid = str(row.get("id") or uuid.uuid4())
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT INTO video_person_reports (
                    id, job_id, time_analyze_s, frame_index, sample_index, img_url,
                    id_tracking, video_clip, face_id, display_name, distance, match_score,
                    match_candidates_json,
                    det_score, gender, age, box_face, face_img, box_person, person_img,
                    features_face, armed, weapon_status, weapon_label, weapon_types_json, weapon_score,
                    weapon_img, weapon_boxes_json, weapon_crops_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    rid,
                    row["job_id"],
                    float(row["time_analyze_s"]),
                    row.get("frame_index"),
                    row.get("sample_index"),
                    row.get("img_url"),
                    int(row["id_tracking"]),
                    int(row.get("video_clip") or 1),
                    row.get("face_id"),
                    row.get("display_name"),
                    row.get("distance"),
                    row.get("match_score"),
                    row.get("match_candidates_json"),
                    row.get("det_score"),
                    row.get("gender"),
                    row.get("age"),
                    row.get("box_face"),
                    row.get("face_img"),
                    row.get("box_person"),
                    row.get("person_img"),
                    row.get("features_face"),
                    int(row.get("armed") or 0),
                    row.get("weapon_status"),
                    row.get("weapon_label"),
                    row.get("weapon_types_json"),
                    row.get("weapon_score"),
                    row.get("weapon_img"),
                    row.get("weapon_boxes_json"),
                    row.get("weapon_crops_json"),
                ),
            )
        return rid

    def insert_person_reports_batch(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        sql = """
            INSERT INTO video_person_reports (
                id, job_id, time_analyze_s, frame_index, sample_index, img_url,
                id_tracking, video_clip, face_id, display_name, distance, match_score,
                match_candidates_json,
                det_score, gender, age, box_face, face_img, box_person, person_img,
                features_face, armed, weapon_status, weapon_label, weapon_types_json, weapon_score,
                weapon_img, weapon_boxes_json, weapon_crops_json
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        params: List[tuple] = []
        for row in rows:
            rid = str(row.get("id") or uuid.uuid4())
            params.append(
                (
                    rid,
                    row["job_id"],
                    float(row["time_analyze_s"]),
                    row.get("frame_index"),
                    row.get("sample_index"),
                    row.get("img_url"),
                    int(row["id_tracking"]),
                    int(row.get("video_clip") or 1),
                    row.get("face_id"),
                    row.get("display_name"),
                    row.get("distance"),
                    row.get("match_score"),
                    row.get("match_candidates_json"),
                    row.get("det_score"),
                    row.get("gender"),
                    row.get("age"),
                    row.get("box_face"),
                    row.get("face_img"),
                    row.get("box_person"),
                    row.get("person_img"),
                    row.get("features_face"),
                    int(row.get("armed") or 0),
                    row.get("weapon_status"),
                    row.get("weapon_label"),
                    row.get("weapon_types_json"),
                    row.get("weapon_score"),
                    row.get("weapon_img"),
                    row.get("weapon_boxes_json"),
                    row.get("weapon_crops_json"),
                )
            )
        with self._lock, self._conn() as c:
            c.executemany(sql, params)
        return len(params)

    def insert_weapon_report(self, row: Dict[str, Any]) -> str:
        rid = str(row.get("id") or uuid.uuid4())
        with self._lock, self._conn() as c:
            c.execute(
                """
                INSERT INTO video_weapon_reports (
                    id, job_id, time_analyze_s, frame_index, sample_index, img_url,
                    id_tracking, video_clip, person_bbox, image_status, armed,
                    weapon_types_json, weapon_score, scene_crop_path
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    rid,
                    row["job_id"],
                    float(row["time_analyze_s"]),
                    row.get("frame_index"),
                    row.get("sample_index"),
                    row.get("img_url"),
                    int(row["id_tracking"]),
                    int(row.get("video_clip") or 1),
                    row.get("person_bbox"),
                    row.get("image_status"),
                    int(row.get("armed") or 0),
                    row.get("weapon_types_json"),
                    row.get("weapon_score"),
                    row.get("scene_crop_path"),
                ),
            )
        return rid

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            cur = c.execute("SELECT * FROM video_analyze_jobs WHERE id=?", (job_id,))
            r = cur.fetchone()
            if not r:
                return None
            return self._job_row_to_dict(r)

    def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT * FROM video_analyze_jobs
                ORDER BY time_upload_utc DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            )
            return [self._job_row_to_dict(r) for r in cur.fetchall()]

    def count_reports(self, job_id: str) -> Dict[str, int]:
        with self._conn() as c:
            p = c.execute(
                "SELECT COUNT(*) FROM video_person_reports WHERE job_id=?", (job_id,)
            ).fetchone()[0]
            w = c.execute(
                "SELECT COUNT(*) FROM video_weapon_reports WHERE job_id=?", (job_id,)
            ).fetchone()[0]
            return {"person_reports": int(p), "weapon_reports": int(w)}

    def list_person_reports(
        self,
        job_id: str,
        *,
        start_time_s: float = 0.0,
        end_time_s: float = 0.0,
        gender: Optional[int] = None,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        clauses = ["job_id=?"]
        params: List[Any] = [job_id]
        if start_time_s > 0:
            clauses.append("time_analyze_s >= ?")
            params.append(start_time_s)
        if end_time_s > 0:
            clauses.append("time_analyze_s <= ?")
            params.append(end_time_s)
        if gender is not None:
            clauses.append("gender=?")
            params.append(gender)
        params.append(max(1, min(20000, int(limit))))
        sql = f"""
            SELECT * FROM video_person_reports
            WHERE {' AND '.join(clauses)}
            ORDER BY time_analyze_s ASC, id ASC
            LIMIT ?
        """
        with self._conn() as c:
            cur = c.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def get_person_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            cur = c.execute("SELECT * FROM video_person_reports WHERE id=?", (report_id,))
            r = cur.fetchone()
            return dict(r) if r else None

    def list_person_reports_by_tracking(
        self,
        job_id: str,
        id_tracking: int,
        *,
        start_time_s: float = 0.0,
        end_time_s: float = 0.0,
        limit: int = 5000,
        use_vm_time_filter: bool = True,
    ) -> List[Dict[str, Any]]:
        clauses = ["r.job_id=?", "r.id_tracking=?"]
        params: List[Any] = [job_id, int(id_tracking)]
        if use_vm_time_filter:
            t_clauses, t_params = self._vm_time_filter_clauses(
                float(start_time_s), float(end_time_s)
            )
            clauses.extend(t_clauses)
            params.extend(t_params)
        else:
            if start_time_s > 0:
                clauses.append("r.time_analyze_s >= ?")
                params.append(start_time_s)
            if end_time_s > 0:
                clauses.append("r.time_analyze_s <= ?")
                params.append(end_time_s)
        params.append(max(1, min(20000, int(limit))))
        sql = f"""
            SELECT r.*, j.time_upload_utc AS time_video
            FROM video_person_reports r
            INNER JOIN video_analyze_jobs j ON j.id = r.job_id
            WHERE {' AND '.join(clauses)}
            ORDER BY r.time_analyze_s ASC, r.id ASC
            LIMIT ?
        """
        with self._conn() as c:
            cur = c.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    @staticmethod
    def _vm_time_filter_clauses(
        start_time: float,
        end_time: float,
        *,
        time_col: str = "r.time_analyze_s",
        job_alias: str = "j",
    ) -> tuple[List[str], List[Any]]:
        """VideoMaster: time_analyze + Video.time_video trong khoảng [start, end]."""
        clauses: List[str] = []
        params: List[Any] = []
        eff = f"({time_col} + COALESCE({job_alias}.time_upload_utc, 0))"
        if start_time > 0:
            clauses.append(f"{eff} >= ?")
            params.append(float(start_time))
        if end_time > 0:
            clauses.append(f"{eff} <= ?")
            params.append(float(end_time))
        return clauses, params

    def list_faces_person_reports(
        self,
        job_ids: List[str],
        *,
        start_time_s: float = 0.0,
        end_time_s: float = 0.0,
        gender: Optional[int] = None,
        start_age: int = 0,
        end_age: int = 1000,
        limit: int = 50000,
        use_vm_time_filter: bool = True,
    ) -> List[Dict[str, Any]]:
        if not job_ids:
            return []
        placeholders = ",".join("?" * len(job_ids))
        clauses = [
            f"r.job_id IN ({placeholders})",
            "r.features_face IS NOT NULL",
            "TRIM(r.features_face) != ''",
            "r.box_face IS NOT NULL",
            "TRIM(r.box_face) != ''",
            "LOWER(TRIM(r.box_face)) != 'none'",
        ]
        params: List[Any] = list(job_ids)
        if use_vm_time_filter:
            t_clauses, t_params = self._vm_time_filter_clauses(
                float(start_time_s), float(end_time_s)
            )
            clauses.extend(t_clauses)
            params.extend(t_params)
        else:
            if start_time_s > 0:
                clauses.append("r.time_analyze_s >= ?")
                params.append(start_time_s)
            if end_time_s > 0:
                clauses.append("r.time_analyze_s <= ?")
                params.append(end_time_s)
        if gender is not None:
            clauses.append("r.gender=?")
            params.append(gender)
        if start_age > 0:
            clauses.append("(r.age IS NULL OR r.age >= ?)")
            params.append(int(start_age))
        if end_age < 1000:
            clauses.append("(r.age IS NULL OR r.age <= ?)")
            params.append(int(end_age))
        params.append(max(1, min(50000, int(limit))))
        sql = f"""
            SELECT r.*, j.time_upload_utc AS time_video, j.display_name AS job_display_name
            FROM video_person_reports r
            INNER JOIN video_analyze_jobs j ON j.id = r.job_id
            WHERE {' AND '.join(clauses)}
            ORDER BY r.time_analyze_s ASC, r.id ASC
            LIMIT ?
        """
        with self._conn() as c:
            cur = c.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def list_person_clips_summary(
        self,
        job_ids: List[str],
        *,
        start_time_s: float = 0.0,
        end_time_s: float = 0.0,
        gender: Optional[int] = None,
        start_age: int = 0,
        end_age: int = 1000,
        sleeve_length: Optional[str] = None,
        type_of_lower_body_clothing: Optional[str] = None,
        length_of_lower_body_clothing: Optional[str] = None,
        carrying_handbag: Optional[str] = None,
        wearing_hat: Optional[str] = None,
        color: Optional[str] = None,
        mask: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Gom theo job + video_clip — giống GET /api/reports/persons VideoMaster."""
        if not job_ids:
            return []
        placeholders = ",".join("?" * len(job_ids))
        clauses = [f"r.job_id IN ({placeholders})"]
        params: List[Any] = list(job_ids)
        t_clauses, t_params = self._vm_time_filter_clauses(float(start_time_s), float(end_time_s))
        clauses.extend(t_clauses)
        params.extend(t_params)
        if gender is not None:
            clauses.append("r.gender=?")
            params.append(gender)
        if start_age > 0:
            clauses.append("(r.age IS NULL OR r.age >= ?)")
            params.append(int(start_age))
        if end_age < 1000:
            clauses.append("(r.age IS NULL OR r.age <= ?)")
            params.append(int(end_age))
        if sleeve_length is not None:
            clauses.append("r.sleeve_length=?")
            params.append(sleeve_length)
        if type_of_lower_body_clothing is not None:
            clauses.append("r.type_of_lower_body_clothing=?")
            params.append(type_of_lower_body_clothing)
        if length_of_lower_body_clothing is not None:
            clauses.append("r.length_of_lower_body_clothing=?")
            params.append(length_of_lower_body_clothing)
        if carrying_handbag is not None:
            clauses.append("r.carrying_handbag=?")
            params.append(carrying_handbag)
        if wearing_hat is not None:
            clauses.append("r.wearing_hat=?")
            params.append(wearing_hat)
        if mask is not None:
            clauses.append("r.mask=?")
            params.append(mask)
        if color is not None:
            clauses.append("r.color LIKE ?")
            params.append(f"%{color}%")
        sql = f"""
            SELECT
                r.job_id AS video_id,
                r.video_clip,
                MIN(r.id) AS id,
                MIN(j.time_upload_utc) AS time_video,
                MIN(r.time_analyze_s) AS time_analyze,
                MAX(r.time_analyze_s) AS end_time,
                COUNT(DISTINCT r.id_tracking) AS count_persons
            FROM video_person_reports r
            INNER JOIN video_analyze_jobs j ON j.id = r.job_id
            WHERE {' AND '.join(clauses)}
            GROUP BY r.job_id, r.video_clip
            ORDER BY time_analyze ASC
        """
        with self._conn() as c:
            cur = c.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
        out: List[Dict[str, Any]] = []
        for r in rows:
            jid = str(r.get("video_id") or "")
            job = self.get_job(jid) or {}
            r["video_name"] = self.job_title(job)
            r["job_id"] = jid
            ta = r.get("time_analyze")
            if ta is not None:
                try:
                    fv = float(ta)
                    r["time_analyze"] = int(fv) if fv == int(fv) else fv
                except (TypeError, ValueError):
                    pass
            et = r.get("end_time")
            if et is not None:
                try:
                    fv = float(et)
                    r["end_time"] = int(fv) if fv == int(fv) else fv
                except (TypeError, ValueError):
                    pass
            out.append(r)
        return out

    def list_person_reports_by_clip(
        self,
        job_id: str,
        video_clip: int,
        *,
        start_time_s: float = 0.0,
        end_time_s: float = 0.0,
        limit: int = 5000,
        use_vm_time_filter: bool = True,
    ) -> List[Dict[str, Any]]:
        clauses = ["r.job_id=?", "r.video_clip=?"]
        params: List[Any] = [job_id, int(video_clip)]
        if use_vm_time_filter:
            t_clauses, t_params = self._vm_time_filter_clauses(
                float(start_time_s), float(end_time_s)
            )
            clauses.extend(t_clauses)
            params.extend(t_params)
        else:
            if start_time_s > 0:
                clauses.append("r.time_analyze_s >= ?")
                params.append(start_time_s)
            if end_time_s > 0:
                clauses.append("r.time_analyze_s <= ?")
                params.append(end_time_s)
        params.append(max(1, min(20000, int(limit))))
        sql = f"""
            SELECT r.*, j.time_upload_utc AS time_video
            FROM video_person_reports r
            INNER JOIN video_analyze_jobs j ON j.id = r.job_id
            WHERE {' AND '.join(clauses)}
            ORDER BY r.time_analyze_s ASC, r.id ASC
            LIMIT ?
        """
        with self._conn() as c:
            cur = c.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]

    def list_person_reports_with_embeddings(
        self,
        job_id: str,
        *,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT * FROM video_person_reports
                WHERE job_id=? AND features_face IS NOT NULL AND features_face != ''
                ORDER BY time_analyze_s ASC
                LIMIT ?
                """,
                (job_id, max(1, min(50000, int(limit)))),
            )
            return [dict(r) for r in cur.fetchall()]

    def list_person_reports_merged(
        self, job_id: str, *, include_clip: bool = False
    ) -> List[Dict[str, Any]]:
        """Gom theo id_tracking; track_names = top N tên nghi ngờ (bỏ phiếu đa khung)."""
        raw = self.list_person_reports(job_id, limit=50000)
        if not raw:
            return []
        from identity_vm_app.services.video_report_merge import merge_reports

        merged = merge_reports(raw, include_clip=include_clip)
        job = self.get_job(job_id) or {}
        vname = self.job_title(job)
        for d in merged:
            d["video_name"] = vname
            d["count_persons"] = 1
            if d.get("time_analyze") is None and d.get("time_analyze_s") is not None:
                d["time_analyze"] = d["time_analyze_s"]
        return merged

    def list_person_reports_merged_sql(self, job_id: str) -> List[Dict[str, Any]]:
        """Legacy SQL GROUP BY — giữ cho tương thích nội bộ."""
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT
                    MIN(id) AS id,
                    job_id,
                    id_tracking,
                    video_clip,
                    MIN(time_analyze_s) AS time_analyze,
                    MAX(time_analyze_s) AS end_time,
                    MIN(img_url) AS img_url,
                    MAX(display_name) AS display_name,
                    MAX(face_id) AS face_id,
                    MAX(face_img) AS face_img,
                    MAX(box_face) AS box_face,
                    MAX(box_person) AS box_person,
                    MAX(armed) AS armed,
                    COUNT(*) AS hit_count
                FROM video_person_reports
                WHERE job_id=?
                GROUP BY job_id, id_tracking, video_clip
                ORDER BY time_analyze ASC
                """,
                (job_id,),
            )
            rows = []
            job = self.get_job(job_id) or {}
            vname = self.job_title(job)
            for r in cur.fetchall():
                d = dict(r)
                d["video_name"] = vname
                d["count_persons"] = 1
                rows.append(d)
            return rows

    def list_weapon_reports(
        self,
        job_id: str,
        *,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT * FROM video_weapon_reports
                WHERE job_id=?
                ORDER BY time_analyze_s ASC
                LIMIT ?
                """,
                (job_id, max(1, min(20000, int(limit)))),
            )
            return [dict(r) for r in cur.fetchall()]

    def list_weapon_reports_merged(self, job_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            cur = c.execute(
                """
                SELECT
                    MIN(id) AS id,
                    job_id,
                    id_tracking,
                    video_clip,
                    MIN(time_analyze_s) AS time_analyze,
                    MAX(time_analyze_s) AS end_time,
                    MIN(img_url) AS img_url,
                    MAX(image_status) AS image_status,
                    MAX(armed) AS armed,
                    MAX(scene_crop_path) AS scene_crop_path,
                    COUNT(*) AS hit_count
                FROM video_weapon_reports
                WHERE job_id=?
                GROUP BY job_id, id_tracking, video_clip
                ORDER BY time_analyze ASC
                """,
                (job_id,),
            )
            job = self.get_job(job_id) or {}
            vname = self.job_title(job)
            return [{**dict(r), "video_name": vname} for r in cur.fetchall()]

    def clear_job_reports(self, job_id: str) -> int:
        """Xóa báo cáo người/vũ khí; giữ bản ghi job."""
        with self._lock, self._conn() as c:
            n = c.execute(
                "SELECT COUNT(*) FROM video_person_reports WHERE job_id=?", (job_id,)
            ).fetchone()[0]
            c.execute("DELETE FROM video_person_reports WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM video_weapon_reports WHERE job_id=?", (job_id,))
            c.execute(
                "UPDATE video_analyze_jobs SET index_frame=0, total_sample_frames=0 WHERE id=?",
                (job_id,),
            )
            return int(n)

    def delete_job_data(self, job_id: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM video_person_reports WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM video_weapon_reports WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM video_analyze_jobs WHERE id=?", (job_id,))

    @staticmethod
    def _job_row_to_dict(r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        raw = d.get("feature_analyze_json")
        if raw:
            try:
                d["feature_analyze"] = json.loads(raw)
            except Exception:
                d["feature_analyze"] = {}
        else:
            d["feature_analyze"] = {}
        st = int(d.get("status") or 0)
        d["status_name"] = {
            VA_STATUS_PENDING: "pending",
            VA_STATUS_PROCESSING: "running",
            VA_STATUS_COMPLETED: "done",
            VA_STATUS_ERROR: "error",
        }.get(st, "unknown")
        return d


_store_singleton: Optional[VideoAnalyzeStore] = None
_store_lock = threading.Lock()


def get_video_analyze_store() -> VideoAnalyzeStore:
    global _store_singleton
    with _store_lock:
        if _store_singleton is None:
            _store_singleton = VideoAnalyzeStore()
        return _store_singleton
