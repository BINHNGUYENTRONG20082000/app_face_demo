"""Worker một camera: thread riêng + model stack riêng (VisionMaster), queue giữ mọi khung sample."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from module_ai.config import settings as s
from identity_vm_app.camera_analyze_control import get_analyze_enabled
from module_ai.camera.activity_log import record as log_activity
from module_ai.camera.analyze_recording import record_visual_frame
from module_ai.camera.camera_model_stack import CameraModelStack
from module_ai.camera.display import build_analyze_visual_bgr
from module_ai.camera.infer import (
    identify_and_scale,
    identify_frame_person_first_detailed,
    ingest_armed_persons,
    ingest_faces,
)
from module_ai.camera.weapon import weapon_detection_available
from identity_vm_app.services.video_analyze_fps import frame_skip_for_sample
from packages.camera_stream import StreamConnectionConfig
from packages.camera_stream.opencv_reader import StableCameraReader

logger = logging.getLogger("camera_recognition.worker")


@dataclass
class CameraRecognitionConfig:
    camera_id: str
    source: Any
    api_base: str
    interval_s: Optional[float] = None
    distance_threshold: Optional[float] = None
    analyze_target_fps: float = field(default_factory=lambda: float(s.IVM_ANALYZE_TARGET_FPS))
    analyze_even_frames_only: bool = field(
        default_factory=lambda: bool(s.IVM_ANALYZE_EVEN_FRAMES_ONLY)
    )
    max_frame_width: int = field(default_factory=lambda: int(s.IVM_ANALYZE_MAX_FRAME_WIDTH))
    jpeg_quality: int = field(default_factory=lambda: int(s.IVM_ANALYZE_JPEG_QUALITY))
    identify_timeout_s: float = field(default_factory=lambda: float(s.IVM_ANALYZE_IDENTIFY_TIMEOUT_S))
    ingest_timeout_s: float = field(default_factory=lambda: float(s.IVM_ANALYZE_INGEST_TIMEOUT_S))
    use_in_process: bool = field(default_factory=lambda: bool(s.IVM_USE_IN_PROCESS_INFER))
    log_unknown_events: bool = field(default_factory=lambda: bool(s.IVM_LOG_UNKNOWN_EVENTS))
    display_jpeg_quality: int = field(default_factory=lambda: int(s.IVM_INFER_DISPLAY_JPEG_QUALITY))


@dataclass
class _QueuedInferFrame:
    frame_bgr: np.ndarray
    frame_count: int
    stream_fps: float
    job_id: Optional[str]
    live_mode: bool
    infer_key: int = 0
    sample_index: int = 0
    rel_frame_index: int = 0
    t_analyze_s: float = 0.0


class CameraRecognitionWorker(threading.Thread):
    """Thread nền / camera: model riêng + hàng đợi infer (không bỏ khung sample)."""

    def __init__(self, cfg: CameraRecognitionConfig) -> None:
        super().__init__(name=f"ivm-recog-{cfg.camera_id}", daemon=True)
        self.cfg = cfg
        self._stop = threading.Event()
        self._display_lock = threading.Lock()
        self._last_display_jpeg: Optional[bytes] = None
        self._meta: Dict[str, Any] = {
            "camera_id": cfg.camera_id,
            "infer_ms": 0.0,
            "n_faces": 0,
            "error": None,
            "recognition_enabled": False,
            "infer_queue_depth": 0,
            "dedicated_models": bool(s.IVM_CAMERA_DEDICATED_MODELS),
        }
        self._models = CameraModelStack(cfg.camera_id)
        self._queue_lock = threading.Lock()
        self._queue_cond = threading.Condition(self._queue_lock)
        self._infer_queue: Deque[_QueuedInferFrame] = deque()
        self._last_enqueued_key = -1
        self._last_enqueued_fc = 0
        self._queue_job_id: Optional[str] = None
        self._queue_warn_ts = 0.0
        self._infer_in_progress = False
        self._stats_count = 0
        self._stats_batch_from = 0
        self._stats_sum_infer = 0.0
        self._stats_sum_detect = 0.0
        self._stats_sum_embed = 0.0
        self._stats_sum_search = 0.0
        self._stats_sum_weapon = 0.0
        self._stats_sum_person = 0.0
        self._stats_sum_pose = 0.0
        self._stats_first_batch_pending = True
        stream_cfg = StreamConnectionConfig(
            read_fails_before_reconnect=int(s.IVM_PIPELINE_READ_FAILS_BEFORE_RECONNECT),
            reconnect_delay_s=float(s.IVM_RTSP_RECONNECT_DELAY_S),
            cap_buffer_size=int(s.IVM_CAP_PROP_BUFFERSIZE),
            rtsp_stale_reconnect_s=float(s.IVM_RTSP_STALE_RECONNECT_S),
            rtsp_proactive_reconnect_s=float(s.IVM_RTSP_PROACTIVE_RECONNECT_S),
        )
        self._reader = StableCameraReader(cfg.camera_id, cfg.source, config=stream_cfg)
        self._reader.set_frame_decoded_callback(self._on_reader_frame_decoded)

    def stop(self) -> None:
        self._stop.set()
        with self._queue_cond:
            self._queue_cond.notify_all()
        self._reader.stop()
        self._models.dispose()

    @property
    def reader(self):
        return self._reader

    def ensure_rtsp_reader(self) -> None:
        """Mở RTSP chỉ khi BẬT nhận diện — hiển thị lưới dùng /ivm/preview (hub riêng)."""
        if not self._reader.is_running:
            self._reader.start()
            logger.info("[%s] Bật đọc RTSP (nhận diện)", self.cfg.camera_id)

    def release_rtsp_reader(self) -> None:
        if self._reader.is_running:
            self._reader.stop()
            logger.info(
                "[%s] Ngắt đọc RTSP (nhận diện tắt — xem trước: POST /ivm/preview/warm hoặc MJPEG)",
                self.cfg.camera_id,
            )

    def sync_rtsp_reader(self) -> None:
        if get_analyze_enabled(self.cfg.camera_id):
            self.ensure_rtsp_reader()
        else:
            self.release_rtsp_reader()

    @property
    def infer_in_progress(self) -> bool:
        return self._infer_in_progress

    def get_meta(self) -> Dict[str, Any]:
        with self._display_lock:
            return dict(self._meta)

    def get_display_jpeg(self) -> Optional[bytes]:
        with self._display_lock:
            return self._last_display_jpeg

    def infer_queue_size(self) -> int:
        with self._queue_lock:
            return len(self._infer_queue)

    def clear_infer_queue(self) -> int:
        """Xóa khung chờ infer (khi tắt nhận diện — không drain). Trả số khung đã bỏ."""
        with self._queue_lock:
            n = len(self._infer_queue)
            self._infer_queue.clear()
            self._set_queue_depth_meta(0)
            self._queue_cond.notify_all()
        return int(n)

    def wait_infer_queue_drained(self, timeout: float = 300.0) -> bool:
        deadline = time.monotonic() + max(0.1, float(timeout))
        with self._queue_cond:
            while self._infer_queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._queue_cond.wait(timeout=min(0.25, remaining))
            return True

    def dispose_session_models(self) -> None:
        self._models.dispose()

    def _set_queue_depth_meta(self, depth: int) -> None:
        with self._display_lock:
            self._meta["infer_queue_depth"] = int(depth)

    def _reset_infer_stats(self, *, first_batch: bool = False) -> None:
        self._stats_count = 0
        self._stats_batch_from = 0
        self._stats_sum_infer = 0.0
        self._stats_sum_detect = 0.0
        self._stats_sum_embed = 0.0
        self._stats_sum_search = 0.0
        self._stats_sum_weapon = 0.0
        self._stats_sum_person = 0.0
        self._stats_sum_pose = 0.0
        if first_batch:
            self._stats_first_batch_pending = True

    def _should_log_infer_ok(self, infer_cycle: int, q_depth: int) -> bool:
        """Giảm spam infer_ok khi queue sâu — infer_stats vẫn log đủ."""
        thr = int(s.IVM_CAMERA_INFER_OK_LOG_QUEUE_THROTTLE)
        if thr <= 0 or q_depth < thr:
            return True
        every_n = max(1, int(s.IVM_CAMERA_INFER_OK_LOG_EVERY_N_WHEN_BUSY))
        return int(infer_cycle) % every_n == 0

    def _infer_stats_window(self) -> int:
        every_n = int(s.IVM_INFER_STATS_EVERY_N)
        if every_n <= 0:
            return 0
        if self._stats_first_batch_pending:
            return min(every_n, max(1, int(s.IVM_INFER_STATS_FIRST_N)))
        return every_n

    @staticmethod
    def _fmt_stage_ms(label: str, ms: float, total_ms: float) -> str:
        if ms <= 0.05:
            return f"{label} 0"
        if total_ms > 0:
            return f"{label} {ms:.0f} ({100.0 * ms / total_ms:.0f}%)"
        return f"{label} {ms:.0f}"

    @staticmethod
    def vm_sample_slot(
        fc: int,
        start_fc: int,
        stream_fps: float,
        sample_fps: float,
    ) -> Tuple[bool, int]:
        rel = int(fc) - int(start_fc)
        if rel < 0:
            return False, -1
        skip = frame_skip_for_sample(stream_fps, sample_fps)
        if skip <= 1:
            return True, rel
        if rel % skip != 0:
            return False, -1
        return True, rel // skip

    def _accumulate_infer_stats(self, cycle: int, timing: Dict[str, Any], infer_ms: float) -> None:
        window = self._infer_stats_window()
        if window <= 0:
            return
        if self._stats_count == 0:
            self._stats_batch_from = int(cycle)
        self._stats_count += 1
        self._stats_sum_infer += float(infer_ms)
        self._stats_sum_detect += float(timing.get("detect_ms") or 0.0)
        self._stats_sum_embed += float(timing.get("embedding_ms") or 0.0)
        self._stats_sum_search += float(timing.get("search_ms") or 0.0)
        self._stats_sum_weapon += float(timing.get("weapon_ms") or 0.0)
        self._stats_sum_person += float(timing.get("person_track_ms") or 0.0)
        self._stats_sum_pose += float(timing.get("pose_refine_ms") or 0.0)
        if self._stats_count < window:
            return

        n = self._stats_count
        avg_infer = self._stats_sum_infer / n
        avg_detect = self._stats_sum_detect / n
        avg_embed = self._stats_sum_embed / n
        avg_search = self._stats_sum_search / n
        avg_weapon = self._stats_sum_weapon / n
        avg_person = self._stats_sum_person / n
        avg_pose = self._stats_sum_pose / n
        avg_parts = avg_person + avg_detect + avg_pose + avg_weapon + avg_embed + avg_search
        eff_fps = 1000.0 / avg_infer if avg_infer > 0 else 0.0
        q_depth = self.infer_queue_size()
        stream_fps = max(1.0, float(self._reader.fps_actual) or 0.0)
        target_fps = float(self.cfg.analyze_target_fps)
        try:
            from identity_vm_app.services.camera_live_session import get_active_session

            live = get_active_session(self.cfg.camera_id)
            if live is not None and float(live.analyze_fps_eff) > 0:
                target_fps = float(live.analyze_fps_eff)
        except Exception:
            pass
        stage_avgs = [
            ("person_track", avg_person),
            ("face_det", avg_detect),
            ("pose", avg_pose),
            ("weapon", avg_weapon),
            ("embed", avg_embed),
            ("search", avg_search),
        ]
        bottleneck_label, bottleneck_ms = max(stage_avgs, key=lambda item: item[1])
        batch_tag = "lô đầu" if self._stats_first_batch_pending else "định kỳ"
        parts_txt = " | ".join(
            self._fmt_stage_ms(label, ms, avg_parts if avg_parts > 0 else avg_infer)
            for label, ms in stage_avgs
        )
        msg = (
            f"📊 [{batch_tag}] TB {n} khung infer #{self._stats_batch_from}–{cycle}: "
            f"tổng {avg_infer:.0f} ms (≈{eff_fps:.2f} fps) | {parts_txt} | "
            f"nghẽn: {bottleneck_label} {bottleneck_ms:.0f} ms"
        )
        if abs(avg_infer - avg_parts) > 2.0:
            msg += f" | lệch {avg_infer - avg_parts:.0f} ms"
        msg += f" | queue={q_depth}"
        if stream_fps > 0:
            msg += f" | camera ~{stream_fps:.0f} fps"
        if target_fps > 0:
            msg += f" | mục tiêu sample ~{target_fps:.0f} fps"
            if eff_fps > 0 and eff_fps < target_fps * 0.95:
                msg += " ⚠ chậm hơn sample_fps"
            elif eff_fps >= target_fps * 0.95:
                msg += " ✓ đủ sample_fps"
        logger.info("[%s] %s", self.cfg.camera_id, msg)
        log_activity(
            self.cfg.camera_id,
            "infer_stats",
            msg,
            also_console=False,
            extra={
                "cycle_from": self._stats_batch_from,
                "cycle_to": int(cycle),
                "n_frames": n,
                "batch_tag": batch_tag,
                "bottleneck": bottleneck_label,
                "bottleneck_ms": round(bottleneck_ms, 2),
                "avg_infer_ms": round(avg_infer, 2),
                "avg_person_track_ms": round(avg_person, 2),
                "avg_detect_ms": round(avg_detect, 2),
                "avg_pose_refine_ms": round(avg_pose, 2),
                "avg_weapon_ms": round(avg_weapon, 2),
                "avg_embedding_ms": round(avg_embed, 2),
                "avg_search_ms": round(avg_search, 2),
                "avg_parts_ms": round(avg_parts, 2),
                "effective_fps": round(eff_fps, 3),
                "target_fps": round(target_fps, 2),
                "stream_fps": round(stream_fps, 2),
                "infer_queue_depth": q_depth,
                "dedicated_models": bool(s.IVM_CAMERA_DEDICATED_MODELS),
            },
        )
        with self._display_lock:
            self._meta["infer_avg_ms"] = round(avg_infer, 2)
            self._meta["infer_avg_person_ms"] = round(avg_person, 2)
            self._meta["infer_avg_pose_ms"] = round(avg_pose, 2)
            self._meta["infer_stats_window"] = n
        self._stats_first_batch_pending = False
        self._reset_infer_stats()

    def _enqueue_infer(self, item: _QueuedInferFrame) -> None:
        max_q = int(s.IVM_CAMERA_INFER_QUEUE_MAX)
        blocked_logged = False
        while not self._stop.is_set():
            with self._queue_lock:
                if max_q <= 0 or len(self._infer_queue) < max_q:
                    self._infer_queue.append(item)
                    depth = len(self._infer_queue)
                    self._queue_cond.notify_all()
                    self._set_queue_depth_meta(depth)
                    warn_at = max(1, int(s.IVM_CAMERA_INFER_QUEUE_WARN_DEPTH))
                    now = time.monotonic()
                    if depth >= warn_at and now - self._queue_warn_ts > 5.0:
                        self._queue_warn_ts = now
                        logger.warning(
                            "[%s] Hàng đợi infer sâu %d khung (model chậm — vẫn giữ khung sample)",
                            self.cfg.camera_id,
                            depth,
                        )
                    return
            if not blocked_logged:
                blocked_logged = True
                logger.warning(
                    "[%s] Hàng đợi infer đầy (%d) — chờ drain (không bỏ khung)",
                    self.cfg.camera_id,
                    max_q,
                )
            self._stop.wait(0.05)

    def _dequeue_infer(self) -> Optional[_QueuedInferFrame]:
        with self._queue_lock:
            if not self._infer_queue:
                self._set_queue_depth_meta(0)
                self._queue_cond.notify_all()
                return None
            item = self._infer_queue.popleft()
            depth = len(self._infer_queue)
            self._set_queue_depth_meta(depth)
            if not self._infer_queue:
                self._queue_cond.notify_all()
            return item

    def _reset_enqueue_state(self) -> None:
        self._last_enqueued_key = -1
        self._last_enqueued_fc = 0
        self._queue_job_id = None

    def _publish_display(
        self,
        frame: np.ndarray,
        faces: List[Dict[str, Any]],
        meta: Dict[str, Any],
        *,
        armed_persons: Optional[List[Dict[str, Any]]] = None,
        visual_bgr: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        cap_fps = float(self._reader.fps_actual)
        vis = visual_bgr
        if vis is None:
            vis = build_analyze_visual_bgr(
                frame,
                camera_id=self.cfg.camera_id,
                faces_payload=faces,
                meta=meta,
                capture_fps=cap_fps,
                armed_persons=armed_persons,
            )
        if vis is not None and meta.get("recognition_enabled"):
            record_visual_frame(self.cfg.camera_id, vis)
        chunk = None
        if vis is not None:
            ok, buf = cv2.imencode(
                ".jpg",
                vis,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.display_jpeg_quality)],
            )
            chunk = buf.tobytes() if ok else None
        if chunk:
            with self._display_lock:
                self._last_display_jpeg = chunk
                self._meta = meta
        return vis

    @staticmethod
    def _frame_ready_for_infer(
        fc: int,
        last_enqueued_fc: int,
        *,
        even_frames_only: bool,
    ) -> bool:
        if fc <= 0 or fc <= last_enqueued_fc:
            return False
        if even_frames_only:
            return fc % 2 == 0
        return True

    def _interval_s(self) -> float:
        if self.cfg.analyze_even_frames_only:
            return 0.0
        iv = self.cfg.interval_s
        if iv is not None and float(iv) > 0:
            return float(iv)
        env_iv = float(s.IVM_ANALYZE_INTERVAL_S)
        if env_iv > 0:
            return env_iv
        fps = float(self.cfg.analyze_target_fps)
        return max(0.2, 1.0 / fps) if fps > 0 else 1.0

    def _refresh_idle_display(self, frame: np.ndarray) -> None:
        meta = {
            "camera_id": self.cfg.camera_id,
            "infer_ms": 0.0,
            "n_faces": 0,
            "error": None,
            "recognition_enabled": False,
            "frame_count": self._reader.frame_count,
            "analyze_target_fps": self.cfg.analyze_target_fps,
            "infer_queue_depth": 0,
            "dedicated_models": bool(s.IVM_CAMERA_DEDICATED_MODELS),
        }
        self._publish_display(frame, [], meta)

    def _on_reader_frame_decoded(self, fc: int, frame_bgr: np.ndarray, stream_fps: float) -> None:
        """Enqueue mọi khung sample ngay khi decode (không bỏ slot khi infer chậm)."""
        if self._stop.is_set():
            return
        if not get_analyze_enabled(self.cfg.camera_id):
            return
        try:
            from identity_vm_app.services.camera_live_session import get_active_session

            live_sess = get_active_session(self.cfg.camera_id)
        except Exception:
            return
        if live_sess is None:
            return

        job_id = str(live_sess.job_id)
        if self._queue_job_id != job_id:
            self._queue_job_id = job_id
            self._last_enqueued_key = -1

        vfps = max(1.0, float(stream_fps) if stream_fps > 0 else float(self._reader.fps_actual) or 10.0)
        slot, infer_key = self.vm_sample_slot(
            int(fc),
            int(live_sess.start_frame_count),
            vfps,
            float(live_sess.sample_fps),
        )
        if not slot:
            return

        with self._queue_lock:
            if int(infer_key) <= int(self._last_enqueued_key):
                return
            self._last_enqueued_key = int(infer_key)

        rel = max(0, int(fc) - int(live_sess.start_frame_count))
        t_analyze_s = float(rel) / vfps

        self._enqueue_infer(
            _QueuedInferFrame(
                frame_bgr=frame_bgr,
                frame_count=int(fc),
                stream_fps=vfps,
                job_id=job_id,
                live_mode=True,
                infer_key=int(infer_key),
                sample_index=int(infer_key),
                rel_frame_index=int(rel),
                t_analyze_s=t_analyze_s,
            )
        )

    def _maybe_enqueue_legacy(
        self,
        frame: np.ndarray,
        fc: int,
        stream_fps: float,
        *,
        even_only: bool,
        last_infer: float,
        interval: float,
        now: float,
    ) -> bool:
        if not even_only and now - last_infer < interval:
            return False
        if not self._frame_ready_for_infer(fc, self._last_enqueued_fc, even_frames_only=even_only):
            return False
        infer_key = fc // 2 if even_only else fc
        if infer_key <= self._last_enqueued_key:
            return False
        self._enqueue_infer(
            _QueuedInferFrame(
                frame_bgr=frame.copy(),
                frame_count=int(fc),
                stream_fps=float(stream_fps),
                job_id=None,
                live_mode=False,
                infer_key=int(infer_key),
            )
        )
        self._last_enqueued_fc = int(fc)
        self._last_enqueued_key = int(infer_key)
        return True

    def _execute_infer(
        self,
        frame_snap: np.ndarray,
        fc: int,
        stream_fps: float,
        *,
        live_sess: Any,
        default_thr: float,
        even_only: bool,
        infer_cycle: int,
        t_analyze_s: Optional[float] = None,
        rel_frame_index: Optional[int] = None,
        sample_index: Optional[int] = None,
    ) -> Tuple[int, float, int]:
        cfg = self.cfg
        engine, person_tracker, face_db = self._models.ensure_loaded()
        run_weapon = weapon_detection_available()
        thr = live_sess.distance_threshold if live_sess else default_thr
        use_live_reports = (
            live_sess is not None
            and cfg.use_in_process
            and person_tracker is not None
            and bool(s.IVM_USE_PERSON_FIRST_PIPELINE)
        )

        if use_live_reports:
            from identity_vm_app.services.camera_live_session import on_frame as live_on_frame

            faces, assignments, faces_with_matches, timing = identify_frame_person_first_detailed(
                frame_snap,
                engine=engine,
                face_db=face_db,
                person_tracker=person_tracker,
                thr=float(thr),
                max_width=cfg.max_frame_width,
                camera_id=cfg.camera_id,
                run_weapon=run_weapon,
            )
            weapon_scene: Dict[str, Any] = dict(timing.get("weapon_scene") or {})
            armed_persons = list(timing.get("armed_persons") or []) if run_weapon else []
            meta_pre = {
                "camera_id": cfg.camera_id,
                "recognition_enabled": True,
                "frame_count": fc,
                "analyze_target_fps": live_sess.analyze_fps_eff if live_sess else cfg.analyze_target_fps,
            }
            vis = build_analyze_visual_bgr(
                frame_snap,
                camera_id=cfg.camera_id,
                faces_payload=faces,
                meta=meta_pre,
                capture_fps=stream_fps,
                armed_persons=armed_persons if run_weapon else None,
            )
            report_fc = int(rel_frame_index) if rel_frame_index is not None else int(fc)
            n_report = live_on_frame(
                cfg.camera_id,
                frame_snap,
                frame_count=report_fc,
                assignments=assignments,
                faces_with_matches=faces_with_matches,
                visual_bgr=vis,
                t_analyze_s=t_analyze_s,
                sample_index=sample_index,
            )
            n_ingested = n_report
            if run_weapon and live_sess is not None:
                from module_ai.camera.weapon_alerts import emit_weapon_track_alerts

                alert_frame = vis if vis is not None else frame_snap
                new_weapon_alerts = emit_weapon_track_alerts(
                    cfg.camera_id,
                    job_id=str(live_sess.job_id),
                    faces=faces,
                    armed_persons=armed_persons,
                    weapon_track_rows=list(timing.get("weapon_track_rows") or []),
                    frame_bgr=alert_frame,
                    frame_count=int(fc),
                    t_analyze_s=t_analyze_s,
                    sample_index=sample_index,
                )
                if new_weapon_alerts:
                    weapon_scene = dict(weapon_scene)
                    weapon_scene["last_alerts"] = new_weapon_alerts
            if s.IVM_CAMERA_LEGACY_EVENTS:
                n_ingested += ingest_faces(
                    cfg.camera_id,
                    frame_snap,
                    faces,
                    api_base=cfg.api_base,
                    use_in_process=cfg.use_in_process,
                    ingest_timeout_s=cfg.ingest_timeout_s,
                    jpeg_quality=cfg.jpeg_quality,
                    log_unknown=cfg.log_unknown_events,
                )
                if run_weapon and armed_persons:
                    n_ingested += ingest_armed_persons(
                        cfg.camera_id,
                        frame_snap,
                        armed_persons,
                        api_base=cfg.api_base,
                        use_in_process=cfg.use_in_process,
                        ingest_timeout_s=cfg.ingest_timeout_s,
                        jpeg_quality=cfg.jpeg_quality,
                    )
        else:
            faces, timing, _, _ = identify_and_scale(
                frame_snap,
                api_base=cfg.api_base,
                thr=float(thr),
                max_width=cfg.max_frame_width,
                use_in_process=cfg.use_in_process,
                http_timeout_s=cfg.identify_timeout_s,
                jpeg_quality=cfg.jpeg_quality,
                person_tracker=person_tracker,
                camera_id=cfg.camera_id,
                run_weapon=run_weapon,
            )
            weapon_scene = dict(timing.get("weapon_scene") or {})
            armed_persons = list(timing.get("armed_persons") or []) if run_weapon else []
            n_ingested = ingest_faces(
                cfg.camera_id,
                frame_snap,
                faces,
                api_base=cfg.api_base,
                use_in_process=cfg.use_in_process,
                ingest_timeout_s=cfg.ingest_timeout_s,
                jpeg_quality=cfg.jpeg_quality,
                log_unknown=cfg.log_unknown_events,
            )
            if run_weapon and armed_persons:
                n_ingested += ingest_armed_persons(
                    cfg.camera_id,
                    frame_snap,
                    armed_persons,
                    api_base=cfg.api_base,
                    use_in_process=cfg.use_in_process,
                    ingest_timeout_s=cfg.ingest_timeout_s,
                    jpeg_quality=cfg.jpeg_quality,
                )
            vis = None

        infer_cycle += 1
        names: List[str] = []
        armed_n = 0
        for f in faces:
            ms = f.get("matches") or []
            if ms:
                names.append(str(ms[0].get("name") or ms[0].get("display_name") or "?"))
            else:
                names.append("unknown")
            if (f.get("weapon") or {}).get("armed"):
                armed_n += 1
        infer_ms = float(timing.get("infer_ms", 0.0))
        q_depth = self.infer_queue_size()
        if self._should_log_infer_ok(infer_cycle, q_depth):
            log_activity(
                cfg.camera_id,
                "infer_ok",
                (
                    f"infer #{infer_cycle}: {len(faces)} mặt, ghi {n_ingested} row/event, "
                    f"{infer_ms:.0f} ms"
                    + (f" — {', '.join(names)}" if names else " — không có mặt")
                    + (f" | vũ khí: {armed_n} người" if weapon_detection_available() else "")
                    + (f" | queue={q_depth}" if q_depth else "")
                ),
                extra={
                    "cycle": infer_cycle,
                    "n_faces": len(faces),
                    "n_ingested": n_ingested,
                    "infer_ms": round(infer_ms, 2),
                    "names": names,
                    "armed_faces": armed_n,
                    "frame_count": fc,
                    "job_id": live_sess.job_id if live_sess else None,
                    "infer_queue_depth": q_depth,
                },
            )
        self._accumulate_infer_stats(infer_cycle, timing, infer_ms)
        meta = {
            "camera_id": cfg.camera_id,
            "infer_ms": round(float(timing.get("infer_ms", 0.0)), 2),
            "person_track_ms": round(float(timing.get("person_track_ms", 0.0)), 2),
            "detect_ms": round(float(timing.get("detect_ms", 0.0)), 2),
            "pose_refine_ms": round(float(timing.get("pose_refine_ms", 0.0)), 2),
            "embedding_ms": round(float(timing.get("embedding_ms", 0.0)), 2),
            "search_ms": round(float(timing.get("search_ms", 0.0)), 2),
            "weapon_ms": round(float(timing.get("weapon_ms", 0.0)), 2),
            "n_faces": len(faces),
            "armed_faces": armed_n,
            "weapon_scene": weapon_scene,
            "weapon_alert_tracks": int(timing.get("weapon_alert_tracks") or 0),
            "error": None,
            "recognition_enabled": True,
            "frame_count": fc,
            "analyze_target_fps": live_sess.analyze_fps_eff if live_sess else cfg.analyze_target_fps,
            "analyze_even_frames_only": even_only,
            "job_id": live_sess.job_id if live_sess else None,
            "infer_queue_depth": q_depth,
            "dedicated_models": bool(s.IVM_CAMERA_DEDICATED_MODELS),
        }
        if use_live_reports and vis is not None:
            ok, buf = cv2.imencode(
                ".jpg",
                vis,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.display_jpeg_quality)],
            )
            if ok:
                with self._display_lock:
                    self._last_display_jpeg = buf.tobytes()
                    self._meta = meta
        else:
            self._publish_display(
                frame_snap,
                faces,
                meta,
                armed_persons=armed_persons if run_weapon else None,
            )
        return infer_cycle, time.monotonic(), int(fc)

    def _process_queued_infer(
        self,
        item: _QueuedInferFrame,
        *,
        default_thr: float,
        even_only: bool,
        infer_cycle: int,
    ) -> Tuple[int, float, int]:
        from identity_vm_app.services.camera_live_session import get_active_session

        cfg = self.cfg
        fc = int(item.frame_count)
        live_sess = get_active_session(cfg.camera_id) if item.live_mode else None
        from identity_vm_app.camera_analyze_control import is_analyze_stopping

        if (
            item.live_mode
            and is_analyze_stopping(cfg.camera_id)
            and not bool(s.IVM_CAMERA_DRAIN_INFER_ON_STOP)
        ):
            return infer_cycle, time.monotonic(), fc
        if item.live_mode and live_sess is None:
            logger.warning("[%s] Bỏ khung queue fc=%s — phiên live đã đóng", cfg.camera_id, fc)
            return infer_cycle, time.monotonic(), fc
        if item.live_mode and item.job_id and live_sess and live_sess.job_id != item.job_id:
            logger.warning(
                "[%s] Bỏ khung queue fc=%s — job_id không khớp (%s != %s)",
                cfg.camera_id,
                fc,
                item.job_id,
                live_sess.job_id,
            )
            return infer_cycle, time.monotonic(), fc

        self._infer_in_progress = True
        try:
            return self._execute_infer(
                item.frame_bgr,
                fc,
                item.stream_fps,
                live_sess=live_sess,
                default_thr=default_thr,
                even_only=even_only,
                infer_cycle=infer_cycle,
                t_analyze_s=item.t_analyze_s if item.live_mode else None,
                rel_frame_index=item.rel_frame_index if item.live_mode else None,
                sample_index=item.sample_index if item.live_mode else None,
            )
        except Exception as exc:
            infer_cycle += 1
            log_activity(
                cfg.camera_id,
                "infer_error",
                f"infer #{infer_cycle} LỖI: {exc}",
                level="error",
                extra={"cycle": infer_cycle, "error": str(exc), "frame_count": fc},
            )
            meta = {
                "camera_id": cfg.camera_id,
                "infer_ms": 0.0,
                "n_faces": 0,
                "error": str(exc),
                "recognition_enabled": True,
                "frame_count": fc,
                "analyze_target_fps": cfg.analyze_target_fps,
                "analyze_even_frames_only": even_only,
                "infer_queue_depth": self.infer_queue_size(),
            }
            self._publish_display(item.frame_bgr, [], meta)
            return infer_cycle, time.monotonic(), fc
        finally:
            self._infer_in_progress = False
            self._maybe_finish_stopping()

    def _maybe_finish_stopping(self) -> None:
        from identity_vm_app.camera_analyze_control import finish_analyze_stop, is_analyze_stopping

        if is_analyze_stopping(self.cfg.camera_id) and self.infer_queue_size() == 0 and not self._infer_in_progress:
            finish_analyze_stop(self.cfg.camera_id)
            self._models.dispose()

    def run(self) -> None:
        cfg = self.cfg
        default_thr = (
            cfg.distance_threshold if cfg.distance_threshold is not None else s.IVM_DISTANCE_THRESHOLD
        )
        interval = self._interval_s()
        last_infer = 0.0
        last_idle_refresh = 0.0
        was_enabled = False
        infer_cycle = 0
        even_only_cfg = bool(cfg.analyze_even_frames_only)

        logger.info(
            "[%s] Worker started (RTSP khi BẬT nhận diện; hiển thị=/ivm/preview, dedicated_models=%s, target_fps=%.1f)",
            cfg.camera_id,
            bool(s.IVM_CAMERA_DEDICATED_MODELS),
            cfg.analyze_target_fps,
        )

        while not self._stop.is_set():
            self.sync_rtsp_reader()
            enabled = get_analyze_enabled(cfg.camera_id)
            from identity_vm_app.camera_analyze_control import is_analyze_stopping
            from identity_vm_app.services.camera_live_session import get_active_session

            stopping = is_analyze_stopping(cfg.camera_id)
            drain_on_stop = bool(s.IVM_CAMERA_DRAIN_INFER_ON_STOP)
            live_sess = get_active_session(cfg.camera_id) if enabled else None
            even_only = even_only_cfg and live_sess is None

            if stopping and not drain_on_stop:
                if self._infer_in_progress:
                    self._stop.wait(0.002)
                    continue
                dropped = self.clear_infer_queue()
                if dropped > 0:
                    logger.info(
                        "[%s] Tắt nhận diện — bỏ %d khung chờ (không drain)",
                        cfg.camera_id,
                        dropped,
                    )
                self._maybe_finish_stopping()
            else:
                queued = self._dequeue_infer()
                if queued is not None:
                    infer_cycle, last_infer, _ = self._process_queued_infer(
                        queued,
                        default_thr=default_thr,
                        even_only=even_only,
                        infer_cycle=infer_cycle,
                    )
                    continue

            if enabled and not was_enabled:
                try:
                    self._models.ensure_loaded()
                except Exception as ex:
                    logger.error("[%s] Không load model stack: %s", cfg.camera_id, ex)
                    self._stop.wait(1.0)
                    continue
                mode = (
                    f"phiên live sample_fps={live_sess.sample_fps if live_sess else '?'}"
                    if live_sess
                    else (
                        "chỉ infer frame chẵn (2, 4, 6, …)"
                        if even_only
                        else f"~{cfg.analyze_target_fps:.0f} fps (interval={interval:.2f}s)"
                    )
                )
                log_activity(
                    cfg.camera_id,
                    "worker_on",
                    f"Worker BẬT — {mode} | model riêng/camera, queue giữ khung sample",
                    extra={
                        "even_frames_only": even_only,
                        "interval_s": interval,
                        "in_process": cfg.use_in_process,
                        "job_id": live_sess.job_id if live_sess else None,
                        "dedicated_models": bool(s.IVM_CAMERA_DEDICATED_MODELS),
                    },
                )
                was_enabled = True
                self._reset_infer_stats(first_batch=True)
                stats_every = int(s.IVM_INFER_STATS_EVERY_N)
                if stats_every > 0:
                    first_n = min(stats_every, max(1, int(s.IVM_INFER_STATS_FIRST_N)))
                    logger.info(
                        "[%s] Log TB giai đoạn infer: sau %d khung, rồi mỗi %d khung",
                        cfg.camera_id,
                        first_n,
                        stats_every,
                    )
            elif not enabled and was_enabled:
                if stopping and drain_on_stop:
                    if self.infer_queue_size() > 0 or self._infer_in_progress:
                        self._stop.wait(0.002)
                        continue
                    self._maybe_finish_stopping()
                elif not stopping:
                    self._models.dispose()
                log_activity(cfg.camera_id, "worker_off", "Worker dừng infer (nhận diện TẮT)")
                was_enabled = False
                self._reset_enqueue_state()

            if not self._reader.is_running:
                self._stop.wait(0.15)
                continue

            frame = self._reader.get_frame()
            if frame is None:
                if enabled and self._reader.frame_count == 0 and infer_cycle == 0:
                    self._stop.wait(0.2)
                else:
                    self._stop.wait(0.05)
                continue

            now = time.monotonic()
            if not enabled:
                self._maybe_finish_stopping()
                if now - last_idle_refresh > 0.5:
                    self._refresh_idle_display(frame)
                    last_idle_refresh = now
                self._stop.wait(0.05)
                continue

            fc = self._reader.frame_count
            stream_fps = max(1.0, float(self._reader.fps_actual) or 10.0)

            if live_sess is not None:
                from identity_vm_app.services.camera_live_session import record_stream_frame

                record_stream_frame(cfg.camera_id, frame, stream_fps=stream_fps)
                self._stop.wait(0.002)
                continue

            if self._maybe_enqueue_legacy(
                frame,
                fc,
                stream_fps,
                even_only=even_only,
                last_infer=last_infer,
                interval=interval,
                now=now,
            ):
                continue
            self._stop.wait(0.002)

        self._reader.stop()
        self._models.dispose()
        logger.info("[%s] Recognition worker stopped", cfg.camera_id)
