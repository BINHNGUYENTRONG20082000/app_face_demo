"""Đăng ký hàng loạt từ thư mục đĩa: batch DB, checkpoint resume, lỗi từng file không chặn job."""

from __future__ import annotations

import json
import gc
import os
import queue
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from identity_vm_app import settings as s
from module_ai.engine.insightface_engine import InsightFaceEngine
from identity_vm_app.store.sqlite_store import IdentityVmStore
from module_ai.persistence.face_database import FaceDatabase
from module_ai.utils.text import remove_accents

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# (fp, path_key, display_name, path_idx)
PendingItem = Tuple[Path, str, str, int]

_bulk_terminal_lock = threading.Lock()

# Chỉ sửa dưới stats_lock trong _run_pipeline; sau join copy sang stats dict.
InferTotals = Dict[str, int]


def _infer_totals_new() -> InferTotals:
    return {"ok": 0, "fail_infer": 0}


def _bulk_emit_line(msg: str) -> None:
    """Luôn log bulk ra stderr (uvicorn thường hiển thị) + append file (để máy chủ bulk không mất log)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"{ts} {msg}"
    with _bulk_terminal_lock:
        try:
            print(f"[ivm-bulk] {line}", file=sys.stderr, flush=True)
        except OSError:
            pass
        try:
            lp = s.IVM_REGISTER_FOLDER_BULK_LOG
            lp.parent.mkdir(parents=True, exist_ok=True)
            with open(lp, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    """Đọc ảnh qua buffer để hỗ trợ đường dẫn Unicode trên Windows."""
    p = str(path)
    try:
        data = np.fromfile(p, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except OSError:
        return cv2.imread(p)


def assert_bulk_root_allowed(root: Path) -> Path:
    r = root.expanduser().resolve()
    if not r.is_dir():
        raise ValueError(f"Không phải thư mục: {r}")
    allowed = [Path(p).expanduser().resolve() for p in s.IVM_BULK_ALLOWED_ROOTS]
    if not allowed:
        return r
    for a in allowed:
        try:
            r.relative_to(a)
            return r
        except ValueError:
            continue
    raise PermissionError(
        f"Thư mục {r} không nằm trong IVM_BULK_ALLOWED_ROOTS — thiết lập biến môi trường tương ứng."
    )


def _display_name_from_disk(root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(root)
    synthetic = str(rel).replace(os.sep, "_").replace("/", "_")
    stem = Path(synthetic).stem.strip()
    if not stem:
        raise ValueError("Không suy ra được tên từ đường dẫn tương đối.")
    name = remove_accents(stem)
    name = name.replace("_", " ").strip()
    name = " ".join(name.split())
    if not name:
        raise ValueError("Tên sau chuẩn hoá bị rỗng.")
    return name


def _image_path_for_person(clean_name: str, idx: int) -> str:
    images_dir = Path(s.IVM_FACE_DB_DIR) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in clean_name)
    name = f"{safe}_{int(time.time() * 1000)}_{idx}.jpg"
    return str(images_dir / name)


def _failure_raw_bytes(path: Path) -> Optional[bytes]:
    lim = int(s.IVM_BULK_FAILURE_SAMPLE_MAX_BYTES)
    if lim <= 0:
        return None
    try:
        st = path.stat()
        if st.st_size > lim:
            return None
        return path.read_bytes()
    except OSError:
        return None


def _log_failure(store: IdentityVmStore, path: Path, message: str) -> None:
    raw = _failure_raw_bytes(path)
    store.insert_registration_failure(
        original_filename=str(path.resolve()),
        error_message=str(message),
        raw_bytes=raw,
    )


def iter_image_files(
    root: Path, *, recursive: bool, max_collect: Optional[int] = None
) -> List[Path]:
    """Liệt kê ảnh; có thể giới hạn sớm bằng max_collect — caller sắp xếp (sort_bulk_paths) nếu cần."""
    root = root.resolve()
    out: List[Path] = []
    if recursive:
        it = root.rglob("*")
    else:
        it = root.iterdir()
    for p in it:
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in _IMAGE_EXT:
            continue
        try:
            p.resolve().relative_to(root)
        except ValueError:
            continue
        out.append(p)
        if max_collect is not None and len(out) >= max_collect:
            break
    return out


def sort_bulk_paths(paths: List[Path]) -> None:
    paths.sort(key=lambda x: str(x).lower())


# Bulk infer: mỗi worker thread một FaceAnalysis (số luồng = IVM_BULK_INFER_WORKERS trong settings).
def _split_pending_even(pending: List[PendingItem], k: int) -> List[List[PendingItem]]:
    """Chia pending thành k phần liên tiếp, số phần tử lệch nhau tối đa 1 (gần đều)."""
    if not pending:
        return []
    if k <= 1:
        return [pending]
    k = min(k, len(pending))
    n = len(pending)
    base, extra = divmod(n, k)
    out: List[List[PendingItem]] = []
    start = 0
    for i in range(k):
        sz = base + (1 if i < extra else 0)
        out.append(pending[start : start + sz])
        start += sz
    return out


def _dispose_worker_engine(eng: Optional[InsightFaceEngine]) -> None:
    """Gỡ FaceAnalysis / ONNX trong worker (engine tạm, không phải state.engine boot)."""
    from module_ai.engine.gpu_cleanup import dispose_insightface_engine

    dispose_insightface_engine(eng)


def _release_bulk_infer_resources() -> None:
    """Sau job multi-worker: GC + dọn cache CUDA nếu có."""
    from module_ai.engine.gpu_cleanup import gpu_soft_cleanup

    gpu_soft_cleanup(log_label="bulk_register")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(raw, encoding="utf-8")
    tmp.replace(path)


def write_register_folder_progress(
    path: Path,
    *,
    stats: Dict[str, Any],
    running: bool,
    phase: str,
    processed: int,
    total: int,
    message: str = "",
    error: str = "",
) -> None:
    """Ghi trạng thái tiến trình (poll bởi GET admin/register-folder/progress)."""
    tot = max(1, int(total) if total else 1)
    proc = int(processed)
    pct = min(100.0, 100.0 * float(proc) / float(tot))
    out: Dict[str, Any] = {
        "running": bool(running),
        "phase": phase,
        "root": str(stats.get("root", "")),
        "processed": proc,
        "total": int(total),
        "registered": int(stats.get("success", 0)),
        "success": int(stats.get("success", 0)),
        "failed": int(stats.get("failed", 0)),
        "skipped_checkpoint": int(stats.get("skipped_checkpoint", 0)),
        "progress_pct": round(pct, 2),
        "candidates_total": stats.get("candidates_total"),
        "message": message,
        "updated_at": time.time(),
        "started_at": stats.get("_progress_started_at"),
        "elapsed_s": stats.get("elapsed_s"),
        "model_tag": stats.get("model_tag"),
        "last_errors": stats.get("last_errors", []),
    }
    if error:
        out["error"] = error
    for fk in (
        "parallel_workers",
        "parallel_mode",
        "face_db_embeddings_before",
        "face_db_embeddings_after",
        "face_db_embeddings_delta_this_run",
        "infer_accounting_gap",
        "failed_preprocess",
        "failed_infer",
    ):
        if fk in stats and stats[fk] is not None:
            out[fk] = stats[fk]
    _atomic_write_json(path, out)


def collect_pending_items(
    *,
    files: List[Path],
    root: Path,
    store: IdentityVmStore,
    resume: bool,
    resume_skip_failed: bool,
    stats_out: Dict[str, Any],
    on_soft_error: Optional[Callable[[str], None]] = None,
) -> List[PendingItem]:
    """Danh sách ảnh cần chạy infer (đã lọc checkpoint + lỗi tên)."""
    pending: List[PendingItem] = []
    resume_map: Dict[str, str] = {}
    path_keys_ordered: Optional[List[str]] = None
    if resume and files:
        path_keys_ordered = [str(fp.resolve()) for fp in files]
        resume_map = store.bulk_checkpoint_lookup_many(
            path_keys_ordered, chunk_size=s.IVM_BULK_CHECKPOINT_LOOKUP_CHUNK
        )

    for path_idx, fp in enumerate(files, start=1):
        key = path_keys_ordered[path_idx - 1] if path_keys_ordered else str(fp.resolve())
        if resume:
            st_cp = resume_map.get(key)
            if st_cp == "ok":
                stats_out["skipped_checkpoint"] += 1
                continue
            if resume_skip_failed and st_cp == "fail":
                stats_out["skipped_checkpoint"] += 1
                continue
        try:
            display_name = _display_name_from_disk(root, fp)
        except ValueError as e:
            stats_out["failed_preprocess"] += 1
            store.bulk_checkpoint_set(key, "fail")
            _log_failure(store, fp, str(e))
            if on_soft_error:
                on_soft_error(f"{fp}: {e}")
            continue
        pending.append((fp, key, display_name, path_idx))
    return pending


def resolve_bulk_infer_workers(requested: Optional[int]) -> int:
    """
    Clamp số worker infer bulk:
    requested None → IVM_BULK_INFER_WORKERS (env).
    luôn ≤ IVM_BULK_API_MAX_INFER_WORKERS và ≤ 16.
    """
    cap_w = max(1, min(16, int(s.IVM_BULK_API_MAX_INFER_WORKERS)))
    env_w = max(1, min(cap_w, int(s.IVM_BULK_INFER_WORKERS)))
    if requested is None:
        return env_w
    return max(1, min(cap_w, int(requested)))


def run_folder_register(
    *,
    root: Path,
    engine: Optional[InsightFaceEngine],
    db: FaceDatabase,
    store: IdentityVmStore,
    recursive: bool = True,
    resume: bool = True,
    resume_skip_failed: Optional[bool] = None,
    clear_checkpoint_first: bool = False,
    db_batch_size: Optional[int] = None,
    max_files: Optional[int] = None,
    progress_every: int = 500,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_status: Optional[Callable[[str], None]] = None,
    progress_json_path: Optional[Path] = None,
    infer_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Quét thư mục, detect + embed từng ảnh, ghi DB theo batch.
    Số luồng infer: **`infer_workers`** (API/UI); None = **`IVM_BULK_INFER_WORKERS`** trong env/settings.
    Luôn clamp theo **`IVM_BULK_API_MAX_INFER_WORKERS`** (mặc định 16).
    `engine` giữ để tương thích call site — luồng bulk không dùng engine toàn cục.
    Gom embedding theo **IVM_BULK_MULTI_THREAD_DB_FLUSH** rồi `add_faces_batch`.
    Checkpoint: `path_key = str(path.resolve())` — resume bỏ qua `ok` (và `fail` nếu bật skip).
    `progress_json_path`: ghi file JSON để UI/GET API poll (tương thích luồng `/progress` app cũ).
    `on_status`: callback chuỗi ngắn (UI / log) tại các mốc quét checkpoint, khởi worker.
    """
    _ = engine
    def _emit(msg: str) -> None:
        _bulk_emit_line(msg)
        if on_status:
            on_status(msg)

    if resume_skip_failed is None:
        resume_skip_failed = bool(s.IVM_BULK_RESUME_SKIP_FAILED)

    root = assert_bulk_root_allowed(root)
    _emit(f"Đã chọn root hợp lệ: {root}")

    # Checkpoint "ok" gắn với embedding đã ghi vào Face DB; nếu DB đã bị xoá tay / chỉ reset file
    # nhưng SQLite checkpoint còn, resume=True sẽ bỏ qua hết ảnh — không có bản ghi mới (gây hiểu nhầm).
    if resume and not clear_checkpoint_first:
        try:
            if len(db.metadata) == 0:
                ck_del = store.bulk_checkpoint_clear()
                if ck_del:
                    _emit(
                        f"Face DB đang không có embedding — đã xóa **{ck_del}** mục checkpoint cũ "
                        "(để chạy lại đăng ký không bị bỏ qua)."
                    )
        except Exception:  # noqa: BLE001
            pass

    if clear_checkpoint_first:
        store.bulk_checkpoint_clear()

    batch_n = int(db_batch_size or s.IVM_BULK_DB_WRITE_BATCH)
    nw = resolve_bulk_infer_workers(infer_workers)
    db_flush_per_thread = max(batch_n, int(s.IVM_BULK_MULTI_THREAD_DB_FLUSH))
    effective_limit = max_files if max_files is not None and max_files > 0 else None
    pj = progress_json_path.resolve() if progress_json_path else None
    file_progress_every = int(progress_every)
    if file_progress_every <= 0:
        file_progress_every = 1 if pj else 500

    files = iter_image_files(root, recursive=recursive, max_collect=effective_limit)
    total_scan = len(files)
    thr_sort = int(s.IVM_BULK_MAX_SORT_FILES)
    if thr_sort > 0 and total_scan > thr_sort and files:
        _emit(
            f"Số file **{total_scan}** vượt **IVM_BULK_MAX_SORT_FILES={thr_sort}** — "
            "bỏ sort toàn cục (đỡ RAM/CPU; thứ tự theo quét đĩa)."
        )
    elif thr_sort > 0 and files:
        sort_bulk_paths(files)

    _emit(f"Đã quét đĩa: **{total_scan}** file ảnh hợp lệ (trước checkpoint).")

    stats: Dict[str, Any] = {
        "root": str(root),
        "recursive": recursive,
        "resume": resume,
        "resume_skip_failed": resume_skip_failed,
        "db_batch_size": batch_n,
        "db_flush_per_thread": db_flush_per_thread,
        "parallel_workers": nw,
        "parallel_mode": f"multi_infer_{nw}w",
        "candidates_total": total_scan,
        "candidates_run": total_scan,
        "skipped_checkpoint": 0,
        "success": 0,
        "failed_infer": 0,
        "failed_preprocess": 0,
        "failed": 0,
        "_progress_started_at": time.time(),
    }

    last_errors: List[str] = []
    last_err_lock = threading.Lock()

    def push_err(msg: str) -> None:
        with last_err_lock:
            last_errors.append(msg)
            if len(last_errors) > 32:
                last_errors.pop(0)

    t0 = time.perf_counter()

    if pj:
        write_register_folder_progress(
            pj,
            stats=stats,
            running=True,
            phase="scanning",
            processed=0,
            total=max(1, total_scan),
            message="Đang lọc checkpoint / chuẩn bị",
        )

    try:
        faces_before_job = len(db.metadata)
        stats["face_db_embeddings_before"] = faces_before_job
        _emit("Đang lọc checkpoint và lập danh sách ảnh cần infer…")
        pending = collect_pending_items(
            files=files,
            root=root,
            store=store,
            resume=resume,
            resume_skip_failed=resume_skip_failed,
            stats_out=stats,
            on_soft_error=push_err,
        )
        tot_p = len(pending)
        stats["_bulk_total_files"] = max(1, tot_p)
        infer_totals = _infer_totals_new()
        _emit(
            f"Checkpoint xong: **{tot_p}** ảnh cần xử lý — khởi động **{nw}** worker infer (song song)."
        )

        if pj:
            write_register_folder_progress(
                pj,
                stats=stats,
                running=True,
                phase="processing",
                processed=0,
                total=stats["_bulk_total_files"],
                message="Đang detect + embedding",
            )

        def _snapshot() -> Dict[str, Any]:
            with last_err_lock:
                errs = list(last_errors[-16:])
            with stats_lock:
                ok_snap = int(infer_totals["ok"])
                finf_snap = int(infer_totals["fail_infer"])
            snap = dict(stats)
            pre = int(snap.get("failed_preprocess") or 0)
            snap["success"] = ok_snap
            snap["failed_infer"] = finf_snap
            snap["failed"] = pre + finf_snap
            snap["last_errors"] = errs
            snap["elapsed_s"] = round(time.perf_counter() - t0, 3)
            snap["model_tag"] = s.IVM_MODEL_TAG
            return snap

        def _progress_cb(d: Dict[str, Any]) -> None:
            if on_progress:
                on_progress(d)
            if pj:
                write_register_folder_progress(
                    pj,
                    stats=_snapshot(),
                    running=True,
                    phase="processing",
                    processed=int(d.get("file_index", 0)),
                    total=int(d.get("total_files", 1)),
                    message="",
                )

        use_cb = _progress_cb if (on_progress or pj) else None
        n_shards = min(nw, max(1, tot_p))
        shards = _split_pending_even(pending, n_shards)
        ctx_ids = s.ivm_bulk_worker_ctx_ids()
        stats_lock = threading.Lock()
        progress_lock = threading.Lock()
        gp: List[int] = [0]

        def _shard_worker(shard: List[PendingItem], wid: int) -> None:
            shard_engine: Optional[InsightFaceEngine] = None
            try:
                ctx = ctx_ids[wid % len(ctx_ids)]
                shard_engine = InsightFaceEngine(ctx_id=ctx)
                _run_pipeline(
                    engine=shard_engine,
                    db=db,
                    store=store,
                    pending=shard,
                    batch_n=db_flush_per_thread,
                    infer_totals=infer_totals,
                    push_err=push_err,
                    total_files=tot_p,
                    progress_every=file_progress_every,
                    on_progress=use_cb,
                    stats_lock=stats_lock,
                    progress_lock=progress_lock,
                    global_processed_counter=gp,
                    stats_for_skipped=stats,
                    worker_id=wid,
                )
            finally:
                try:
                    _dispose_worker_engine(shard_engine)
                finally:
                    shard_engine = None

        threads: List[threading.Thread] = []
        for i, shard in enumerate(shards):
            if not shard:
                continue
            t = threading.Thread(
                target=_shard_worker,
                args=(shard, i),
                name=f"ivm_bulk_infer_{i}",
                daemon=True,
            )
            t.start()
            threads.append(t)
        _emit(f"Đang chạy infer — **{len(threads)}** luồng (đã start).")
        for t in threads:
            t.join()

        stats["success"] = int(infer_totals["ok"])
        stats["failed_infer"] = int(infer_totals["fail_infer"])
        stats["failed"] = int(stats["failed_preprocess"]) + int(stats["failed_infer"])
        stats["infer_accounting_gap"] = int(tot_p) - int(stats["success"]) - int(stats["failed_infer"])

        if use_cb and tot_p > 0:
            use_cb(
                {
                    "file_index": tot_p,
                    "total_files": tot_p,
                    "success": stats["success"],
                    "failed": stats["failed"],
                    "skipped_checkpoint": stats["skipped_checkpoint"],
                    "pending_batch": 0,
                }
            )
        _release_bulk_infer_resources()
        _emit("Worker đã kết thúc; đã gọi GC / dọn VRAM (nếu có CUDA).")

        faces_after_job = len(db.metadata)
        delta_emb = faces_after_job - faces_before_job
        stats["face_db_embeddings_after"] = faces_after_job
        stats["face_db_embeddings_delta_this_run"] = delta_emb

        igap = int(stats["infer_accounting_gap"])
        if igap != 0:
            _emit(
                f"Cảnh báo nhất quán infer: tot_p(**{tot_p}**) − success (**{stats['success']}**) "
                f"− failed_infer (**{stats['failed_infer']}**) = **{igap}** (mong đợi 0). "
                f"Kiểm tra exception trong worker bulk."
            )
        tol = max(5, tot_p // 10000 + int(stats.get("failed_preprocess", 0)))
        if abs(int(delta_emb) - int(stats["success"])) > tol:
            _emit(
                f"Cảnh báo Face DB: metadata tăng **{delta_emb}** vector trong job này, "
                f"nhưng `success` (đếm ảnh enroll) báo **{stats['success']}** — sai lệch > {tol}: "
                f"xem embeddings.npy/metadata hoặc lỗi ghi đĩa."
            )

        stats["last_errors"] = last_errors
        stats["elapsed_s"] = round(time.perf_counter() - t0, 3)
        stats["model_tag"] = s.IVM_MODEL_TAG
        if pj:
            write_register_folder_progress(
                pj,
                stats=stats,
                running=False,
                phase="done",
                processed=tot_p,
                total=stats["_bulk_total_files"],
                message="Hoàn thành",
            )
        return stats
    except BaseException as exc:
        if pj:
            stats["last_errors"] = last_errors
            stats["elapsed_s"] = round(time.perf_counter() - t0, 3)
            stats["model_tag"] = s.IVM_MODEL_TAG
            tot = int(stats.get("_bulk_total_files") or max(1, total_scan))
            write_register_folder_progress(
                pj,
                stats=stats,
                running=False,
                phase="error",
                processed=0,
                total=tot,
                error=str(exc),
            )
        raise


def _run_pipeline(
    *,
    engine: InsightFaceEngine,
    db: FaceDatabase,
    store: IdentityVmStore,
    pending: List[PendingItem],
    batch_n: int,
    infer_totals: InferTotals,
    push_err: Callable[[str], None],
    total_files: int,
    progress_every: int,
    on_progress: Optional[Callable[[Dict[str, Any]], None]],
    stats_lock: Optional[threading.Lock] = None,
    progress_lock: Optional[threading.Lock] = None,
    global_processed_counter: Optional[List[int]] = None,
    stats_for_skipped: Dict[str, Any],
    worker_id: Optional[int] = None,
) -> None:
    """Mỗi item trong pending được đếm đúng một lần: infer_totals.ok (mặt ghi được) hoặc fail_infer."""
    sctx = stats_lock if stats_lock is not None else nullcontext()
    prefetch = max(1, int(s.IVM_BULK_PREFETCH))
    # Thống kê thời gian theo lô 100 ảnh (decode + infer + ghi batch cục bộ) — log terminal.
    win_t0 = time.perf_counter()
    win_n = 0
    w_id = worker_id
    q: queue.Queue[Optional[Tuple[str, Path, str, str, int, Optional[np.ndarray]]]] = queue.Queue(
        maxsize=prefetch
    )
    # ("ok", fp, key, name, path_idx, img) | ("read_fail", fp, key, name, path_idx, None)

    def producer() -> None:
        for fp, key, display_name, path_idx in pending:
            try:
                img = imread_unicode(fp)
                if img is None or img.size == 0:
                    q.put(("read_fail", fp, key, display_name, path_idx, None))
                else:
                    q.put(("ok", fp, key, display_name, path_idx, img))
            except Exception:  # noqa: BLE001
                q.put(("read_fail", fp, key, display_name, path_idx, None))
        q.put(None)

    threading.Thread(target=producer, name="ivm_bulk_decode", daemon=True).start()

    batch_emb: List[np.ndarray] = []
    batch_names: List[str] = []
    batch_paths: List[str] = []
    batch_keys: List[str] = []
    processed = 0

    def flush_batch() -> None:
        if not batch_emb:
            return
        arr = np.stack(batch_emb, axis=0)
        db.add_faces_batch(arr, batch_names, batch_paths)
        for k in batch_keys:
            store.bulk_checkpoint_set(k, "ok")
        n_ok = len(batch_emb)
        with sctx:
            infer_totals["ok"] += n_ok
        batch_emb.clear()
        batch_names.clear()
        batch_paths.clear()
        batch_keys.clear()

    while True:
        item = q.get()
        if item is None:
            break
        kind, fp, key, display_name, path_idx, img = item
        if kind == "read_fail":
            with sctx:
                infer_totals["fail_infer"] += 1
            store.bulk_checkpoint_set(key, "fail")
            _log_failure(store, fp, "cannot read image")
            push_err(f"{fp}: cannot read image")
        else:
            assert img is not None
            try:
                det = engine.analyze_bgr(img)
                if not det:
                    raise RuntimeError("no face detected")
                face = max(det, key=lambda f: f.det_score)
                dest = _image_path_for_person(display_name, path_idx)
                if not cv2.imwrite(dest, img):
                    raise RuntimeError("failed to write enrolled image")
                batch_emb.append(face.embedding.astype(np.float32))
                batch_names.append(display_name)
                batch_paths.append(dest)
                batch_keys.append(key)
                if len(batch_emb) >= batch_n:
                    flush_batch()
            except Exception as e:  # noqa: BLE001
                with sctx:
                    infer_totals["fail_infer"] += 1
                store.bulk_checkpoint_set(key, "fail")
                _log_failure(store, fp, str(e))
                push_err(f"{fp}: {e}")

            if w_id is not None:
                win_n += 1
                if win_n >= 100:
                    dt = time.perf_counter() - win_t0
                    _bulk_emit_line(
                        f"worker {w_id} — 100 ảnh (infer): {dt:.2f}s, TB {dt / 100.0:.3f}s/ảnh"
                    )
                    win_t0 = time.perf_counter()
                    win_n = 0

        if global_processed_counter is not None and progress_lock is not None:
            with progress_lock:
                global_processed_counter[0] += 1
                gp = global_processed_counter[0]
            if progress_every > 0 and gp % progress_every == 0 and on_progress:
                with sctx:
                    succ = infer_totals["ok"]
                    finf = infer_totals["fail_infer"]
                pre_fp = int(stats_for_skipped.get("failed_preprocess", 0))
                sk_cp = int(stats_for_skipped.get("skipped_checkpoint", 0))
                on_progress(
                    {
                        "file_index": gp,
                        "total_files": total_files,
                        "success": succ,
                        "failed": pre_fp + finf,
                        "skipped_checkpoint": sk_cp,
                        "pending_batch": len(batch_emb),
                    }
                )
        else:
            processed += 1
            if progress_every > 0 and processed % progress_every == 0 and on_progress:
                with sctx:
                    succ2 = infer_totals["ok"]
                    finf2 = infer_totals["fail_infer"]
                pre_fp2 = int(stats_for_skipped.get("failed_preprocess", 0))
                on_progress(
                    {
                        "file_index": processed,
                        "total_files": total_files,
                        "success": succ2,
                        "failed": pre_fp2 + finf2,
                        "skipped_checkpoint": int(stats_for_skipped.get("skipped_checkpoint", 0)),
                        "pending_batch": len(batch_emb),
                    }
                )

    if on_progress and pending and global_processed_counter is None:
        with sctx:
            succ3 = infer_totals["ok"]
            finf3 = infer_totals["fail_infer"]
        pre_fp3 = int(stats_for_skipped.get("failed_preprocess", 0))
        on_progress(
            {
                "file_index": processed,
                "total_files": total_files,
                "success": succ3,
                "failed": pre_fp3 + finf3,
                "skipped_checkpoint": int(stats_for_skipped.get("skipped_checkpoint", 0)),
                "pending_batch": len(batch_emb),
            }
        )

    if w_id is not None and win_n > 0:
        dt = time.perf_counter() - win_t0
        _bulk_emit_line(
            f"worker {w_id} — {win_n} ảnh cuối (infer): {dt:.2f}s, TB {dt / float(win_n):.3f}s/ảnh"
        )

    flush_batch()
