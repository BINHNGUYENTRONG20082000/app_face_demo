"""CLI: đăng ký hàng loạt từ thư mục — chạy trên máy có GPU/model, không qua HTTP từng ảnh.

Ví dụ:
  python -m identity_vm_app.cli_bulk_register --root "E:\\datasets\\faces" --resume
  IVM_USE_FAISS=1 IVM_BULK_DB_WRITE_BATCH=128 python -m identity_vm_app.cli_bulk_register --root ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from identity_vm_app import settings as s
from module_ai.jobs.bulk_folder_register import run_folder_register
from identity_vm_app.lifecycle import release_inference_engine, startup


def main() -> int:
    p = argparse.ArgumentParser(description="Bulk register images from a local folder (Identity VM).")
    p.add_argument("--root", type=Path, required=True, help="Thư mục chứa ảnh")
    p.add_argument("--no-recurse", action="store_true", help="Chỉ tầng trên cùng")
    p.add_argument("--no-resume", action="store_true", help="Bỏ qua checkpoint (vẫn merge vào DB nếu trùng path — nên tránh)")
    p.add_argument("--clear-checkpoint", action="store_true", help="Xóa checkpoint rồi chạy (không xóa face DB)")
    p.add_argument("--batch", type=int, default=None, help="Ghi DB mỗi N ảnh (mặc định IVM_BULK_DB_WRITE_BATCH)")
    p.add_argument("--max-files", type=int, default=None, help="Giới hạn số file (debug)")
    p.add_argument(
        "--infer-workers",
        type=int,
        default=None,
        help="Số luồng infer bulk (1–16); mặc định IVM_BULK_INFER_WORKERS (env)",
    )
    p.add_argument("--progress-every", type=int, default=1000, help="In tiến độ mỗi N ảnh (0 = tắt)")
    args = p.parse_args()

    startup(load_face_model=False)

    from identity_vm_app.state import state

    if state.face_db is None or state.store is None:
        print("Không khởi tạo được DB / store.", file=sys.stderr)
        return 1

    def _on_progress(d: dict) -> None:
        print(json.dumps({"progress": d}, ensure_ascii=False), flush=True)

    on_prog = _on_progress if int(args.progress_every) > 0 else None

    try:
        stats = run_folder_register(
            root=args.root,
            engine=None,
            db=state.face_db,
            store=state.store,
            recursive=not args.no_recurse,
            resume=not args.no_resume,
            clear_checkpoint_first=bool(args.clear_checkpoint),
            db_batch_size=args.batch,
            max_files=args.max_files,
            infer_workers=args.infer_workers,
            progress_every=max(0, int(args.progress_every)),
            on_progress=on_prog,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    finally:
        release_inference_engine()

    print(f"# IVM_FACE_DB_DIR={s.IVM_FACE_DB_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
