"""
Kiểm tra luồng đăng ký thư mục bulk (bulk_folder_register.run_folder_register)
không phụ thuộc InsightFace hay GPU — mock FaceAnalysis trong worker threads.

Chạy từ thư mục gốc repo:
  pip install numpy opencv-python-headless pytest  # pytest tùy chọn
  set IVM_DATA_DIR=%TEMP%\\ivm_bulk_test   (tùy chọn; test tự đặt)
  pytest tests/test_bulk_folder_register_flow.py -v

Hoặc:
  python -m unittest tests.test_bulk_folder_register_flow -v
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _reload_ivm_settings(data_dir: Path) -> None:
    os.environ["IVM_DATA_DIR"] = str(data_dir)
    import identity_vm_app.settings as ivm_s

    importlib.reload(ivm_s)


class FakeFace:
    def __init__(self) -> None:
        self.det_score = 0.95
        self.embedding = np.linspace(0.01, 0.5, 512, dtype=np.float32)


class FakeInsightFaceEngine:
    def __init__(self, ctx_id: int = 0, **kwargs: object) -> None:
        self.ctx_id = int(ctx_id)

    def analyze_bgr(self, img: np.ndarray, timing_out: dict | None = None, **_kw: object) -> list[FakeFace]:
        _ = img
        if timing_out is not None:
            timing_out.clear()
            timing_out.update({"detect_ms": 0.1, "embedding_ms": 0.2})
        return [FakeFace()]


class TestBulkFolderRegisterFlow(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.td = Path(self._td.name)
        self.ivm_data = self.td / "ivm_data"
        self.ivm_data.mkdir(parents=True, exist_ok=True)
        _reload_ivm_settings(self.ivm_data)

        self.in_dir = self.td / "input_images"
        self.in_dir.mkdir(parents=True, exist_ok=True)
        # Ảnh tối giản; engine đã mock nên không cần mặt thật
        try:
            import cv2
        except ImportError as e:
            self.skipTest(f"Need opencv: {e}")
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        cv2.imwrite(str(self.in_dir / "test_person.jpg"), img)

    def tearDown(self) -> None:
        os.environ.pop("IVM_BULK_INFER_WORKERS", None)
        self._td.cleanup()

    def test_run_folder_register_writes_face_db(self) -> None:
        from identity_vm_app.bulk_folder_register import run_folder_register
        from identity_vm_app.store.sqlite_store import IdentityVmStore
        from packages.persistence.face_database import FaceDatabase

        import identity_vm_app.settings as s

        store = IdentityVmStore()
        db = FaceDatabase(str(s.IVM_FACE_DB_DIR), use_faiss=False)

        with patch(
            "identity_vm_app.bulk_folder_register.InsightFaceEngine",
            FakeInsightFaceEngine,
        ):
            stats = run_folder_register(
                root=self.in_dir,
                engine=None,
                db=db,
                store=store,
                recursive=False,
                resume=False,
                clear_checkpoint_first=True,
                progress_every=0,
            )

        self.assertEqual(int(stats.get("success", 0)), 1, stats)
        self.assertEqual(len(db.metadata), 1)
        emb_path = Path(s.IVM_FACE_DB_DIR) / "embeddings.npy"
        meta_path = Path(s.IVM_FACE_DB_DIR) / "metadata.json"
        self.assertTrue(emb_path.is_file(), "embeddings.npy phải được persist")
        self.assertTrue(meta_path.is_file(), "metadata.json phải được persist")
        key = str((self.in_dir / "test_person.jpg").resolve())
        self.assertEqual(store.bulk_checkpoint_get(key), "ok")
        self.assertEqual(int(stats.get("infer_accounting_gap", -1)), 0, stats)
        self.assertEqual(int(stats.get("failed_infer", -1)), 0, stats)
        self.assertEqual(int(stats.get("failed_preprocess", -1)), 0, stats)
        self.assertEqual(
            int(stats["success"]) + int(stats["failed_infer"]),
            int(stats["_bulk_total_files"]),
            stats,
        )

    def test_empty_face_db_clears_stale_checkpoint_then_registers(self) -> None:
        """Checkpoint 'ok' cũ trong khi face DB trống phải được xóa tự động (resume=True)."""
        from identity_vm_app.bulk_folder_register import run_folder_register
        from identity_vm_app.store.sqlite_store import IdentityVmStore
        from packages.persistence.face_database import FaceDatabase

        import identity_vm_app.settings as s

        store = IdentityVmStore()
        db = FaceDatabase(str(s.IVM_FACE_DB_DIR), use_faiss=False)

        fp = self.in_dir / "test_person.jpg"
        path_key = str(fp.resolve())
        store.bulk_checkpoint_set(path_key, "ok")
        self.assertEqual(len(db.metadata), 0)

        with patch(
            "identity_vm_app.bulk_folder_register.InsightFaceEngine",
            FakeInsightFaceEngine,
        ):
            stats = run_folder_register(
                root=self.in_dir,
                engine=None,
                db=db,
                store=store,
                recursive=False,
                resume=True,
                clear_checkpoint_first=False,
                progress_every=0,
            )

        self.assertEqual(int(stats.get("skipped_checkpoint", -1)), 0, stats)
        self.assertEqual(int(stats.get("success", 0)), 1, stats)
        self.assertEqual(len(db.metadata), 1)
        self.assertEqual(int(stats.get("infer_accounting_gap", -999)), 0, stats)

    def test_infer_balance_four_workers_partial_failures(self) -> None:
        """Đa luồng + một phần ảnh không có mặt: tot_p == success + failed_infer."""

        os.environ["IVM_BULK_INFER_WORKERS"] = "4"
        _reload_ivm_settings(Path(self.ivm_data))

        from identity_vm_app.bulk_folder_register import run_folder_register
        from identity_vm_app.store.sqlite_store import IdentityVmStore
        from packages.persistence.face_database import FaceDatabase

        import identity_vm_app.settings as s_mod

        n_img = 60
        batch_d = self.in_dir / "batch60"
        batch_d.mkdir(exist_ok=True)
        tmpl = self.in_dir / "test_person.jpg"
        for i in range(n_img):
            shutil.copy(tmpl, batch_d / f"p_{i}.jpg")

        class EveryThirdNoFace(FakeInsightFaceEngine):
            def __init__(self, ctx_id: int = 0, **kwargs: object) -> None:
                super().__init__(ctx_id=ctx_id, **kwargs)
                self._ctr = [0]

            def analyze_bgr(self, img: np.ndarray, timing_out: dict | None = None, **_kw: object) -> list:
                self._ctr[0] += 1
                if self._ctr[0] % 3 == 0:
                    if timing_out is not None:
                        timing_out.clear()
                        timing_out.update({"detect_ms": 0.0, "embedding_ms": 0.0})
                    return []
                return super().analyze_bgr(img, timing_out=timing_out)

        store = IdentityVmStore()
        db = FaceDatabase(str(s_mod.IVM_FACE_DB_DIR), use_faiss=False)

        with patch(
            "identity_vm_app.bulk_folder_register.InsightFaceEngine",
            EveryThirdNoFace,
        ):
            stats = run_folder_register(
                root=batch_d,
                engine=None,
                db=db,
                store=store,
                recursive=False,
                resume=False,
                clear_checkpoint_first=True,
                progress_every=0,
            )

        tot_p = int(stats.get("_bulk_total_files", 0))
        succ = int(stats.get("success", 0))
        finf = int(stats.get("failed_infer", 0))
        fpre = int(stats.get("failed_preprocess", 0))
        self.assertEqual(fpre, 0, stats)
        self.assertEqual(int(stats.get("infer_accounting_gap", -1)), 0, stats)
        self.assertEqual(tot_p, succ + finf, f"{stats} tot≠ok+infer_fail")
        self.assertGreater(finf, 0)
        self.assertEqual(tot_p, n_img)


class TestBulkCheckpointLookupMany(unittest.TestCase):
    """lookup many thay cho N lần get — không cần mock engine."""

    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.td = Path(self._td.name)
        self.ivm_data = self.td / "ivm_data"
        self.ivm_data.mkdir(parents=True, exist_ok=True)
        _reload_ivm_settings(self.ivm_data)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_lookup_many_matches_individual_get(self) -> None:
        from identity_vm_app.store.sqlite_store import IdentityVmStore

        store = IdentityVmStore()
        k1 = str((self.td / "a.jpg").resolve())
        k_missing = str((self.td / "b.jpg").resolve())
        store.bulk_checkpoint_set(k1, "ok")
        mp = store.bulk_checkpoint_lookup_many([k1, k_missing])
        self.assertEqual(mp.get(k1), "ok")
        self.assertNotIn(k_missing, mp)
        self.assertEqual(store.bulk_checkpoint_get(k1), "ok")

    def test_chunked_lookup_all_keys(self) -> None:
        from identity_vm_app.store.sqlite_store import IdentityVmStore

        store = IdentityVmStore()
        keys_ns: List[str] = []
        for i in range(5):
            path = self.td / f"f{i}.txt"
            path.write_text(".", encoding="utf-8")
            ks = str(path.resolve())
            keys_ns.append(ks)
            store.bulk_checkpoint_set(ks, "ok" if i % 2 == 0 else "fail")
        merged = store.bulk_checkpoint_lookup_many(keys_ns, chunk_size=2)
        self.assertEqual(len(merged), 5)


class TestCliBulkRegisterSmoke(unittest.TestCase):
    """Import smoke: module CLI không lỗi cú pháp / circular import."""

    def test_cli_module_import(self) -> None:
        td = tempfile.mkdtemp()
        try:
            _reload_ivm_settings(Path(td))
            import identity_vm_app.cli_bulk_register as cli_mod  # noqa: F401

            self.assertTrue(hasattr(cli_mod, "main"))
        finally:
            shutil.rmtree(td, ignore_errors=True)


class TestBulkProgressJsonWrites(unittest.TestCase):
    """write_register_folder_progress không raise."""

    def test_write_progress_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td)
            pj = tp / "progress.json"
            from identity_vm_app.bulk_folder_register import write_register_folder_progress

            write_register_folder_progress(
                pj,
                stats={
                    "root": str(tp),
                    "success": 0,
                    "failed": 0,
                    "skipped_checkpoint": 0,
                },
                running=True,
                phase="scanning",
                processed=0,
                total=1,
            )
            self.assertTrue(pj.is_file())
            payload = json.loads(pj.read_text(encoding="utf-8"))
            self.assertEqual(payload.get("phase"), "scanning")
            self.assertTrue(payload.get("running"))


if __name__ == "__main__":
    unittest.main()
