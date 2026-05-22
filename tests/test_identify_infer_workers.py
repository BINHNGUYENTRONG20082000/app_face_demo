"""Unit tests cho resolve_identify_infer_workers (không cần GPU)."""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class TestResolveIdentifyInferWorkers(unittest.TestCase):
    def tearDown(self) -> None:
        for k in (
            "IVM_IDENTIFY_BATCH_INFER_WORKERS",
            "IVM_IDENTIFY_BATCH_API_MAX_INFER_WORKERS",
        ):
            os.environ.pop(k, None)
        import identity_vm_app.settings as ivm_s

        importlib.reload(ivm_s)

    def test_defaults_and_clamp(self) -> None:
        os.environ["IVM_IDENTIFY_BATCH_INFER_WORKERS"] = "1"
        os.environ["IVM_IDENTIFY_BATCH_API_MAX_INFER_WORKERS"] = "16"
        import identity_vm_app.settings as ivm_s

        importlib.reload(ivm_s)
        self.assertEqual(ivm_s.resolve_identify_infer_workers(None), 1)
        self.assertEqual(ivm_s.resolve_identify_infer_workers(4), 4)
        self.assertEqual(ivm_s.resolve_identify_infer_workers(99), 16)

    def test_api_cap_limits_env_default(self) -> None:
        os.environ["IVM_IDENTIFY_BATCH_INFER_WORKERS"] = "8"
        os.environ["IVM_IDENTIFY_BATCH_API_MAX_INFER_WORKERS"] = "4"
        import identity_vm_app.settings as ivm_s

        importlib.reload(ivm_s)
        self.assertEqual(ivm_s.resolve_identify_infer_workers(None), 4)
        self.assertEqual(ivm_s.resolve_identify_infer_workers(3), 3)


if __name__ == "__main__":
    unittest.main()
