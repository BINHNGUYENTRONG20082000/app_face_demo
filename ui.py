#!/usr/bin/env python
"""
Giao diện Streamlit — Camera trực tiếp, bật/tắt nhận diện, báo cáo CSV.

Chạy (sau `python backend.py`):

  streamlit run ui.py --server.port 8510

Giao diện test API đầy đủ (cũ): đổi trong file này thành `streamlit_test.py`.

Biến môi trường: IVM_UI_API_URL (mặc định cổng IVM_API_PORT).
"""

from __future__ import annotations

import runpy
from pathlib import Path

_root = Path(__file__).resolve().parent
_ui = _root / "identity_vm_app" / "camera_dashboard.py"
if not _ui.is_file():
    raise SystemExit(f"Không tìm thấy {_ui}")

runpy.run_path(str(_ui), run_name="__main__")
