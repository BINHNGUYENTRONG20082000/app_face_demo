"""Kiểm tra embedding từ infer được ghi vào features_face (camera live)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from identity_vm_app.services.camera_report_writer import person_report_row  # noqa: E402


def main() -> int:
    emb = np.arange(512, dtype=np.float32)
    row = person_report_row(
        "cam0",
        "live-cam0-test",
        t_s=0.0,
        frame_index=0,
        sample_index=0,
        video_clip=0,
        img_url="scene.jpg",
        id_tracking=1,
        box_person="[10, 20, 200, 400]",
        face={
            "bbox": [50, 60, 120, 140],
            "det_score": 0.9,
            "matches": [{"face_id": 1, "name": "Alice", "distance": 0.2}],
            "embedding": emb,
        },
        persist_embeddings=True,
    )
    assert row.get("features_face"), "features_face must be set when embedding present"
    assert row.get("box_face"), row
    assert row.get("display_name") == "Alice", row
    print("OK camera_report_embedding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
