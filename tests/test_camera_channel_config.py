"""Tests cho load_camera_channel_config — nhiều camera, gap số, JSON lỏng."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from camera_channel_config import load_camera_channel_specs, _parse_json_loose


class TestParseJsonLoose(unittest.TestCase):
    def test_trailing_comma_object(self) -> None:
        raw = '{"a": 1, "b": 2,}'
        out = _parse_json_loose(raw)
        self.assertEqual(out, {"a": 1, "b": 2})

    def test_valid_json(self) -> None:
        self.assertEqual(_parse_json_loose('{"x": 3}'), {"x": 3})


class TestNumberedCameraKeys(unittest.TestCase):
    def test_gap_uses_source_index_in_id(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(
                {
                    "camera_source0": "rtsp://a/0",
                    "camera_source1": "rtsp://a/1",
                    "camera_source3": "rtsp://a/3",
                },
                f,
            )
            path = f.name
        try:
            specs = load_camera_channel_specs(path)
            ids = [s["id"] for s in specs]
            self.assertEqual(ids, ["cam0", "cam1", "cam3"])
            self.assertEqual(specs[2]["source"], "rtsp://a/3")
        finally:
            os.unlink(path)

    def test_many_numbered_sorted(self) -> None:
        d = {f"camera_source{i}": f"u{i}" for i in [5, 2, 9, 1]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(d, f)
            path = f.name
        try:
            specs = load_camera_channel_specs(path)
            self.assertEqual([s["id"] for s in specs], ["cam1", "cam2", "cam5", "cam9"])
        finally:
            os.unlink(path)


class TestCamerasArray(unittest.TestCase):
    def test_duplicate_id_skipped(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(
                {
                    "cameras": [
                        {"id": "c1", "source": "a"},
                        {"id": "c1", "source": "b"},
                        {"id": "c2", "source": "c"},
                    ]
                },
                f,
            )
            path = f.name
        try:
            specs = load_camera_channel_specs(path)
            self.assertEqual(len(specs), 2)
            self.assertEqual(specs[0]["source"], "a")
            self.assertEqual(specs[1]["id"], "c2")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
