import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


class FaceStorageRepository:
    """Đóng gói thao tác lưu trữ embeddings/metadata trên đĩa."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.mkdir(exist_ok=True)
        self.embeddings_path = self.db_path / "embeddings.npy"
        self.metadata_path = self.db_path / "metadata.json"

    def current_embeddings_signature(self, embeddings: np.ndarray) -> Dict[str, int]:
        stat = self.embeddings_path.stat() if self.embeddings_path.exists() else None
        shape = embeddings.shape if embeddings is not None else (0, 0)
        return {
            "vector_count": int(shape[0]) if len(shape) >= 1 else 0,
            "dimension": int(shape[1]) if len(shape) >= 2 else 0,
            "embeddings_size": int(stat.st_size) if stat else 0,
            "embeddings_mtime_ns": int(stat.st_mtime_ns) if stat else 0,
        }

    def exists(self) -> bool:
        return self.embeddings_path.exists() and self.metadata_path.exists()

    def load(self) -> Tuple[np.ndarray, dict]:
        embeddings = np.load(self.embeddings_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not isinstance(metadata, dict):
            raise ValueError("metadata.json must be a JSON object")

        return embeddings, metadata

    def backup_corrupted_files(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        for file_path in [self.embeddings_path, self.metadata_path]:
            if file_path.exists():
                backup_path = self.db_path / f"{file_path.stem}.corrupt.{timestamp}{file_path.suffix}"
                os.replace(file_path, backup_path)

    def save(self, embeddings: np.ndarray, metadata: dict) -> None:
        temp_suffix = uuid.uuid4().hex
        temp_embeddings = self.db_path / f"{self.embeddings_path.stem}.tmp.{temp_suffix}{self.embeddings_path.suffix}"
        temp_metadata = self.db_path / f"{self.metadata_path.stem}.tmp.{temp_suffix}{self.metadata_path.suffix}"

        try:
            with open(temp_embeddings, "wb") as f:
                np.save(f, embeddings)
            with open(temp_metadata, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            os.replace(temp_embeddings, self.embeddings_path)
            os.replace(temp_metadata, self.metadata_path)
        finally:
            if temp_embeddings.exists():
                temp_embeddings.unlink(missing_ok=True)
            if temp_metadata.exists():
                temp_metadata.unlink(missing_ok=True)
