import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


class FaissSearchBackend:
    def __init__(self, db_path: Path, use_faiss: bool = True) -> None:
        self.db_path = db_path
        self.use_faiss = bool(use_faiss)
        self.faiss_index_path = self.db_path / "faiss.index"
        self.faiss_meta_path = self.db_path / "faiss.meta.json"
        self._faiss_index = None

    def is_available(self) -> bool:
        return faiss is not None

    def can_use(self) -> bool:
        return self.use_faiss and self.is_available()

    def is_ready(self) -> bool:
        return self._faiss_index is not None

    def _is_index_compatible(self, index, embeddings: np.ndarray) -> bool:
        if embeddings is None:
            return False
        if not hasattr(index, "ntotal") or not hasattr(index, "d"):
            return False
        return int(index.ntotal) == int(embeddings.shape[0]) and int(index.d) == int(embeddings.shape[1])

    def load_persisted(self, signature: dict, embeddings: np.ndarray) -> bool:
        if not self.can_use():
            return False
        if not self.faiss_index_path.exists() or not self.faiss_meta_path.exists():
            return False

        try:
            with open(self.faiss_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            saved_signature = meta.get("signature", {}) if isinstance(meta, dict) else {}
            if saved_signature != signature:
                return False

            loaded_index = faiss.read_index(str(self.faiss_index_path))
            if not self._is_index_compatible(loaded_index, embeddings):
                return False

            self._faiss_index = loaded_index
            return True
        except Exception:
            return False

    def persist(self, signature: dict, temp_suffix: str) -> None:
        if not self.can_use() or self._faiss_index is None:
            return

        temp_index = self.db_path / f"{self.faiss_index_path.stem}.tmp.{temp_suffix}{self.faiss_index_path.suffix}"
        temp_meta = self.db_path / f"{self.faiss_meta_path.stem}.tmp.{temp_suffix}{self.faiss_meta_path.suffix}"

        meta = {
            "backend": "faiss",
            "signature": signature,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }

        try:
            faiss.write_index(self._faiss_index, str(temp_index))
            with open(temp_meta, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            os.replace(temp_index, self.faiss_index_path)
            os.replace(temp_meta, self.faiss_meta_path)
        finally:
            if temp_index.exists():
                temp_index.unlink(missing_ok=True)
            if temp_meta.exists():
                temp_meta.unlink(missing_ok=True)

    def invalidate(self, remove_persisted: bool = True) -> None:
        self._faiss_index = None
        if remove_persisted:
            self.remove_persisted()

    def remove_persisted(self) -> None:
        for path in (self.faiss_index_path, self.faiss_meta_path):
            if path.exists():
                path.unlink(missing_ok=True)

    def rebuild(self, normalized_database: np.ndarray, dimension: int) -> None:
        if not self.can_use():
            self._faiss_index = None
            return

        index = faiss.IndexFlatIP(int(dimension))
        if normalized_database.shape[0] > 0:
            index.add(normalized_database.astype(np.float32))
        self._faiss_index = index

    def search(
        self,
        normalized_queries: np.ndarray,
        total_database: int,
        k: int,
        distance_threshold: float,
        name_by_id: List[Optional[str]],
    ) -> Tuple[List[List[dict]], float, float]:
        top_k = max(1, min(int(k), int(total_database)))
        similarities, indices = self._faiss_index.search(normalized_queries.astype(np.float32), top_k)

        min_similarity = 1.0 - float(distance_threshold)
        results: List[List[dict]] = []
        for row_idx in range(indices.shape[0]):
            row_results = []
            for col_idx, face_id in enumerate(indices[row_idx]):
                idx_int = int(face_id)
                if idx_int < 0 or idx_int >= len(name_by_id):
                    continue

                similarity = float(similarities[row_idx, col_idx])
                if similarity < min_similarity:
                    continue

                person_name = name_by_id[idx_int]
                if person_name is None:
                    continue

                row_results.append(
                    {
                        "distance": 1.0 - similarity,
                        "name": person_name,
                        "face_id": idx_int,
                    }
                )

            results.append(row_results)

        return results, 0.0, 0.0
