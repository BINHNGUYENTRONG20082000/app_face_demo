import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np

from .backends import FaissSearchBackend, SklearnSearchBackend
from .storage import FaceStorageRepository


class FaceDatabase:
    """Facade quản lý database khuôn mặt, tách backend tìm kiếm riêng."""

    def __init__(
        self,
        db_path: str = "face_db",
        use_faiss: bool | None = None,
    ):
        if use_faiss is None:
            use_faiss = os.getenv("IVM_USE_FAISS", "0").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._storage = FaceStorageRepository(self.db_path)
        self._write_lock = threading.RLock()

        self.embeddings = None  # Shape: (N, 512)
        self.metadata = {}
        self._name_by_id_cache: Optional[List[Optional[str]]] = None

        self.use_faiss = bool(use_faiss)
        self._sklearn_backend = SklearnSearchBackend()
        self._faiss_backend = FaissSearchBackend(db_path=self.db_path, use_faiss=self.use_faiss)

        self.load_or_create_db()

    def _can_use_faiss(self) -> bool:
        return self._faiss_backend.can_use()

    def _current_embeddings_signature(self) -> dict:
        return self._storage.current_embeddings_signature(self.embeddings)

    def _persist_faiss_index(self) -> None:
        self._faiss_backend.persist(
            signature=self._current_embeddings_signature(),
            temp_suffix=uuid.uuid4().hex,
        )

    def _invalidate_faiss_index(self) -> None:
        self._faiss_backend.invalidate(remove_persisted=True)

    def _remove_persisted_faiss_index(self) -> None:
        self._faiss_backend.remove_persisted()

    def _load_persisted_faiss_index(self) -> bool:
        return self._faiss_backend.load_persisted(
            signature=self._current_embeddings_signature(),
            embeddings=self.embeddings,
        )

    def _rebuild_faiss_index(self) -> None:
        if self.embeddings is None:
            return
        normalized_database = self._normalize_embeddings(self.embeddings)
        dim = int(self.embeddings.shape[1]) if self.embeddings.size > 0 else 512
        self._faiss_backend.rebuild(normalized_database=normalized_database, dimension=dim)

    def rebuild_search_index(self) -> dict:
        """Rebuild search index thủ công (dùng từ API/UI)."""
        start = time.perf_counter()
        with self._write_lock:
            self._rebuild_faiss_index()
            self._persist_faiss_index()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "backend": "faiss" if self._can_use_faiss() else "sklearn",
            "faiss_enabled": bool(self.use_faiss),
            "faiss_available": bool(self._faiss_backend.is_available()),
            "index_ready": bool(self._faiss_backend.is_ready()) if self._can_use_faiss() else False,
            "index_persisted": bool(
                self._faiss_backend.faiss_index_path.exists() and self._faiss_backend.faiss_meta_path.exists()
            )
            if self._can_use_faiss()
            else False,
            "vector_count": int(self.embeddings.shape[0]),
            "elapsed_ms": round(elapsed_ms, 3),
        }

    def get_search_backend_info(self) -> dict:
        """Thông tin backend tìm kiếm để hiển thị trong giao diện."""
        return {
            "backend": "faiss" if self._can_use_faiss() else "sklearn",
            "faiss_enabled": bool(self.use_faiss),
            "faiss_available": bool(self._faiss_backend.is_available()),
            "index_ready": bool(self._faiss_backend.is_ready()) if self._can_use_faiss() else False,
            "index_persisted": bool(
                self._faiss_backend.faiss_index_path.exists() and self._faiss_backend.faiss_meta_path.exists()
            )
            if self._can_use_faiss()
            else False,
            "vector_count": int(self.embeddings.shape[0]),
        }

    def _invalidate_runtime_caches(self):
        self._sklearn_backend.invalidate()
        self._name_by_id_cache = None

    def _build_name_cache(self) -> List[Optional[str]]:
        if self.embeddings is None:
            return []

        db_count = int(self.embeddings.shape[0])
        names: List[Optional[str]] = [None] * db_count
        for key, value in self.metadata.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < db_count:
                names[idx] = value.get("name")

        self._name_by_id_cache = names
        return names

    def _get_name_by_id_cache(self) -> List[Optional[str]]:
        if self._name_by_id_cache is None:
            return self._build_name_cache()
        return self._name_by_id_cache

    def load_or_create_db(self):
        """Load hoặc tạo mới database"""
        if self._storage.exists():
            try:
                self.embeddings, self.metadata = self._storage.load()
                self._invalidate_runtime_caches()

                print(f"✓ Loaded {len(self.metadata)} faces từ database")
            except (OSError, ValueError) as e:
                print(f"⚠ Database corrupted, recreating clean database: {e}")
                self._storage.backup_corrupted_files()
                self.embeddings = np.empty((0, 512), dtype=np.float32)
                self._invalidate_runtime_caches()
                self.metadata = {}
                self.save()
                print("✓ Recreated clean database")
        else:
            self.embeddings = np.empty((0, 512), dtype=np.float32)
            self._invalidate_runtime_caches()
            self.metadata = {}
            self.save()
            print("✓ Created new database")

        self._load_persisted_faiss_index()

    def add_face(self, embedding: np.ndarray, person_name: str, image_path: str = None) -> int:
        """Thêm khuôn mặt vào database."""
        with self._write_lock:
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)

            embedding = embedding.astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            face_id = len(self.metadata)
            self.embeddings = np.vstack([self.embeddings, embedding])
            self._invalidate_runtime_caches()
            self._invalidate_faiss_index()

            self.metadata[str(face_id)] = {
                "name": person_name,
                "image_path": image_path,
                "id": face_id,
            }

            self.save()
            return face_id

    def add_faces_batch(
        self,
        embeddings: np.ndarray,
        person_names: List[str],
        image_paths: Optional[List[str]] = None,
    ) -> List[int]:
        """Thêm nhiều khuôn mặt và chỉ lưu DB một lần để tăng tốc."""
        with self._write_lock:
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)

            if embeddings.shape[0] == 0:
                return []

            if len(person_names) != embeddings.shape[0]:
                raise ValueError("person_names length must match embeddings rows")

            if image_paths is None:
                image_paths = [None] * embeddings.shape[0]
            if len(image_paths) != embeddings.shape[0]:
                raise ValueError("image_paths length must match embeddings rows")

            normalized_embeddings = self._normalize_embeddings(embeddings)
            start_id = len(self.metadata)
            batch_size = normalized_embeddings.shape[0]
            face_ids = list(range(start_id, start_id + batch_size))

            if self.embeddings.shape[0] == 0:
                self.embeddings = normalized_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, normalized_embeddings])
            self._invalidate_runtime_caches()
            self._invalidate_faiss_index()

            for idx, face_id in enumerate(face_ids):
                self.metadata[str(face_id)] = {
                    "name": person_names[idx],
                    "image_path": image_paths[idx],
                    "id": face_id,
                }

            self.save()
            return face_ids

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    def search(self, embedding: np.ndarray, k: int = 5, distance_threshold: float = 0.7) -> List[dict]:
        """Tìm kiếm khuôn mặt tương tự trong database."""
        return self.search_batch(
            embeddings=embedding,
            k=k,
            distance_threshold=distance_threshold,
        )[0]

    def search_batch(
        self,
        embeddings: np.ndarray,
        k: int = 5,
        distance_threshold: float = 0.7,
        return_timing: bool = False,
    ):
        """Tìm kiếm nhiều embedding trong database bằng một lần tính ma trận khoảng cách."""
        if self.embeddings.shape[0] == 0:
            if return_timing:
                return [], {
                    "query_count": 0,
                    "database_count": 0,
                    "backend": "empty-db",
                    "module_ms": {
                        "normalize_query": 0.0,
                        "normalize_db": 0.0,
                        "distance": 0.0,
                        "build_results": 0.0,
                        "total": 0.0,
                    },
                }
            return []

        total_start = time.perf_counter()

        t0 = time.perf_counter()
        normalized_queries = self._normalize_embeddings(embeddings)
        normalize_query_ms = (time.perf_counter() - t0) * 1000

        name_by_id = self._get_name_by_id_cache()

        if self._can_use_faiss() and self._faiss_backend.is_ready():
            backend_used = "faiss"
            with self._write_lock:
                t0 = time.perf_counter()
                results, normalize_db_ms, build_results_ms = self._faiss_backend.search(
                    normalized_queries=normalized_queries,
                    total_database=int(self.embeddings.shape[0]),
                    k=k,
                    distance_threshold=distance_threshold,
                    name_by_id=name_by_id,
                )
                distance_ms = (time.perf_counter() - t0) * 1000
        else:
            backend_used = "sklearn"
            results, normalize_db_ms, distance_ms, build_results_ms = self._sklearn_backend.search(
                normalized_queries=normalized_queries,
                embeddings=self.embeddings,
                k=k,
                distance_threshold=distance_threshold,
                name_by_id=name_by_id,
            )

        total_ms = (time.perf_counter() - total_start) * 1000

        if return_timing:
            return results, {
                "query_count": int(normalized_queries.shape[0]),
                "database_count": int(self.embeddings.shape[0]),
                "backend": backend_used,
                "module_ms": {
                    "normalize_query": round(normalize_query_ms, 3),
                    "normalize_db": round(normalize_db_ms, 3),
                    "distance": round(distance_ms, 3),
                    "build_results": round(build_results_ms, 3),
                    "total": round(total_ms, 3),
                },
            }

        return results

    def save(self):
        """Lưu embeddings và metadata"""
        with self._write_lock:
            self._storage.save(embeddings=self.embeddings, metadata=self.metadata)

    def get_all_faces(self) -> List[dict]:
        """Lấy toàn bộ thông tin khuôn mặt"""
        return list(self.metadata.values())

    def delete_face(self, face_id: int):
        """Xóa khuôn mặt (rebuild array)"""
        with self._write_lock:
            if str(face_id) in self.metadata:
                del self.metadata[str(face_id)]

                self.embeddings = np.delete(self.embeddings, face_id, axis=0)
                self._invalidate_runtime_caches()

                new_metadata = {}
                for idx, (_, data) in enumerate(self.metadata.items()):
                    data["id"] = idx
                    new_metadata[str(idx)] = data

                self.metadata = new_metadata
                self._invalidate_faiss_index()
                self.save()

    def reset(self):
        """Reset toàn bộ database"""
        with self._write_lock:
            self.embeddings = np.empty((0, 512), dtype=np.float32)
            self._invalidate_runtime_caches()
            self.metadata = {}
            self._invalidate_faiss_index()
            images_dir = self.db_path / "images"
            if images_dir.exists():
                shutil.rmtree(images_dir, ignore_errors=True)
            images_dir.mkdir(parents=True, exist_ok=True)
            self.save()
            print("✓ Database reset")

    def get_stats(self) -> dict:
        """Lấy thống kê database"""
        total_faces = len(self.metadata)
        total_people = len(set([m["name"] for m in self.metadata.values()]))

        return {
            "total_faces": total_faces,
            "total_people": total_people,
            "embeddings_shape": self.embeddings.shape,
            "db_size_mb": (self.embeddings.nbytes + sum(len(str(m)) for m in self.metadata.values())) / (1024 * 1024),
            "search_backend": "faiss" if self._can_use_faiss() else "sklearn",
            "search_backend_info": self.get_search_backend_info(),
        }
