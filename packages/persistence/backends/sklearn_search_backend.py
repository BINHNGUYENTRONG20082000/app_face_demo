import time
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_distances


class SklearnSearchBackend:
    def __init__(self) -> None:
        self._normalized_db_cache: Optional[np.ndarray] = None

    def invalidate(self) -> None:
        self._normalized_db_cache = None

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    def _build_search_results(
        self,
        distances: np.ndarray,
        k: int,
        distance_threshold: float,
        name_by_id: List[Optional[str]],
    ) -> List[List[dict]]:
        db_count = distances.shape[1]
        if db_count == 0:
            return [[] for _ in range(distances.shape[0])]

        top_k = max(1, min(int(k), db_count))

        if top_k < db_count:
            partition_indices = np.argpartition(distances, kth=top_k - 1, axis=1)[:, :top_k]
            partition_distances = np.take_along_axis(distances, partition_indices, axis=1)
            local_order = np.argsort(partition_distances, axis=1)
            top_indices = np.take_along_axis(partition_indices, local_order, axis=1)
        else:
            top_indices = np.argsort(distances, axis=1)

        batch_results: List[List[dict]] = []
        for row_idx, row_indices in enumerate(top_indices):
            results = []
            for idx in row_indices:
                idx_int = int(idx)
                dist = float(distances[row_idx, idx_int])
                if dist > distance_threshold:
                    continue

                if idx_int < 0 or idx_int >= len(name_by_id):
                    continue

                person_name = name_by_id[idx_int]
                if person_name is None:
                    continue

                results.append(
                    {
                        "distance": dist,
                        "name": person_name,
                        "face_id": idx_int,
                    }
                )

            batch_results.append(results)

        return batch_results

    def search(
        self,
        normalized_queries: np.ndarray,
        embeddings: np.ndarray,
        k: int,
        distance_threshold: float,
        name_by_id: List[Optional[str]],
    ) -> Tuple[List[List[dict]], float, float, float]:
        t0 = time.perf_counter()
        if self._normalized_db_cache is None:
            self._normalized_db_cache = self._normalize_embeddings(embeddings)
        normalized_database = self._normalized_db_cache
        normalize_db_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        distances = cosine_distances(normalized_queries, normalized_database)
        distance_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        results = self._build_search_results(
            distances=distances,
            k=k,
            distance_threshold=distance_threshold,
            name_by_id=name_by_id,
        )
        build_results_ms = (time.perf_counter() - t0) * 1000

        return results, normalize_db_ms, distance_ms, build_results_ms
