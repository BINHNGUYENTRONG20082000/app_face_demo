from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from module_ai.config import settings as s

_INSIGHTFACE_DOWNLOAD_PATCHED = False


def _patch_insightface_download_policy() -> None:
    """Chặn insightface.utils.storage tự tải zip khi IVM_INSIGHTFACE_AUTO_DOWNLOAD=0."""
    global _INSIGHTFACE_DOWNLOAD_PATCHED
    if _INSIGHTFACE_DOWNLOAD_PATCHED:
        return
    import os.path as osp

    import insightface.utils.storage as storage

    _orig_download = storage.download

    def _download(sub_dir, name, force=False, root="~/.insightface"):
        dir_path = osp.join(osp.expanduser(root), sub_dir, name)
        if osp.isdir(dir_path) and any(
            f.lower().endswith(".onnx") for f in os.listdir(dir_path)
        ):
            return dir_path
        if not s.IVM_INSIGHTFACE_AUTO_DOWNLOAD:
            pack = s.insightface_pack_dir(name)
            raise RuntimeError(
                "InsightFace pack không có sẵn local và IVM_INSIGHTFACE_AUTO_DOWNLOAD=0 — "
                f"không tải từ mạng. Đặt file .onnx vào: {pack} "
                f"(hoặc IVM_INSIGHTFACE_ROOT={s.IVM_INSIGHTFACE_ROOT})."
            )
        return _orig_download(sub_dir, name, force=force, root=root)

    storage.download = _download
    storage.ensure_available = lambda sub_dir, name, root="~/.insightface": _download(
        sub_dir, name, force=False, root=root
    )
    _INSIGHTFACE_DOWNLOAD_PATCHED = True


@dataclass
class DetectedFace:
    bbox: np.ndarray  # (4,) xyxy
    det_score: float
    embedding: np.ndarray  # L2-normalized float32 (512,)
    gender: Optional[int] = None
    age: Optional[int] = None


class InsightFaceEngine:
    """InsightFace detection + recognition; `get_feat` theo lô (chunk IVM_REC_GET_FEAT_MAX_BATCH), có thể gom crop từ nhiều ảnh."""

    _ALLOWED_MODULES = ("detection", "recognition")

    def __init__(
        self,
        model_name: Optional[str] = None,
        root: Optional[str] = None,
        providers: Optional[List[str]] = None,
        ctx_id: Optional[int] = None,
        det_size: Optional[Tuple[int, int]] = None,
    ):
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align

        _patch_insightface_download_policy()
        self._face_align = face_align

        self.model_name = model_name or s.IVM_INSIGHTFACE_MODEL_NAME
        root = root if root is not None else s.IVM_INSIGHTFACE_ROOT
        s.validate_insightface_pack(self.model_name)
        providers = list(providers or s.IVM_INSIGHTFACE_PROVIDERS)
        ctx_id = s.IVM_CTX_ID if ctx_id is None else ctx_id
        det_size = tuple(det_size or s.IVM_DET_SIZE)

        kwargs: Dict[str, Any] = {
            "name": self.model_name,
            "providers": providers,
            "allowed_modules": list(self._ALLOWED_MODULES),
        }
        if root:
            kwargs["root"] = root
        self._app = FaceAnalysis(**kwargs)
        self._app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=float(s.IVM_DET_THRESH))

        if "recognition" not in self._app.models:
            raise RuntimeError(
                f"Pack `{self.model_name}` không có model recognition sau khi lọc "
                f"{self._ALLOWED_MODULES}; kiểm tra thư mục model InsightFace."
            )

        self._rec_model = self._app.models["recognition"]
        self._align_image_size = int(self._rec_model.input_size[0])

    def log_loaded_models(self) -> None:
        """In đường dẫn file ONNX và Ort providers đang hoạt động (CUDAExecutionProvider = GPU)."""
        root = Path(s.IVM_INSIGHTFACE_ROOT).expanduser().resolve()
        pack_dir = s.insightface_pack_dir(self.model_name)
        lines = [
            "=== IVM InsightFace ONNX ===",
            f"pack_name={self.model_name}",
            f"models_root={root}",
            f"pack_dir={pack_dir}",
            f"requested_providers={list(s.IVM_INSIGHTFACE_PROVIDERS)}",
            f"IVM_CTX_ID={s.IVM_CTX_ID}",
        ]
        try:
            import onnxruntime as ort

            lines.append(f"ort_available_providers={list(ort.get_available_providers())}")
        except Exception as ex:
            lines.append(f"ort_available_providers=(error: {ex})")

        for label, model in (
            ("detection", self._app.det_model),
            ("recognition", self._rec_model),
        ):
            mf = getattr(model, "model_file", None)
            sess = getattr(model, "session", None)
            mp = ""
            if mf:
                try:
                    mp = str(Path(str(mf)).expanduser().resolve())
                except Exception:
                    mp = str(mf)
            prov = list(sess.get_providers()) if sess is not None else []
            lines.append(f"  [{label}] onnx_file={mp or '?'}")
            lines.append(f"  [{label}] ort_session_providers(active)={prov}")
            if sess is not None:
                try:
                    opts = sess.get_provider_options()
                    if opts:
                        lines.append(f"  [{label}] ort_provider_options={opts}")
                except Exception:
                    pass
        lines.append("=== end IVM ONNX ===")
        print("\n".join(lines), flush=True)

    def get_runtime_info(self) -> Dict[str, Any]:
        try:
            import onnxruntime as ort

            avail = list(ort.get_available_providers())
        except Exception:
            avail = []
        return {
            "model": self.model_name,
            "requested_providers": list(s.IVM_INSIGHTFACE_PROVIDERS),
            "available_ort_providers": avail,
            "det_size": list(s.IVM_DET_SIZE),
            "det_thresh": float(s.IVM_DET_THRESH),
            "allowed_modules": list(self._ALLOWED_MODULES),
            "rec_get_feat_max_batch": int(s.IVM_REC_GET_FEAT_MAX_BATCH),
        }

    def recognition_feature_dim(self) -> int:
        try:
            sh = getattr(self._rec_model, "output_shape", None)
            if sh is None:
                return 512
            v = sh[-1]
            return int(v) if v is not None else 512
        except Exception:
            return 512

    def detect_align_faces(
        self, image_bgr: np.ndarray
    ) -> Tuple[float, List[np.ndarray], List[Tuple[np.ndarray, float]]]:
        """
        Chỉ detect + norm_crop; không gọi recognition.
        Trả (detect_ms, aligned_crops, meta) với meta[i] = (bbox_xyxy, det_score).
        """
        if image_bgr is None or image_bgr.size == 0:
            return 0.0, [], []

        t_det = time.perf_counter()
        det, kpss = self._app.det_model.detect(image_bgr)
        detect_ms = (time.perf_counter() - t_det) * 1000

        if det.shape[0] == 0:
            return detect_ms, [], []

        aligned: List[np.ndarray] = []
        meta: List[Tuple[np.ndarray, float]] = []

        for i in range(det.shape[0]):
            bbox = det[i, :4].astype(np.float32, copy=False)
            det_score = float(det[i, 4])
            if kpss is None:
                continue
            kps = np.asarray(kpss[i], dtype=np.float32)
            if kps.shape != (5, 2):
                continue
            try:
                crop = self._face_align.norm_crop(
                    image_bgr,
                    landmark=kps,
                    image_size=self._align_image_size,
                )
            except Exception:
                continue
            aligned.append(crop)
            meta.append((bbox, det_score))

        return detect_ms, aligned, meta

    def embed_aligned_crops(
        self,
        aligned: List[np.ndarray],
        *,
        max_batch: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        ONNX recognition theo lô (có thể gom crop từ nhiều ảnh).
        Trả (features (N, D) đã L2-normalize theo từng hàng, embedding_ms).
        """
        dim = self.recognition_feature_dim()
        if not aligned:
            return np.empty((0, dim), dtype=np.float32), 0.0

        bs = int(max_batch) if max_batch is not None else int(s.IVM_REC_GET_FEAT_MAX_BATCH)
        bs = max(1, min(256, bs))

        t0 = time.perf_counter()
        parts: List[np.ndarray] = []
        for start in range(0, len(aligned), bs):
            chunk = aligned[start : start + bs]
            feat = self._rec_model.get_feat(chunk)
            feat = np.asarray(feat, dtype=np.float32)
            if feat.ndim == 1:
                feat = feat.reshape(1, -1)
            parts.append(feat)
        feats = np.vstack(parts) if len(parts) > 1 else parts[0]

        out_rows: List[np.ndarray] = []
        for j in range(feats.shape[0]):
            emb = feats[j].reshape(-1).astype(np.float32, copy=False)
            n = float(np.linalg.norm(emb))
            if n > 0:
                emb = emb / n
            out_rows.append(emb)
        stacked = np.stack(out_rows, axis=0) if out_rows else np.empty((0, dim), dtype=np.float32)
        embedding_ms = (time.perf_counter() - t0) * 1000
        return stacked, embedding_ms

    def analyze_bgr(
        self,
        image_bgr: np.ndarray,
        timing_out: Optional[Dict[str, float]] = None,
    ) -> List[DetectedFace]:
        """`timing_out`: ghi `detect_ms`, `embedding_ms` (align + ONNX recognition, có chunk theo IVM_REC_GET_FEAT_MAX_BATCH)."""
        z = {"detect_ms": 0.0, "embedding_ms": 0.0}

        if image_bgr is None or image_bgr.size == 0:
            if timing_out is not None:
                timing_out.clear()
                timing_out.update(z)
            return []

        detect_ms, aligned, meta = self.detect_align_faces(image_bgr)

        if not aligned:
            if timing_out is not None:
                timing_out.clear()
                timing_out.update({"detect_ms": detect_ms, "embedding_ms": 0.0})
            return []

        feats, embedding_ms = self.embed_aligned_crops(
            aligned, max_batch=int(s.IVM_REC_GET_FEAT_MAX_BATCH)
        )

        out: List[DetectedFace] = []
        for j in range(feats.shape[0]):
            emb = feats[j].reshape(-1).astype(np.float32, copy=False)
            bbox, score = meta[j]
            out.append(
                DetectedFace(
                    bbox=bbox,
                    det_score=score,
                    embedding=emb,
                    gender=None,
                    age=None,
                )
            )

        if timing_out is not None:
            timing_out.clear()
            timing_out.update({"detect_ms": detect_ms, "embedding_ms": embedding_ms})

        return out
