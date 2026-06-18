"""ONNX on-device embeddings — run embedding models locally via ONNX Runtime.

Requires: onnxruntime, numpy
Optional: huggingface_hub (for downloading models)
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ONNXEmbeddingModel:
    """Run an ONNX embedding model on-device.

    Downloads the model from HuggingFace Hub on first use if not cached.
    Uses the "onnx" subfolder convention (model.onnx, tokenizer.json).

    Args:
        model_id: HuggingFace model ID (e.g. "sentence-transformers/all-MiniLM-L6-v2")
        provider: ONNX Runtime provider (default: "CPUExecutionProvider")
        cache_dir: Optional cache directory for downloaded models
    """

    def __init__(
        self,
        model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
        provider: str = "CPUExecutionProvider",
        cache_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.provider = provider
        self.cache_dir = cache_dir
        self._session = None
        self._tokenizer = None

    def _ensure_loaded(self):
        if self._session is not None:
            return
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "ONNX embeddings require 'onnxruntime' and 'numpy'. "
                "Install with: pip install onnxruntime numpy"
            ) from e

        model_path = self._get_model_path()
        self._session = ort.InferenceSession(model_path, providers=[self.provider])
        self._load_tokenizer()

    def _get_model_path(self) -> str:
        model_path = self._try_local_cache()
        if model_path:
            return model_path
        return self._download_model()

    def _try_local_cache(self) -> Optional[str]:
        import os
        candidates = [
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
            os.path.join(os.path.dirname(__file__), "models"),
        ]
        for base in candidates:
            path = os.path.join(base, self.model_id.replace("/", "_"), "onnx", "model.onnx")
            if os.path.exists(path):
                return path
        return None

    def _download_model(self) -> str:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "To download ONNX models automatically, install: pip install huggingface_hub"
            )
        model_path = hf_hub_download(
            repo_id=self.model_id,
            filename="onnx/model.onnx",
            cache_dir=self.cache_dir,
        )
        return model_path

    def _load_tokenizer(self):
        try:
            from tokenizers import Tokenizer
            path = self._get_tokenizer_path()
            if path:
                self._tokenizer = Tokenizer.from_file(path)
        except ImportError:
            logger.warning("tokenizers not installed; using manual tokenization fallback")

    def _get_tokenizer_path(self) -> Optional[str]:
        import os
        dir_path = os.path.dirname(self._get_model_path() if self._session is None else "")
        if not dir_path:
            return None
        for name in ("tokenizer.json", "tokenizer_config.json"):
            path = os.path.join(dir_path, name)
            if os.path.exists(path):
                return path
        return None

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure_loaded()
        import numpy as np

        if self._tokenizer:
            encoded = self._tokenizer.encode_batch(texts)
            input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        else:
            input_ids = self._simple_tokenize(texts)
            attention_mask = np.ones_like(input_ids)

        outputs = self._session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        embeddings = outputs[0]
        embeddings = self._mean_pooling(embeddings, attention_mask)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms
        return embeddings.tolist()

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    @staticmethod
    def _simple_tokenize(texts: List[str], max_length: int = 128) -> "np.ndarray":
        import numpy as np
        ids = []
        for text in texts:
            tokens = [ord(c) for c in text[:max_length]]
            tokens = tokens[:max_length]
            ids.append(tokens + [0] * (max_length - len(tokens)))
        return np.array(ids, dtype=np.int64)

    @staticmethod
    def _mean_pooling(embeddings: "np.ndarray", attention_mask: "np.ndarray") -> "np.ndarray":
        import numpy as np
        mask = attention_mask.astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)
        masked = embeddings * mask
        summed = masked.sum(axis=1)
        counts = mask.sum(axis=1).clip(min=1e-9)
        return summed / counts


def get_onnx_embedding(model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> ONNXEmbeddingModel:
    """Get or create a cached ONNX embedding model instance."""
    return ONNXEmbeddingModel(model_id=model_id)
