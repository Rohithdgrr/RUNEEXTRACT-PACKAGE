"""RuneExtract Embeddings — on-device embedding generation."""

from runeextract.embeddings.onnx import ONNXEmbeddingModel, get_onnx_embedding

__all__ = [
    "ONNXEmbeddingModel", "get_onnx_embedding",
]
