"""Tests for ONNX on-device embeddings."""

import pytest

from runeextract.embeddings.onnx import ONNXEmbeddingModel


class TestONNXInit:
    def test_create_model(self):
        model = ONNXEmbeddingModel("test-model")
        assert model.model_id == "test-model"
        assert model.provider == "CPUExecutionProvider"

    def test_no_onnx_runtime(self, monkeypatch):
        monkeypatch.setattr("runeextract.embeddings.onnx.ONNXEmbeddingModel._ensure_loaded", lambda self: (_ for _ in ()).throw(ImportError("no onnxruntime")))
        model = ONNXEmbeddingModel("test")
        with pytest.raises(ImportError):
            model.embed(["hello"])


class TestEmbed:
    def test_embed_one_method(self):
        model = ONNXEmbeddingModel("test")
        assert hasattr(model, "embed_one")

    def test_simple_tokenize(self):
        model = ONNXEmbeddingModel("test")
        import numpy as np
        tokens = model._simple_tokenize(["hello world"])
        assert isinstance(tokens, np.ndarray)
        assert tokens.shape[0] == 1

    def test_simple_tokenize_long(self):
        model = ONNXEmbeddingModel("test")
        tokens = model._simple_tokenize(["a" * 200], max_length=50)
        assert tokens.shape[1] == 50

    def test_mean_pooling(self):
        model = ONNXEmbeddingModel("test")
        import numpy as np
        emb = np.array([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]], dtype=np.float32)
        mask = np.array([[1, 1, 0]], dtype=np.int64)
        pooled = model._mean_pooling(emb, mask)
        assert pooled.shape == (1, 2)
        assert pooled[0][0] == pytest.approx(2.0)
        assert pooled[0][1] == pytest.approx(3.0)


class TestGetONNXEmbedding:
    def test_importable(self):
        from runeextract.embeddings import get_onnx_embedding
        fn = get_onnx_embedding
        assert callable(fn)
