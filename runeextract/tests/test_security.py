"""
Security tests covering deserialization, SSRF, path traversal, and prompt injection.
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

import pytest

from runeextract.exceptions import SSRFBlockedError, URLBlockedError, PathTraversalError


# ---------------------------------------------------------------------------
# FAISS metadata — no pickle deserialization
# ---------------------------------------------------------------------------

class TestFAISSMetadata:
    def test_metadata_written_as_json(self, tmp_path):
        pytest.importorskip("faiss")
        from runeextract.models.document import Document
        doc = Document(text="test", source_type="text")
        doc._chunks = [type("Chunk", (), {"text": "test", "chunk_id": "0",
                                            "metadata": {}} )()]
        path = os.path.join(tmp_path, "faiss_index")
        doc.to_faiss(path)
        meta_path = path + ".meta.json"
        assert os.path.exists(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["text"] == "test"

    def test_metadata_no_pickle_file(self, tmp_path):
        pytest.importorskip("faiss")
        from runeextract.models.document import Document
        doc = Document(text="test", source_type="text")
        doc._chunks = [type("Chunk", (), {"text": "test", "chunk_id": "0",
                                            "metadata": {}} )()]
        path = os.path.join(tmp_path, "faiss_index")
        doc.to_faiss(path)
        assert not os.path.exists(path + ".meta.pkl")

    def test_faiss_retriever_reads_json(self, tmp_path):
        pytest.importorskip("faiss")
        import faiss
        import numpy as np
        from runeextract.rag.retriever import FAISSRetriever
        path = os.path.join(tmp_path, "idx")
        index = faiss.IndexFlatL2(384)
        index.add(np.random.random((2, 384)).astype(np.float32))
        faiss.write_index(index, path + ".index")
        meta = [{"text": "doc a", "source": "a.txt", "source_type": "text",
                 "document_id": "", "chunk_id": "0", "page": None},
                {"text": "doc b", "source": "b.txt", "source_type": "text",
                 "document_id": "", "chunk_id": "1", "page": None}]
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
        retriever = FAISSRetriever(index_path=path)
        retriever._load()
        assert len(retriever._metadata) == 2

    def test_faiss_retriever_fallback_pkl(self, tmp_path):
        pytest.importorskip("faiss")
        import faiss
        import numpy as np
        import pickle
        from runeextract.rag.retriever import FAISSRetriever
        path = os.path.join(tmp_path, "idx")
        index = faiss.IndexFlatL2(384)
        index.add(np.random.random((2, 384)).astype(np.float32))
        faiss.write_index(index, path + ".index")
        meta = [{"text": "legacy"}]
        with open(path + ".meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        retriever = FAISSRetriever(index_path=path)
        retriever._load()
        assert retriever._metadata[0]["text"] == "legacy"

    def test_faiss_retriever_index_not_found(self, tmp_path):
        pytest.importorskip("faiss")
        from runeextract.rag.retriever import FAISSRetriever
        path = os.path.join(tmp_path, "nonexistent")
        retriever = FAISSRetriever(index_path=path)
        with pytest.raises(FileNotFoundError):
            retriever._load()


# ---------------------------------------------------------------------------
# SSRF protection — DNS-based
# ---------------------------------------------------------------------------

class TestSSRFProtection:
    def test_private_ip_blocked(self):
        from runeextract.core.router import URLValidator
        with pytest.raises(SSRFBlockedError):
            URLValidator.validate("http://192.168.1.1/admin")

    def test_localhost_blocked(self):
        from runeextract.core.router import URLValidator
        with pytest.raises(SSRFBlockedError):
            URLValidator.validate("http://localhost:8080/secret")

    def test_loopback_blocked(self):
        from runeextract.core.router import URLValidator
        with pytest.raises(SSRFBlockedError):
            URLValidator.validate("http://127.0.0.1/api")

    def test_dns_to_private_blocked(self):
        from runeextract.core.router import URLValidator
        with patch("socket.gethostbyname", return_value="10.0.0.1"):
            with pytest.raises(SSRFBlockedError):
                URLValidator.validate("http://evil.example.com/file.pdf")

    def test_dns_to_public_allowed(self):
        from runeextract.core.router import URLValidator
        with patch("socket.gethostbyname", return_value="93.184.216.34"):
            URLValidator.validate("http://example.com/file.pdf")

    def test_dns_rebinding_revalidated(self):
        from runeextract.core.router import URLValidator
        """DNS rebinding attack: first resolve public, second resolve private."""
        calls = {"count": 0}
        def _dns_side_effect(hostname):
            calls["count"] += 1
            return "93.184.216.34" if calls["count"] == 1 else "10.0.0.1"

        with patch("socket.gethostbyname", side_effect=_dns_side_effect):
            URLValidator.validate("http://evil.example.com/file.pdf")
            with pytest.raises(SSRFBlockedError):
                URLValidator.validate("http://evil.example.com/file.pdf")


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------

class TestPathTraversal:
    def test_basic_path_traversal(self):
        from runeextract.core.router import _check_path_traversal
        with pytest.raises(PathTraversalError):
            _check_path_traversal("../../../etc/passwd")

    def test_null_byte_attack(self):
        from runeextract.core.router import _check_path_traversal
        with pytest.raises(PathTraversalError):
            _check_path_traversal("file.txt\x00.exe")

    def test_normal_absolute_path_allowed(self):
        from runeextract.core.router import _check_path_traversal
        _check_path_traversal("C:\\Users\\test\\file.txt")

    def test_windows_unc_path_blocked(self):
        from runeextract.core.router import _check_path_traversal
        with pytest.raises(PathTraversalError):
            _check_path_traversal("\\\\?\\C:\\Windows\\System32\\config\\SAM")

    def test_normal_path_allowed(self):
        from runeextract.core.router import _check_path_traversal
        _check_path_traversal("report.pdf")

    def test_double_slash_blocked(self):
        from runeextract.core.router import _check_path_traversal
        with pytest.raises(PathTraversalError):
            _check_path_traversal("//server/share/malicious.exe")

    def test_trailing_dotdot_blocked(self):
        from runeextract.core.router import _check_path_traversal
        with pytest.raises(PathTraversalError):
            _check_path_traversal("folder/..")


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------

class TestSecurityImports:
    def test_ssrf_exceptions_importable(self):
        from runeextract.exceptions import SSRFBlockedError, URLBlockedError
        assert SSRFBlockedError is not None
        assert URLBlockedError is not None
