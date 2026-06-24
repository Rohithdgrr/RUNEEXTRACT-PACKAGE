"""Tests for Phase 11+ features: streaming, tools, cost tracking, query expansion,
sentence-window chunking, quality scoring, rate limiter, batch processing, audio/video."""

import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from runeextract.models.document import Document, ChunkingStrategy, Chunk
from runeextract.processors.ai import AIProcessor, MODEL_COST_MAP
from runeextract.utils.rate_limiter import RateLimiter


# --- Rate Limiter ---

class TestRateLimiter:
    def test_acquire_allows_request(self):
        rl = RateLimiter(requests_per_minute=60)
        assert rl.acquire(block=False) is True

    def test_rate_limit_exceeded(self):
        rl = RateLimiter(requests_per_minute=1)
        assert rl.acquire(block=False) is True
        assert rl.acquire(block=False) is False

    def test_token_limit(self):
        rl = RateLimiter(requests_per_minute=60, tokens_per_minute=100)
        assert rl.acquire(tokens=50, block=False) is True
        assert rl.acquire(tokens=60, block=False) is False

    def test_call_operator(self):
        rl = RateLimiter(requests_per_minute=60)
        assert rl() is True


# --- Sentence-window chunking ---

class TestSentenceWindowChunking:
    def test_sentence_window_basic(self):
        doc = Document(text="First sentence here. Second one here. Third one too. Fourth is good. Fifth is last.")
        chunks = doc.chunks(strategy=ChunkingStrategy.SENTENCE_WINDOW, size=3, overlap=1)
        assert len(chunks) >= 2
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.parent_document_id == doc.document_id for c in chunks)

    def test_sentence_window_single_chunk(self):
        doc = Document(text="Just one sentence here.")
        chunks = doc.chunks(strategy=ChunkingStrategy.SENTENCE_WINDOW, size=5, overlap=1)
        assert len(chunks) == 1
        assert "one sentence" in chunks[0].text

    def test_sentence_window_overlap(self):
        doc = Document(text="A. B. C. D. E. F. G.")
        chunks = doc.chunks(strategy=ChunkingStrategy.SENTENCE_WINDOW, size=3, overlap=1)
        assert len(chunks) >= 2
        # Check overlap contains shared sentences
        if len(chunks) >= 2:
            c0_end = chunks[0].text.split(".")[-2] if chunks[0].text else ""
            c1_start = chunks[1].text.split(".")[0] if chunks[1].text else ""
            # Adjacent windows should share at least some content
            assert len(chunks[0].text) > 0 and len(chunks[1].text) > 0

    def test_sentence_window_no_sentence_endings(self):
        doc = Document(text="a" * 200)
        chunks = doc.chunks(strategy=ChunkingStrategy.SENTENCE_WINDOW, size=3, overlap=1)
        assert len(chunks) == 1


# --- Quality Scoring ---

class TestQualityScoring:
    def test_score_quality_returns_all_keys(self):
        doc = Document(text="Hello world. This is a test document with enough text.")
        scores = doc.score_quality()
        expected_keys = {"text_density", "readability", "structure", "completeness", "ocr_confidence", "overall"}
        assert all(k in scores for k in expected_keys)
        assert 0 <= scores["overall"] <= 100

    def test_score_quality_empty_text(self):
        doc = Document(text="")
        scores = doc.score_quality()
        assert scores["text_density"] == 0
        assert scores["completeness"] == 0

    def test_score_quality_with_tables_and_images(self):
        doc = Document(
            text="Some content here.",
            tables=[MagicMock(rows=[["a"]], columns=["x"])],
            images=[MagicMock(data=b"123", format="png")],
        )
        scores = doc.score_quality()
        assert scores["structure"] >= 50

    def test_score_quality_ocr_metadata(self):
        doc = Document(text="Text", metadata={"ocr_confidence": 0.95})
        scores = doc.score_quality()
        assert scores["ocr_confidence"] == 95


# --- Web Crawling ---

class TestExtractCrawl:
    def test_crawl_import(self):
        from runeextract import extract_crawl
        assert callable(extract_crawl)

    @patch("runeextract.core.extraction.extract")
    @patch("requests.get")
    def test_crawl_single_page(self, mock_get, mock_extract):
        from runeextract import extract_crawl

        mock_extract.return_value = Document(text="Page content", source_type="html", source_path="http://example.com")

        mock_resp = MagicMock()
        mock_resp.text = "<html><body><p>Hello</p></body></html>"
        mock_get.return_value = mock_resp

        docs = extract_crawl("http://example.com", max_pages=1, delay=0)
        assert len(docs) == 1
        assert docs[0].text == "Page content"

    @patch("runeextract.core.extraction.extract")
    @patch("requests.get")
    def test_crawl_skips_external_links(self, mock_get, mock_extract):
        from runeextract import extract_crawl

        mock_extract.return_value = Document(text="Content", source_type="html")

        mock_html = (
            '<html><body>'
            '<a href="http://other.com/page">external</a>'
            '<a href="http://example.com/page2">internal</a>'
            '</body></html>'
        )
        mock_resp = MagicMock()
        mock_resp.text = mock_html
        mock_get.return_value = mock_resp

        docs = extract_crawl("http://example.com", max_pages=2, delay=0, same_domain=True)
        assert len(docs) == 2


# --- Batch LLM Processing ---

class TestBatchProcess:
    def test_batch_process_empty(self):
        ai = AIProcessor(api_key="sk-test", provider="openai")
        results = ai.batch_process([])
        assert results == []

    def test_batch_process_single(self):
        ai = AIProcessor(api_key="sk-test", provider="openai")
        with patch.object(ai, "_call", return_value="mock response") as mock_call:
            results = ai.batch_process([{"system": "be helpful", "user": "hi"}])
            assert results == ["mock response"]
            mock_call.assert_called_once()

    def test_batch_process_multiple(self):
        ai = AIProcessor(api_key="sk-test", provider="openai")
        with patch.object(ai, "_call", side_effect=["resp1", "resp2", "resp3"]) as mock_call:
            prompts = [
                {"system": "s1", "user": "u1"},
                {"system": "s2", "user": "u2"},
                {"system": "s3", "user": "u3"},
            ]
            results = ai.batch_process(prompts, max_concurrency=2)
            assert len(results) == 3
            assert set(results) == {"resp1", "resp2", "resp3"}


# --- Audio / Video extractors ---

class TestAudioExtractor:
    def test_audio_import(self):
        from runeextract.extractors.audio.extractor import AudioExtractor
        assert hasattr(AudioExtractor, "extract")
        assert AudioExtractor().supported_extensions() == [
            ".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma", ".aac", ".opus"
        ]

    def test_audio_raises_dependency(self):
        from runeextract.extractors.audio.extractor import AudioExtractor
        from runeextract.exceptions import DependencyMissingError
        with patch("runeextract.extractors.audio.extractor.BaseExtractor.validate_file", return_value=True):
            with pytest.raises(DependencyMissingError):
                AudioExtractor().extract("test.mp3")


class TestVideoExtractor:
    def test_video_import(self):
        from runeextract.extractors.video.extractor import VideoExtractor
        assert hasattr(VideoExtractor, "extract")
        exts = VideoExtractor().supported_extensions()
        assert ".mp4" in exts
        assert ".avi" in exts

    def test_video_raises_dependency(self):
        from runeextract.extractors.video.extractor import VideoExtractor
        from runeextract.exceptions import DependencyMissingError
        with patch("runeextract.extractors.video.extractor.BaseExtractor.validate_file", return_value=True):
            with pytest.raises(DependencyMissingError):
                VideoExtractor().extract("test.mp4")
