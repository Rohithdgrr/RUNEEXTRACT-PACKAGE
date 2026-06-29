"""Tests for Tier 3 features: signing, server, OCR, TOC, quality, export, fine-tuning."""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---- Signing ----


class TestSigning:
    def test_sign_document_needs_cryptography(self):
        try:
            import cryptography  # noqa: F401
            pytest.skip("cryptography is installed — skipping ImportError test")
        except ImportError:
            from runeextract.models.document import Document
            from runeextract.signing import sign_document
            doc = Document(text="hello world")
            with pytest.raises(ImportError):
                sign_document(doc, b"\x00" * 32)

    @patch.dict("sys.modules", {"cryptography": MagicMock()})
    def test_compute_document_hash(self):
        from runeextract.models.document import Document
        from runeextract.signing import compute_document_hash
        doc = Document(text="test content", source_path="/path/to/file.pdf", source_type="pdf")
        h = compute_document_hash(doc)
        assert isinstance(h, str)
        assert len(h) == 64

    @patch.dict("sys.modules", {"cryptography": MagicMock()})
    def test_compute_hash_different_text(self):
        from runeextract.models.document import Document
        from runeextract.signing import compute_document_hash
        doc1 = Document(text="hello")
        doc2 = Document(text="world")
        assert compute_document_hash(doc1) != compute_document_hash(doc2)

    @patch.dict("sys.modules", {"cryptography": MagicMock()})
    def test_signature_info_roundtrip(self):
        from runeextract.signing import SignatureInfo
        info = SignatureInfo(hash="abc123", public_key_fingerprint="deadbeef", signed_fields=["text"])
        d = info.to_dict()
        restored = SignatureInfo.from_dict(d)
        assert restored.hash == "abc123"
        assert restored.public_key_fingerprint == "deadbeef"
        assert restored.signed_fields == ["text"]

    def test_signing_error_exception(self):
        from runeextract.signing import SigningError
        e = SigningError("test error")
        assert str(e) == "test error"
        assert isinstance(e, Exception)

    def test_verification_error_exception(self):
        from runeextract.signing import VerificationError
        e = VerificationError("verify failed")
        assert str(e) == "verify failed"

    @patch.dict("sys.modules", {"cryptography": MagicMock()})
    def test_verify_without_signature_raises(self):
        from runeextract.models.document import Document
        from runeextract.signing import verify_document
        doc = Document(text="test")
        with pytest.raises(Exception):
            verify_document(doc, b"\x00" * 32)

    @patch.dict("sys.modules", {"cryptography": MagicMock()})
    def test_sign_document_bad_key(self):
        from runeextract.models.document import Document
        from runeextract.signing import sign_document
        doc = Document(text="test")
        with pytest.raises(Exception):
            sign_document(doc, b"bad-key")


# ---- Server (WebSocket + HTTP) ----


class TestServer:
    def test_extraction_job_dataclass(self):
        from runeextract.server import ExtractionServer
        server = ExtractionServer(host="127.0.0.1", port=9999)
        assert server.host == "127.0.0.1"
        assert server.port == 9999

    @pytest.mark.asyncio
    async def test_handler_extract_no_file_path(self):
        from runeextract.server import WebSocketHandler
        handler = WebSocketHandler()
        ws = AsyncMock()
        await handler.handle_extract(ws, {})
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "error"
        assert "file_path" in sent["message"]

    @pytest.mark.asyncio
    async def test_handler_invalid_json(self):
        from runeextract.server import WebSocketHandler
        handler = WebSocketHandler()
        ws = AsyncMock()
        await handler.handle_message(ws, "not json")
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "error"

    @pytest.mark.asyncio
    async def test_handler_unknown_type(self):
        from runeextract.server import WebSocketHandler
        handler = WebSocketHandler()
        ws = AsyncMock()
        await handler.handle_message(ws, json.dumps({"type": "unknown_type"}))
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "error"
        assert "unknown" in sent["message"]

    @pytest.mark.asyncio
    async def test_handler_status_not_found(self):
        from runeextract.server import WebSocketHandler
        handler = WebSocketHandler()
        ws = AsyncMock()
        await handler.handle_message(ws, json.dumps({"type": "status", "job_id": "nonexistent"}))
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "error"

    @pytest.mark.asyncio
    async def test_handler_list_jobs(self):
        from runeextract.server import WebSocketHandler
        handler = WebSocketHandler()
        ws = AsyncMock()
        await handler.handle_message(ws, json.dumps({"type": "list"}))
        ws.send.assert_called_once()
        sent = json.loads(ws.send.call_args[0][0])
        assert sent["type"] == "job_list"

    @pytest.mark.asyncio
    async def test_start_stop_server(self):
        mock_runner = MagicMock()
        mock_runner.setup = AsyncMock()
        mock_runner.cleanup = AsyncMock()
        with patch("aiohttp.web.AppRunner", return_value=mock_runner):
            from runeextract.server import ExtractionServer
            server = ExtractionServer()
            await server.start()
            await server.stop()
            assert mock_runner.setup.called


# ---- OCR Improvements ----


class TestOCREnhanced:
    def test_detect_english(self):
        from runeextract.ocr import OCRLanguageDetector
        detector = OCRLanguageDetector()
        result = detector.detect("Hello world, this is English text.")
        assert result[0][0] == "eng"

    def test_detect_cjk(self):
        from runeextract.ocr import OCRLanguageDetector
        detector = OCRLanguageDetector()
        result = detector.detect("你好世界")
        assert result[0][0] == "cmn"

    def test_detect_cyrillic(self):
        from runeextract.ocr import OCRLanguageDetector
        detector = OCRLanguageDetector()
        result = detector.detect("Привет мир")
        assert result[0][0] == "rus"

    def test_multi_lang(self):
        from runeextract.ocr import OCRLanguageDetector
        detector = OCRLanguageDetector()
        langs = detector.multi_lang("Hello 你好", threshold=0.01)
        assert "cmn" in langs or "jpn" in langs

    def test_tesseract_langs_multi(self):
        from runeextract.ocr import OCRLanguageDetector
        detector = OCRLanguageDetector()
        # Mixed English and CJK
        langs = detector.tesseract_langs("Hello 你好 world")
        assert isinstance(langs, list)
        assert len(langs) >= 1

    def test_empty_text(self):
        from runeextract.ocr import OCRLanguageDetector
        detector = OCRLanguageDetector()
        assert detector.detect("")[0][0] == "eng"
        assert detector.detect(None if False else "")[0][0] == "eng"

    def test_detect_text_script(self):
        from runeextract.ocr import detect_text_script
        assert detect_text_script("Hello") in ("Latin",)
        assert detect_text_script("你好") in ("CJK", "Latin")

    def test_supported_languages_expanded(self):
        from runeextract.ocr import OCRLanguageDetector
        detector = OCRLanguageDetector()
        langs = detector.supported_languages()
        assert "fra" in langs
        assert "deu" in langs
        assert "vie" in langs
        assert "tur" in langs
        assert "pol" in langs


# ---- TOC Extraction ----


class TestTOCExtraction:
    def test_parse_markdown_flat(self):
        from runeextract.toc import TOCParser
        parser = TOCParser()
        entries = parser.parse_markdown("# Title\n\n## Section 1\n\n### Subsection\n")
        assert len(entries) == 1
        assert entries[0].title == "Title"
        assert entries[0].level == 1
        assert len(entries[0].children) == 1
        assert entries[0].children[0].title == "Section 1"

    def test_parse_markdown_tree(self):
        from runeextract.toc import TOCParser
        parser = TOCParser()
        entries = parser.parse_markdown("# A\n## B\n## C\n### D\n# E\n")
        assert len(entries) == 2
        assert entries[0].title == "A"
        assert len(entries[0].children) == 2
        assert entries[0].children[0].title == "B"

    def test_parse_html(self):
        from runeextract.toc import TOCParser
        parser = TOCParser()
        entries = parser.parse_html("<h1>Title</h1><p>text</p><h2>Section</h2>")
        assert len(entries) == 1
        assert entries[0].title == "Title"
        assert len(entries[0].children) == 1
        assert entries[0].children[0].title == "Section"

    def test_extract_toc_markdown(self):
        from runeextract.toc import extract_toc
        entries = extract_toc("# Header\n\nSome text.\n\n## Sub\n", format="markdown")
        assert len(entries) == 1

    def test_extract_toc_html(self):
        from runeextract.toc import extract_toc
        entries = extract_toc("<h1>A</h1><h2>B</h2>", format="html")
        assert len(entries) == 1

    def test_toc_to_markdown(self):
        from runeextract.toc import TOCEntry, toc_to_markdown
        entries = [TOCEntry(title="Test", level=1)]
        md = toc_to_markdown(entries)
        assert "- Test" in md

    def test_toc_to_dict(self):
        from runeextract.toc import TOCEntry, toc_to_dict
        entries = [TOCEntry(title="Test", level=1, page_number=5)]
        d = toc_to_dict(entries)
        assert d[0]["title"] == "Test"
        assert d[0]["page_number"] == 5

    def test_toc_to_json(self):
        from runeextract.toc import TOCEntry, toc_to_json
        entries = [TOCEntry(title="Test", level=1)]
        j = toc_to_json(entries)
        assert '"Test"' in j

    def test_parse_pdf_toc(self):
        from runeextract.toc import TOCParser
        parser = TOCParser()
        text = "Introduction........3\n  Methods........7\n    Details........10"
        entries = parser.parse_pdf_toc(text)
        assert len(entries) == 1
        assert entries[0].title == "Introduction"
        assert entries[0].page_number == 3
        assert len(entries[0].children) == 1
        assert entries[0].children[0].children[0].title == "Details"


# ---- Quality Levels / Fast Mode ----


class TestQualityLevels:
    def test_draft_config(self):
        from runeextract.quality import QualityLevel, get_quality_config
        config = get_quality_config(QualityLevel.DRAFT)
        assert config.ocr is False
        assert config.tables is False

    def test_normal_config(self):
        from runeextract.quality import QualityLevel, get_quality_config
        config = get_quality_config(QualityLevel.NORMAL)
        assert config.ocr is True
        assert config.tables is True

    def test_verified_config(self):
        from runeextract.quality import QualityLevel, get_quality_config
        config = get_quality_config(QualityLevel.VERIFIED)
        assert config.ocr is True
        assert config.ai_analysis is True
        assert config.image_dpi == 300

    def test_get_quality_config_string(self):
        from runeextract.quality import get_quality_config
        config = get_quality_config("draft")
        assert config.level.value == "draft"

    def test_get_quality_config_invalid(self):
        from runeextract.quality import get_quality_config
        with pytest.raises(ValueError):
            get_quality_config("nonexistent")

    def test_to_extract_kwargs(self):
        from runeextract.quality import QualityConfig
        config = QualityConfig(ocr=True, tables=False)
        kwargs = config.to_extract_kwargs()
        assert kwargs["ocr"] is True
        assert kwargs["tables"] is False

    def test_extract_with_quality_file_not_found(self):
        from runeextract.quality import extract_with_quality, QualityLevel
        with pytest.raises(FileNotFoundError):
            extract_with_quality("/nonexistent/file.pdf", level=QualityLevel.DRAFT)

    def test_resolution_map(self):
        from runeextract.quality import resolution_map, QualityLevel
        assert resolution_map[QualityLevel.DRAFT] == 72
        assert resolution_map[QualityLevel.NORMAL] == 150
        assert resolution_map[QualityLevel.VERIFIED] == 300

    def test_quality_level_enum(self):
        from runeextract.quality import QualityLevel
        assert list(QualityLevel) == [QualityLevel.DRAFT, QualityLevel.NORMAL, QualityLevel.VERIFIED]


# ---- RAGAS / DSPy / Haystack Export ----


class TestExportFormats:
    def test_to_ragas(self):
        from runeextract.export import to_ragas, ExportRecord
        records = [ExportRecord(question="Q1", answer="A1", contexts=["ctx1"], ground_truth="GT1")]
        result = to_ragas(records)
        data = json.loads(result)
        assert data["questions"] == ["Q1"]
        assert data["answers"] == ["A1"]
        assert data["contexts"] == [["ctx1"]]
        assert data["ground_truths"] == [["GT1"]]

    def test_to_ragas_with_file(self):
        from runeextract.export import to_ragas, ExportRecord
        records = [ExportRecord(question="Q", answer="A")]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            to_ragas(records, output_path=f.name)
            with open(f.name) as f2:
                data = json.load(f2)
            assert data["questions"] == ["Q"]
        os.unlink(f.name)

    def test_to_dspy(self):
        from runeextract.export import to_dspy, ExportRecord
        records = [ExportRecord(question="Q", answer="A", contexts=["c1"], ground_truth="GT")]
        result = to_dspy(records)
        data = json.loads(result)
        assert data[0]["question"] == "Q"
        assert "context" in data[0]

    def test_to_haystack(self):
        from runeextract.export import to_haystack, ExportRecord
        records = [ExportRecord(question="Q", answer="A", source="file.pdf")]
        result = to_haystack(records)
        data = json.loads(result)
        assert "documents" in data
        assert data["documents"][0]["content"] == "A"

    def test_to_llama_index(self):
        from runeextract.export import to_llama_index, ExportRecord
        records = [ExportRecord(question="Q", answer="A")]
        result = to_llama_index(records)
        data = json.loads(result)
        assert data[0]["text"] == "A"
        assert "metadata" in data[0]

    def test_to_jsonl(self):
        from runeextract.export import to_jsonl, ExportRecord
        records = [
            ExportRecord(question="Q1", answer="A1"),
            ExportRecord(question="Q2", answer="A2"),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            to_jsonl(records, f.name)
            with open(f.name) as f2:
                lines = f2.readlines()
            assert len(lines) == 2
        os.unlink(f.name)

    def test_export_document_unsupported(self):
        from runeextract.export import export_document, ExportRecord, ExportFormat, ExportError
        with pytest.raises(ExportError):
            export_document([], "unsupported")

    def test_export_document_ragas(self):
        from runeextract.export import export_document, ExportRecord, ExportFormat
        records = [ExportRecord(question="Q", answer="A")]
        result = export_document(records, ExportFormat.RAGAS)
        data = json.loads(result)
        assert data["questions"] == ["Q"]

    def test_export_record_defaults(self):
        from runeextract.export import ExportRecord
        r = ExportRecord()
        assert r.question == ""
        assert r.answer == ""
        assert r.contexts == []
        assert r.ground_truth == ""


# ---- Fine-Tuning Data Generation ----


class TestFineTuning:
    def test_fine_tune_example_to_dict(self):
        from runeextract.rag.fine_tune import FineTuneExample
        ex = FineTuneExample(instruction="test", input="input text", output="output text")
        d = ex.to_dict()
        assert d["instruction"] == "test"
        assert d["input"] == "input text"
        assert d["output"] == "output text"

    def test_fine_tune_dataset_add_and_save(self):
        from runeextract.rag.fine_tune import FineTuneExample, FineTuneDataset
        dataset = FineTuneDataset()
        dataset.add(FineTuneExample(instruction="i", input="in", output="out"))
        assert dataset.size == 1

    def test_fine_tune_dataset_save_alpaca(self):
        from runeextract.rag.fine_tune import FineTuneExample, FineTuneDataset
        dataset = FineTuneDataset()
        dataset.add(FineTuneExample(instruction="i", output="o"))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            dataset.save(f.name, fmt="alpaca")
            with open(f.name) as f2:
                data = json.load(f2)
            assert data[0]["instruction"] == "i"
        os.unlink(f.name)

    def test_fine_tune_dataset_save_sharegpt(self):
        from runeextract.rag.fine_tune import FineTuneExample, FineTuneDataset
        dataset = FineTuneDataset()
        dataset.add(FineTuneExample(instruction="i", input="in", output="out"))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            dataset.save(f.name, fmt="sharegpt")
            with open(f.name) as f2:
                data = json.load(f2)
            assert "conversations" in data[0]
        os.unlink(f.name)

    def test_fine_tune_dataset_save_messages(self):
        from runeextract.rag.fine_tune import FineTuneExample, FineTuneDataset
        dataset = FineTuneDataset()
        dataset.add(FineTuneExample(instruction="i", input="in", output="out"))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            dataset.save(f.name, fmt="messages")
            with open(f.name) as f2:
                data = json.load(f2)
            assert "messages" in data[0]
        os.unlink(f.name)

    def test_extract_topics(self):
        from runeextract.rag.fine_tune import extract_topics
        text = "John Smith presented the Annual Report. The Board Meeting discussed Revenue Growth."
        topics = extract_topics(text, max_topics=3)
        assert len(topics) >= 1

    def test_extract_topics_short_text(self):
        from runeextract.rag.fine_tune import extract_topics
        assert extract_topics("hello") == []

    def test_generate_examples_empty_doc(self):
        from runeextract.models.document import Document
        from runeextract.rag.fine_tune import generate_examples_from_document
        doc = Document(text="Hi")
        dataset = generate_examples_from_document(doc, num_examples=2)
        assert dataset.size == 0

    def test_generate_examples_normal_doc(self):
        from runeextract.models.document import Document
        from runeextract.rag.fine_tune import generate_examples_from_document
        text = "Artificial Intelligence is transforming how we process documents." * 20
        doc = Document(text=text)
        dataset = generate_examples_from_document(doc, num_examples=3)
        assert dataset.size >= 3, f"Expected >=3, got {dataset.size}"
        for ex in dataset.examples:
            assert ex.instruction
            assert ex.output

    def test_generate_examples_includes_summaries(self):
        from runeextract.models.document import Document
        from runeextract.rag.fine_tune import generate_examples_from_document
        text = ("Machine learning is a subset of artificial intelligence. " * 30)
        text += "\n\n" + ("Deep learning uses neural networks. " * 30)
        text += "\n\n" + ("Natural language processing is a field of AI. " * 30)
        doc = Document(text=text)
        dataset = generate_examples_from_document(doc, num_examples=2, include_summaries=True)
        assert dataset.size >= 2 + 3

    def test_generate_fine_tuning_data_multiple_docs(self):
        from runeextract.models.document import Document
        from runeextract.rag.fine_tune import generate_fine_tuning_data
        docs = [
            Document(text=("Deep learning is powerful. " * 30)),
            Document(text=("Reinforcement learning is fascinating. " * 30)),
        ]
        dataset = generate_fine_tuning_data(docs, examples_per_doc=2)
        assert dataset.size >= 2

    def test_generate_fine_tuning_data_with_output(self):
        from runeextract.models.document import Document
        from runeextract.rag.fine_tune import generate_fine_tuning_data
        docs = [Document(text=("Quantum computing is an emerging technology. " * 30))]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            dataset = generate_fine_tuning_data(docs, examples_per_doc=2, output_path=f.name)
            with open(f.name) as f2:
                data = json.load(f2)
            assert len(data) >= 1
        os.unlink(f.name)
        assert dataset.size >= 1


# ---- Top-level Exports ----


class TestTopLevelExports:
    def test_import_quality_level(self):
        from runeextract import QualityLevel
        assert callable(QualityLevel)

    def test_import_document_signer(self):
        from runeextract import DocumentSigner
        assert callable(DocumentSigner)

    def test_import_extraction_server(self):
        from runeextract import ExtractionServer
        assert callable(ExtractionServer)

    def test_import_export_functions(self):
        from runeextract import to_ragas, to_dspy, to_haystack, to_llama_index
        assert callable(to_ragas)
        assert callable(to_dspy)
        assert callable(to_haystack)
        assert callable(to_llama_index)

    def test_import_fine_tune(self):
        from runeextract import FineTuneExample, FineTuneDataset
        assert callable(FineTuneExample)
        assert callable(FineTuneDataset)
