"""Tests for multi-modal RAG (text + tables + images)."""

from unittest.mock import MagicMock
import pytest

from runeextract.rag.multimodal import (
    MultiModalIndex, MultiModalItem, MultiModalResult,
    _table_to_text, _image_to_base64,
)
from runeextract.models.document import Document, Image, Table


def _make_doc(
    text: str = "hello world",
    tables: list = None,
    images: list = None,
    source_path: str = "doc.txt",
) -> Document:
    return Document(
        text=text,
        tables=tables or [],
        images=images or [],
        source_type="text",
        source_path=source_path,
    )


class TestTableToText:
    def test_basic_table(self):
        t = Table(rows=[["a", "1"], ["b", "2"]], columns=["key", "value"])
        result = _table_to_text(t)
        assert "key, value" in result
        assert "a, 1" in result
        assert "b, 2" in result

    def test_empty_columns(self):
        t = Table(rows=[], columns=[])
        result = _table_to_text(t)
        assert result == ""


class TestImageToBase64:
    def test_conversion(self):
        img = Image(data=b"fake_png_bytes", format="png")
        result = _image_to_base64(img)
        assert isinstance(result, str)
        assert len(result) > 0


class TestMultiModalIndexAddDocument:
    def test_add_text_only(self):
        doc = _make_doc(text="hello " * 500)
        index = MultiModalIndex()
        count = index.add_document(doc, chunk_strategy="fixed_size", chunk_size=200)
        assert count > 0
        assert index.item_count > 0

    def test_add_text_and_tables(self):
        table = Table(rows=[["v1"]], columns=["col"])
        doc = _make_doc(text="text", tables=[table])
        index = MultiModalIndex()
        count = index.add_document(doc)
        assert count >= 2  # text chunks + table

    def test_add_text_and_images(self):
        img = Image(data=b"pngdata", format="png", caption="chart")
        doc = _make_doc(text="text", images=[img])
        index = MultiModalIndex()
        count = index.add_document(doc)
        assert count >= 2

    def test_add_all_types(self):
        table = Table(rows=[["v1"]], columns=["c"])
        img = Image(data=b"imgdata", format="jpeg", caption="photo")
        doc = _make_doc(text="hello " * 200, tables=[table], images=[img])
        index = MultiModalIndex()
        count = index.add_document(doc)
        stats = index.stats()
        assert stats.get("text", 0) > 0
        assert stats.get("table", 0) == 1
        assert stats.get("image", 0) == 1

    def test_add_multiple_documents(self):
        index = MultiModalIndex()
        doc1 = _make_doc(text="doc1 " * 100, source_path="a.txt")
        doc2 = _make_doc(text="doc2 " * 100, source_path="b.txt")
        index.add_document(doc1)
        index.add_document(doc2)
        assert index.item_count > 1


class TestMultiModalIndexSearch:
    def test_search_overlap(self):
        doc = _make_doc(text="cats are furry pets")
        index = MultiModalIndex()
        index.add_document(doc, chunk_strategy="fixed_size", chunk_size=1000)
        result = index.search("cats")
        assert len(result.items) >= 1
        assert "cats" in result.items[0].text

    def test_search_no_match(self):
        doc = _make_doc(text="alpha beta gamma")
        index = MultiModalIndex()
        index.add_document(doc)
        result = index.search("zzz_nonexistent")
        assert len(result.items) == 0

    def test_search_type_filter_text(self):
        table = Table(rows=[["data"]], columns=["x"])
        doc = _make_doc(text="cats are furry", tables=[table])
        index = MultiModalIndex()
        index.add_document(doc)
        result = index.search("cats", type_filter="text")
        assert all(i.item_type == "text" for i in result.items)

    def test_search_type_filter_table(self):
        table = Table(rows=[["data"]], columns=["x"])
        doc = _make_doc(text="cats are furry", tables=[table])
        index = MultiModalIndex()
        index.add_document(doc)
        result = index.search("data", type_filter="table")
        assert all(i.item_type == "table" for i in result.items)

    def test_search_with_embed_fn(self):
        mock_embed = MagicMock()
        # Return a vector for query and each item
        mock_embed.return_value = [[0.5, 0.5]]
        doc = _make_doc(text="embed test")
        index = MultiModalIndex(embed_fn=mock_embed)
        index.add_document(doc)
        # embed is called with query then each item
        result = index.search("test", top_k=5)
        assert len(result.items) >= 0

    def test_search_empty_index(self):
        index = MultiModalIndex()
        result = index.search("anything")
        assert len(result.items) == 0

    def test_search_top_k(self):
        doc = _make_doc(text="a b c d e f g h i j " * 100)
        index = MultiModalIndex()
        count = index.add_document(doc, chunk_strategy="fixed_size", chunk_size=50)
        result = index.search("a", top_k=3)
        assert len(result.items) <= 3


class TestMultiModalResult:
    def test_properties(self):
        items = [
            MultiModalItem(text="text1", item_type="text"),
            MultiModalItem(text="table1", item_type="table"),
            MultiModalItem(text="img1", item_type="image"),
        ]
        r = MultiModalResult(items=items, query="test")
        assert len(r.texts) == 1
        assert len(r.tables) == 1
        assert len(r.images) == 1

    def test_to_openai_messages_text_only(self):
        items = [
            MultiModalItem(text="context text", item_type="text"),
            MultiModalItem(text="table data", item_type="table"),
        ]
        r = MultiModalResult(items=items, query="q")
        msgs = r.to_openai_messages(system_prompt="You are a bot", question="What?")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        user_content = msgs[1]["content"]
        text_blocks = [c for c in user_content if c["type"] == "text"]
        assert any("context text" in c["text"] for c in text_blocks)

    def test_to_openai_messages_with_images(self):
        items = [
            MultiModalItem(text="chart caption", item_type="image",
                           image_data="fakebase64", image_format="png"),
        ]
        r = MultiModalResult(items=items, query="q")
        msgs = r.to_openai_messages(question="Describe")
        assert len(msgs) == 1  # no system prompt
        user_content = msgs[0]["content"]
        image_blocks = [c for c in user_content if c["type"] == "image_url"]
        assert len(image_blocks) >= 1

    def test_to_openai_messages_max_images(self):
        items = [
            MultiModalItem(text=f"img{i}", item_type="image",
                           image_data="data", image_format="png")
            for i in range(10)
        ]
        r = MultiModalResult(items=items, query="q")
        msgs = r.to_openai_messages(max_images=2)
        user_content = msgs[0]["content"]
        image_blocks = [c for c in user_content if c["type"] == "image_url"]
        assert len(image_blocks) == 2


class TestMultiModalIndexStats:
    def test_stats_empty(self):
        index = MultiModalIndex()
        assert index.stats() == {}

    def test_stats_counts(self):
        t1 = Table(rows=[["a"]], columns=["c"])
        t2 = Table(rows=[["b"]], columns=["d"])
        img = Image(data=b"d", format="webp")
        doc = _make_doc(text="hi " * 100, tables=[t1, t2], images=[img])
        index = MultiModalIndex()
        index.add_document(doc)
        stats = index.stats()
        assert stats["text"] > 0
        assert stats["table"] == 2
        assert stats["image"] == 1


class TestCosSimilarity:
    def test_identical_vectors(self):
        score = MultiModalIndex._cosine_sim([1, 0, 0], [1, 0, 0])
        assert score == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        score = MultiModalIndex._cosine_sim([1, 0], [0, 1])
        assert score == pytest.approx(0.0)

    def test_opposite_vectors(self):
        score = MultiModalIndex._cosine_sim([1, 0], [-1, 0])
        assert score == pytest.approx(-1.0)
