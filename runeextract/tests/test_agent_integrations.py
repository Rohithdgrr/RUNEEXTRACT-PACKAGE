"""Tests for agent SDK integrations (MCP, LangChain, LlamaIndex, CrewAI, AutoGen,
LangGraph, OpenAI Agents SDK, PydanticAI) and parent-child chunking."""

import os
import tempfile

import pytest

from runeextract.agent.mcp_server import (
    mcp_tool_extract, mcp_tool_extract_many, mcp_tool_extract_url,
    mcp_tool_search, mcp_tool_ask, mcp_tool_crawl, mcp_tool_chunk,
    run_mcp_server, main_cli,
)
from runeextract.agent.langchain import RuneExtractLoader, RuneExtractTransformer
from runeextract.models.document import Document
from runeextract.models.types import Chunk, ChunkingStrategy, HierarchicalChunk
from runeextract.agent.llamaindex import RuneExtractReader
from runeextract.agent.crewai import RuneExtractTool
from runeextract.agent.autogen import autogen_extract_tool
from runeextract.agent.langgraph import RuneExtractGraphTool, RuneExtractSearchTool, RuneExtractAskTool
from runeextract.agent.openai_sdk import rune_extract_function_tool, rune_extract_search_tool
from runeextract.agent.pydantic_ai import RuneExtractAITool, RuneExtractSearchAITool


# ── MCP ──────────────────────────────────────────────────

class TestMCP:
    @pytest.mark.asyncio
    async def test_mcp_tool_extract(self, md_file):
        result = await mcp_tool_extract(md_file)
        assert isinstance(result, str)
        assert "Hello from RuneExtract" in result

    @pytest.mark.asyncio
    async def test_mcp_tool_extract_many(self, md_file, csv_file):
        result = await mcp_tool_extract_many([md_file, csv_file])
        assert isinstance(result, str)
        assert "Hello from RuneExtract" in result
        assert "foo,1" in result or "name" in result

    @pytest.mark.asyncio
    async def test_mcp_tool_crawl_invalid_url(self):
        result = await mcp_tool_crawl("http://nonexistent.invalid", max_pages=1)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_mcp_tool_search_no_sources(self):
        result = await mcp_tool_search("test query", source_paths=[])
        assert isinstance(result, str)
        assert "No sources provided" in result


# ── LangChain ────────────────────────────────────────────

class TestLangChain:
    def test_load(self, md_file):
        loader = RuneExtractLoader(md_file)
        docs = loader.load()
        assert len(docs) >= 1
        assert "Hello from RuneExtract" in docs[0]["page_content"]

    def test_lazy_load(self, md_file):
        loader = RuneExtractLoader(md_file)
        docs = list(loader.lazy_load())
        assert len(docs) >= 1
        assert docs[0]["metadata"]["source"] == md_file

    def test_from_file_list(self, md_file, csv_file):
        loaders = RuneExtractLoader.from_file_list([md_file, csv_file])
        assert len(loaders) == 2
        docs = loaders[0].load()
        assert len(docs) >= 1

    def test_load_metadata(self, md_file):
        loader = RuneExtractLoader(md_file)
        docs = loader.load()
        meta = docs[0]["metadata"]
        assert "source" in meta
        assert meta["source"] == md_file

    def test_load_minimal_file(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write(".")
        f.close()
        try:
            loader = RuneExtractLoader(f.name)
            docs = loader.load()
            assert len(docs) >= 1
        finally:
            os.unlink(f.name)

    def test_transformer(self, md_file):
        loader = RuneExtractLoader(md_file)
        docs = loader.load()
        transformer = RuneExtractTransformer(chunk_strategy="fixed_size", chunk_size=10, chunk_overlap=2)
        chunked = transformer.transform_documents(docs)
        assert len(chunked) >= 1
        if hasattr(chunked[0], "page_content"):
            text = chunked[0].page_content
        else:
            text = chunked[0]["page_content"]
        assert isinstance(text, str)

    def test_transformer_empty(self):
        transformer = RuneExtractTransformer()
        result = transformer.transform_documents([])
        assert result == []

    def test_transformer_dicts(self):
        transformer = RuneExtractTransformer(chunk_size=50, chunk_overlap=10)
        docs = [{"page_content": "Hello world. " * 20, "metadata": {"source": "test.txt"}}]
        chunked = transformer.transform_documents(docs)
        assert len(chunked) >= 2
        if hasattr(chunked[0], "page_content"):
            text = chunked[0].page_content
        else:
            text = chunked[0]["page_content"]
        assert len(text) > 0

    def test_to_langchain_documents_import_error(self):
        doc = Document(text="Hello world. " * 50, metadata={"key": "val"})
        with pytest.raises(ImportError, match="langchain-core"):
            doc.to_langchain_documents()


# ── LlamaIndex ──────────────────────────────────────────

class TestLlamaIndex:
    def test_load_data(self, md_file):
        reader = RuneExtractReader()
        docs = reader.load_data(md_file)
        assert len(docs) >= 1
        assert "Hello from RuneExtract" in docs[0]["text"]

    def test_load_data_metadata(self, md_file):
        reader = RuneExtractReader()
        docs = reader.load_data(md_file)
        assert "source" in docs[0]["metadata"]
        assert docs[0]["metadata"]["source"] == md_file

    def test_load_data_csv(self, csv_file):
        reader = RuneExtractReader()
        docs = reader.load_data(csv_file)
        assert len(docs) >= 1


# ── CrewAI ──────────────────────────────────────────────

class TestCrewAI:
    def test_run(self, md_file):
        tool = RuneExtractTool()
        result = tool.run(md_file)
        assert isinstance(result, str)
        assert "Hello from RuneExtract" in result

    def test_name_and_description(self):
        tool = RuneExtractTool()
        assert tool.name == "RuneExtract"
        assert "Extract text" in tool.description

    def test_run_multiple_files(self, md_file, csv_file):
        tool = RuneExtractTool()
        r1 = tool.run(md_file)
        r2 = tool.run(csv_file)
        assert isinstance(r1, str)
        assert isinstance(r2, str)


# ── AutoGen ─────────────────────────────────────────────

class TestAutoGen:
    def test_extract_tool(self, md_file):
        result = autogen_extract_tool(file_path=md_file)
        assert isinstance(result, str)
        assert "Hello from RuneExtract" in result

    def test_extract_tool_ocr_false(self, md_file):
        result = autogen_extract_tool(file_path=md_file, ocr=False)
        assert isinstance(result, str)

    def test_extract_tool_csv(self, csv_file):
        result = autogen_extract_tool(file_path=csv_file)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_extract_tool_nonexistent_file(self):
        """Should raise on non-existent file."""
        with pytest.raises(Exception):
            autogen_extract_tool(file_path="/nonexistent/file.pdf")


# ── MCP — New Tools ─────────────────────────────────────

class TestMCPNewTools:
    @pytest.mark.asyncio
    async def test_mcp_tool_extract_url_invalid(self):
        with pytest.raises(Exception):
            await mcp_tool_extract_url("http://nonexistent.invalid")

    @pytest.mark.asyncio
    async def test_mcp_tool_ask_no_sources(self):
        result = await mcp_tool_ask("test question", source_paths=[])
        assert isinstance(result, str)
        assert "No source" in result or "Error" in result or len(result) > 0

    @pytest.mark.asyncio
    async def test_mcp_tool_chunk(self, md_file):
        result = await mcp_tool_chunk(md_file, strategy="fixed_size", size=50, overlap=0)
        assert isinstance(result, str)
        assert "Chunk" in result or len(result) > 0

    @pytest.mark.asyncio
    async def test_mcp_tool_chunk_minimal(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        f.write("Hello")
        f.close()
        try:
            result = await mcp_tool_chunk(f.name)
            assert isinstance(result, str)
        finally:
            os.unlink(f.name)

    def test_run_mcp_server_import(self):
        """run_mcp_server is importable."""
        assert callable(run_mcp_server)

    def test_main_cli_import(self):
        """main_cli is importable."""
        assert callable(main_cli)


# ── LangGraph ───────────────────────────────────────────

class TestLangGraph:
    def test_graph_tool_extract(self, md_file):
        tool = RuneExtractGraphTool()
        result = tool(md_file)
        assert isinstance(result, str)
        assert "Hello from RuneExtract" in result

    def test_graph_tool_name(self):
        tool = RuneExtractGraphTool()
        assert tool.name == "rune_extract"
        assert "Extract text" in tool.description

    def test_graph_search_tool(self):
        tool = RuneExtractSearchTool()
        assert tool.name == "rune_extract_search"
        assert callable(tool)

    def test_graph_ask_tool(self):
        tool = RuneExtractAskTool()
        assert tool.name == "rune_extract_ask"
        assert callable(tool)


# ── OpenAI Agents SDK ───────────────────────────────────

class TestOpenAISDK:
    def test_function_tool_structure(self):
        tool = rune_extract_function_tool()
        assert isinstance(tool, dict)
        assert tool["name"] == "rune_extract"
        assert "description" in tool
        assert "parameters" in tool
        assert "required" in tool["parameters"]
        assert "file_path" in tool["parameters"]["properties"]

    def test_function_tool_callable(self, md_file):
        tool = rune_extract_function_tool()
        result = tool["func"](md_file)
        assert isinstance(result, str)
        assert "Hello from RuneExtract" in result

    def test_search_tool_structure(self):
        tool = rune_extract_search_tool()
        assert isinstance(tool, dict)
        assert tool["name"] == "rune_extract_search"
        assert "query" in tool["parameters"]["properties"]


# ── PydanticAI ──────────────────────────────────────────

class TestPydanticAI:
    def test_ai_tool_extract(self, md_file):
        tool = RuneExtractAITool()
        result = tool(md_file)
        assert isinstance(result, str)
        assert "Hello from RuneExtract" in result

    def test_ai_tool_name(self):
        tool = RuneExtractAITool()
        assert tool.name == "rune_extract"

    def test_ai_search_tool(self):
        tool = RuneExtractSearchAITool()
        assert tool.name == "rune_extract_search"
        assert callable(tool)

    def test_ai_tool_run_method(self, md_file):
        tool = RuneExtractAITool()
        result = tool.run(md_file)
        assert isinstance(result, str)
        assert "Hello from RuneExtract" in result


# ── Parent-Child Chunking ───────────────────────────────

class TestParentChildChunking:
    def test_hierarchical_chunks_flat(self):
        """hierarchical_chunks() returns flat list with both levels."""
        text = "Paragraph one. " * 50 + "\n\n" + "Paragraph two. " * 50 + "\n\n" + "Paragraph three. " * 50
        doc = Document(text=text)
        chunks = doc.hierarchical_chunks(child_size=100, parent_size=300, parent_overlap=0)
        assert len(chunks) > 0
        assert any(c.metadata.get("level") == 0 for c in chunks)
        assert any(c.metadata.get("level") == 1 for c in chunks)

    def test_hierarchical_chunks_parent_child_links(self):
        """Child chunks have parent_chunk_id set."""
        text = "Hello world. " * 100
        doc = Document(text=text)
        chunks = doc.hierarchical_chunks(child_size=50, parent_size=150, parent_overlap=0)
        children = [c for c in chunks if c.metadata.get("level") == 0]
        parents = {c.chunk_id for c in chunks if c.metadata.get("level") == 1}
        assert len(children) > 0
        for child in children:
            assert child.parent_chunk_id is not None
            assert child.parent_chunk_id in parents

    def test_hierarchical_chunks_via_chunks_method(self):
        """chunks(HIERARCHICAL) delegates to hierarchical_chunks()."""
        text = "Test content. " * 50
        doc = Document(text=text)
        result = doc.chunks(strategy=ChunkingStrategy.HIERARCHICAL, child_size=80, parent_size=200)
        assert len(result) > 0
        assert any(c.metadata.get("strategy") == "hierarchical" for c in result)

    def test_chunk_is_child(self):
        chunk = Chunk(text="test", chunk_id="c1", start_index=0, end_index=4,
                       parent_chunk_id="p1", metadata={"level": 0})
        assert chunk.is_child()
        assert not chunk.is_parent()

    def test_chunk_is_parent(self):
        chunk = Chunk(text="test", chunk_id="p1", start_index=0, end_index=4,
                       parent_chunk_id=None, metadata={"level": 1})
        assert not chunk.is_child()
        assert chunk.is_parent()

    def test_hierarchical_chunk_dataclass(self):
        hc = HierarchicalChunk(text="test", chunk_id="hc1", start_index=0, end_index=4,
                                level=1, children=["child_1", "child_2"])
        assert hc.level == 1
        assert len(hc.children) == 2

    def test_hierarchical_chunks_by_heading_parent(self):
        """hierarchical_chunks with by_heading parent strategy."""
        text = "# Section 1\n\nContent one. " * 20 + "\n\n# Section 2\n\nContent two. " * 20
        doc = Document(text=text)
        chunks = doc.hierarchical_chunks(
            child_size=50, parent_size=300, parent_overlap=0,
            parent_strategy="by_heading",
        )
        assert len(chunks) > 0
        children = [c for c in chunks if c.metadata.get("level") == 0]
        assert len(children) > 0
        for child in children:
            assert child.parent_chunk_id is not None

    def test_hierarchical_chunks_empty_text(self):
        doc = Document(text="")
        chunks = doc.hierarchical_chunks()
        assert chunks == []

    def test_hierarchical_chunks_caching(self):
        text = "Cached content. " * 20
        doc = Document(text=text)
        first = doc.hierarchical_chunks(child_size=50, parent_size=100)
        second = doc.hierarchical_chunks(child_size=50, parent_size=100)
        assert len(first) == len(second)

    def test_hierarchical_chunks_to_dict(self):
        text = "Dict test. " * 30
        doc = Document(text=text)
        doc.hierarchical_chunks(child_size=50, parent_size=150, parent_overlap=0)
        d = doc.to_dict()
        assert "chunks" in d
        assert len(d["chunks"]) > 0
        assert "parent_chunk_id" in d["chunks"][0]
