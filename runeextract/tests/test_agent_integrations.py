"""Tests for agent SDK integrations (MCP, LangChain, LlamaIndex, CrewAI, AutoGen)."""

import tempfile
import os

import pytest

from runeextract.agent.mcp_server import mcp_tool_extract, mcp_tool_extract_many, mcp_tool_crawl
from runeextract.agent.langchain import RuneExtractLoader
from runeextract.agent.llamaindex import RuneExtractReader
from runeextract.agent.crewai import RuneExtractTool
from runeextract.agent.autogen import autogen_extract_tool


@pytest.fixture
def md_file():
    """Create a temporary markdown file for extraction tests."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    f.write("Hello from RuneExtract agent test!")
    f.close()
    yield f.name
    try:
        os.unlink(f.name)
    except OSError:
        pass


@pytest.fixture
def csv_file():
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write("name,value\nfoo,1\nbar,2")
    f.close()
    yield f.name
    try:
        os.unlink(f.name)
    except OSError:
        pass


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
