"""LangGraph integration — RuneExtractGraphTool for LangGraph agent nodes.

Usage:
    from runeextract.agent import RuneExtractGraphTool

    # In your LangGraph graph:
    from langgraph.prebuilt import ToolNode
    tool = RuneExtractGraphTool()
    tool_node = ToolNode([tool])
"""

from typing import Any, Callable, Dict, List, Optional, Type, Union


class RuneExtractGraphTool:
    """LangGraph-compatible tool for document extraction.

    Implements the LangGraph tool interface (name, description, func).
    Use with ToolNode or as a callable in your graph.
    """

    name: str = "rune_extract"
    description: str = (
        "Extract text content from documents (PDF, DOCX, HTML, images, YouTube, etc.). "
        "Input: a file path or URL. Returns the extracted text."
    )

    def __init__(
        self,
        ocr: bool = False,
        tables: bool = True,
        images: bool = True,
        **kwargs,
    ):
        self.ocr = ocr
        self.tables = tables
        self.images = images
        self.kwargs = kwargs

    def _run(self, path: str) -> str:
        from runeextract import extract
        doc = extract(path, ocr=self.ocr, tables=self.tables, images=self.images, **self.kwargs)
        return doc.text or ""

    async def _arun(self, path: str) -> str:
        from runeextract import extract
        import asyncio
        from functools import partial
        fn = partial(extract, path, ocr=self.ocr, tables=self.tables, images=self.images, **self.kwargs)
        doc = await asyncio.get_running_loop().run_in_executor(None, fn)
        return doc.text or ""

    def __call__(self, path: str) -> str:
        return self._run(path)


class RuneExtractSearchTool:
    """LangGraph-compatible tool for searching extracted documents via RAG.

    Use with ToolNode in LangGraph for RAG-powered search.
    """

    name: str = "rune_extract_search"
    description: str = (
        "Search documents using RAG (hybrid search + optional reranking). "
        "Input: a query string. Returns relevant chunks with scores. "
        "Documents must be ingested first via rune_extract."
    )

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents",
        top_k: int = 5,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.top_k = top_k

    def _run(self, query: str) -> str:
        from runeextract.rag.auto_pipeline import AutoRAG
        import os
        rag = AutoRAG()
        if os.path.isdir(self.persist_directory):
            rag._load_index(self.persist_directory, self.collection_name)
        result = rag.query(query, top_k=self.top_k)
        lines = [f"Query: {query}"]
        for i, chunk in enumerate(result.chunks):
            lines.append(f"\n[{i+1}] (score: {chunk.score:.3f}) {chunk.text[:500]}")
        return "\n".join(lines)

    def __call__(self, query: str) -> str:
        return self._run(query)


class RuneExtractAskTool:
    """LangGraph-compatible tool for asking questions about documents with RAG.

    Use with ToolNode in LangGraph for Q&A over ingested documents.
    """

    name: str = "rune_extract_ask"
    description: str = (
        "Ask a question about ingested documents and get an answer with citations. "
        "Input: a natural language question. Returns answer with source citations."
    )

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "documents",
        top_k: int = 5,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.top_k = top_k

    def _run(self, question: str) -> str:
        from runeextract.rag.auto_pipeline import AutoRAG
        import os
        rag = AutoRAG()
        if os.path.isdir(self.persist_directory):
            rag._load_index(self.persist_directory, self.collection_name)
        result = rag.query(question, top_k=self.top_k)
        return f"Answer: {result.answer}\n\nSources: {len(result.citations)} citations"

    def __call__(self, question: str) -> str:
        return self._run(question)
