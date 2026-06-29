"""PydanticAI integration — RuneExtractAITool for PydanticAI agents.

Usage:
    from runeextract.agent import RuneExtractAITool

    # In your PydanticAI agent:
    from pydantic_ai import Agent

    agent = Agent(
        "openai:gpt-4o",
        tools=[RuneExtractAITool()],
    )
"""

from typing import Any, Dict, List, Optional


class RuneExtractAITool:
    """PydanticAI-compatible tool for document extraction.

    PydanticAI tools follow the pattern of a class with a ``name``,
    ``description``, and an ``__call__`` method or a ``run`` method.
    """

    name: str = "rune_extract"
    description: str = (
        "Extract text content from documents (PDF, DOCX, HTML, images, YouTube, etc.). "
        "Provide a file path or URL as input."
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

    def run(self, file_path: str) -> str:
        """Execute document extraction.

        Args:
            file_path: Path or URL to the document file.

        Returns:
            Extracted text content.
        """
        from runeextract import extract
        doc = extract(file_path, ocr=self.ocr, tables=self.tables, images=self.images, **self.kwargs)
        return doc.text or ""

    async def arun(self, file_path: str) -> str:
        """Execute document extraction asynchronously.

        Args:
            file_path: Path or URL to the document file.

        Returns:
            Extracted text content.
        """
        from runeextract import extract
        import asyncio
        from functools import partial
        fn = partial(extract, file_path, ocr=self.ocr, tables=self.tables, images=self.images, **self.kwargs)
        doc = await asyncio.get_running_loop().run_in_executor(None, fn)
        return doc.text or ""

    def __call__(self, file_path: str) -> str:
        return self.run(file_path)


class RuneExtractSearchAITool:
    """PydanticAI-compatible tool for RAG search over documents."""

    name: str = "rune_extract_search"
    description: str = (
        "Search documents using RAG. Provide a search query. "
        "Documents must have been ingested first."
    )

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def run(self, query: str) -> str:
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG()
        result = rag.query(query, top_k=self.top_k)
        lines = [f"Query: {query}"]
        for i, chunk in enumerate(result.chunks):
            lines.append(f"\n[{i+1}] (score: {chunk.score:.3f}) {chunk.text[:500]}")
        return "\n".join(lines)

    def __call__(self, query: str) -> str:
        return self.run(query)
