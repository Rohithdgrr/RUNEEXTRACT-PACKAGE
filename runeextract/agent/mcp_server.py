"""MCP (Model Context Protocol) integration — expose RuneExtract as MCP tools.

Usage:
    # Run as server:
    from runeextract.agent import run_mcp_server
    run_mcp_server()

    # Or from CLI:
    # runeextract-mcp

    # Or use individual tools in your own MCP app:
    from runeextract.agent import mcp_tool_extract
    @app.tool()
    async def extract_document(file_path: str) -> str:
        return await mcp_tool_extract(file_path)
"""

from functools import partial
from typing import List, Optional


async def mcp_tool_extract(
    file_path: str,
    ocr: bool = False,
    tables: bool = True,
    images: bool = True,
    metadata: bool = True,
) -> str:
    """Extract a document and return its text content as a string.

    This is designed to be registered as an MCP tool.
    """
    import asyncio
    from runeextract import extract
    loop = asyncio.get_running_loop()
    fn = partial(extract, file_path, ocr=ocr, tables=tables, images=images, metadata=metadata)
    doc = await loop.run_in_executor(None, fn)
    return doc.text


async def mcp_tool_extract_many(
    file_paths: List[str],
    ocr: bool = False,
) -> str:
    """Extract multiple documents and return concatenated text.

    This is designed to be registered as an MCP tool.
    """
    import asyncio
    from runeextract import extract_many
    loop = asyncio.get_running_loop()
    fn = partial(extract_many, file_paths, ocr=ocr)
    docs = await loop.run_in_executor(None, fn)
    return "\n\n---\n\n".join(d.text for d in docs if d.text)


async def mcp_tool_extract_url(
    url: str,
    ocr: bool = False,
) -> str:
    """Extract content from a URL (web page, YouTube, etc.).

    This is designed to be registered as an MCP tool.
    """
    import asyncio
    from runeextract import extract
    loop = asyncio.get_running_loop()
    fn = partial(extract, url, ocr=ocr)
    doc = await loop.run_in_executor(None, fn)
    return doc.text


async def mcp_tool_search(
    query: str,
    source_paths: Optional[List[str]] = None,
    top_k: int = 5,
) -> str:
    """Search extracted documents using RAG.

    This is designed to be registered as an MCP tool.
    If source_paths is provided, extracts those files first, then searches.
    """
    import asyncio
    from runeextract import extract_many
    from runeextract.rag.auto_pipeline import AutoRAG
    loop = asyncio.get_running_loop()

    if not source_paths:
        return f"Query: {query}\nNo sources provided — ingest documents first."

    def _run():
        rag = AutoRAG()
        rag.ingest(source_paths)
        return rag.query(query)

    result = await loop.run_in_executor(None, _run)
    return (
        f"Query: {query}\n"
        + "\n".join(f"[{i+1}] {c.text[:500]}" for i, c in enumerate(result.chunks))
    )


async def mcp_tool_ask(
    question: str,
    source_paths: List[str],
    top_k: int = 5,
) -> str:
    """Ask a question about documents using RAG and return an answer with citations.

    This is designed to be registered as an MCP tool.
    """
    import asyncio
    from runeextract import extract_many
    from runeextract.rag.auto_pipeline import AutoRAG
    loop = asyncio.get_running_loop()

    if not source_paths:
        return "No source documents provided."

    def _run():
        rag = AutoRAG()
        rag.ingest(source_paths)
        return rag.query(question, return_all=True)

    try:
        result = await loop.run_in_executor(None, _run)
        citations = result.get("citations", [])
        chunks = result.get("chunks", [])
        answer = result.get("answer", "")
        parts = [f"Answer: {answer}"]
        if citations:
            parts.append("\nCitations:")
            for i, c in enumerate(citations):
                parts.append(f"  [{i+1}] {c.get('text', '')[:300]} (source: {c.get('source', 'unknown')})")
        if chunks:
            parts.append("\nRetrieved chunks:")
            for i, c in enumerate(chunks):
                parts.append(f"  [{i+1}] {c.text[:300]}")
        return "\n".join(parts)
    except Exception as exc:
        return f"Error answering question: {exc}"


async def mcp_tool_crawl(
    start_url: str,
    max_pages: int = 10,
    same_domain: bool = True,
) -> str:
    """Crawl web pages and extract their content.

    This is designed to be registered as an MCP tool.
    """
    import asyncio
    from runeextract import extract_crawl
    loop = asyncio.get_running_loop()
    fn = partial(extract_crawl, start_url, max_pages=max_pages, same_domain=same_domain)
    docs = await loop.run_in_executor(None, fn)
    parts = []
    for i, doc in enumerate(docs):
        parts.append(f"=== Page {i+1}: {doc.source_path} ===\n{doc.text[:2000]}")
    return "\n\n".join(parts)


async def mcp_tool_chunk(
    file_path: str,
    strategy: str = "fixed_size",
    size: int = 1000,
    overlap: int = 100,
) -> str:
    """Chunk a document using the specified strategy.

    This is designed to be registered as an MCP tool.

    Args:
        file_path: Path to the document file.
        strategy: Chunking strategy (fixed_size, by_page, by_heading, semantic, by_token, sentence_window).
        size: Target chunk size.
        overlap: Overlap between chunks.

    Returns:
        Formatted string of chunks with metadata.
    """
    import asyncio
    from runeextract import extract
    loop = asyncio.get_running_loop()
    fn = partial(extract, file_path, chunking_strategy=strategy, chunk_size=size, chunk_overlap=overlap)
    doc = await loop.run_in_executor(None, fn)
    if not doc._chunks:
        return "No chunks generated."
    parts = [f"Document: {doc.source_path} ({len(doc._chunks)} chunks)"]
    for i, c in enumerate(doc._chunks):
        parts.append(f"\n--- Chunk {i+1} (id={c.chunk_id}, chars={len(c.text)}) ---")
        parts.append(c.text[:500])
    return "\n".join(parts)


def run_mcp_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start a standalone MCP server exposing all RuneExtract tools.

    Requires the optional ``mcp`` extra (``pip install runeextract[mcp]``).

    Args:
        host: Bind address (default ``"127.0.0.1"``).
        port: Port number (default ``8000``).

    Usage::

        from runeextract.agent import run_mcp_server
        run_mcp_server()
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "The 'mcp' extra is required. Install with: pip install runeextract[mcp]"
        )

    mcp = FastMCP("RuneExtract", host=host, port=port)

    @mcp.tool()
    async def extract_document(
        file_path: str,
        ocr: bool = False,
        tables: bool = True,
        images: bool = True,
        metadata: bool = True,
    ) -> str:
        """Extract text content from a document."""
        return await mcp_tool_extract(file_path, ocr=ocr, tables=tables, images=images, metadata=metadata)

    @mcp.tool()
    async def extract_documents(
        file_paths: List[str],
        ocr: bool = False,
    ) -> str:
        """Extract multiple documents and return concatenated text."""
        return await mcp_tool_extract_many(file_paths, ocr=ocr)

    @mcp.tool()
    async def extract_url(
        url: str,
        ocr: bool = False,
    ) -> str:
        """Extract content from a URL (web page, YouTube video, etc.)."""
        return await mcp_tool_extract_url(url, ocr=ocr)

    @mcp.tool()
    async def search_documents(
        query: str,
        source_paths: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> str:
        """Search documents using RAG and return relevant chunks."""
        return await mcp_tool_search(query, source_paths=source_paths, top_k=top_k)

    @mcp.tool()
    async def ask_documents(
        question: str,
        source_paths: List[str],
        top_k: int = 5,
    ) -> str:
        """Ask a question about documents and get an answer with citations."""
        return await mcp_tool_ask(question, source_paths=source_paths, top_k=top_k)

    @mcp.tool()
    async def crawl_website(
        start_url: str,
        max_pages: int = 10,
        same_domain: bool = True,
    ) -> str:
        """Crawl a website and extract content from its pages."""
        return await mcp_tool_crawl(start_url, max_pages=max_pages, same_domain=same_domain)

    @mcp.tool()
    async def chunk_document(
        file_path: str,
        strategy: str = "fixed_size",
        size: int = 1000,
        overlap: int = 100,
    ) -> str:
        """Chunk a document using the specified chunking strategy."""
        return await mcp_tool_chunk(file_path, strategy=strategy, size=size, overlap=overlap)

    mcp.run()


def main_cli():
    """CLI entry point for runeextract-mcp."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(
        prog="runeextract-mcp",
        description="RuneExtract MCP server — expose document extraction as MCP tools.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--version", "-v", action="store_true", help="Show version and exit")
    args = parser.parse_args()

    if args.version:
        from runeextract import __version__
        print(f"runeextract-mcp {__version__}")
        sys.exit(0)

    run_mcp_server(host=args.host, port=args.port)
