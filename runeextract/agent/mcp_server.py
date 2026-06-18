"""MCP (Model Context Protocol) integration — expose RuneExtract as MCP tools.

Usage:
    from runeextract.agent import mcp_tool_extract

    # In your MCP server:
    @app.tool()
    async def extract_document(file_path: str) -> str:
        return await mcp_tool_extract(file_path)
"""

from typing import List, Optional

from runeextract import extract, extract_many, extract_crawl
from runeextract.rag.auto_pipeline import auto_rag


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
    doc = extract(file_path, ocr=ocr, tables=tables, images=images, metadata=metadata)
    return doc.text


async def mcp_tool_extract_many(
    file_paths: List[str],
    ocr: bool = False,
) -> str:
    """Extract multiple documents and return concatenated text.

    This is designed to be registered as an MCP tool.
    """
    docs = extract_many(file_paths, ocr=ocr)
    return "\n\n---\n\n".join(d.text for d in docs if d.text)


async def mcp_tool_search(
    query: str,
    source_paths: Optional[List[str]] = None,
    top_k: int = 5,
) -> str:
    """Search extracted documents using RAG.

    This is designed to be registered as an MCP tool.
    If source_paths is provided, extracts those files first, then searches.
    """
    if source_paths:
        docs = extract_many(source_paths)
        texts = [d.text for d in docs if d.text]
        corpus = "\n".join(texts)
        result = auto_rag(query, corpus=corpus, top_k=top_k)
    else:
        result = auto_rag(query, top_k=top_k)
    return (
        f"Query: {query}\n"
        + "\n".join(f"[{i+1}] {c.text[:500]}" for i, c in enumerate(result.chunks))
    )


async def mcp_tool_crawl(
    start_url: str,
    max_pages: int = 10,
    same_domain: bool = True,
) -> str:
    """Crawl web pages and extract their content.

    This is designed to be registered as an MCP tool.
    """
    docs = extract_crawl(start_url, max_pages=max_pages, same_domain=same_domain)
    parts = []
    for i, doc in enumerate(docs):
        parts.append(f"=== Page {i+1}: {doc.source_path} ===\n{doc.text[:2000]}")
    return "\n\n".join(parts)
