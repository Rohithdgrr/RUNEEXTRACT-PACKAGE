"""OpenAI Agents SDK integration — RuneExtract as an OpenAI function tool.

Usage:
    from runeextract.agent import rune_extract_function_tool

    # With OpenAI Agents SDK:
    from agents import Agent, Runner
    import asyncio

    agent = Agent(
        name="extractor",
        instructions="Extract documents when asked.",
        tools=[rune_extract_function_tool()],
    )
    result = asyncio.run(Runner.run(agent, "Extract document.pdf"))
"""

from typing import Any, Dict, List, Optional


def rune_extract_function_tool(
    ocr: bool = False,
    tables: bool = True,
    images: bool = True,
) -> Dict[str, Any]:
    """Create an OpenAI function tool definition for document extraction.

    Returns a dictionary compatible with the OpenAI Agents SDK ``function_tool``
    format, containing name, description, and a parameters schema.

    Args:
        ocr: Enable OCR for scanned documents.
        tables: Extract tables (default True).
        images: Extract images (default True).

    Returns:
        A dict with ``name``, ``description``, ``parameters``, and ``func``
        keys ready for use with OpenAI's function calling.
    """

    def _extract(file_path: str) -> str:
        from runeextract import extract
        doc = extract(file_path, ocr=ocr, tables=tables, images=images)
        return doc.text or ""

    return {
        "name": "rune_extract",
        "description": "Extract text content from any document (PDF, DOCX, HTML, images, YouTube, etc.). Provide a file path or URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path or URL to the document file.",
                }
            },
            "required": ["file_path"],
        },
        "func": _extract,
    }


def rune_extract_search_tool(
    top_k: int = 5,
) -> Dict[str, Any]:
    """Create an OpenAI function tool definition for RAG search.

    Returns a dictionary compatible with OpenAI function calling for
    searching previously extracted and indexed documents.

    Args:
        top_k: Number of top results to return (default 5).

    Returns:
        A dict with ``name``, ``description``, ``parameters``, and ``func``.
    """

    def _search(query: str) -> str:
        from runeextract.rag.auto_pipeline import AutoRAG
        rag = AutoRAG()
        result = rag.query(query, top_k=top_k)
        lines = [f"Query: {query}"]
        for i, chunk in enumerate(result.chunks):
            lines.append(f"\n[{i+1}] (score: {chunk.score:.3f}) {chunk.text[:500]}")
        return "\n".join(lines)

    return {
        "name": "rune_extract_search",
        "description": "Search previously extracted documents using RAG. Provide a search query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant document content.",
                }
            },
            "required": ["query"],
        },
        "func": _search,
    }
