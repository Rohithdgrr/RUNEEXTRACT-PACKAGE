"""AutoGen integration — autogen_extract_tool for AutoGen agents.

Usage:
    from runeextract.agent import autogen_extract_tool

    # Register with AutoGen:
    # assistant = autogen.AssistantAgent(
    #     name="extractor",
    #     llm_config=llm_config,
    #     system_message="You can extract documents.",
    #     functions=[autogen_extract_tool],
    # )
"""

from typing import Optional

from runeextract import extract


def autogen_extract_tool(
    file_path: str,
    ocr: bool = False,
) -> str:
    """Extract text content from a document.

    AutoGen-compatible function tool. Returns the extracted text content.

    Args:
        file_path: Path to the document file (PDF, DOCX, HTML, etc.)
        ocr: Enable OCR for scanned documents and images

    Returns:
        Extracted text content of the document
    """
    doc = extract(file_path, ocr=ocr)
    return doc.text or ""
