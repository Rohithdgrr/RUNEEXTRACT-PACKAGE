"""CrewAI integration — RuneExtractTool for CrewAI agents.

Usage:
    from runeextract.agent import RuneExtractTool

    tool = RuneExtractTool()
    result = tool.run("extract document.pdf")
"""

from typing import Optional

from runeextract import extract


class RuneExtractTool:
    """CrewAI-compatible tool for document extraction.

    Mimics the CrewAI BaseTool interface.
    The `run` method accepts a file path and returns extracted text.
    """

    name: str = "RuneExtract"
    description: str = "Extract text content from documents (PDF, DOCX, HTML, images, etc.)"

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
        doc = extract(file_path, ocr=self.ocr, tables=self.tables, images=self.images, **self.kwargs)
        return doc.text or ""
