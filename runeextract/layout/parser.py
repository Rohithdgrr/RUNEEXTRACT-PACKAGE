"""Layout-aware document parser — bounding boxes, columns, figures, reading order."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class ElementType(Enum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    FIGURE = "figure"
    LIST = "list"
    CODE = "code"


@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float
    page: int = 0

    def width(self) -> float:
        return self.x1 - self.x0

    def height(self) -> float:
        return self.y1 - self.y0

    def area(self) -> float:
        return self.width() * self.height()

    def overlaps(self, other: "BoundingBox", threshold: float = 0.1) -> bool:
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        if ix0 >= ix1 or iy0 >= iy1:
            return False
        inter = (ix1 - ix0) * (iy1 - iy0)
        return inter / min(self.area(), other.area()) > threshold


@dataclass
class LayoutElement:
    bbox: BoundingBox
    element_type: ElementType
    text: str = ""
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


class LayoutParser:
    """Parse document layout into typed elements with bounding boxes.

    Uses heuristic rules based on text analysis when no native PDF layout
    information is available. For PDFs with embedded layout data, delegates
    to the PDF extractor.
    """

    def __init__(self, extractor=None):
        self._extractor = extractor

    def parse(self, text: str, source_type: str = "text", **kwargs) -> List[LayoutElement]:
        if source_type == "html":
            return self._parse_html(text)
        return self._parse_text(text)

    def _parse_text(self, text: str) -> List[LayoutElement]:
        elements = []
        lines = text.split("\n")
        y = 0.0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                y += 1.0
                continue
            etype = self._detect_type(stripped)
            bbox = BoundingBox(x0=0.0, y0=y, x1=len(line) * 0.5, y1=y + 1.0)
            elements.append(LayoutElement(bbox=bbox, element_type=etype, text=stripped))
            y += 1.0
        return elements

    def _parse_html(self, html: str) -> List[LayoutElement]:
        elements = []
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            y = 0.0
            for tag in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "table"]):
                text = tag.get_text(strip=True)
                if not text:
                    continue
                etype = self._tag_to_type(tag.name)
                bbox = BoundingBox(x0=0.0, y0=y, x1=len(text) * 0.5, y1=y + 1.0)
                elements.append(LayoutElement(bbox=bbox, element_type=etype, text=text))
                y += 1.0
        except ImportError:
            elements = self._parse_text(html)
        return elements

    def _detect_type(self, line: str) -> ElementType:
        if line.startswith("#"):
            return ElementType.HEADING
        if line.startswith("|") and "|" in line[1:]:
            return ElementType.TABLE
        if line.startswith("- ") or line.startswith("* "):
            return ElementType.LIST
        if line.startswith("```") or line.startswith("    "):
            return ElementType.CODE
        return ElementType.TEXT

    def _tag_to_type(self, tag: str) -> ElementType:
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            return ElementType.HEADING
        if tag == "table":
            return ElementType.TABLE
        if tag == "li":
            return ElementType.LIST
        if tag == "pre":
            return ElementType.CODE
        if tag == "figure" or tag == "img":
            return ElementType.FIGURE
        return ElementType.TEXT

    def get_columns(self, elements: List[LayoutElement], threshold: float = 0.3) -> List[List[LayoutElement]]:
        if not elements:
            return []
        mid_x = sum(e.bbox.x0 for e in elements) / len(elements)
        left = [e for e in elements if e.bbox.x0 < mid_x]
        right = [e for e in elements if e.bbox.x0 >= mid_x]
        return [left, right]

    def get_reading_order(self, elements: List[LayoutElement]) -> List[LayoutElement]:
        return sorted(elements, key=lambda e: (e.bbox.page, e.bbox.y0, e.bbox.x0))


def parse_layout(text: str, source_type: str = "text", **kwargs) -> List[LayoutElement]:
    parser = LayoutParser()
    return parser.parse(text, source_type=source_type, **kwargs)


def get_reading_order(elements: List[LayoutElement]) -> List[LayoutElement]:
    return sorted(elements, key=lambda e: (e.bbox.page, e.bbox.y0, e.bbox.x0))
