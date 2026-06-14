"""
EPUB extractor using ebooklib and BeautifulSoup.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document, Table, Image
from runeextract.exceptions import CorruptFileError, DependencyMissingError

logger = logging.getLogger(__name__)

try:
    import ebooklib
    from ebooklib import epub
except ImportError:
    ebooklib = None
    epub = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


class EpubExtractor(BaseExtractor):
    """Extractor for EPUB e-book files."""

    def extract(self, file_path: str) -> Document:
        self.validate_file(file_path)

        if ebooklib is None:
            raise DependencyMissingError(
                file_path,
                "ebooklib and beautifulsoup4 (pip install runeextract[epub])"
            )

        text_parts: List[str] = []
        tables: List[Table] = []
        images: List[Image] = []
        metadata: Dict[str, Any] = {}

        try:
            book = epub.read_epub(file_path)
        except Exception as exc:
            raise CorruptFileError(file_path, detail=str(exc))

        metadata["title"] = _get_metadata(book, "title", "")
        metadata["author"] = _get_metadata(book, "creator", "")
        metadata["language"] = _get_metadata(book, "language", "")
        metadata["publisher"] = _get_metadata(book, "publisher", "")
        metadata["description"] = _get_metadata(book, "description", "")
        metadata["identifier"] = _get_metadata(book, "identifier", "")

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_body_content()
                if not content:
                    continue
                soup = BeautifulSoup(content, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text_parts.append(soup.get_text(separator="\n", strip=True))

                html_tables = soup.find_all("table")
                for html_table in html_tables:
                    rows: List[List[str]] = []
                    columns: List[str] = []
                    for tr in html_table.find_all("tr"):
                        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                        if not columns:
                            columns = [th.get_text(strip=True) for th in tr.find_all("th")] or cells
                        if cells:
                            rows.append(cells)
                    if columns or rows:
                        tables.append(Table(rows=rows, columns=columns, metadata={"source": item.get_name()}))

            elif item.get_type() == ebooklib.ITEM_IMAGE:
                img_data = item.get_content()
                img_name = item.get_name()
                if img_data:
                    images.append(Image(data=img_data, metadata={"source": img_name}))

        combined_text = "\n\n".join(text_parts)
        combined_text = self.clean_text(combined_text)

        return Document(
            text=combined_text, tables=tables, images=images, metadata=metadata,
            source_type="epub", source_path=file_path
        )

    def supported_extensions(self) -> list[str]:
        return [".epub"]


def _get_metadata(book: Any, key: str, default: str = "") -> str:
    """Safely extract metadata from an EPUB book."""
    try:
        values = book.get_metadata("DC", key)
        if values and values[0]:
            return str(values[0][0])
    except Exception:
        pass
    return default
