"""
HTML extractor using BeautifulSoup4.
"""

import logging
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import requests
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document as RuneDocument, Table, Image

logger = logging.getLogger(__name__)


class HtmlExtractor(BaseExtractor):
    """Extractor for HTML files."""

    def extract(self, file_path: str) -> RuneDocument:
        if file_path.startswith(("http://", "https://")):
            return self._extract_from_url(file_path)
        else:
            self.validate_file(file_path)
            return self._extract_from_file(file_path)

    def _extract_from_file(self, file_path: str) -> RuneDocument:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return self._parse_html(html_content, file_path)

    def _extract_from_url(self, url: str) -> RuneDocument:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            html_content = response.text
            return self._parse_html(html_content, url)
        except Exception as e:
            raise ValueError(f"Failed to fetch URL: {e}")

    def _parse_html(self, html_content: str, source: str) -> RuneDocument:
        soup = BeautifulSoup(html_content, "lxml")
        text = ""
        tables: List[Table] = []
        images: List[Image] = []
        metadata: Dict[str, Any] = {}
        metadata = self._extract_metadata(soup)
        text = self._extract_text(soup)
        if self.options.get("tables", True):
            tables = self._extract_tables(soup)
        if self.options.get("images", True):
            images = self._extract_images(soup, source)
        text = self.clean_text(text)
        return RuneDocument(text=text, tables=tables, images=images, metadata=metadata,
                            source_type="html", source_path=source)

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        metadata = {}
        title_tag = soup.find("title")
        metadata["title"] = title_tag.get_text().strip() if title_tag else ""
        meta_tags = soup.find_all("meta")
        for tag in meta_tags:
            name = tag.get("name") or tag.get("property")
            content = tag.get("content")
            if name and content:
                metadata[name] = content
        html_tag = soup.find("html")
        if html_tag:
            metadata["language"] = html_tag.get("lang", "")
        return metadata

    def _extract_text(self, soup: BeautifulSoup) -> str:
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        text = soup.get_text(separator="\n")
        return text

    def _extract_tables(self, soup: BeautifulSoup) -> List[Table]:
        tables = []
        for table_index, table in enumerate(soup.find_all("table"), start=1):
            rows_raw = table.find_all("tr")
            if not rows_raw:
                continue

            # Build expanded grid with colspan/rowspan
            grid = []
            columns = []
            used_first_row = False

            for row in rows_raw:
                cells = row.find_all(["th", "td"])
                row_data = []
                col = 0
                for cell in cells:
                    # Skip cells occupied by rowspan from above
                    while any(
                        len(g) > col and g[col] is not None
                        for g in grid
                    ):
                        col += 1

                    text = cell.get_text().strip()
                    colspan = int(cell.get("colspan", 1))
                    rowspan = int(cell.get("rowspan", 1))

                    # Store (col, colspan, rowspan_remaining, text)
                    row_data.append({
                        "col": col, "text": text,
                        "colspan": colspan, "rowspan": rowspan
                    })

                    for _ in range(colspan):
                        row_data.append(None)  # placeholder
                    col += colspan

                grid.append(row_data)

            # Extract columns from first row
            if grid and grid[0]:
                first_cells = [c for c in grid[0] if c is not None]
                columns = [c["text"] for c in first_cells]
                used_first_row = True

            # Expand colspan/rowspan into a flat 2D array
            expanded = []
            row_idx = 0
            while row_idx < len(grid):
                flat_row = []
                col = 0
                pending = []
                col_idx = 0
                for cell_info in grid[row_idx]:
                    if cell_info is None:
                        continue
                    c = cell_info
                    # Fill gaps
                    while col < c["col"]:
                        flat_row.append("")
                        col += 1
                    flat_row.append(c["text"])
                    if c["rowspan"] > 1:
                        pending.append((c["col"], c["colspan"], c["rowspan"] - 1, c["text"]))
                    col += 1
                expanded.append(flat_row)
                row_idx += 1

            data_rows = expanded[1:] if used_first_row and len(expanded) > 1 else expanded
            data_rows = [r for r in data_rows if any(r)]

            if data_rows:
                tables.append(Table(
                    rows=data_rows, columns=columns,
                    metadata={"table_index": table_index}
                ))

        return tables

    def _extract_images(self, soup: BeautifulSoup, source: str) -> List[Image]:
        images = []
        for img_index, img in enumerate(soup.find_all("img"), start=1):
            src = img.get("src")
            alt = img.get("alt", "")
            if src:
                if source.startswith(("http://", "https://")):
                    src = urljoin(source, src)
                images.append(Image(data=b"", format="unknown",
                                    metadata={"src": src, "alt": alt, "index": img_index}))
        return images

    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]
