"""
DOCX extractor using python-docx.
"""

from docx import Document
from pathlib import Path
from typing import List, Dict, Any
from io import BytesIO
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document as RuneDocument, Table, Image


class DocxExtractor(BaseExtractor):
    """Extractor for DOCX files."""

    def extract(self, file_path: str) -> RuneDocument:
        self.validate_file(file_path)
        doc = Document(file_path)
        text = ""
        tables: List[Table] = []
        images: List[Image] = []
        metadata: Dict[str, Any] = {}
        metadata = self._extract_metadata(doc)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n\n"
        if self.options.get('tables', True):
            tables = self._extract_tables(doc)
        if self.options.get('images', True):
            images = self._extract_images(doc, file_path)
        text = self.clean_text(text)
        return RuneDocument(text=text, tables=tables, images=images, metadata=metadata,
                            source_type="docx", source_path=file_path)

    def _extract_metadata(self, doc: Document) -> Dict[str, Any]:
        metadata = {}
        core_props = doc.core_properties
        metadata['title'] = core_props.title or ""
        metadata['author'] = core_props.author or ""
        metadata['subject'] = core_props.subject or ""
        metadata['keywords'] = core_props.keywords or ""
        metadata['created'] = core_props.created or ""
        metadata['modified'] = core_props.modified or ""
        metadata['last_modified_by'] = core_props.last_modified_by or ""
        metadata['revision'] = core_props.revision or ""
        metadata['category'] = core_props.category or ""
        metadata['comments'] = core_props.comments or ""
        return metadata

    def _extract_tables(self, doc: Document) -> List[Table]:
        tables = []
        for table_index, table in enumerate(doc.tables, start=1):
            rows = []
            columns = []
            if table.rows:
                first_row = table.rows[0]
                columns = [cell.text.strip() for cell in first_row.cells]
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    if any(row_data):
                        rows.append(row_data)
            if rows:
                tables.append(Table(rows=rows, columns=columns, metadata={'table_index': table_index}))
        return tables

    def _extract_images(self, doc: Document, file_path: str) -> List[Image]:
        images = []
        try:
            for para_index, para in enumerate(doc.paragraphs):
                for run in para.runs:
                    for inline in run._element.xpath('.//w:drawing//wp:inline'):
                        for blip in inline.xpath('.//a:blip'):
                            embed = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                            if embed:
                                try:
                                    image_part = doc.part.related_part(embed)
                                    image_data = image_part.blob
                                    content_type = image_part.content_type
                                    image_format = content_type.split('/')[-1] if '/' in content_type else 'png'
                                    images.append(Image(
                                        data=image_data,
                                        format=image_format,
                                        metadata={'paragraph_index': para_index, 'rId': embed}
                                    ))
                                except Exception:
                                    pass
        except Exception:
            pass
        return images

    def supported_extensions(self) -> list[str]:
        return [".docx", ".doc"]
