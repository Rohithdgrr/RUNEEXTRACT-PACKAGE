"""
PPTX extractor using python-pptx.
"""

from pptx import Presentation
from pptx.util import Inches
from pathlib import Path
from typing import List, Dict, Any
from io import BytesIO
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document as RuneDocument, Table, Image


class PptxExtractor(BaseExtractor):
    """Extractor for PPTX files."""

    def extract(self, file_path: str) -> RuneDocument:
        self.validate_file(file_path)
        prs = Presentation(file_path)
        text = ""
        tables: List[Table] = []
        images: List[Image] = []
        metadata: Dict[str, Any] = {}
        metadata = self._extract_metadata(prs)
        slide_count = 0

        for slide_num, slide in enumerate(prs.slides, start=1):
            slide_count = slide_num
            if hasattr(slide, 'slide_layout') and slide.slide_layout:
                text += f"\n--- Slide {slide_num} ---\n\n"
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        para_text = para.text.strip()
                        if para_text:
                            text += para_text + "\n"
                    text += "\n"
                if shape.has_table and self.options.get('tables', True):
                    table_shape = shape.table
                    rows = []
                    columns = []
                    if table_shape.rows:
                        first_row = table_shape.rows[0]
                        columns = [cell.text.strip() for cell in first_row.cells]
                        for row_idx in range(1, len(table_shape.rows)):
                            row_data = [table_shape.cell(row_idx, col).text.strip()
                                       for col in range(len(table_shape.columns))]
                            if any(row_data):
                                rows.append(row_data)
                    if rows:
                        tables.append(Table(rows=rows, columns=columns, page_number=slide_num,
                                            metadata={'table_index': len(tables) + 1}))
                if shape.shape_type == 13 and self.options.get('images', True):
                    try:
                        image_data = shape.image.blob
                        content_type = shape.image.content_type
                        image_format = content_type.split('/')[-1] if '/' in content_type else 'png'
                        images.append(Image(data=image_data, format=image_format, page_number=slide_num,
                                            metadata={'shape_name': shape.name, 'shape_id': shape.shape_id}))
                    except Exception:
                        pass

        metadata['slide_count'] = slide_count
        text = self.clean_text(text)
        return RuneDocument(text=text, tables=tables, images=images, metadata=metadata,
                            source_type="pptx", source_path=file_path)

    def _extract_metadata(self, prs: Presentation) -> Dict[str, Any]:
        metadata = {}
        try:
            props = prs.core_properties
            metadata['title'] = props.title or ""
            metadata['author'] = props.author or ""
            metadata['subject'] = props.subject or ""
            metadata['keywords'] = props.keywords or ""
            metadata['created'] = str(props.created) if props.created else ""
            metadata['modified'] = str(props.modified) if props.modified else ""
            metadata['last_modified_by'] = props.last_modified_by or ""
            metadata['revision'] = props.revision or ""
            metadata['category'] = props.category or ""
            metadata['comments'] = props.comments or ""
        except Exception:
            pass
        return metadata

    def supported_extensions(self) -> list[str]:
        return [".pptx", ".ppt"]
