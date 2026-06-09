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
        """
        Extract content from a DOCX file.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Document object with extracted content
        """
        self.validate_file(file_path)
        
        doc = Document(file_path)
        
        text = ""
        tables: List[Table] = []
        images: List[Image] = []
        metadata: Dict[str, Any] = {}
        
        # Extract metadata
        metadata = self._extract_metadata(doc)
        
        # Extract paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n\n"
        
        # Extract tables
        if self.options.get('tables', True):
            tables = self._extract_tables(doc)
        
        # Extract images
        if self.options.get('images', True):
            images = self._extract_images(doc, file_path)
        
        # Clean text
        text = self.clean_text(text)
        
        return RuneDocument(
            text=text,
            tables=tables,
            images=images,
            metadata=metadata,
            source_type="docx",
            source_path=file_path
        )
    
    def _extract_metadata(self, doc: Document) -> Dict[str, Any]:
        """Extract metadata from DOCX document."""
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
        """Extract tables from DOCX document."""
        tables = []
        
        for table_index, table in enumerate(doc.tables, start=1):
            rows = []
            columns = []
            
            # Extract headers (first row)
            if table.rows:
                first_row = table.rows[0]
                columns = [cell.text.strip() for cell in first_row.cells]
                
                # Extract data rows
                for row in table.rows[1:]:
                    row_data = [cell.text.strip() for cell in row.cells]
                    if any(row_data):  # Only add non-empty rows
                        rows.append(row_data)
            
            if rows:
                tables.append(Table(
                    rows=rows,
                    columns=columns,
                    metadata={'table_index': table_index}
                ))
        
        return tables
    
    def _extract_images(self, doc: Document, file_path: str) -> List[Image]:
        """Extract images from DOCX document."""
        images = []
        
        # Extract images from document relationships
        try:
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            from docx.text.paragraph import Paragraph
            from docx.table import _Cell, Table
            from docx.document import Document as DocxDocument
            
            # Iterate through document elements
            for block in doc.element.body:
                if isinstance(block, CT_P):
                    paragraph = Paragraph(block, doc)
                    self._extract_images_from_paragraph(paragraph, images)
                elif isinstance(block, CT_Tbl):
                    table = Table(block, doc)
                    for row in table.rows:
                        for cell in row.cells:
                            for paragraph in cell.paragraphs:
                                self._extract_images_from_paragraph(paragraph, images)
        except Exception:
            # If image extraction fails, continue without images
            pass
        
        return images
    
    def _extract_images_from_paragraph(self, paragraph, images: List[Image]):
        """Extract images from a paragraph."""
        try:
            for run in paragraph.runs:
                for inline in run._element.xpath('.//w:drawing//wp:inline'):
                    for blip in inline.xpath('.//a:blip'):
                        embed = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                        if embed:
                            # This is a simplified version - full implementation would need
                            # to extract the actual image data from the ZIP file
                            pass
        except Exception:
            pass
    
    def supported_extensions(self) -> list[str]:
        """Return supported file extensions."""
        return [".docx", ".doc"]
