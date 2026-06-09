"""
PDF extractor using PyMuPDF and pdfplumber.
"""

import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document, Table, Image


class PDFExtractor(BaseExtractor):
    """Extractor for PDF files."""
    
    def extract(self, file_path: str) -> Document:
        """
        Extract content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Document object with extracted content
        """
        self.validate_file(file_path)
        
        text = ""
        tables: List[Table] = []
        images: List[Image] = []
        metadata: Dict[str, Any] = {}
        
        # Extract using PyMuPDF for text and images
        doc = fitz.open(file_path)
        
        # Extract metadata
        metadata.update(self._extract_metadata(doc))
        
        # Extract text and images page by page
        for page_num, page in enumerate(doc, start=1):
            # Extract text
            page_text = page.get_text()
            text += page_text + "\n\n"
            
            # Extract images
            if self.options.get('images', True):
                page_images = self._extract_images(page, page_num)
                images.extend(page_images)
        
        doc.close()
        
        # Extract tables using pdfplumber
        if self.options.get('tables', True):
            tables = self._extract_tables(file_path)
        
        # Clean text
        text = self.clean_text(text)
        
        return Document(
            text=text,
            tables=tables,
            images=images,
            metadata=metadata,
            source_type="pdf",
            source_path=file_path
        )
    
    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """Extract metadata from PDF document."""
        metadata = {}
        
        if doc.metadata:
            meta = doc.metadata
            metadata['title'] = meta.get('title', '')
            metadata['author'] = meta.get('author', '')
            metadata['subject'] = meta.get('subject', '')
            metadata['keywords'] = meta.get('keywords', '')
            metadata['creator'] = meta.get('creator', '')
            metadata['producer'] = meta.get('producer', '')
            metadata['creation_date'] = meta.get('creationDate', '')
            metadata['modification_date'] = meta.get('modDate', '')
        
        metadata['page_count'] = len(doc)
        
        return metadata
    
    def _extract_images(self, page: fitz.Page, page_num: int) -> List[Image]:
        """Extract images from a PDF page."""
        images = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list, start=1):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_data = base_image["image"]
                image_format = base_image["ext"]
                
                images.append(Image(
                    data=image_data,
                    format=image_format,
                    page_number=page_num,
                    metadata={'xref': xref, 'index': img_index}
                ))
            except Exception:
                # Skip images that fail to extract
                continue
        
        return images
    
    def _extract_tables(self, file_path: str) -> List[Table]:
        """Extract tables from PDF using pdfplumber."""
        tables = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = page.extract_tables()
                    
                    for table_index, table_data in enumerate(page_tables, start=1):
                        if not table_data:
                            continue
                        
                        # Convert to rows and columns
                        rows = []
                        columns = []
                        
                        if table_data:
                            # First row as headers
                            columns = [str(cell) if cell else "" for cell in table_data[0]]
                            
                            # Remaining rows as data
                            for row in table_data[1:]:
                                if row:
                                    rows.append([str(cell) if cell else "" for cell in row])
                        
                        if rows:
                            tables.append(Table(
                                rows=rows,
                                columns=columns,
                                page_number=page_num,
                                metadata={'table_index': table_index}
                            ))
        except Exception as e:
            # Log error but don't fail entire extraction
            pass
        
        return tables
    
    def supported_extensions(self) -> list[str]:
        """Return supported file extensions."""
        return [".pdf"]
