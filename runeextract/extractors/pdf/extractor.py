"""
PDF extractor using PyMuPDF and pdfplumber.
"""

import logging
import fitz
from pathlib import Path
from typing import List, Dict, Any, Optional
from runeextract.core.extractor import BaseExtractor, StreamingExtractor
from runeextract.models.document import Document, Table, Image
from runeextract.exceptions import CorruptFileError, ExtractionError, WrongPasswordError

logger = logging.getLogger(__name__)


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
        
        text_parts: List[str] = []
        tables: List[Table] = []
        images: List[Image] = []
        metadata: Dict[str, Any] = {}
        
        extract_images = self.options.get('images', True)
        extract_tables = self.options.get('tables', True)
        password = self.options.get('password')
        
        doc = fitz.open(file_path)
        try:
            if doc.is_encrypted:
                if not password:
                    raise WrongPasswordError(file_path)
                if not doc.authenticate(password):
                    raise WrongPasswordError(file_path)
            
            metadata.update(self._extract_metadata(doc))
            
            page_breaks = []
            char_count = 0
            
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text()
                
                if self.ocr and len(page_text.strip()) < 20:
                    ocr_text = self._ocr_page(page)
                    if ocr_text:
                        page_text = ocr_text
                        metadata.setdefault("ocr_pages", []).append(page_num)
                
                text_parts.append(page_text)
                text_parts.append("\n\n")
                char_count += len(page_text) + 2
                page_breaks.append(char_count)
                
                if extract_images:
                    page_images = self._extract_images(page, page_num)
                    images.extend(page_images)
        finally:
            doc.close()
        
        metadata["page_breaks"] = page_breaks
        
        if extract_tables:
            tables = self._extract_tables(file_path)
        
        text = self.clean_text("".join(text_parts))
        
        if not text.strip():
            logger.warning(f"PDF produced no text content: {file_path}")
        
        return Document(
            text=text,
            tables=tables,
            images=images,
            metadata=metadata,
            source_type="pdf",
            source_path=file_path
        )
    
    def _ocr_page(self, page: fitz.Page) -> str:
        """Run OCR on a single PDF page rendered as image."""
        try:
            from runeextract.processors.ocr import extract_text
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            return extract_text(img_bytes)
        except ImportError:
            logger.debug("OCR processor not available (easyocr/Pillow missing)")
            return ""
        except Exception as exc:
            logger.debug(f"Page OCR failed: {exc}")
            return ""
    
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
            except Exception as exc:
                logger.debug(f"Skipping image xref={xref} on page {page_num}: {exc}")
        
        return images
    
    def _extract_tables(self, file_path: str) -> List[Table]:
        """Extract tables from PDF using pdfplumber (optional dep)."""
        tables = []
        
        try:
            import pdfplumber
        except ImportError:
            logger.debug("pdfplumber not available; skipping PDF table extraction")
            return tables

        try:
            password = self.options.get('password')
            with pdfplumber.open(file_path, password=password) as pdf:
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
            logger.warning(f"Table extraction failed for {file_path}: {e}")
        
        return tables
    
    def supported_extensions(self) -> list[str]:
        """Return supported file extensions."""
        return [".pdf"]


class PdfStreamingExtractor(StreamingExtractor, PDFExtractor):
    """PDF extractor with page-by-page streaming support."""

    async def extract_stream(self, file_path: str):
        """Yield a Document per page."""
        self.validate_file(file_path)
        import fitz
        password = self.options.get('password')
        doc = fitz.open(file_path)
        if doc.is_encrypted:
            if not password:
                raise WrongPasswordError(file_path)
            if not doc.authenticate(password):
                raise WrongPasswordError(file_path)
        try:
            total_pages = len(doc)
            cumulative = 0

            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text()
                if self.ocr and len(page_text.strip()) < 20:
                    ocr_text = self._ocr_page(page)
                    if ocr_text:
                        page_text = ocr_text

                cumulative += len(page_text) + 2
                page_doc = Document(
                    text=page_text,
                    source_type="pdf",
                    source_path=file_path,
                    metadata={
                        "page": page_num,
                        "total_pages": total_pages,
                        "page_breaks": [cumulative],
                    }
                )
                yield page_doc
        finally:
            doc.close()
