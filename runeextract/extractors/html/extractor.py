"""
HTML extractor using BeautifulSoup4.
"""

from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import requests
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document as RuneDocument, Table, Image


class HtmlExtractor(BaseExtractor):
    """Extractor for HTML files."""
    
    def extract(self, file_path: str) -> RuneDocument:
        """
        Extract content from an HTML file.
        
        Args:
            file_path: Path to the HTML file or URL
            
        Returns:
            Document object with extracted content
        """
        # Check if it's a URL or local file
        if file_path.startswith(('http://', 'https://')):
            return self._extract_from_url(file_path)
        else:
            self.validate_file(file_path)
            return self._extract_from_file(file_path)
    
    def _extract_from_file(self, file_path: str) -> RuneDocument:
        """Extract content from local HTML file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return self._parse_html(html_content, file_path)
    
    def _extract_from_url(self, url: str) -> RuneDocument:
        """Extract content from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            html_content = response.text
            return self._parse_html(html_content, url)
        except Exception as e:
            raise ValueError(f"Failed to fetch URL: {e}")
    
    def _parse_html(self, html_content: str, source: str) -> RuneDocument:
        """Parse HTML content and extract structured data."""
        soup = BeautifulSoup(html_content, 'lxml')
        
        text = ""
        tables: List[Table] = []
        images: List[Image] = []
        metadata: Dict[str, Any] = {}
        
        # Extract metadata
        metadata = self._extract_metadata(soup)
        
        # Extract text content
        text = self._extract_text(soup)
        
        # Extract tables
        if self.options.get('tables', True):
            tables = self._extract_tables(soup)
        
        # Extract images
        if self.options.get('images', True):
            images = self._extract_images(soup, source)
        
        # Clean text
        text = self.clean_text(text)
        
        return RuneDocument(
            text=text,
            tables=tables,
            images=images,
            metadata=metadata,
            source_type="html",
            source_path=source
        )
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from HTML."""
        metadata = {}
        
        # Title
        title_tag = soup.find('title')
        metadata['title'] = title_tag.get_text().strip() if title_tag else ""
        
        # Meta tags
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            name = tag.get('name') or tag.get('property')
            content = tag.get('content')
            if name and content:
                metadata[name] = content
        
        # Language
        html_tag = soup.find('html')
        if html_tag:
            metadata['language'] = html_tag.get('lang', '')
        
        return metadata
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract text content from HTML."""
        # Remove script and style elements
        for script in soup(['script', 'style', 'nav', 'footer', 'header']):
            script.decompose()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        return text
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Table]:
        """Extract tables from HTML."""
        tables = []
        
        for table_index, table in enumerate(soup.find_all('table'), start=1):
            rows = []
            columns = []
            
            # Find headers
            thead = table.find('thead')
            if thead:
                header_row = thead.find('tr')
                if header_row:
                    columns = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
            
            # If no thead, use first row of tbody
            if not columns:
                tbody = table.find('tbody')
                if tbody:
                    first_row = tbody.find('tr')
                    if first_row:
                        columns = [th.get_text().strip() for th in first_row.find_all(['th', 'td'])]
            
            # Extract data rows
            tbody = table.find('tbody')
            if not tbody:
                tbody = table
            
            for row in tbody.find_all('tr'):
                row_data = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                if row_data and any(row_data):
                    rows.append(row_data)
            
            if rows:
                tables.append(Table(
                    rows=rows,
                    columns=columns,
                    metadata={'table_index': table_index}
                ))
        
        return tables
    
    def _extract_images(self, soup: BeautifulSoup, source: str) -> List[Image]:
        """Extract image information from HTML."""
        images = []
        
        for img_index, img in enumerate(soup.find_all('img'), start=1):
            src = img.get('src')
            alt = img.get('alt', '')
            
            if src:
                # Resolve relative URLs
                if source.startswith(('http://', 'https://')):
                    src = urljoin(source, src)
                
                images.append(Image(
                    data=b'',  # Would need to download actual image data
                    format='unknown',
                    metadata={
                        'src': src,
                        'alt': alt,
                        'index': img_index
                    }
                ))
        
        return images
    
    def supported_extensions(self) -> list[str]:
        """Return supported file extensions."""
        return [".html", ".htm"]
