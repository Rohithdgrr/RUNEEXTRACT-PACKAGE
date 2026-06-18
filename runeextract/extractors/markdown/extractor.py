"""Markdown extractor (zero additional dependencies)."""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document as RuneDocument, Table

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)$', re.MULTILINE)
_BULLET_RE = re.compile(r'^[\s]*[-*+]\s+(.*)$', re.MULTILINE)
_ORDERED_RE = re.compile(r'^[\s]*\d+[.)]\s+(.*)$', re.MULTILINE)
_FENCE_RE = re.compile(r'^```', re.MULTILINE)
_HR_RE = re.compile(r'^[-*_]{3,}\s*$', re.MULTILINE)
_INLINE_CODE_RE = re.compile(r'`([^`]+)`')
_LINK_RE = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\([^)]+\)')
_BOLD_RE = re.compile(r'\*\*(.+?)\*\*')
_ITALIC_RE = re.compile(r'\*(.+?)\*')


class MarkdownExtractor(BaseExtractor):
    """Extractor for Markdown files (zero extra deps)."""

    def extract(self, file_path: str) -> RuneDocument:
        self.validate_file(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        metadata = self._extract_frontmatter(content)
        body = self._strip_frontmatter(content)
        text = self._render_body(body)
        tables = self._extract_tables(body)
        return RuneDocument(
            text=self.clean_text(text),
            tables=tables,
            images=[],
            metadata=metadata,
            source_type="markdown",
            source_path=file_path,
        )

    def _strip_frontmatter(self, content: str) -> str:
        lines = content.split('\n')
        if lines and lines[0].strip() == '---':
            for i, line in enumerate(lines[1:], start=1):
                if line.strip() == '---':
                    return '\n'.join(lines[i + 1:])
        return content

    def _render_body(self, body: str) -> str:
        lines = body.split('\n')
        out = []
        in_code = False
        for line in lines:
            if _FENCE_RE.match(line):
                in_code = not in_code
                out.append('')
                continue
            if in_code:
                out.append(line)
                continue
            if _HR_RE.match(line):
                out.append('')
                continue
            line = _IMAGE_RE.sub(r'[image: \1]', line)
            line = _LINK_RE.sub(r'\1', line)
            line = _BOLD_RE.sub(r'\1', line)
            line = _ITALIC_RE.sub(r'\1', line)
            line = _INLINE_CODE_RE.sub(r'`\1`', line)
            m = _HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                out.append(f"{'#' * level} {m.group(2)}")
                continue
            m = _BULLET_RE.match(line)
            if m:
                out.append(m.group(1))
                continue
            out.append(line)
        text = '\n'.join(out)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text
    
    def _extract_frontmatter(self, markdown_content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from markdown."""
        metadata = {}
        
        lines = markdown_content.split('\n')
        if lines and lines[0].strip() == '---':
            # Find end of frontmatter
            end_idx = -1
            for i, line in enumerate(lines[1:], start=1):
                if line.strip() == '---':
                    end_idx = i
                    break
            
            if end_idx > 0:
                frontmatter_lines = lines[1:end_idx]
                # Simple parsing - for full YAML support, would use PyYAML
                for line in frontmatter_lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
        
        return metadata
    
    def _extract_tables(self, markdown_content: str) -> List[Table]:
        """Extract tables from markdown content."""
        tables = []
        lines = markdown_content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this looks like a table row
            if '|' in line:
                # Collect table rows
                table_rows = []
                while i < len(lines) and '|' in lines[i]:
                    table_rows.append(lines[i])
                    i += 1
                
                # Parse table
                if len(table_rows) >= 2:
                    parsed_table = self._parse_markdown_table(table_rows)
                    if parsed_table:
                        tables.append(parsed_table)
            else:
                i += 1
        
        return tables
    
    def _parse_markdown_table(self, rows: List[str]) -> Table:
        """Parse a markdown table into structured format."""
        # Remove leading/trailing pipes and split
        parsed_rows = []
        
        for row in rows:
            # Remove leading/trailing |
            row = row.strip()
            if row.startswith('|'):
                row = row[1:]
            if row.endswith('|'):
                row = row[:-1]
            
            # Split by |
            cells = [cell.strip() for cell in row.split('|')]
            parsed_rows.append(cells)
        
        if not parsed_rows:
            return None
        
        # Check if second row is separator (contains only -, :, |)
        if len(parsed_rows) >= 2:
            separator_row = parsed_rows[1]
            is_separator = all(
                cell.strip().replace('-', '').replace(':', '').replace(' ', '') == ''
                for cell in separator_row
            )
            
            if is_separator:
                # First row is headers
                columns = parsed_rows[0]
                data_rows = parsed_rows[2:]
            else:
                # No separator row, first row is data
                columns = [f"Column {i+1}" for i in range(len(parsed_rows[0]))]
                data_rows = parsed_rows
        else:
            columns = [f"Column {i+1}" for i in range(len(parsed_rows[0]))]
            data_rows = parsed_rows
        
        # Filter empty rows
        data_rows = [row for row in data_rows if any(row)]
        
        if data_rows:
            return Table(
                rows=data_rows,
                columns=columns,
                metadata={}
            )
        
        return None
    
    def supported_extensions(self) -> list[str]:
        """Return supported file extensions."""
        return [".md", ".markdown"]
