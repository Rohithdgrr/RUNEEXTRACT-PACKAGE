"""
Markdown extractor using markdown-it-py.
"""

from markdown_it import MarkdownIt
from pathlib import Path
from typing import List, Dict, Any
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document as RuneDocument, Table


class MarkdownExtractor(BaseExtractor):
    """Extractor for Markdown files."""
    
    def extract(self, file_path: str) -> RuneDocument:
        """
        Extract content from a Markdown file.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            Document object with extracted content
        """
        self.validate_file(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        text = ""
        tables: List[Table] = []
        metadata: Dict[str, Any] = {}
        
        # Parse markdown
        md = MarkdownIt()
        tokens = md.parse(markdown_content)
        
        # Extract content
        text, tables, metadata = self._parse_tokens(tokens, markdown_content)
        
        # Clean text
        text = self.clean_text(text)
        
        return RuneDocument(
            text=text,
            tables=tables,
            images=[],
            metadata=metadata,
            source_type="markdown",
            source_path=file_path
        )
    
    def _parse_tokens(self, tokens, markdown_content: str) -> tuple[str, List[Table], Dict[str, Any]]:
        """Parse markdown tokens to extract structured content."""
        text = ""
        tables: List[Table] = []
        metadata: Dict[str, Any] = {}
        
        # Extract frontmatter (YAML metadata)
        metadata = self._extract_frontmatter(markdown_content)
        
        # Extract text and tables from tokens
        for token in tokens:
            if token.type == 'heading_open':
                level = token.tag
                text += "#" * int(level[1:]) + "\n\n"
            elif token.type == 'heading_close':
                text += "\n\n"
            elif token.type == 'paragraph_open':
                text += "\n"
            elif token.type == 'paragraph_close':
                text += "\n\n"
            elif token.type == 'inline':
                if token.children:
                    for child in token.children:
                        if child.type == 'text':
                            text += child.content
                        elif child.type == 'code_inline':
                            text += f"`{child.content}`"
                        elif child.type == 'softbreak':
                            text += " "
            elif token.type == 'bullet_list_open':
                text += "\n"
            elif token.type == 'list_item_open':
                text += "• "
            elif token.type == 'list_item_close':
                text += "\n"
            elif token.type == 'ordered_list_open':
                text += "\n"
            elif token.type == 'fence':
                text += f"\n```\n{token.content}\n```\n\n"
            elif token.type == 'code_block':
                text += f"\n```\n{token.content}\n```\n\n"
            elif token.type == 'table_open':
                # Start of table
                pass
            elif token.type == 'table_close':
                # End of table - tables are handled separately
                pass
            elif token.type == 'hr':
                text += "\n---\n\n"
        
        # Extract tables separately
        tables = self._extract_tables(markdown_content)
        
        return text, tables, metadata
    
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
