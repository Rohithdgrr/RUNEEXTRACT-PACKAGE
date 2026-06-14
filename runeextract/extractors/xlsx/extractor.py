"""
XLSX extractor using openpyxl.
"""

from openpyxl import load_workbook
from pathlib import Path
from typing import List, Dict, Any
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document as RuneDocument, Table


class XlsxExtractor(BaseExtractor):
    """Extractor for XLSX files."""

    def extract(self, file_path: str) -> RuneDocument:
        self.validate_file(file_path)
        wb = load_workbook(file_path, read_only=True, data_only=True)
        text = ""
        tables: List[Table] = []
        metadata: Dict[str, Any] = {}
        metadata = self._extract_metadata(wb)
        metadata['sheet_count'] = len(wb.sheetnames)
        metadata['sheet_names'] = wb.sheetnames

        for sheet_index, sheet_name in enumerate(wb.sheetnames, start=1):
            ws = wb[sheet_name]
            text += f"\n--- Sheet: {sheet_name} ---\n\n"
            rows_data = []
            if ws.max_row and ws.max_column:
                for row in ws.iter_rows(values_only=True):
                    row_values = [str(cell) if cell is not None else "" for cell in row]
                    rows_data.append(row_values)
            if rows_data:
                columns = rows_data[0] if rows_data else []
                data_rows = rows_data[1:]
                has_data = any(any(cell for cell in row) for row in data_rows)
                if has_data:
                    tables.append(Table(rows=data_rows, columns=columns,
                                        metadata={'sheet_name': sheet_name, 'sheet_index': sheet_index,
                                                  'max_row': len(data_rows), 'max_column': len(columns)}))
                text_lines = []
                for row in rows_data:
                    text_lines.append('\t'.join(row))
                text += '\n'.join(text_lines) + '\n\n'

        wb.close()
        text = self.clean_text(text)
        return RuneDocument(text=text, tables=tables, images=[], metadata=metadata,
                            source_type="xlsx", source_path=file_path)

    def _extract_metadata(self, wb) -> Dict[str, Any]:
        metadata = {}
        try:
            props = wb.properties
            if props:
                metadata['title'] = props.title or ""
                metadata['creator'] = props.creator or ""
                metadata['description'] = props.description or ""
                metadata['keywords'] = props.keywords or ""
                metadata['created'] = str(props.created) if props.created else ""
                metadata['modified'] = str(props.modified) if props.modified else ""
                metadata['category'] = props.category or ""
        except Exception:
            pass
        return metadata

    def supported_extensions(self) -> list[str]:
        return [".xlsx", ".xls"]
