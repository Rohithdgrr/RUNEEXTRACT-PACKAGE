"""
CSV extractor using standard library csv module.
"""

import csv
import logging
import io
from pathlib import Path
from typing import List, Dict, Any
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document as RuneDocument, Table
from runeextract.exceptions import CorruptFileError

logger = logging.getLogger(__name__)


class CsvExtractor(BaseExtractor):
    """Extractor for CSV files."""

    def extract(self, file_path: str) -> RuneDocument:
        self.validate_file(file_path)

        text = ""
        tables: List[Table] = []
        metadata: Dict[str, Any] = {}

        try:
            with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except Exception as exc:
            raise CorruptFileError(file_path, detail=str(exc))

        if not rows:
            return RuneDocument(text="", source_type="csv", source_path=file_path)

        metadata["row_count"] = len(rows)
        metadata["column_count"] = max(len(r) for r in rows) if rows else 0

        columns = rows[0] if rows else []
        data_rows = rows[1:] if len(rows) > 1 else []
        has_data = any(any(cell for cell in row) for row in data_rows)

        if has_data:
            tables.append(Table(
                rows=data_rows,
                columns=columns,
                metadata={"source": file_path}
            ))

        for row in rows:
            text += "\t".join(row) + "\n"

        text = self.clean_text(text)

        return RuneDocument(
            text=text, tables=tables, images=[], metadata=metadata,
            source_type="csv", source_path=file_path
        )

    def supported_extensions(self) -> list[str]:
        return [".csv"]
