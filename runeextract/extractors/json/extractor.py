"""
JSON extractor for structured JSON data.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document as RuneDocument, Table
from runeextract.exceptions import CorruptFileError

logger = logging.getLogger(__name__)


class JsonExtractor(BaseExtractor):
    """Extractor for JSON files."""

    def extract(self, file_path: str) -> RuneDocument:
        self.validate_file(file_path)

        text = ""
        tables: List[Table] = []
        metadata: Dict[str, Any] = {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            raise CorruptFileError(file_path, detail=str(exc))

        metadata["type"] = type(data).__name__
        if isinstance(data, list):
            metadata["length"] = len(data)
        elif isinstance(data, dict):
            metadata["keys"] = list(data.keys())

        text = json.dumps(data, indent=2, default=str)

        # Extract tables from list of dicts (array of objects)
        if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
            columns = list(data[0].keys())
            rows = []
            for item in data:
                rows.append([str(item.get(k, "")) for k in columns])
            if rows:
                tables.append(Table(
                    rows=rows,
                    columns=columns,
                    metadata={"source": file_path}
                ))

        text = self.clean_text(text)

        return RuneDocument(
            text=text, tables=tables, images=[], metadata=metadata,
            source_type="json", source_path=file_path
        )

    def supported_extensions(self) -> list[str]:
        return [".json"]
