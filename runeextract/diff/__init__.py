"""RuneExtract Diff — document diff and version tracking."""

from runeextract.diff.comparator import (
    DiffChange, DiffResult, DocumentComparator,
    diff_documents, compare_files,
)

__all__ = [
    "DiffChange", "DiffResult", "DocumentComparator",
    "diff_documents", "compare_files",
]
