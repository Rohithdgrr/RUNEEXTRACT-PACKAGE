"""Document diff — compare two documents and find changes between versions."""

import difflib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from runeextract import extract
from runeextract.models.document import Document


class ChangeType(Enum):
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


@dataclass
class DiffChange:
    change_type: ChangeType
    text: str
    line_number_old: Optional[int] = None
    line_number_new: Optional[int] = None


@dataclass
class DiffResult:
    changes: List[DiffChange] = field(default_factory=list)
    old_text: str = ""
    new_text: str = ""

    @property
    def added_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == ChangeType.ADDED)

    @property
    def removed_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == ChangeType.REMOVED)

    @property
    def modified_count(self) -> int:
        return sum(1 for c in self.changes if c.change_type == ChangeType.MODIFIED)

    @property
    def total_changes(self) -> int:
        return len(self.changes)

    def summary(self) -> str:
        return (
            f"Changes: {self.total_changes} total "
            f"({self.added_count} added, {self.removed_count} removed, "
            f"{self.modified_count} modified)"
        )


class DocumentComparator:
    """Compare two documents and produce a structured diff."""

    def compare_text(self, old_text: str, new_text: str) -> DiffResult:
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)

        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        changes = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            old_chunk = "".join(old_lines[i1:i2]).rstrip("\n")
            new_chunk = "".join(new_lines[j1:j2]).rstrip("\n")

            if tag == "replace":
                if old_chunk:
                    changes.append(DiffChange(
                        change_type=ChangeType.REMOVED,
                        text=old_chunk,
                        line_number_old=i1 + 1,
                    ))
                if new_chunk:
                    changes.append(DiffChange(
                        change_type=ChangeType.ADDED,
                        text=new_chunk,
                        line_number_new=j1 + 1,
                    ))
            elif tag == "delete":
                if old_chunk:
                    changes.append(DiffChange(
                        change_type=ChangeType.REMOVED,
                        text=old_chunk,
                        line_number_old=i1 + 1,
                    ))
            elif tag == "insert":
                if new_chunk:
                    changes.append(DiffChange(
                        change_type=ChangeType.ADDED,
                        text=new_chunk,
                        line_number_new=j1 + 1,
                    ))

        return DiffResult(changes=changes, old_text=old_text, new_text=new_text)

    def compare_documents(self, old_doc: Document, new_doc: Document) -> DiffResult:
        return self.compare_text(old_doc.text or "", new_doc.text or "")

    def compare_files(self, old_path: str, new_path: str, **kwargs) -> DiffResult:
        old_doc = extract(old_path, **kwargs)
        new_doc = extract(new_path, **kwargs)
        return self.compare_documents(old_doc, new_doc)


def diff_documents(old_text: object, new_text: object) -> DiffResult:
    from runeextract.models.document import Document
    if isinstance(old_text, Document):
        old_text = old_text.text or ""
    if isinstance(new_text, Document):
        new_text = new_text.text or ""
    comparator = DocumentComparator()
    return comparator.compare_text(str(old_text), str(new_text))


def compare_files(old_path: str, new_path: str, **kwargs) -> DiffResult:
    comparator = DocumentComparator()
    return comparator.compare_files(old_path, new_path, **kwargs)
