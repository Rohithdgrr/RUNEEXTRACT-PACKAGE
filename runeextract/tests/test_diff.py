"""Tests for document diff / version tracking."""

import tempfile
import os

import pytest

from runeextract.diff.comparator import (
    DiffChange, DiffResult, DocumentComparator, ChangeType,
    diff_documents, compare_files,
)


class TestDiffChange:
    def test_create(self):
        c = DiffChange(change_type=ChangeType.ADDED, text="hello")
        assert c.change_type == ChangeType.ADDED
        assert c.text == "hello"

    def test_with_lines(self):
        c = DiffChange(ChangeType.REMOVED, "old", line_number_old=5)
        assert c.line_number_old == 5
        assert c.line_number_new is None


class TestDiffResult:
    def test_empty(self):
        r = DiffResult()
        assert r.total_changes == 0
        assert r.added_count == 0
        assert r.removed_count == 0

    def test_counts(self):
        r = DiffResult(changes=[
            DiffChange(ChangeType.ADDED, "a"),
            DiffChange(ChangeType.REMOVED, "r"),
            DiffChange(ChangeType.REMOVED, "r2"),
        ])
        assert r.added_count == 1
        assert r.removed_count == 2
        assert r.total_changes == 3

    def test_summary(self):
        r = DiffResult(changes=[
            DiffChange(ChangeType.ADDED, "a"),
            DiffChange(ChangeType.REMOVED, "r"),
        ])
        s = r.summary()
        assert "1 added" in s
        assert "1 removed" in s


class TestDocumentComparatorIdentical:
    def test_no_changes(self):
        result = DocumentComparator().compare_text("hello\nworld", "hello\nworld")
        assert result.total_changes == 0

    def test_empty_strings(self):
        result = DocumentComparator().compare_text("", "")
        assert result.total_changes == 0


class TestDocumentComparatorAdditions:
    def test_add_line(self):
        result = DocumentComparator().compare_text("hello", "hello\nworld")
        assert result.added_count >= 1

    def test_add_empty_to_content(self):
        result = DocumentComparator().compare_text("", "new content")
        assert result.added_count >= 1


class TestDocumentComparatorRemovals:
    def test_remove_line(self):
        result = DocumentComparator().compare_text("hello\nworld", "hello")
        assert result.removed_count >= 1

    def test_remove_all(self):
        result = DocumentComparator().compare_text("hello\nworld", "")
        assert result.removed_count >= 1


class TestDocumentComparatorReplacements:
    def test_replace_content(self):
        old = "line1\nline2\nline3"
        new = "line1\nCHANGED\nline3"
        result = DocumentComparator().compare_text(old, new)
        assert result.total_changes >= 1

    def test_mixed_changes(self):
        old = "a\nb\nc"
        new = "a\nX\nc\nd"
        result = DocumentComparator().compare_text(old, new)
        assert result.total_changes >= 1


class TestDocumentComparatorCompareFiles:
    def test_compare_same_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("content")
            path = f.name
        try:
            result = DocumentComparator().compare_files(path, path)
            assert result.total_changes == 0
        finally:
            os.unlink(path)

    def test_compare_different_files(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f1:
            f1.write("old content")
            p1 = f1.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f2:
            f2.write("new content")
            p2 = f2.name
        try:
            result = DocumentComparator().compare_files(p1, p2)
            assert result.total_changes >= 1
        finally:
            os.unlink(p1)
            os.unlink(p2)


class TestConvenienceFunctions:
    def test_diff_documents(self):
        result = diff_documents("old", "new")
        assert isinstance(result, DiffResult)
        assert result.total_changes >= 1

    def test_compare_files(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f1:
            f1.write("a")
            p1 = f1.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f2:
            f2.write("b")
            p2 = f2.name
        try:
            result = compare_files(p1, p2)
            assert isinstance(result, DiffResult)
        finally:
            os.unlink(p1)
            os.unlink(p2)
