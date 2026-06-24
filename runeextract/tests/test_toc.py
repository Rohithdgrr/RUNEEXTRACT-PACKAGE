"""Tests for Table of Contents extraction."""

import pytest

from runeextract.toc import TOCEntry, TOCParser, extract_toc, toc_to_markdown, toc_to_dict, toc_to_json


class TestTOCEntry:
    def test_create_minimal(self):
        e = TOCEntry(title="Intro", level=1)
        assert e.title == "Intro"
        assert e.level == 1
        assert e.page_number is None
        assert e.children == []

    def test_create_full(self):
        e = TOCEntry(title="Intro", level=1, page_number=3)
        assert e.page_number == 3


class TestTOCParser:
    def test_empty(self):
        p = TOCParser()
        assert p.parse_markdown("") == []
        assert p.parse_html("") == []
        assert p.parse_pdf_toc("") == []

    def test_markdown_headings(self):
        text = "# Intro\n\nSome text\n\n## Details\n\nMore\n\n### Deep"
        p = TOCParser()
        entries = p.parse_markdown(text)
        assert len(entries) == 1
        assert entries[0].title == "Intro"
        assert len(entries[0].children) == 1
        assert entries[0].children[0].title == "Details"
        assert len(entries[0].children[0].children) == 1
        assert entries[0].children[0].children[0].title == "Deep"

    def test_markdown_multiple_top_level(self):
        text = "# A\n\n# B\n\n## B1"
        p = TOCParser()
        entries = p.parse_markdown(text)
        assert len(entries) == 2
        assert entries[1].title == "B"
        assert len(entries[1].children) == 1

    def test_html_headings(self):
        html = "<h1>Intro</h1><p>Text</p><h2>Details</h2><h3>Sub</h3>"
        p = TOCParser()
        entries = p.parse_html(html)
        assert len(entries) == 1
        assert entries[0].title == "Intro"
        assert entries[0].children[0].title == "Details"

    def test_html_with_inline_tags(self):
        html = "<h1>Hello <em>World</em></h1>"
        p = TOCParser()
        entries = p.parse_html(html)
        assert entries[0].title == "Hello World"

    def test_pdf_toc_dots(self):
        text = "Introduction........3\n  Basics..............5\n    Sub-topic.........7\nConclusion........10"
        p = TOCParser()
        entries = p.parse_pdf_toc(text)
        assert len(entries) > 0
        assert any(e.title == "Introduction" for e in entries)
        intro = next(e for e in entries if e.title == "Introduction")
        assert intro.page_number == 3
        assert len(intro.children) > 0

    def test_pdf_toc_no_dots(self):
        text = "Intro\n  Details\n    Sub"
        p = TOCParser()
        entries = p.parse_pdf_toc(text)
        assert len(entries) == 1
        assert entries[0].title == "Intro"

    def test_build_tree_flat(self):
        e = TOCParser()._build_tree([
            TOCEntry(title="A", level=1),
            TOCEntry(title="B", level=1),
        ])
        assert len(e) == 2

    def test_build_tree_nested(self):
        e = TOCParser()._build_tree([
            TOCEntry(title="A", level=1),
            TOCEntry(title="A1", level=2),
            TOCEntry(title="A2", level=2),
            TOCEntry(title="B", level=1),
        ])
        assert len(e) == 2
        assert len(e[0].children) == 2

    def test_build_tree_skip_levels(self):
        e = TOCParser()._build_tree([
            TOCEntry(title="A", level=1),
            TOCEntry(title="C", level=3),
        ])
        assert len(e[0].children) == 1
        assert e[0].children[0].level == 3


class TestExtractTOC:
    def test_markdown_default(self):
        entries = extract_toc("# Hello\n\n## World")
        assert len(entries) == 1
        assert entries[0].title == "Hello"

    def test_html_format(self):
        entries = extract_toc("<h1>Hello</h1>", format="html")
        assert len(entries) == 1

    def test_unknown_format_fallback(self):
        entries = extract_toc("# Hello", format="unknown")
        assert len(entries) == 1


class TestTOCToMarkdown:
    def test_flat(self):
        entries = [TOCEntry(title="Intro", level=1, page_number=3)]
        md = toc_to_markdown(entries)
        assert "- Intro … 3" in md

    def test_no_page(self):
        entries = [TOCEntry(title="Intro", level=1)]
        md = toc_to_markdown(entries)
        assert "- Intro" in md

    def test_nested(self):
        e = TOCEntry(title="A", level=1, children=[TOCEntry(title="A1", level=2)])
        md = toc_to_markdown([e])
        assert "- A" in md
        assert "  - A1" in md


class TestTOCToDict:
    def test_simple(self):
        entries = [TOCEntry(title="Intro", level=1)]
        j = toc_to_dict(entries)
        assert j[0]["title"] == "Intro"

    def test_with_page(self):
        entries = [TOCEntry(title="Intro", level=1, page_number=3)]
        j = toc_to_dict(entries)
        assert j[0]["page_number"] == 3

    def test_nested(self):
        e = TOCEntry(title="A", level=1, children=[TOCEntry(title="A1", level=2)])
        j = toc_to_dict([e])
        assert j[0]["children"][0]["title"] == "A1"


class TestTOCToJson:
    def test_returns_string(self):
        entries = [TOCEntry(title="Intro", level=1)]
        result = toc_to_json(entries)
        assert isinstance(result, str)
        assert '"Intro"' in result

    def test_with_indent(self):
        entries = [TOCEntry(title="Intro", level=1)]
        result = toc_to_json(entries, indent=2)
        assert "\n" in result
