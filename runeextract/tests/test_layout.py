"""Tests for layout-aware parsing."""

import pytest

from runeextract.layout.parser import (
    BoundingBox, LayoutElement, LayoutParser, ElementType,
    parse_layout, get_reading_order,
)


class TestBoundingBox:
    def test_basic(self):
        b = BoundingBox(10, 20, 100, 200)
        assert b.width() == 90
        assert b.height() == 180
        assert b.area() == 16200

    def test_overlap_true(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 5, 15, 15)
        assert a.overlaps(b)

    def test_overlap_false(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(20, 20, 30, 30)
        assert not a.overlaps(b)

    def test_overlap_threshold(self):
        a = BoundingBox(0, 0, 100, 100)
        b = BoundingBox(90, 90, 200, 200)
        assert not a.overlaps(b, threshold=0.5)

    def test_page_default(self):
        b = BoundingBox(0, 0, 10, 10)
        assert b.page == 0


class TestLayoutElement:
    def test_create(self):
        bbox = BoundingBox(0, 0, 10, 10)
        el = LayoutElement(bbox=bbox, element_type=ElementType.HEADING, text="Title", confidence=0.95)
        assert el.element_type == ElementType.HEADING
        assert el.text == "Title"
        assert el.confidence == 0.95

    def test_default_conf(self):
        el = LayoutElement(bbox=BoundingBox(0, 0, 1, 1), element_type=ElementType.TEXT)
        assert el.confidence == 1.0


class TestLayoutParserParseText:
    def test_empty_text(self):
        result = LayoutParser().parse("")
        assert result == []

    def test_plain_text(self):
        result = LayoutParser().parse("hello\nworld")
        assert len(result) == 2
        assert result[0].element_type == ElementType.TEXT
        assert result[1].element_type == ElementType.TEXT

    def test_heading(self):
        result = LayoutParser().parse("# Title\nparagraph")
        assert len(result) == 2
        assert result[0].element_type == ElementType.HEADING

    def test_list_detection(self):
        result = LayoutParser().parse("- item\n* another")
        assert all(e.element_type == ElementType.LIST for e in result)

    def test_table_detection(self):
        result = LayoutParser().parse("| a | b |\n| 1 | 2 |")
        assert result[0].element_type == ElementType.TABLE

    def test_blank_lines_skipped(self):
        result = LayoutParser().parse("a\n\n\nb")
        assert len(result) == 2


class TestLayoutParserParseHtml:
    def test_html_headings(self):
        html = "<h1>Title</h1><p>Para</p>"
        result = LayoutParser().parse(html, source_type="html")
        assert len(result) >= 1
        assert result[0].element_type == ElementType.HEADING

    def test_html_table(self):
        html = "<table><tr><td>x</td></tr></table>"
        result = LayoutParser().parse(html, source_type="html")
        table_els = [e for e in result if e.element_type == ElementType.TABLE]
        assert len(table_els) >= 1


class TestLayoutParserGetColumns:
    def test_two_columns(self):
        elements = [
            LayoutElement(bbox=BoundingBox(0, 0, 5, 1), element_type=ElementType.TEXT),
            LayoutElement(bbox=BoundingBox(10, 0, 15, 1), element_type=ElementType.TEXT),
        ]
        cols = LayoutParser().get_columns(elements)
        assert len(cols) == 2
        assert len(cols[0]) == 1
        assert len(cols[1]) == 1

    def test_empty(self):
        assert LayoutParser().get_columns([]) == []


class TestReadingOrder:
    def test_top_to_bottom(self):
        els = [
            LayoutElement(bbox=BoundingBox(0, 10, 5, 11), element_type=ElementType.TEXT),
            LayoutElement(bbox=BoundingBox(0, 0, 5, 1), element_type=ElementType.TEXT),
        ]
        ordered = get_reading_order(els)
        assert ordered[0].bbox.y0 == 0
        assert ordered[1].bbox.y0 == 10

    def test_page_sorting(self):
        els = [
            LayoutElement(bbox=BoundingBox(0, 0, 1, 1, page=1), element_type=ElementType.TEXT),
            LayoutElement(bbox=BoundingBox(0, 0, 1, 1, page=0), element_type=ElementType.TEXT),
        ]
        ordered = get_reading_order(els)
        assert ordered[0].bbox.page == 0

    def test_left_to_right(self):
        els = [
            LayoutElement(bbox=BoundingBox(10, 0, 15, 1), element_type=ElementType.TEXT),
            LayoutElement(bbox=BoundingBox(0, 0, 5, 1), element_type=ElementType.TEXT),
        ]
        ordered = get_reading_order(els)
        assert ordered[0].bbox.x0 == 0


class TestParseLayout:
    def test_convenience(self):
        result = parse_layout("hello\nworld")
        assert len(result) == 2
