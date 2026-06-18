"""Tests for visual document understanding."""

import pytest

from runeextract.vision.analyzer import (
    VisionAnalyzer, ChartInterpretation, FigureCaption,
    describe_image, interpret_chart, caption_figure,
    DocImage,
)


class TestChartInterpretation:
    def test_create(self):
        ci = ChartInterpretation(chart_type="bar", title="Sales", data_summary="Q1: 100", key_insights=["Up trend"])
        assert ci.chart_type == "bar"
        assert ci.title == "Sales"
        assert len(ci.key_insights) == 1

    def test_defaults(self):
        ci = ChartInterpretation()
        assert ci.chart_type == ""
        assert ci.key_insights == []


class TestFigureCaption:
    def test_create(self):
        fc = FigureCaption(caption="A cat", description="A cat sitting on a mat", objects_detected=["cat"])
        assert fc.caption == "A cat"
        assert "cat" in fc.objects_detected

    def test_defaults(self):
        fc = FigureCaption()
        assert fc.caption == ""


class TestVisionAnalyzer:
    def test_init(self):
        va = VisionAnalyzer(provider="openai", model="gpt-4o")
        assert va.provider == "openai"
        assert va.model == "gpt-4o"

    def test_image_to_data_url(self):
        va = VisionAnalyzer()
        img = DocImage(data=b"fake_png", format="png")
        url = va._image_to_data_url(img)
        assert url.startswith("data:image/png;base64,")

    def test_image_to_data_url_unknown_format(self):
        va = VisionAnalyzer()
        img = DocImage(data=b"data", format="png")
        url = va._image_to_data_url(img)
        assert url.startswith("data:image/png;base64,")

    def test_extract_field(self):
        text = "1. Chart type: bar\n2. Title: Sales Growth\n"
        result = VisionAnalyzer._extract_field(text, "Chart type")
        assert result == "bar"

    def test_extract_field_not_found(self):
        result = VisionAnalyzer._extract_field("no fields here", "Something")
        assert result == ""

    def test_extract_field_numbered(self):
        text = "1. Chart type: bar\n2. Title: Test"
        result = VisionAnalyzer._extract_field(text, "Chart type")
        assert result == "bar"


class TestDescribeImage:
    def test_describe_no_ai_provider(self):
        img = DocImage(data=b"test", format="png")
        with pytest.raises(Exception):
            describe_image(img)


class TestInterpretChart:
    def test_interpret_no_ai_provider(self):
        img = DocImage(data=b"test", format="png")
        with pytest.raises(Exception):
            interpret_chart(img)


class TestCaptionFigure:
    def test_caption_no_ai_provider(self):
        img = DocImage(data=b"test", format="png")
        with pytest.raises(Exception):
            caption_figure(img)
