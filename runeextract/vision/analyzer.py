"""Vision analyzer — interpret charts, figures, and diagrams using vision LLMs.

Requires an AI provider with vision capabilities (OpenAI GPT-4V, Anthropic Claude 3).
Uses the existing runeextract AIProcessor infrastructure.
"""

import base64
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from runeextract.models.document import Image as DocImage

logger = logging.getLogger(__name__)


@dataclass
class ChartInterpretation:
    chart_type: str = ""
    title: str = ""
    axes: dict = field(default_factory=dict)
    data_summary: str = ""
    key_insights: List[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class FigureCaption:
    caption: str = ""
    description: str = ""
    objects_detected: List[str] = field(default_factory=list)
    text_in_image: str = ""
    raw_response: str = ""


class VisionAnalyzer:
    """Analyze images/charts/figures using vision-capable LLMs.

    Uses the AIProcessor infrastructure with a vision model prompt.
    """

    def __init__(self, provider: str = "openai", model: str = "gpt-4o", **kwargs):
        self.provider = provider
        self.model = model
        self.kwargs = kwargs
        self._ai = None

    def _get_ai(self):
        if self._ai is None:
            from runeextract.processors.ai import AIProcessor
            self._ai = AIProcessor(provider=self.provider, model=self.model, **self.kwargs)
        return self._ai

    def _image_to_data_url(self, image: DocImage) -> str:
        fmt = image.format or "png"
        b64 = base64.b64encode(image.data).decode("utf-8")
        return f"data:image/{fmt};base64,{b64}"

    def describe_image(self, image: DocImage, prompt: str = "Describe this image in detail.") -> FigureCaption:
        ai = self._get_ai()
        data_url = self._image_to_data_url(image)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        response = ai._call(messages=messages, model=self.model)
        text = response.get("text", "") or response.get("content", "") or str(response)
        return FigureCaption(
            caption=text[:200],
            description=text,
            raw_response=text,
        )

    def interpret_chart(self, image: DocImage, chart_context: str = "") -> ChartInterpretation:
        ai = self._get_ai()
        data_url = self._image_to_data_url(image)
        prompt = (
            "You are a chart analysis expert. Analyze this chart/figure and return:\n"
            "1. Chart type (bar, line, pie, scatter, etc.)\n"
            "2. Title\n"
            "3. X and Y axes with labels and ranges\n"
            "4. Data summary\n"
            "5. 3-5 key insights\n"
        )
        if chart_context:
            prompt += f"\nContext: {chart_context}\n"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        response = ai._call(messages=messages, model=self.model)
        text = response.get("text", "") or response.get("content", "") or str(response)
        return ChartInterpretation(
            chart_type=self._extract_field(text, "Chart type", "unknown"),
            title=self._extract_field(text, "Title", ""),
            data_summary=self._extract_field(text, "Data summary", text[:500]),
            raw_response=text,
        )

    def caption_figure(self, image: DocImage, context: str = "") -> FigureCaption:
        ai = self._get_ai()
        data_url = self._image_to_data_url(image)
        prompt = "Generate a concise figure caption and detailed description for this image."
        if context:
            prompt += f"\nDocument context: {context}\n"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        response = ai._call(messages=messages, model=self.model)
        text = response.get("text", "") or response.get("content", "") or str(response)
        return FigureCaption(
            caption=text[:200],
            description=text,
            raw_response=text,
        )

    @staticmethod
    def _extract_field(text: str, field_name: str, default: str = "") -> str:
        import re
        patterns = [
            rf"{field_name}[\s:]*([^\n]+)",
            rf"\*\*{field_name}[\s:]*\*\*([^\n]+)",
            rf"\d+\.\s+\*?\*?{field_name}\*?\*?[\s:]*([^\n]+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return default


def describe_image(image: DocImage, provider: str = "openai", model: str = "gpt-4o") -> FigureCaption:
    analyzer = VisionAnalyzer(provider=provider, model=model)
    return analyzer.describe_image(image)


def interpret_chart(image: DocImage, provider: str = "openai", model: str = "gpt-4o") -> ChartInterpretation:
    analyzer = VisionAnalyzer(provider=provider, model=model)
    return analyzer.interpret_chart(image)


def caption_figure(image: DocImage, provider: str = "openai", model: str = "gpt-4o") -> FigureCaption:
    analyzer = VisionAnalyzer(provider=provider, model=model)
    return analyzer.caption_figure(image)
