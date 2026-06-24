"""Vision analyzer — interpret charts, figures, and diagrams using vision LLMs.

Requires an AI provider with vision capabilities (OpenAI GPT-4V, Anthropic Claude 3).
Uses the existing runeextract AIProcessor infrastructure.
"""

import base64
import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from runeextract.models.document import Image as DocImage
from runeextract.utils.maturity import beta

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


@beta(name="vision.analyzer")
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
        # Embed data URL in user prompt for vision-capable model
        user_content = f"{prompt}\n\nImage data URL: {data_url}"
        text = ai._call("You are a vision analyst.", user_content)
        return FigureCaption(
            caption=text[:200],
            description=text,
            raw_response=text,
        )

    def interpret_chart(self, image: DocImage, chart_context: str = "") -> ChartInterpretation:
        ai = self._get_ai()
        data_url = self._image_to_data_url(image)
        user_content = (
            "You are a chart analysis expert. Analyze this chart/figure and return:\n"
            "1. Chart type (bar, line, pie, scatter, etc.)\n"
            "2. Title\n"
            "3. X and Y axes with labels and ranges\n"
            "4. Data summary\n"
            "5. 3-5 key insights\n"
        )
        if chart_context:
            user_content += f"\nContext: {chart_context}\n"
        user_content += f"\nImage data URL: {data_url}"
        text = ai._call("You are a chart analysis expert.", user_content)
        return ChartInterpretation(
            chart_type=self._extract_field(text, "Chart type", "unknown"),
            title=self._extract_field(text, "Title", ""),
            data_summary=self._extract_field(text, "Data summary", text[:500]),
            raw_response=text,
        )

    def caption_figure(self, image: DocImage, context: str = "") -> FigureCaption:
        ai = self._get_ai()
        data_url = self._image_to_data_url(image)
        user_content = "Generate a concise figure caption and detailed description for this image."
        if context:
            user_content += f"\nDocument context: {context}\n"
        user_content += f"\nImage data URL: {data_url}"
        text = ai._call("You are a figure captioning assistant.", user_content)
        return FigureCaption(
            caption=text[:200],
            description=text,
            raw_response=text,
        )

    @staticmethod
    def _extract_field(text: str, field_name: str, default: str = "") -> str:
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


def describe_image(image, provider: str = "openai", model: str = "gpt-4o") -> FigureCaption:
    if isinstance(image, (str, os.PathLike)):
        from runeextract import extract
        doc = extract(str(image))
        found = doc.images
        if found:
            image = found[0]
        else:
            from PIL import Image as PILImage
            from runeextract.models.document import Image as DocImage
            import io
            pil_img = PILImage.open(str(image))
            buf = io.BytesIO()
            pil_img.save(buf, format=pil_img.format or "PNG")
            image = DocImage(data=buf.getvalue(), format=(pil_img.format or "PNG").lower())
    analyzer = VisionAnalyzer(provider=provider, model=model)
    return analyzer.describe_image(image)


def _resolve_image(image):
    if isinstance(image, (str, os.PathLike)):
        from runeextract import extract
        doc = extract(str(image))
        found = doc.images
        if found:
            return found[0]
        from PIL import Image as PILImage
        from runeextract.models.document import Image as DocImage
        import io
        pil_img = PILImage.open(str(image))
        buf = io.BytesIO()
        pil_img.save(buf, format=pil_img.format or "PNG")
        return DocImage(data=buf.getvalue(), format=(pil_img.format or "PNG").lower())
    return image


def interpret_chart(image, provider: str = "openai", model: str = "gpt-4o") -> ChartInterpretation:
    analyzer = VisionAnalyzer(provider=provider, model=model)
    return analyzer.interpret_chart(_resolve_image(image))


def caption_figure(image, provider: str = "openai", model: str = "gpt-4o") -> FigureCaption:
    analyzer = VisionAnalyzer(provider=provider, model=model)
    return analyzer.caption_figure(_resolve_image(image))
