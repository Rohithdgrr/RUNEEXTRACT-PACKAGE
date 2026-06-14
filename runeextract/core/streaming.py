"""
Streaming extractor resolution.
"""

import logging
from pathlib import Path
from runeextract.core.extractor import BaseExtractor, StreamingExtractor
from runeextract.core.router import ExtractorRouter
from runeextract.exceptions import UnsupportedFormatError

logger = logging.getLogger(__name__)

# Extensions that support streaming
_STREAMING_EXTRACTORS = {
    '.pdf': 'runeextract.extractors.pdf.extractor.PdfStreamingExtractor',
}


def get_streaming_extractor(file_path: str, **kwargs) -> StreamingExtractor:
    """
    Get a streaming-capable extractor for the given file.

    Falls back to the standard extractor wrapped in StreamingExtractor.
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension in _STREAMING_EXTRACTORS:
        module_path = _STREAMING_EXTRACTORS[extension]
        try:
            module_name, class_name = module_path.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            extractor_class = getattr(module, class_name)
            return extractor_class(**kwargs)
        except ImportError as exc:
            logger.warning(f"Streaming extractor unavailable for {extension}: {exc}")

    # Fallback: wrap standard extractor
    base = ExtractorRouter.get_extractor(file_path, **kwargs)
    return _WrappedStreamingExtractor(base)


class _WrappedStreamingExtractor(StreamingExtractor):
    """Wraps a non-streaming extractor for the streaming interface."""

    def __init__(self, extractor: BaseExtractor):
        self._extractor = extractor
        super().__init__(
            ocr=getattr(extractor, 'ocr', False),
            **getattr(extractor, 'options', {})
        )

    def extract(self, file_path: str):
        return self._extractor.extract(file_path)

    def supported_extensions(self) -> list[str]:
        return self._extractor.supported_extensions()

    async def extract_stream(self, file_path: str):
        import asyncio
        loop = asyncio.get_running_loop()
        doc = await loop.run_in_executor(None, self._extractor.extract, file_path)
        yield doc
