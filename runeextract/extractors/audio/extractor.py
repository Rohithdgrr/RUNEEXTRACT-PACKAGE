"""
Audio extractor using Whisper (openai-whisper or local transformers).

Transcribes speech to text from audio files.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document, Image
from runeextract.exceptions import DependencyMissingError, CorruptFileError

logger = logging.getLogger(__name__)


class AudioExtractor(BaseExtractor):
    """Extractor for audio files.

    Uses openai-whisper for transcription. Falls back to local transformers
    pipeline if whisper is not installed.
    """

    def extract(self, file_path: str) -> Document:
        self.validate_file(file_path)

        text = ""
        images: List[Image] = []
        metadata: Dict[str, Any] = {}

        try:
            import whisper
        except ImportError:
            whisper = None

        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            hf_pipeline = None

        if whisper is None and hf_pipeline is None:
            raise DependencyMissingError(
                file_path,
                "openai-whisper (pip install openai-whisper) or transformers (pip install transformers[torch])"
            )

        try:
            if whisper is not None:
                model_name = self.options.get("whisper_model", "base")
                model = whisper.load_model(model_name)
                result = model.transcribe(file_path, language=self.options.get("language", None))
                text = result.get("text", "")
                segments = result.get("segments", [])
                if segments:
                    metadata["segments"] = [
                        {"start": s["start"], "end": s["end"], "text": s["text"]}
                        for s in segments
                    ]
                    metadata["duration"] = segments[-1]["end"] if segments else 0
                metadata["whisper_model"] = model_name
                metadata["language"] = result.get("language", "unknown")
            else:
                pipe = hf_pipeline(
                    "automatic-speech-recognition",
                    model=self.options.get("hf_model", "openai/whisper-base"),
                )
                result = pipe(file_path, return_timestamps=True)
                text = result.get("text", "")
                chunks = result.get("chunks", [])
                if chunks:
                    metadata["segments"] = [
                        {"start": c["timestamp"][0], "end": c["timestamp"][1], "text": c["text"]}
                        for c in chunks
                    ]
                    if chunks:
                        metadata["duration"] = chunks[-1]["timestamp"][1]
                metadata["hf_model"] = True

        except Exception as exc:
            raise CorruptFileError(file_path, detail=str(exc))

        self._extract_file_metadata(file_path, metadata)
        text = self.clean_text(text)

        return Document(
            text=text, tables=[], images=images, metadata=metadata,
            source_type="audio", source_path=file_path
        )

    def _extract_file_metadata(self, file_path: str, metadata: Dict[str, Any]):
        """Extract file-level metadata like size and format."""
        try:
            stat = os.stat(file_path)
            metadata["file_size"] = stat.st_size
        except OSError:
            pass

        ext = Path(file_path).suffix.lower()
        metadata["format"] = ext.lstrip(".")

        try:
            import mutagen
            audio = mutagen.File(file_path)
            if audio:
                info = audio.info
                metadata["sample_rate"] = getattr(info, "sample_rate", None)
                metadata["channels"] = getattr(info, "channels", None)
                metadata["bitrate"] = getattr(info, "bitrate", None)
                if hasattr(info, "length"):
                    metadata["duration"] = info.length
        except ImportError:
            pass

    def supported_extensions(self) -> list[str]:
        return [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".wma", ".aac", ".opus"]
