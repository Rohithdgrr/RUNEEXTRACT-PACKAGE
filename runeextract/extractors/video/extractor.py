"""
Video extractor using OpenCV for frames and Whisper for audio transcription.

Extracts key frames as images and transcribes audio track to text.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document, Image
from runeextract.exceptions import DependencyMissingError, CorruptFileError

logger = logging.getLogger(__name__)


class VideoExtractor(BaseExtractor):
    """Extractor for video files.

    Extracts key frames as images and transcribes audio to text.
    Requires opencv-python-headless for frames and openai-whisper for audio.
    """

    def extract(self, file_path: str) -> Document:
        self.validate_file(file_path)

        text = ""
        images: List[Image] = []
        metadata: Dict[str, Any] = {}
        image_data_list: List[Image] = []

        try:
            import cv2
        except ImportError:
            raise DependencyMissingError(file_path, "opencv-python-headless")

        try:
            import whisper
        except ImportError:
            whisper = None

        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            hf_pipeline = None

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise CorruptFileError(file_path, detail="Cannot open video file")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            metadata["fps"] = fps
            metadata["total_frames"] = total_frames
            metadata["width"] = width
            metadata["height"] = height
            metadata["duration"] = duration

            frame_interval = max(1, int(fps * self.options.get("frame_interval_seconds", 5)))
            max_frames = self.options.get("max_frames", 20)
            frame_count = 0
            extracted = 0

            while extracted < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    ret_buf, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if ret_buf:
                        ts = frame_count / fps if fps > 0 else 0
                        image_data_list.append(Image(
                            data=buf.tobytes(),
                            format="jpg",
                            width=width,
                            height=height,
                            metadata={"timestamp": ts, "frame_index": frame_count},
                        ))
                        extracted += 1
                frame_count += 1

        finally:
            cap.release()

        images = image_data_list

        ext = Path(file_path).suffix.lower()
        temp_audio = None
        try:
            import subprocess
            ffmpeg_paths = ["ffmpeg", "ffmpeg.exe"]
            ffmpeg_available = any(
                os.path.isfile(p) or os.path.sep in p or not os.path.isfile(p)
                for p in ffmpeg_paths
            )
            has_ffmpeg = False
            for cmd in ffmpeg_paths:
                try:
                    import subprocess
                    subprocess.run([cmd, "-version"], capture_output=True, check=True)
                    has_ffmpeg = True
                    break
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue

            if has_ffmpeg:
                temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_audio.close()
                subprocess.run(
                    ["ffmpeg", "-i", "./" + file_path, "-vn", "-acodec", "pcm_s16le",
                     "-ar", "16000", "-ac", "1", "--", temp_audio.name],
                    capture_output=True, check=True,
                )

                if whisper is not None:
                    model_name = self.options.get("whisper_model", "base")
                    model = whisper.load_model(model_name)
                    result = model.transcribe(temp_audio.name, language=self.options.get("language", None))
                    text = result.get("text", "")
                    segments = result.get("segments", [])
                    if segments:
                        metadata["segments"] = [
                            {"start": s["start"], "end": s["end"], "text": s["text"]}
                            for s in segments
                        ]
                    metadata["whisper_model"] = model_name
                    metadata["audio_language"] = result.get("language", "unknown")
                elif hf_pipeline is not None:
                    pipe = hf_pipeline(
                        "automatic-speech-recognition",
                        model=self.options.get("hf_model", "openai/whisper-base"),
                    )
                    result = pipe(temp_audio.name, return_timestamps=True)
                    text = result.get("text", "")
                    chunks = result.get("chunks", [])
                    if chunks:
                        metadata["segments"] = [
                            {"start": c["timestamp"][0], "end": c["timestamp"][1], "text": c["text"]}
                            for c in chunks
                        ]
                    metadata["hf_model"] = True
            else:
                logger.warning("ffmpeg not found; cannot extract audio track for transcription")
        except Exception as exc:
            logger.warning(f"Audio extraction failed: {exc}")
        finally:
            if temp_audio and os.path.exists(temp_audio.name):
                try:
                    os.unlink(temp_audio.name)
                except OSError as e:
                    logger.debug("Failed to remove temp audio file: %s", e)

        try:
            stat = os.stat(file_path)
            metadata["file_size"] = stat.st_size
        except OSError as e:
            logger.debug("Failed to stat video file: %s", e)
        metadata["format"] = ext.lstrip(".")

        text = self.clean_text(text)

        return Document(
            text=text, tables=[], images=images, metadata=metadata,
            source_type="video", source_path=file_path
        )

    def supported_extensions(self) -> list[str]:
        return [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"]
