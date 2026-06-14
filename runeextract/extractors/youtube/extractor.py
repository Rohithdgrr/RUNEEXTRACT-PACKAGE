"""
YouTube extractor using youtube-transcript-api and yt-dlp.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document as RuneDocument, Table
from runeextract.exceptions import DependencyMissingError, ExtractionError

logger = logging.getLogger(__name__)

_YTDL_OPTS = {"quiet": True, "no_warnings": True}

_YOUTUBE_RE = re.compile(
    r'(?:https?://)?(?:www\.)?'
    r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)'
    r'([\w-]{11})'
)


def extract_video_id(url: str) -> Optional[str]:
    match = _YOUTUBE_RE.search(url)
    return match.group(1) if match else None


class YoutubeExtractor(BaseExtractor):
    """Extractor for YouTube videos (transcript + metadata)."""

    def extract(self, url: str) -> RuneDocument:
        video_id = extract_video_id(url)
        if not video_id:
            raise ExtractionError(
                f"Not a valid YouTube URL: {url}",
                file_path=url, error_code="E010"
            )

        text = ""
        tables: List[Table] = []
        metadata: Dict[str, Any] = {}

        metadata["video_id"] = video_id
        metadata["url"] = url

        # 1. Get transcript
        transcript_data = self._get_transcript(video_id)
        if transcript_data:
            metadata["transcript_segments"] = len(transcript_data)
            metadata["duration_seconds"] = (
                transcript_data[-1]["start"] + transcript_data[-1]["duration"]
                if transcript_data else 0
            )
            lines = []
            for seg in transcript_data:
                ts = self._format_timestamp(seg["start"])
                lines.append(f"[{ts}] {seg['text']}")
            text = "\n".join(lines)

        # 2. Get metadata via yt-dlp
        meta = self._get_metadata(video_id)
        metadata.update(meta)

        return RuneDocument(
            text=text, tables=tables, images=[], metadata=metadata,
            source_type="youtube", source_path=url
        )

    def _get_transcript(self, video_id: str) -> Optional[List[Dict]]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise DependencyMissingError(
                f"youtube:{video_id}", "youtube-transcript-api"
            )

        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return transcript
        except Exception as exc:
            logger.debug(f"No transcript for {video_id}: {exc}")
            return None

    def _get_metadata(self, video_id: str) -> Dict[str, Any]:
        meta = {}
        try:
            import yt_dlp
        except ImportError:
            logger.debug("yt-dlp not installed, skipping metadata")
            return meta

        try:
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                info = ydl.extract_info(video_id, download=False)
            if info:
                meta["title"] = info.get("title", "")
                meta["author"] = info.get("uploader", "")
                meta["channel"] = info.get("channel", "")
                meta["duration"] = info.get("duration", 0)
                meta["view_count"] = info.get("view_count", 0)
                meta["like_count"] = info.get("like_count", 0)
                meta["description"] = (info.get("description") or "")[:2000]
                meta["upload_date"] = info.get("upload_date", "")
                meta["tags"] = info.get("tags", [])
                meta["categories"] = info.get("categories", [])
                chapters = info.get("chapters") or []
                if chapters:
                    meta["chapters"] = [
                        {"title": ch["title"], "start": ch["start_time"]}
                        for ch in chapters
                    ]
                    meta["chapter_count"] = len(chapters)
        except Exception as exc:
            logger.debug(f"yt-dlp metadata failed for {video_id}: {exc}")

        return meta

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    async def extract_async(self, url: str) -> RuneDocument:
        """Async extraction using aiohttp for transcript and yt-dlp in executor."""
        video_id = extract_video_id(url)
        if not video_id:
            raise ExtractionError(
                f"Not a valid YouTube URL: {url}",
                file_path=url, error_code="E010"
            )

        import asyncio
        text = ""
        tables: List[Table] = []
        metadata: Dict[str, Any] = {"video_id": video_id, "url": url}

        # Get transcript in executor (blocking)
        loop = asyncio.get_running_loop()
        transcript_data = await loop.run_in_executor(
            None, self._get_transcript, video_id
        )
        if transcript_data:
            metadata["transcript_segments"] = len(transcript_data)
            metadata["duration_seconds"] = (
                transcript_data[-1]["start"] + transcript_data[-1]["duration"]
                if transcript_data else 0
            )
            lines = []
            for seg in transcript_data:
                ts = self._format_timestamp(seg["start"])
                lines.append(f"[{ts}] {seg['text']}")
            text = "\n".join(lines)

        # Get metadata in executor
        meta = await loop.run_in_executor(None, self._get_metadata, video_id)
        metadata.update(meta)

        return RuneDocument(
            text=text, tables=tables, images=[], metadata=metadata,
            source_type="youtube", source_path=url
        )

    def supported_extensions(self) -> list[str]:
        return []  # Uses URL-based routing, not file extension

    def is_url(self) -> bool:
        return True
