"""
Notion extractor using Notion REST API.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from runeextract.core.extractor import BaseExtractor
from runeextract.models.document import Document as RuneDocument, Table
from runeextract.exceptions import ExtractionError, DependencyMissingError

logger = logging.getLogger(__name__)

NOTION_API = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


def _get_api_key() -> str:
    key = os.environ.get("NOTION_API_KEY")
    if not key:
        raise ExtractionError(
            "NOTION_API_KEY environment variable not set",
            error_code="E020"
        )
    return key


def _get_rich_text(block: dict, container: str, field: str) -> str:
    parts = block.get(container, {}).get(field, [])
    return "".join(
        rt.get("plain_text", "") for rt in parts
    )


def _format_todo(block: dict) -> str:
    checked = block.get("to_do", {}).get("checked", False)
    text = _get_rich_text(block, "to_do", "rich_text")
    prefix = "[x]" if checked else "[ ]"
    return f"{prefix} {text}\n"


def _format_code(block: dict) -> str:
    caption = block.get("code", {})
    lang = caption.get("language", "")
    text = _get_rich_text(block, "code", "rich_text")
    return f"```{lang}\n{text}\n```\n\n"


def _format_image(block: dict) -> str:
    image = block.get("image", {})
    caption = ""
    if image.get("caption"):
        caption = _get_rich_text(block, "image", "caption")
    url = image.get("external", {}).get("url") or image.get("file", {}).get("url") or ""
    if url:
        return f"![{caption}]({url})\n\n"
    return f"*image: {caption}*\n\n" if caption else "*[image]*\n\n"


def _format_table(block: dict) -> Table:
    table_block = block.get("table", {})
    table_width = table_block.get("table_width", 0)
    rows = block.get("table", {}).get("children", [])
    table_rows = []
    for row in rows:
        if row.get("type") == "table_row":
            cells = row.get("table_row", {}).get("cells", [])
            row_data = [
                "".join(c.get("plain_text", "") for c in cell)
                for cell in cells
            ]
            table_rows.append(row_data)
    columns = table_rows[0] if table_rows else []
    data = table_rows[1:] if len(table_rows) > 1 else []
    return Table(rows=data, columns=columns)


_BLOCK_MAPPING = {
    "paragraph": lambda b: _get_rich_text(b, "paragraph", "rich_text") + "\n\n",
    "heading_1": lambda b: "# " + _get_rich_text(b, "heading_1", "rich_text") + "\n\n",
    "heading_2": lambda b: "## " + _get_rich_text(b, "heading_2", "rich_text") + "\n\n",
    "heading_3": lambda b: "### " + _get_rich_text(b, "heading_3", "rich_text") + "\n\n",
    "bulleted_list_item": lambda b: "- " + _get_rich_text(b, "bulleted_list_item", "rich_text") + "\n",
    "numbered_list_item": lambda b: "1. " + _get_rich_text(b, "numbered_list_item", "rich_text") + "\n",
    "to_do": _format_todo,
    "code": _format_code,
    "quote": lambda b: "> " + _get_rich_text(b, "quote", "rich_text") + "\n\n",
    "callout": lambda b: _get_rich_text(b, "callout", "rich_text") + "\n\n",
    "divider": lambda b: "---\n\n",
    "toggle": lambda b: _get_rich_text(b, "toggle", "rich_text") + "\n\n",
    "image": _format_image,
    "table": _format_table,
    "child_page": lambda b: "",
    "unsupported": lambda b: "",
}


class NotionExtractor(BaseExtractor):
    """Extractor for Notion pages and databases."""

    def extract(self, identifier: str) -> RuneDocument:
        api_key = _get_api_key()
        page_id = self._resolve_id(identifier)

        text = ""
        tables: List[Table] = []
        metadata: Dict[str, Any] = {}

        try:
            import requests
        except ImportError:
            raise DependencyMissingError(identifier, "requests")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

        # Fetch page metadata
        page_url = f"{NOTION_API}/pages/{page_id}"
        try:
            resp = requests.get(page_url, headers=headers, timeout=30)
            resp.raise_for_status()
            page_data = resp.json()
            if page_data.get("properties"):
                for prop_name, prop in page_data["properties"].items():
                    ptype = prop.get("type", "")
                    if ptype == "title":
                        title_parts = prop.get("title", [])
                        metadata["title"] = "".join(
                            t.get("plain_text", "") for t in title_parts
                        )
                        text += f"# {metadata['title']}\n\n"
                    elif ptype == "rich_text":
                        rt = prop.get("rich_text", [])
                        metadata[prop_name] = "".join(
                            t.get("plain_text", "") for t in rt
                        )
                    else:
                        metadata[prop_name] = str(prop.get(ptype, ""))
        except Exception as exc:
            logger.warning(f"Failed to fetch page metadata: {exc}")

        # Fetch page blocks recursively
        blocks_url = f"{NOTION_API}/blocks/{page_id}/children"
        all_blocks = []
        cursor = None
        seen_cursors = set()
        max_pages = 100
        page_count = 0
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            try:
                resp = requests.get(blocks_url, headers=headers, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                all_blocks.extend(data.get("results", []))
                cursor = data.get("next_cursor")
                page_count += 1
                if not data.get("has_more") or not cursor:
                    break
                if cursor in seen_cursors:
                    logger.warning(f"Cursor cycle detected in Notion blocks fetch for {page_id}")
                    break
                seen_cursors.add(cursor)
                if page_count >= max_pages:
                    logger.warning(f"Reached max {max_pages} pages for Notion blocks fetch")
                    break
            except Exception as exc:
                logger.warning(f"Failed to fetch blocks: {exc}")
                break

        for block in all_blocks:
            block_type = block.get("type", "unsupported")
            handler = _BLOCK_MAPPING.get(block_type, lambda b: "")
            result = handler(block)
            if isinstance(result, Table):
                tables.append(result)
            else:
                text += result

        text = self.clean_text(text)

        return RuneDocument(
            text=text, tables=tables, images=[], metadata=metadata,
            source_type="notion", source_path=identifier
        )

    def query_database(self, database_id: str) -> RuneDocument:
        """Extract all pages from a Notion database."""
        api_key = _get_api_key()
        text = ""
        tables: List[Table] = []
        metadata: Dict[str, Any] = {"type": "database"}

        import requests
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

        url = f"{NOTION_API}/databases/{database_id}/query"
        try:
            resp = requests.post(url, headers=headers, json={}, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            metadata["page_count"] = len(results)

            for i, page in enumerate(results):
                page_id = page.get("id", "")
                text += f"\n## Page {i + 1} (id: {page_id})\n\n"
                props = page.get("properties", {})
                for prop_name, prop in props.items():
                    ptype = prop.get("type", "")
                    if ptype == "title":
                        title_text = "".join(
                            t.get("plain_text", "") for t in prop.get("title", [])
                        )
                        text += f"### {title_text}\n\n"
                    elif ptype == "rich_text":
                        rt_text = "".join(
                            t.get("plain_text", "") for t in prop.get("rich_text", [])
                        )
                        if rt_text:
                            text += f"{prop_name}: {rt_text}\n"
                    else:
                        val = prop.get(ptype)
                        if val is not None:
                            text += f"{prop_name}: {val}\n"
        except Exception as exc:
            logger.warning(f"Database query failed: {exc}")

        return RuneDocument(
            text=text, tables=tables, images=[], metadata=metadata,
            source_type="notion", source_path=f"database:{database_id}"
        )

    @staticmethod
    def _resolve_id(identifier: str) -> str:
        """Convert Notion URL or ID to canonical ID (dashes removed)."""
        import re
        if "/" in identifier:
            parts = identifier.rstrip("/").split("/")
            for part in reversed(parts):
                if len(part) >= 32 and "-" in part:
                    return part.replace("-", "")
                if len(part.replace("-", "")) == 32:
                    return part.replace("-", "")
            raise ExtractionError(
                f"Cannot parse Notion page ID from: {identifier}",
                file_path=identifier, error_code="E021"
            )
        return identifier.replace("-", "")

    async def extract_async(self, identifier: str) -> RuneDocument:
        """Async extraction using aiohttp."""
        api_key = _get_api_key()
        page_id = self._resolve_id(identifier)
        text = ""
        tables: List[Table] = []
        metadata: Dict[str, Any] = {}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

        try:
            import aiohttp
        except ImportError:
            raise DependencyMissingError(identifier, "aiohttp")

        async with aiohttp.ClientSession(headers=headers) as session:
            # Fetch page metadata
            page_url = f"{NOTION_API}/pages/{page_id}"
            try:
                async with session.get(page_url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        page_data = await resp.json()
                        if page_data.get("properties"):
                            for prop_name, prop in page_data["properties"].items():
                                ptype = prop.get("type", "")
                                if ptype == "title":
                                    title_parts = prop.get("title", [])
                                    metadata["title"] = "".join(
                                        t.get("plain_text", "") for t in title_parts
                                    )
                                    text += f"# {metadata['title']}\n\n"
                                elif ptype == "rich_text":
                                    rt = prop.get("rich_text", [])
                                    metadata[prop_name] = "".join(
                                        t.get("plain_text", "") for t in rt
                                    )
                                else:
                                    metadata[prop_name] = str(prop.get(ptype, ""))
            except Exception as exc:
                logger.warning(f"Async page fetch failed: {exc}")

            # Fetch blocks
            blocks_url = f"{NOTION_API}/blocks/{page_id}/children"
            all_blocks = []
            cursor = None
            seen_cursors = set()
            max_pages = 100
            page_count = 0
            while True:
                params = {"page_size": 100}
                if cursor:
                    params["start_cursor"] = cursor
                try:
                    async with session.get(
                        blocks_url, params=params, timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            all_blocks.extend(data.get("results", []))
                            cursor = data.get("next_cursor")
                            page_count += 1
                            if not data.get("has_more") or not cursor:
                                break
                            if cursor in seen_cursors:
                                logger.warning(f"Cursor cycle detected in async Notion blocks fetch for {page_id}")
                                break
                            seen_cursors.add(cursor)
                            if page_count >= max_pages:
                                logger.warning(f"Reached max {max_pages} pages for async Notion blocks fetch")
                                break
                        else:
                            logger.warning(f"Async blocks fetch failed with status {resp.status}")
                            break
                except Exception as exc:
                    logger.warning(f"Async blocks fetch error: {exc}")
                    break

        for block in all_blocks:
            block_type = block.get("type", "unsupported")
            handler = _BLOCK_MAPPING.get(block_type, lambda b: "")
            result = handler(block)
            if isinstance(result, Table):
                tables.append(result)
            else:
                text += result

        text = self.clean_text(text)
        return RuneDocument(
            text=text, tables=tables, images=[], metadata=metadata,
            source_type="notion", source_path=identifier
        )

    def supported_extensions(self) -> list[str]:
        return []

    def is_url(self) -> bool:
        return True
