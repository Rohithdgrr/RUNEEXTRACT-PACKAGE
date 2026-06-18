"""Table of Contents extraction from documents."""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class TOCEntry:
    title: str
    level: int
    page_number: Optional[int] = None
    children: List["TOCEntry"] = field(default_factory=list)


class TOCParser:
    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    MARKDOWN_TOC_RE = re.compile(r"^(\s*[-*+]\s+)?\[(.+?)\]\(#.+?\)", re.MULTILINE)
    INDENT_RE = re.compile(r"^(\s*)(.+)$")

    def parse_markdown(self, text: str) -> List[TOCEntry]:
        entries = []
        for match in self.HEADING_RE.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            entries.append(TOCEntry(title=title, level=level))
        return self._build_tree(entries)

    def parse_html(self, html: str) -> List[TOCEntry]:
        import re as _re
        entries = []
        for m in _re.finditer(r"<h([1-6])[^>]*>(.+?)</h\1>", html, _re.IGNORECASE | _re.DOTALL):
            level = int(m.group(1))
            title = _re.sub(r"<[^>]+>", "", m.group(2)).strip()
            entries.append(TOCEntry(title=title, level=level))
        return self._build_tree(entries)

    def parse_pdf_toc(self, text: str) -> List[TOCEntry]:
        lines = text.splitlines()
        entries = []
        for line in lines:
            if not line.strip():
                continue
            m = self.INDENT_RE.match(line)
            if not m:
                continue
            indent = len(m.group(1))
            content = m.group(2).strip()
            page = None
            page_m = re.search(r"\.{3,}\s*(\d+)$", content)
            if page_m:
                content = content[:page_m.start()].strip()
                try:
                    page = int(page_m.group(1))
                except ValueError:
                    pass
            level = max(1, indent // 2 + 1) if indent else 1
            entries.append(TOCEntry(title=content, level=level, page_number=page))
        return self._build_tree(entries)

    def _build_tree(self, entries: List[TOCEntry]) -> List[TOCEntry]:
        if not entries:
            return []
        root = TOCEntry(title="[root]", level=0, children=[])
        stack = [root]
        for entry in entries:
            node = TOCEntry(title=entry.title, level=entry.level, page_number=entry.page_number)
            while stack and stack[-1].level >= entry.level:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            stack.append(node)
        return root.children


def extract_toc(text: str, format: str = "markdown") -> List[TOCEntry]:
    parser = TOCParser()
    if format == "html":
        return parser.parse_html(text)
    return parser.parse_markdown(text)


def toc_to_markdown(entries: List[TOCEntry], level: int = 1) -> str:
    lines = []
    for entry in entries:
        prefix = "  " * (level - 1)
        page = f" … {entry.page_number}" if entry.page_number is not None else ""
        lines.append(f"{prefix}- {entry.title}{page}")
        if entry.children:
            lines.append(toc_to_markdown(entry.children, level + 1))
    return "\n".join(lines)


def toc_to_json(entries: List[TOCEntry]) -> list:
    def _serialize(e: TOCEntry) -> dict:
        d = {"title": e.title, "level": e.level}
        if e.page_number is not None:
            d["page_number"] = e.page_number
        if e.children:
            d["children"] = [_serialize(c) for c in e.children]
        return d
    return [_serialize(e) for e in entries]
