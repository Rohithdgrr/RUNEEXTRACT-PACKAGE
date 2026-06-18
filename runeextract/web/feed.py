"""RSS / Atom feed discovery and parsing."""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def discover_feed(base_url: str, timeout: float = 15.0) -> Optional[str]:
    """Discover an RSS or Atom feed URL from a webpage.

    Checks:
        1. ``<link>`` tags with ``type="application/rss+xml"`` or ``type="application/atom+xml"``
        2. Common feed URLs (``/feed``, ``/rss``, ``/atom.xml``)

    Returns the first discovered feed URL, or None.
    """
    try:
        resp = requests.get(base_url, timeout=timeout)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "lxml")
        for link in soup.find_all("link", type=True):
            t = link.get("type", "")
            if t in ("application/rss+xml", "application/atom+xml"):
                href = link.get("href")
                if href:
                    return urljoin(base_url, href)
    except requests.RequestException:
        pass

    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    candidates = ["/feed", "/rss", "/atom.xml", "/index.xml", "/feed.xml"]
    for path in candidates:
        url = base + path
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200 and "xml" in resp.headers.get("Content-Type", ""):
                return url
        except requests.RequestException:
            continue

    return None


def parse_feed(
    feed_url: str,
    timeout: float = 15.0,
    max_entries: int = 100,
) -> List[Dict]:
    """Parse an RSS or Atom feed and return entries as dicts.

    Each entry contains:
        - ``title``
        - ``link``
        - ``summary`` (or ``content`` for Atom)
        - ``published`` (parsed datetime or raw string)

    Args:
        feed_url: URL of the RSS/Atom feed.
        timeout: HTTP request timeout.
        max_entries: Maximum entries to return.

    Returns:
        A list of entry dicts.
    """
    try:
        resp = requests.get(feed_url, timeout=timeout)
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
    except Exception as exc:
        logger.warning("Failed to parse feed %s: %s", feed_url, exc)
        return []

    entries: List[Dict] = []

    # Determine namespace — RSS uses no default namespace, Atom uses "http://www.w3.org/2005/Atom"
    ns = ""
    tag = root.tag
    if tag.startswith("{"):
        inner = tag[1:tag.index("}")]
        ns = "{" + inner + "}"
    local = tag[tag.index("}") + 1:] if "}" in tag else tag

    if local == "rss":
        for item in root.iter(f"{ns}item"):
            if len(entries) >= max_entries:
                break
            entry = _parse_rss_item(item, ns)
            if entry:
                entries.append(entry)
    elif local == "feed":
        for item in root.iter(f"{ns}entry"):
            if len(entries) >= max_entries:
                break
            entry = _parse_atom_entry(item, ns)
            if entry:
                entries.append(entry)

    return entries


def _parse_rss_item(item: ElementTree.Element, ns: str) -> Optional[Dict]:
    def _text(tag: str) -> str:
        el = item.find(f"{ns}{tag}")
        return el.text.strip() if el is not None and el.text else ""

    published = _text("pubDate")
    dt = None
    if published:
        try:
            dt = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %z")
        except (ValueError, TypeError):
            pass

    return {
        "title": _text("title"),
        "link": _text("link"),
        "summary": _text("description"),
        "published": dt or published,
    }


def _parse_atom_entry(entry: ElementTree.Element, ns: str) -> Optional[Dict]:
    def _text(tag: str) -> str:
        el = entry.find(f"{ns}{tag}")
        return el.text.strip() if el is not None and el.text else ""

    def _link() -> str:
        for link in entry.findall(f"{ns}link"):
            href = link.get("href", "")
            rel = link.get("rel", "alternate")
            if rel == "alternate" and href:
                return href
        return ""

    published = _text("published") or _text("updated")
    dt = None
    if published:
        try:
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

    return {
        "title": _text("title"),
        "link": _link(),
        "summary": _text("summary") or _text("content"),
        "published": dt or published,
    }
