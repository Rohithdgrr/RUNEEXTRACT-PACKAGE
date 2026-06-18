"""Sitemap discovery and parsing."""

import gzip
import logging
import time
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import requests

logger = logging.getLogger(__name__)

_NAMESPACES = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
}


def discover_sitemap(
    base_url: str,
    timeout: float = 15.0,
) -> Optional[str]:
    """Discover a sitemap URL for a given website.

    Checks common sitemap locations in order:
        1. ``/sitemap.xml``
        2. ``/sitemap_index.xml``
        3. ``/sitemap/sitemap.xml``
        4. ``/robots.txt`` (Sitemap: directive)

    Returns the first discovered sitemap URL, or None.
    """
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    candidates = [
        "/sitemap.xml",
        "/sitemap_index.xml",
        "/sitemap/sitemap.xml",
    ]

    for path in candidates:
        url = base + path
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200 and "xml" in resp.headers.get("Content-Type", ""):
                logger.info("Discovered sitemap at %s", url)
                return url
        except requests.RequestException:
            continue

    # Check robots.txt for Sitemap directive
    try:
        resp = requests.get(base + "/robots.txt", timeout=timeout)
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sitemap_url = line.split(":", 1)[1].strip()
                    if sitemap_url:
                        return sitemap_url
    except requests.RequestException:
        pass

    return None


def parse_sitemap(
    sitemap_url: str,
    timeout: float = 15.0,
    max_urls: int = 10000,
) -> List[str]:
    """Parse a sitemap (or sitemap index) and return all discovered URLs.

    Handles:
        - Standard ``<urlset>`` sitemaps
        - ``<sitemapindex>`` (nested indexes)
        - ``.xml.gz`` compressed sitemaps

    Args:
        sitemap_url: URL of the sitemap file.
        timeout: HTTP request timeout.
        max_urls: Maximum URLs to return (prevents unbounded parsing).

    Returns:
        A list of URLs found in the sitemap (or sub-sitemaps).
    """
    urls: List[str] = []
    seen: Set[str] = set()

    def _fetch_and_parse(url: str) -> None:
        if len(urls) >= max_urls:
            return
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            content = resp.content

            if url.endswith(".gz"):
                content = gzip.decompress(content)

            root = ElementTree.fromstring(content)
            _parse_node(root, url)

        except Exception as exc:
            logger.warning("Failed to parse sitemap %s: %s", url, exc)

    def _parse_node(root: ElementTree.Element, base_url: str) -> None:
        tag = root.tag.replace(f"{{{_NAMESPACES['sm']}}}", "")
        if tag == "sitemapindex":
            for sitemap_elem in root.findall("sm:sitemap/sm:loc", _NAMESPACES):
                if sitemap_elem.text and sitemap_elem.text not in seen:
                    seen.add(sitemap_elem.text)
                    _fetch_and_parse(sitemap_elem.text)
        elif tag == "urlset":
            for url_elem in root.findall("sm:url/sm:loc", _NAMESPACES):
                if url_elem.text and len(urls) < max_urls:
                    urls.append(url_elem.text)

    _fetch_and_parse(sitemap_url)
    return urls
