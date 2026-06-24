"""SmartCrawler — configurable web crawler with sitemap/RSS support."""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse
from collections import deque

import requests
from bs4 import BeautifulSoup

from runeextract.web.sitemap import discover_sitemap, parse_sitemap
from runeextract.web.feed import discover_feed, parse_feed
from runeextract.core.router import URLValidator
from runeextract.exceptions import ResponseSizeError
from runeextract.utils.logging import log_security_event
from runeextract.utils.maturity import beta

logger = logging.getLogger(__name__)

_MAX_RESPONSE_SIZE = 50 * 1024 * 1024  # 50 MB max response body for crawler


@dataclass
class CrawlResult:
    """Result from a single crawled page."""
    url: str
    html: str
    status_code: int
    links: List[str] = field(default_factory=list)
    depth: int = 0
    metadata: Dict = field(default_factory=dict)


@beta(name="web.crawler")
class SmartCrawler:
    """Configurable web crawler with sitemap/RSS ingestion, politeness, and robots.txt respect.

    Usage::

        from runeextract.web import SmartCrawler

        crawler = SmartCrawler(
            start_url="https://example.com",
            max_pages=50,
            same_domain=True,
            use_sitemap=True,
        )
        results = crawler.crawl()
        for result in results:
            print(result.url, result.status_code)
    """

    def __init__(
        self,
        start_url: str,
        max_pages: int = 50,
        same_domain: bool = True,
        respect_robots: bool = True,
        delay: float = 0.5,
        timeout: float = 15.0,
        use_sitemap: bool = False,
        use_feed: bool = False,
        max_depth: int = 3,
        max_response_size: int = _MAX_RESPONSE_SIZE,
        on_page: Optional[Callable[["CrawlResult"], None]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.start_url = start_url
        self.max_pages = max_pages
        self.same_domain = same_domain
        self.respect_robots = respect_robots
        self.delay = delay
        self.timeout = timeout
        self.use_sitemap = use_sitemap
        self.use_feed = use_feed
        self.max_depth = max_depth
        self.max_response_size = max_response_size
        self.on_page = on_page
        self.headers = headers or {
            "User-Agent": "RuneExtractCrawler/1.0",
        }

        self._visited: Set[str] = set()
        self._domain = urlparse(start_url).netloc
        self._rp = None
        self._last_request_time = 0.0

    def crawl(self) -> List[CrawlResult]:
        """Run the crawl starting from ``start_url``.

        If ``use_sitemap`` is True, URLs from the sitemap are added to the queue first.
        If ``use_feed`` is True, entries from the RSS/Atom feed are also added.

        Returns:
            A list of ``CrawlResult`` objects, one per successfully crawled page.
        """
        self._init_robots()

        seed_urls: deque = deque()

        # Seed from sitemap
        if self.use_sitemap:
            sitemap_url = discover_sitemap(self.start_url)
            if sitemap_url:
                sitemap_urls = parse_sitemap(sitemap_url, timeout=self.timeout)
                for u in sitemap_urls:
                    if u not in self._visited:
                        seed_urls.append((u, 0))

        # Seed from feed
        if self.use_feed:
            feed_url = discover_feed(self.start_url)
            if feed_url:
                entries = parse_feed(feed_url)
                for entry in entries:
                    link = entry.get("link")
                    if link and link not in self._visited:
                        seed_urls.append((link, 0))

        # Always include the start URL
        if self.start_url not in self._visited:
            seed_urls.appendleft((self.start_url, 0))

        return self._crawl_from_queue(seed_urls)

    def _crawl_from_queue(self, queue: deque) -> List[CrawlResult]:
        results: List[CrawlResult] = []
        while queue and len(results) < self.max_pages:
            url, depth = queue.popleft()
            if url in self._visited:
                continue
            if not self._is_allowed(url):
                continue
            if depth > self.max_depth:
                continue

            self._respect_delay()
            result = self._fetch(url, depth)
            if result is None:
                continue

            self._visited.add(url)
            results.append(result)

            if self.on_page:
                self.on_page(result)

            # Enqueue discovered links
            for link in result.links:
                if link not in self._visited:
                    queue.append((link, depth + 1))

        return results

    def _fetch(self, url: str, depth: int) -> Optional[CrawlResult]:
        """Fetch a single page and extract links."""
        # Validate URL against SSRF before fetching
        try:
            URLValidator.validate(url)
        except Exception as exc:
            log_security_event("ssrf_blocked", level="WARNING", url=url,
                               reason=f"URL validation failed: {exc}", error_code="E102")
            logger.debug("Blocked SSRF URL %s: %s", url, exc)
            return None

        try:
            resp = requests.get(url, timeout=self.timeout, headers=self.headers, allow_redirects=False)
            # Validate redirect targets
            if resp.status_code in (301, 302, 303, 307, 308):
                redirect_url = resp.headers.get("Location")
                if redirect_url:
                    URLValidator.validate(redirect_url)
                    resp = requests.get(redirect_url, timeout=self.timeout, headers=self.headers, allow_redirects=False)
                    resp.raise_for_status()
                else:
                    return None
            resp.raise_for_status()
            # Enforce response size limit
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > self.max_response_size:
                raise ResponseSizeError(url, int(content_length), self.max_response_size)
            if len(resp.content) > self.max_response_size:
                raise ResponseSizeError(url, len(resp.content), self.max_response_size)
            html = resp.text
        except requests.RequestException as exc:
            logger.debug("Failed to fetch %s: %s", url, exc)
            return None

        links = self._extract_links(url, html)
        return CrawlResult(
            url=url,
            html=html,
            status_code=resp.status_code,
            links=links,
            depth=depth,
            metadata={"content_type": resp.headers.get("Content-Type", "")},
        )

    def _extract_links(self, base_url: str, html: str) -> List[str]:
        """Extract and normalise internal links from HTML."""
        links: List[str] = []
        try:
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("#") or href.startswith("javascript:"):
                    continue
                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)
                if parsed.scheme not in ("http", "https"):
                    continue
                if self.same_domain and parsed.netloc != self._domain:
                    continue
                # Normalise: remove fragment
                clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    clean += f"?{parsed.query}"
                links.append(clean)
        except Exception as exc:
            logger.warning("BS4 link extraction failed: %s", exc)
        return links

    def _init_robots(self) -> None:
        if not self.respect_robots:
            return
        try:
            import urllib.robotparser
            parsed = urlparse(self.start_url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            self._rp = urllib.robotparser.RobotFileParser()
            self._rp.set_url(urljoin(base, "/robots.txt"))
            self._rp.read()
        except Exception as exc:
            logger.warning("Robots.txt init failed: %s", exc)
            self._rp = None

    def _is_allowed(self, url: str) -> bool:
        if self.respect_robots and self._rp is not None:
            return self._rp.can_fetch("*", url)
        return True

    def _respect_delay(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_time = time.time()


def smart_crawl(
    start_url: str,
    max_pages: int = 50,
    same_domain: bool = True,
    use_sitemap: bool = False,
    use_feed: bool = False,
    delay: float = 0.5,
    **kwargs,
) -> List[CrawlResult]:
    """Convenience function for one-off crawling.

    Usage::

        from runeextract.web import smart_crawl

        results = smart_crawl("https://example.com", max_pages=10, use_sitemap=True)
        for r in results:
            print(r.url)
    """
    crawler = SmartCrawler(
        start_url=start_url,
        max_pages=max_pages,
        same_domain=same_domain,
        use_sitemap=use_sitemap,
        use_feed=use_feed,
        delay=delay,
        **kwargs,
    )
    return crawler.crawl()
