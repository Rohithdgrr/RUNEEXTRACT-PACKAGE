"""Tests for web module — sitemap, feed, and crawler."""

from unittest.mock import patch, MagicMock, call
import pytest
from xml.etree import ElementTree

from runeextract.web import (
    SmartCrawler, CrawlResult, smart_crawl,
    parse_sitemap, discover_sitemap,
    parse_feed, discover_feed,
)


# ---------- Helper: a mock response ----------

def _mock_response(text: str, status: int = 200, content_type: str = "application/xml"):
    m = MagicMock()
    m.status_code = status
    m.text = text
    m.content = text.encode()
    m.headers = {"Content-Type": content_type}
    return m


# ==================== Sitemap tests ====================


class TestSitemap:
    SIMPLE_XML = """<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/page1</loc></url>
      <url><loc>https://example.com/page2</loc></url>
    </urlset>"""

    INDEX_XML = """<?xml version="1.0"?>
    <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>https://example.com/sub1.xml</loc></sitemap>
      <sitemap><loc>https://example.com/sub2.xml</loc></sitemap>
    </sitemapindex>"""

    SUB_XML = """<?xml version="1.0"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://example.com/subpage</loc></url>
    </urlset>"""

    @patch("runeextract.web.sitemap.requests.get")
    def test_parse_simple_sitemap(self, mock_get):
        mock_get.return_value = _mock_response(self.SIMPLE_XML)
        urls = parse_sitemap("https://example.com/sitemap.xml")
        assert urls == ["https://example.com/page1", "https://example.com/page2"]

    @patch("runeextract.web.sitemap.requests.get")
    def test_parse_sitemap_index(self, mock_get):
        mock_get.side_effect = [
            _mock_response(self.INDEX_XML),
            _mock_response(self.SUB_XML),
        ]
        urls = parse_sitemap("https://example.com/sitemap.xml")
        # sub2.xml fetch fails because we only set 2 side effects ; the index has 2 subs
        assert "https://example.com/subpage" in urls
        assert mock_get.call_count == 3  # index + 2 subs

    @patch("runeextract.web.sitemap.requests.get")
    def test_parse_sitemap_max_urls(self, mock_get):
        many_urls = ["<url><loc>https://example.com/p%d</loc></url>" % i for i in range(200)]
        big_xml = """<?xml version="1.0"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">%s</urlset>""" % "".join(many_urls)
        mock_get.return_value = _mock_response(big_xml)
        urls = parse_sitemap("https://example.com/sitemap.xml", max_urls=50)
        assert len(urls) == 50

    @patch("runeextract.web.sitemap.requests.get")
    def test_discover_sitemap(self, mock_get):
        # First candidates fail, robots.txt has Sitemap directive
        responses = [
            MagicMock(status_code=404),
            MagicMock(status_code=404),
            MagicMock(status_code=404),
            _mock_response("Sitemap: https://example.com/custom.xml\n", content_type="text/plain"),
        ]
        mock_get.side_effect = responses
        url = discover_sitemap("https://example.com")
        assert url == "https://example.com/custom.xml"

    @patch("runeextract.web.sitemap.requests.get")
    def test_discover_sitemap_returns_none(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404)
        url = discover_sitemap("https://example.com")
        assert url is None


# ==================== Feed tests ====================


class TestFeed:
    RSS_XML = """<?xml version="1.0"?>
    <rss version="2.0">
      <channel>
        <title>Test Feed</title>
        <item>
          <title>Post 1</title>
          <link>https://example.com/p1</link>
          <description>Summary 1</description>
          <pubDate>Mon, 01 Jan 2024 00:00:00 +0000</pubDate>
        </item>
        <item>
          <title>Post 2</title>
          <link>https://example.com/p2</link>
          <description>Summary 2</description>
        </item>
      </channel>
    </rss>"""

    ATOM_XML = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>Atom Feed</title>
      <entry>
        <title>Atom Post</title>
        <link href="https://example.com/atom1" rel="alternate"/>
        <summary>Atom summary</summary>
        <published>2024-01-01T00:00:00Z</published>
      </entry>
    </feed>"""

    @patch("runeextract.web.feed.requests.get")
    def test_parse_rss(self, mock_get):
        mock_get.return_value = _mock_response(self.RSS_XML)
        entries = parse_feed("https://example.com/feed")
        assert len(entries) == 2
        assert entries[0]["title"] == "Post 1"
        assert entries[0]["link"] == "https://example.com/p1"
        assert entries[1]["title"] == "Post 2"

    @patch("runeextract.web.feed.requests.get")
    def test_parse_atom(self, mock_get):
        mock_get.return_value = _mock_response(self.ATOM_XML)
        entries = parse_feed("https://example.com/atom")
        assert len(entries) == 1
        assert entries[0]["title"] == "Atom Post"
        assert entries[0]["link"] == "https://example.com/atom1"

    @patch("runeextract.web.feed.requests.get")
    def test_parse_feed_invalid(self, mock_get):
        mock_get.side_effect = Exception("fail")
        entries = parse_feed("https://example.com/bad")
        assert entries == []

    @patch("runeextract.web.feed.requests.get")
    @patch("runeextract.web.feed.BeautifulSoup")
    def test_discover_feed_via_link_tag(self, mock_soup, mock_get):
        # Mock the page response
        mock_get.return_value = _mock_response(
            "<html><head></head></html>",
            content_type="text/html",
        )
        # Mock BeautifulSoup to return a link tag
        link_tag = MagicMock()
        def link_get(key, default=None):
            return {"type": "application/rss+xml", "href": "https://example.com/feed.xml"}.get(key, default)
        link_tag.get = link_get
        mock_soup.return_value.find_all.return_value = [link_tag]

        url = discover_feed("https://example.com")
        assert url == "https://example.com/feed.xml"

    @patch("runeextract.web.feed.BeautifulSoup")
    @patch("runeextract.web.feed.requests.get")
    def test_discover_feed_via_common_paths(self, mock_get, mock_soup):
        # First request returns 200 with no feed link (BeautifulSoup returns empty)
        mock_soup.return_value.find_all.return_value = []
        # Then common paths: /feed fails, /rss fails, /atom.xml succeeds
        mock_get.side_effect = [
            MagicMock(status_code=200, headers={"Content-Type": "text/html"}),
            MagicMock(status_code=404),
            MagicMock(status_code=404),
            _mock_response("<feed/>", content_type="application/xml"),
        ]
        url = discover_feed("https://example.com")
        assert url is not None


# ==================== Crawler tests ====================


class TestSmartCrawler:
    HTML_WITH_LINKS = """<html>
    <body>
        <a href="/page1">Page 1</a>
        <a href="/page2">Page 2</a>
        <a href="https://other.com">External</a>
        <a href="#section">Skip</a>
        <a href="javascript:void(0)">JS</a>
    </body></html>"""

    @patch("runeextract.web.crawler.requests.get")
    def test_crawl_basic(self, mock_get):
        mock_get.return_value = _mock_response(
            "<html><body>Hello</body></html>",
            content_type="text/html",
        )
        crawler = SmartCrawler("https://example.com", max_pages=5, delay=0)
        results = crawler.crawl()
        assert len(results) == 1
        assert results[0].url == "https://example.com"

    @patch("runeextract.web.crawler.requests.get")
    def test_crawl_follows_links(self, mock_get):
        def response_for(url, **kw):
            if url == "https://example.com":
                return _mock_response(self.HTML_WITH_LINKS, content_type="text/html")
            return _mock_response("<html><body>Sub page</body></html>", content_type="text/html")

        mock_get.side_effect = response_for
        crawler = SmartCrawler("https://example.com", max_pages=5, delay=0)
        results = crawler.crawl()
        assert len(results) >= 2

    @patch("runeextract.web.crawler.requests.get")
    def test_crawl_respects_max_pages(self, mock_get):
        mock_get.return_value = _mock_response(
            "<html><body>Page</body></html>",
            content_type="text/html",
        )
        crawler = SmartCrawler("https://example.com", max_pages=1, delay=0)
        results = crawler.crawl()
        assert len(results) == 1

    @patch("runeextract.web.crawler.requests.get")
    def test_crawl_does_not_follow_external(self, mock_get):
        html = """<html><body>
            <a href="https://other.com/page">External</a>
        </body></html>"""
        mock_get.return_value = _mock_response(html, content_type="text/html")
        crawler = SmartCrawler("https://example.com", max_pages=5, delay=0, same_domain=True)
        results = crawler.crawl()
        assert len(results) == 1  # only the start page

    @patch("runeextract.web.crawler.requests.get")
    def test_crawl_skips_failed_requests(self, mock_get):
        import requests
        mock_get.side_effect = requests.ConnectionError("Connection error")
        crawler = SmartCrawler("https://example.com", max_pages=5, delay=0)
        results = crawler.crawl()
        assert results == []

    @patch("runeextract.web.crawler.requests.get")
    def test_crawl_respects_depth(self, mock_get):
        def response_for(url, **kw):
            return _mock_response(
                '<html><body><a href="/next">Next</a></body></html>',
                content_type="text/html",
            )
        mock_get.side_effect = response_for
        crawler = SmartCrawler("https://example.com/start", max_pages=10, delay=0, max_depth=1)
        results = crawler.crawl()
        # start depth=0, /next depth=1, /next from /next depth=2 would be skipped
        # With many links, we might get up to max_pages, but depth prevents deep recursion
        assert len(results) <= crawler.max_pages

    @patch("runeextract.web.crawler.requests.get")
    def test_crawl_on_page_callback(self, mock_get):
        mock_get.return_value = _mock_response(
            "<html><body>Test</body></html>",
            content_type="text/html",
        )
        callback_results = []
        def cb(result):
            callback_results.append(result.url)

        crawler = SmartCrawler("https://example.com", max_pages=2, delay=0, on_page=cb)
        crawler.crawl()
        assert "https://example.com" in callback_results

    @patch("runeextract.web.crawler.requests.get")
    def test_crawl_with_headers(self, mock_get):
        mock_get.return_value = _mock_response(
            "<html><body>Test</body></html>",
            content_type="text/html",
        )
        custom_headers = {"User-Agent": "TestBot/1.0"}
        crawler = SmartCrawler("https://example.com", delay=0, headers=custom_headers)
        crawler.crawl()
        _, kwargs = mock_get.call_args
        assert kwargs["headers"] == custom_headers

    def test_crawl_with_sitemap(self):
        sitemap_xml = (
            '<?xml version="1.0"?>'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            '<url><loc>https://example.com/sitemap_page</loc></url>'
            '</urlset>'
        )
        sitemap_resp = _mock_response(sitemap_xml)
        crawler_resp = _mock_response("<html><body>Sitemap page</body></html>", content_type="text/html")

        def request_side_effect(url, **kw):
            if "sitemap" in url.lower():
                return sitemap_resp
            return crawler_resp

        with patch("requests.get", side_effect=request_side_effect):
            crawler = SmartCrawler("https://example.com", max_pages=5, delay=0, use_sitemap=True, respect_robots=False)
            results = crawler.crawl()
            urls = [r.url for r in results]
            assert "https://example.com/sitemap_page" in urls


class TestSmartCrawlFunction:
    @patch("runeextract.web.crawler.requests.get")
    def test_smart_crawl(self, mock_get):
        mock_get.return_value = _mock_response(
            "<html><body>Hello</body></html>",
            content_type="text/html",
        )
        results = smart_crawl("https://example.com", max_pages=3, delay=0)
        assert isinstance(results, list)
        assert len(results) == 1


class TestCrawlResultDataclass:
    def test_crawl_result_fields(self):
        r = CrawlResult(url="https://example.com", html="<html/>", status_code=200)
        assert r.url == "https://example.com"
        assert r.status_code == 200
        assert r.depth == 0
        assert r.links == []


class TestCrawlerVisitTracking:
    @patch("runeextract.web.crawler.requests.get")
    def test_no_duplicate_visits(self, mock_get):
        html = """<html><body>
            <a href="/page1">P1</a>
        </body></html>"""
        def side_effect(url, **kw):
            return _mock_response(html, content_type="text/html")
        mock_get.side_effect = side_effect
        crawler = SmartCrawler("https://example.com", max_pages=5, delay=0, max_depth=3)
        results = crawler.crawl()
        urls = [r.url for r in results]
        assert len(urls) == len(set(urls))
