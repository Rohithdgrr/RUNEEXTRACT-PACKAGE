"""
Web module — smart web crawling, sitemap/RSS ingestion, and URL discovery.
"""

from runeextract.web.crawler import SmartCrawler, CrawlResult, smart_crawl
from runeextract.web.sitemap import parse_sitemap, discover_sitemap
from runeextract.web.feed import parse_feed, discover_feed

__all__ = [
    "SmartCrawler", "CrawlResult", "smart_crawl",
    "parse_sitemap", "discover_sitemap",
    "parse_feed", "discover_feed",
]
