import hashlib
from datetime import UTC, datetime
from typing import Any

import feedparser
from bs4 import BeautifulSoup

from rag_src.ingestion.base import BaseIngestionSource
from rag_src.schemas import RawDocument
from rag_src.utils.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_RSS_URL = "https://cointelegraph.com/rss"


def _strip_html(html: str) -> str:
    """Convert HTML content to plain text.

    RSS summaries often come as raw HTML (e.g. wrapped in <p> tags, with
    embedded <img> elements). Chunking/embedding raw HTML would waste tokens
    on markup and dilute the actual semantic content, so this strips tags
    and keeps only the visible text.
    """
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


class CryptoNewsSource(BaseIngestionSource):
    """Concrete ingestion source: crypto news via a public RSS feed.

    This is the concrete example referenced in the roadmap alongside
    BaseIngestionSource. It fetches entries from a public RSS feed (default:
    CoinTelegraph) and normalizes each entry into a RawDocument.

    Deliberately generic: entity_type is "news_article", not tied to any
    specific coin. A single article can mention multiple coins, or none —
    associating an article with a specific coin_id is a retrieval-time
    concern (see hybrid_retriever.py), not an ingestion-time one.
    """

    def __init__(self, feed_url: str = DEFAULT_RSS_URL) -> None:
        self.feed_url = feed_url

    def source_name(self) -> str:
        return "crypto_news_rss"

    def fetch(self) -> list[RawDocument]:
        logger.info("rss_fetch_started", feed_url=self.feed_url)
        parsed = feedparser.parse(self.feed_url)

        if parsed.bozo:
            logger.warning("rss_feed_malformed", error=str(parsed.bozo_exception))

        documents = [self._entry_to_document(entry) for entry in parsed.entries]

        logger.info("rss_fetch_completed", document_count=len(documents))
        return documents

    def _entry_to_document(self, entry: dict[str, Any]) -> RawDocument:
        url = entry.get("link", "")

        entity_id = (
            hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            if url
            else entry.get("id", "unknown")
        )

        published_at = None
        if entry.get("published_parsed"):
            pt = entry["published_parsed"]
            published_at = datetime(pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], tzinfo=UTC)

        raw_content = entry.get("summary") or entry.get("description") or ""
        content = _strip_html(raw_content)

        tags = [tag.get("term") for tag in entry.get("tags", []) if tag.get("term")]

        return RawDocument(
            entity_id=entity_id,
            entity_type="news_article",
            title=entry.get("title", "Untitled"),
            content=content,
            source_url=url or None,
            published_at=published_at,
            metadata={
                "feed_source": self.feed_url,
                "tags": tags,
            },
        )
