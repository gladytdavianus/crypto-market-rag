import hashlib
from datetime import datetime, timezone
from typing import Any

import feedparser

from src.ingestion.base import BaseIngestionSource
from src.schemas import RawDocument
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# CoinDesk's public RSS feed — free, no API key required.
DEFAULT_RSS_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"


class CryptoNewsSource(BaseIngestionSource):
    """Concrete ingestion source: crypto news via a public RSS feed.

    This is the concrete example referenced in the roadmap alongside
    BaseIngestionSource. It fetches entries from a public RSS feed (default:
    CoinDesk) and normalizes each entry into a RawDocument.

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
            # bozo=True means the feed was malformed but feedparser still
            # tried its best to parse it. Log it, don't hard-fail — partial
            # results are still useful.
            logger.warning("rss_feed_malformed", error=str(parsed.bozo_exception))

        documents = [self._entry_to_document(entry) for entry in parsed.entries]

        logger.info("rss_fetch_completed", document_count=len(documents))
        return documents

    def _entry_to_document(self, entry: dict[str, Any]) -> RawDocument:
        url = entry.get("link", "")

        # Stable entity_id derived from the URL, so re-fetching the same
        # article twice produces the same entity_id (idempotent ingestion).
        entity_id = (
            hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
            if url
            else entry.get("id", "unknown")
        )

        published_at = None
        if entry.get("published_parsed"):
            published_at = datetime(*entry["published_parsed"][:6], tzinfo=timezone.utc)

        content = entry.get("summary") or entry.get("description") or ""

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
