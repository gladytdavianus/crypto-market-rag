from abc import ABC, abstractmethod

from rag_src.schemas import RawDocument


class BaseIngestionSource(ABC):
    """Contract every ingestion source must implement.

    Any concrete source (RSS feed, REST API, scraper, database export) must
    produce a list of RawDocument — the generic contract that downstream
    modules (embedding, retrieval) depend on. This is what lets new sources
    be added later without touching embedding/retrieval/generation at all.
    """

    @abstractmethod
    def fetch(self) -> list[RawDocument]:
        """Fetch and normalize documents from this source."""
        ...

    @abstractmethod
    def source_name(self) -> str:
        """Unique, stable identifier for this source.

        Used for tracking (rag.ingestion_runs.source_name) and as part of
        the uniqueness constraint on rag.documents, so re-running ingestion
        doesn't create duplicate chunks.
        """
        ...
