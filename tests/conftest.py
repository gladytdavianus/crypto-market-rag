from datetime import UTC, datetime

import pytest

from rag_src.schemas import RawDocument


@pytest.fixture
def sample_document() -> RawDocument:
    """A representative RawDocument, reused across multiple test files."""
    return RawDocument(
        entity_id="bitcoin",
        entity_type="coin",
        title="Bitcoin Overview",
        content="Bitcoin is a decentralized digital currency. " * 20,
        source_url="https://example.com/bitcoin",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        metadata={"symbol": "btc"},
    )


@pytest.fixture
def fake_embedding() -> list[float]:
    """A 768-dim vector, matching nomic-embed-text's output shape."""
    return [0.1] * 768
