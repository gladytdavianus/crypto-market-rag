from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from rag_src.schemas import DailyReportResponse, RAGResponse, RawDocument


def test_raw_document_requires_entity_id_and_entity_type():
    with pytest.raises(ValidationError):
        RawDocument(title="X", content="Y")


def test_raw_document_optional_fields_default_correctly():
    doc = RawDocument(entity_id="bitcoin", entity_type="coin", title="T", content="C")
    assert doc.source_url is None
    assert doc.published_at is None
    assert doc.metadata == {}


def test_raw_document_metadata_not_shared_between_instances():
    """Guards against the classic Python mutable-default-argument bug -
    each instance's metadata dict must be independent."""
    doc1 = RawDocument(entity_id="a", entity_type="coin", title="T", content="C")
    doc2 = RawDocument(entity_id="b", entity_type="coin", title="T", content="C")

    doc1.metadata["injected"] = True

    assert doc2.metadata == {}


def test_rag_response_rejects_invalid_confidence_value():
    with pytest.raises(ValidationError):
        RAGResponse(
            answer="x",
            confidence="super-high",
            sources=[],
            coins_mentioned=[],
            generated_at=datetime.now(timezone.utc),
            query="q",
        )


def test_daily_report_response_valid_construction():
    report = DailyReportResponse(
        report_date=datetime.now(timezone.utc),
        summary="Market is calm.",
        top_movers=[{"coin_id": "bitcoin", "change_pct": 1.5}],
        market_sentiment="neutral",
        sources=[],
    )
    assert report.market_sentiment == "neutral"
    assert report.top_movers[0]["coin_id"] == "bitcoin"
