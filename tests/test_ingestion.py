from unittest.mock import MagicMock

from rag_src.ingestion.coin_description import CoinDescriptionSource
from rag_src.ingestion.news_scraper import CryptoNewsSource, _strip_html


def test_strip_html_removes_tags_keeps_text():
    html = '<p><img src="x.jpg" alt="pic"/></p><p>Hello world.</p>'
    assert _strip_html(html) == "Hello world."


def test_strip_html_empty_input_returns_empty_string():
    assert _strip_html("") == ""


def test_crypto_news_source_name():
    assert CryptoNewsSource().source_name() == "crypto_news_rss"


def test_coin_description_source_name():
    assert CoinDescriptionSource().source_name() == "dim_coins_description"


def test_coin_description_fetch_maps_db_rows_to_raw_documents(mocker):
    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = [
        ("bitcoin", "Bitcoin", "Bitcoin is digital money."),
    ]
    fake_conn = MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor
    mocker.patch("rag_src.ingestion.coin_description.psycopg.connect", return_value=fake_conn)

    docs = CoinDescriptionSource().fetch()

    assert len(docs) == 1
    assert docs[0].entity_id == "bitcoin"
    assert docs[0].entity_type == "coin"
    assert docs[0].content == "Bitcoin is digital money."


def test_coin_description_fetch_returns_empty_list_when_no_rows(mocker):
    fake_cursor = MagicMock()
    fake_cursor.fetchall.return_value = []
    fake_conn = MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor
    mocker.patch("rag_src.ingestion.coin_description.psycopg.connect", return_value=fake_conn)

    docs = CoinDescriptionSource().fetch()

    assert docs == []


def test_crypto_news_source_fetch_converts_entries_to_raw_documents(mocker):
    fake_entry = {
        "title": "Bitcoin Surges",
        "link": "https://example.com/article",
        "summary": "<p>Bitcoin went up.</p>",
        "published_parsed": (2026, 1, 1, 12, 0, 0, 0, 0, 0),
        "tags": [{"term": "Bitcoin"}, {"term": "Markets"}],
    }
    fake_parsed = mocker.Mock()
    fake_parsed.bozo = False
    fake_parsed.entries = [fake_entry]
    mocker.patch("rag_src.ingestion.news_scraper.feedparser.parse", return_value=fake_parsed)

    docs = CryptoNewsSource().fetch()

    assert len(docs) == 1
    doc = docs[0]
    assert doc.title == "Bitcoin Surges"
    assert doc.content == "Bitcoin went up."
    assert doc.entity_type == "news_article"
    assert doc.source_url == "https://example.com/article"
    assert doc.metadata["tags"] == ["Bitcoin", "Markets"]


def test_crypto_news_source_fetch_handles_malformed_feed_without_crashing(mocker):
    fake_parsed = mocker.Mock()
    fake_parsed.bozo = True
    fake_parsed.bozo_exception = Exception("malformed XML")
    fake_parsed.entries = []
    mocker.patch("rag_src.ingestion.news_scraper.feedparser.parse", return_value=fake_parsed)

    docs = CryptoNewsSource().fetch()

    assert docs == []


def test_crypto_news_source_entry_without_url_uses_entry_id_as_entity_id(mocker):
    fake_entry = {
        "title": "No Link Article",
        "link": "",
        "id": "fallback-id-123",
        "summary": "Some content.",
    }
    fake_parsed = mocker.Mock()
    fake_parsed.bozo = False
    fake_parsed.entries = [fake_entry]
    mocker.patch("rag_src.ingestion.news_scraper.feedparser.parse", return_value=fake_parsed)

    docs = CryptoNewsSource().fetch()

    assert docs[0].entity_id == "fallback-id-123"
    assert docs[0].source_url is None
