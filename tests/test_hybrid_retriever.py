from rag_src.retrieval.hybrid_retriever import get_top_movers, retrieve_context


def test_retrieve_context_only_queries_prices_for_coin_entities(mocker):
    fake_documents = [
        {"entity_id": "bitcoin", "entity_type": "coin", "title": "t", "content": "c"},
        {"entity_id": "abc123", "entity_type": "news_article", "title": "t", "content": "c"},
    ]
    mocker.patch(
        "rag_src.retrieval.hybrid_retriever.search_similar", return_value=fake_documents
    )

    fake_cursor = mocker.MagicMock()
    fake_cursor.fetchall.return_value = [("bitcoin", 65000.0, None, None, "2026-01-01")]
    fake_cursor.description = [
        ("coin_id",), ("price_usd",), ("market_cap_usd",), ("volume_24h_usd",), ("price_date",),
    ]
    fake_conn = mocker.MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor
    mocker.patch(
        "rag_src.retrieval.hybrid_retriever.psycopg.connect", return_value=fake_conn
    )

    result = retrieve_context("how is bitcoin doing", top_k=5)

    assert len(result["documents"]) == 2
    assert len(result["price_data"]) == 1
    queried_coin_ids = fake_cursor.execute.call_args.args[1][0]
    assert queried_coin_ids == ["bitcoin"]


def test_retrieve_context_no_coins_skips_price_query_entirely(mocker):
    fake_documents = [
        {"entity_id": "abc123", "entity_type": "news_article", "title": "t", "content": "c"},
    ]
    mocker.patch(
        "rag_src.retrieval.hybrid_retriever.search_similar", return_value=fake_documents
    )
    mock_connect = mocker.patch("rag_src.retrieval.hybrid_retriever.psycopg.connect")

    result = retrieve_context("random news query")

    assert result["price_data"] == []
    mock_connect.assert_not_called()


def test_get_top_movers_computes_percentage_change_correctly(mocker):
    fake_cursor = mocker.MagicMock()
    fake_cursor.fetchall.return_value = [("bitcoin", 110.0, 100.0)]
    fake_conn = mocker.MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor
    mocker.patch(
        "rag_src.retrieval.hybrid_retriever.psycopg.connect", return_value=fake_conn
    )

    movers = get_top_movers(limit=5)

    assert movers == [{"coin_id": "bitcoin", "change_pct": 10.0}]


def test_get_top_movers_skips_coins_with_zero_previous_price(mocker):
    fake_cursor = mocker.MagicMock()
    fake_cursor.fetchall.return_value = [("bitcoin", 110.0, 0.0)]
    fake_conn = mocker.MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor
    mocker.patch(
        "rag_src.retrieval.hybrid_retriever.psycopg.connect", return_value=fake_conn
    )

    movers = get_top_movers()

    assert movers == []


def test_get_top_movers_sorts_by_absolute_change_descending(mocker):
    fake_cursor = mocker.MagicMock()
    fake_cursor.fetchall.return_value = [
        ("bitcoin", 101.0, 100.0),
        ("ethereum", 80.0, 100.0),
        ("solana", 105.0, 100.0),
    ]
    fake_conn = mocker.MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor
    mocker.patch(
        "rag_src.retrieval.hybrid_retriever.psycopg.connect", return_value=fake_conn
    )

    movers = get_top_movers(limit=5)

    assert [m["coin_id"] for m in movers] == ["ethereum", "solana", "bitcoin"]
