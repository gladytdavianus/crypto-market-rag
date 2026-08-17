from rag_src.retrieval.vector_search import _to_pgvector_literal, search_similar


def test_to_pgvector_literal_formats_list_correctly():
    assert _to_pgvector_literal([0.1, -0.2, 0.35]) == "[0.1,-0.2,0.35]"


def test_to_pgvector_literal_empty_list():
    assert _to_pgvector_literal([]) == "[]"


def test_search_similar_returns_dicts_with_similarity_field(mocker):
    fake_rows = [
        ("bitcoin", "coin", "Bitcoin", "content here", "https://x.com", None, {}, "src", 0.9),
    ]
    fake_cursor = mocker.MagicMock()
    fake_cursor.fetchall.return_value = fake_rows
    fake_cursor.description = [
        ("entity_id",),
        ("entity_type",),
        ("title",),
        ("content",),
        ("source_url",),
        ("published_at",),
        ("metadata",),
        ("source_name",),
        ("similarity",),
    ]
    fake_conn = mocker.MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor

    mocker.patch("rag_src.retrieval.vector_search.psycopg.connect", return_value=fake_conn)
    mocker.patch("rag_src.retrieval.vector_search.embed_text", return_value=[0.1] * 768)

    results = search_similar("what is bitcoin", top_k=5)

    assert len(results) == 1
    assert results[0]["entity_id"] == "bitcoin"
    assert results[0]["similarity"] == 0.9


def test_search_similar_with_entity_type_filter_adds_where_clause(mocker):
    fake_cursor = mocker.MagicMock()
    fake_cursor.fetchall.return_value = []
    fake_cursor.description = [
        ("entity_id",),
        ("entity_type",),
        ("title",),
        ("content",),
        ("source_url",),
        ("published_at",),
        ("metadata",),
        ("source_name",),
        ("similarity",),
    ]
    fake_conn = mocker.MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor

    mocker.patch("rag_src.retrieval.vector_search.psycopg.connect", return_value=fake_conn)
    mocker.patch("rag_src.retrieval.vector_search.embed_text", return_value=[0.1] * 768)

    search_similar("query", entity_type="coin")

    executed_sql = fake_cursor.execute.call_args.args[0]
    assert "WHERE entity_type = %s" in executed_sql
