from unittest.mock import MagicMock


def _make_fake_conn():
    fake_cursor = MagicMock()
    fake_conn = MagicMock()
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor
    fake_conn.__enter__.return_value = fake_conn
    return fake_conn, fake_cursor


def test_index_document_empty_content_returns_zero_and_skips_db(mocker, sample_document):
    from rag_src.embedding.indexer import index_document

    empty_doc = sample_document.model_copy(update={"content": ""})
    mock_connect = mocker.patch("rag_src.embedding.indexer.psycopg.connect")

    result = index_document(empty_doc, source_name="test")

    assert result == 0
    mock_connect.assert_not_called()


def test_index_document_inserts_one_row_per_chunk(mocker, sample_document):
    from rag_src.embedding.indexer import index_document

    fake_conn, fake_cursor = _make_fake_conn()
    mocker.patch("rag_src.embedding.indexer.psycopg.connect", return_value=fake_conn)
    mocker.patch("rag_src.embedding.indexer.embed_text", return_value=[0.1] * 768)
    mocker.patch(
        "rag_src.embedding.indexer.chunk_text",
        return_value=["chunk one", "chunk two"],
    )

    chunk_count = index_document(sample_document, source_name="test_source")

    assert chunk_count == 2
    assert fake_cursor.execute.call_count == 2
    assert fake_conn.commit.called


def test_index_documents_sums_chunk_counts_across_documents(mocker, sample_document):
    from rag_src.embedding.indexer import index_documents

    fake_conn, fake_cursor = _make_fake_conn()
    mocker.patch("rag_src.embedding.indexer.psycopg.connect", return_value=fake_conn)
    mocker.patch("rag_src.embedding.indexer.embed_text", return_value=[0.1] * 768)
    mocker.patch("rag_src.embedding.indexer.chunk_text", return_value=["one chunk"])

    total = index_documents([sample_document, sample_document], source_name="test")

    assert total == 2
