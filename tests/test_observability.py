def test_start_ingestion_run_returns_new_run_id(mocker):
    from rag_src.utils.observability import start_ingestion_run

    fake_cursor = mocker.MagicMock()
    fake_cursor.fetchone.return_value = (7,)
    fake_conn = mocker.MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor
    mocker.patch("rag_src.utils.observability.psycopg.connect", return_value=fake_conn)

    run_id = start_ingestion_run("test_source")

    assert run_id == 7
    assert fake_conn.commit.called


def test_finish_ingestion_run_sends_correct_params(mocker):
    from rag_src.utils.observability import finish_ingestion_run

    fake_cursor = mocker.MagicMock()
    fake_conn = mocker.MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor
    mocker.patch("rag_src.utils.observability.psycopg.connect", return_value=fake_conn)

    finish_ingestion_run(7, status="success", documents_fetched=5, chunks_created=8)

    params = fake_cursor.execute.call_args.args[1]
    assert params == ("success", 5, 8, None, 7)


def test_finish_ingestion_run_records_error_message_on_failure(mocker):
    from rag_src.utils.observability import finish_ingestion_run

    fake_cursor = mocker.MagicMock()
    fake_conn = mocker.MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.cursor.return_value.__enter__.return_value = fake_cursor
    mocker.patch("rag_src.utils.observability.psycopg.connect", return_value=fake_conn)

    finish_ingestion_run(9, status="failed", error_message="RSS server down")

    params = fake_cursor.execute.call_args.args[1]
    assert params == ("failed", None, None, "RSS server down", 9)
