from rag_src.embedding.chunker import chunk_text


def test_chunk_text_empty_input_returns_empty_list():
    assert chunk_text("") == []


def test_chunk_text_whitespace_only_returns_empty_list():
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_short_input_returns_single_chunk():
    result = chunk_text("Bitcoin is digital money.", chunk_size=400)
    assert result == ["Bitcoin is digital money."]


def test_chunk_text_long_input_splits_into_multiple_chunks():
    long_text = "Bitcoin is digital money. " * 300
    result = chunk_text(long_text, chunk_size=100, overlap=20)
    assert len(result) > 1


def test_chunk_text_chunks_overlap_at_boundaries():
    long_text = " ".join(f"word{i}" for i in range(500))
    result = chunk_text(long_text, chunk_size=100, overlap=20)

    assert len(result) >= 2
    chunk0_words = set(result[0].split())
    chunk1_words = set(result[1].split())
    assert chunk0_words & chunk1_words
