from unittest.mock import Mock

from rag_src.embedding.embedder import embed_text, embed_texts


def test_embed_text_returns_768_dim_vector(mocker):
    fake_client = Mock()
    fake_client.embeddings.return_value = {"embedding": [0.1] * 768}
    mocker.patch("rag_src.embedding.embedder.ollama.Client", return_value=fake_client)

    result = embed_text("Bitcoin is digital money.")

    assert len(result) == 768


def test_embed_text_sends_correct_model_and_prompt(mocker):
    fake_client = Mock()
    fake_client.embeddings.return_value = {"embedding": [0.1] * 768}
    mocker.patch("rag_src.embedding.embedder.ollama.Client", return_value=fake_client)

    embed_text("test prompt")

    call_kwargs = fake_client.embeddings.call_args.kwargs
    assert call_kwargs["prompt"] == "test prompt"
    assert call_kwargs["model"] == "nomic-embed-text"


def test_embed_texts_calls_embed_text_once_per_item(mocker):
    fake_client = Mock()
    fake_client.embeddings.return_value = {"embedding": [0.2] * 768}
    mocker.patch("rag_src.embedding.embedder.ollama.Client", return_value=fake_client)

    results = embed_texts(["a", "b", "c"])

    assert len(results) == 3
    assert fake_client.embeddings.call_count == 3
