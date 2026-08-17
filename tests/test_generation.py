import json
from unittest.mock import Mock

from rag_src.generation.llm_client import generate_structured
from rag_src.generation.query_answer import answer_query
from rag_src.generation.report_generator import generate_daily_report
from rag_src.schemas import RAGResponse


def test_generate_structured_parses_json_response(mocker):
    fake_client = Mock()
    fake_client.chat.return_value = {
        "message": {
            "content": json.dumps(
                {
                    "answer": "hi",
                    "confidence": "high",
                    "sources": [],
                    "coins_mentioned": [],
                    "generated_at": "2026-01-01T00:00:00",
                    "query": "q",
                }
            )
        }
    }
    mocker.patch("rag_src.generation.llm_client.ollama.Client", return_value=fake_client)

    result = generate_structured("prompt", schema=RAGResponse)

    assert result["answer"] == "hi"
    call_kwargs = fake_client.chat.call_args.kwargs
    assert "format" in call_kwargs


def test_generate_daily_report_numbers_come_from_sql_not_llm(mocker):
    fake_movers = [{"coin_id": "bitcoin", "change_pct": 5.0}]
    fake_context = {"documents": [], "price_data": []}
    fake_narrative = {"summary": "Calm day.", "market_sentiment": "neutral"}

    mocker.patch("rag_src.generation.report_generator.get_top_movers", return_value=fake_movers)
    mocker.patch("rag_src.generation.report_generator.retrieve_context", return_value=fake_context)
    mock_llm = mocker.patch(
        "rag_src.generation.report_generator.generate_structured",
        return_value=fake_narrative,
    )

    report = generate_daily_report()

    assert report.top_movers == fake_movers
    assert report.summary == "Calm day."
    assert report.market_sentiment == "neutral"
    schema_used = mock_llm.call_args.kwargs.get("schema") or mock_llm.call_args.args[1]
    assert "top_movers" not in schema_used.model_json_schema()["properties"]


def test_answer_query_filters_coins_mentioned_from_documents(mocker):
    fake_context = {
        "documents": [
            {
                "entity_type": "coin",
                "entity_id": "bitcoin",
                "title": "t",
                "content": "c",
                "source_url": None,
            },
            {
                "entity_type": "news_article",
                "entity_id": "abc",
                "title": "t",
                "content": "c",
                "source_url": None,
            },
        ],
        "price_data": [],
    }
    mocker.patch("rag_src.generation.query_answer.retrieve_context", return_value=fake_context)
    mocker.patch(
        "rag_src.generation.query_answer.generate_structured",
        return_value={"answer": "Bitcoin is fine.", "confidence": "high"},
    )

    response = answer_query("how is bitcoin")

    assert response.coins_mentioned == ["bitcoin"]
    assert len(response.sources) == 2
    assert response.query == "how is bitcoin"
