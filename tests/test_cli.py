from datetime import datetime, timezone

from typer.testing import CliRunner

from rag_src.cli.main import app
from rag_src.schemas import DailyReportResponse, RAGResponse

runner = CliRunner()


def test_query_command_prints_answer_and_sources(mocker):
    fake_response = RAGResponse(
        answer="Bitcoin is up.",
        confidence="high",
        sources=[{"type": "news_article", "title": "News", "url": "https://x.com"}],
        coins_mentioned=["bitcoin"],
        generated_at=datetime.now(timezone.utc),
        query="q",
    )
    mocker.patch("rag_src.cli.main.answer_query", return_value=fake_response)

    result = runner.invoke(app, ["query", "how is bitcoin"])

    assert result.exit_code == 0
    assert "Bitcoin is up." in result.stdout
    assert "(news_article)" in result.stdout


def test_report_command_prints_top_movers(mocker):
    fake_report = DailyReportResponse(
        report_date=datetime.now(timezone.utc),
        summary="Calm day.",
        top_movers=[{"coin_id": "bitcoin", "change_pct": 3.3}],
        market_sentiment="neutral",
        sources=[],
    )
    mocker.patch("rag_src.cli.main.generate_daily_report", return_value=fake_report)

    result = runner.invoke(app, ["report"])

    assert result.exit_code == 0
    assert "+3.3%" in result.stdout
    assert "neutral" in result.stdout
