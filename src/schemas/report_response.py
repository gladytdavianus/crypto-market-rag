from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class DailyReportResponse(BaseModel):
    """Structured output for the automated daily market report.

    Same rationale as RAGResponse: structured output over free text, so the
    report is testable and consistent whether it lands in a log file, an
    Airflow task, or a future API/bot interface.
    """

    report_date: datetime
    summary: str
    top_movers: list[dict]
    """Each item shapes like: {"coin_id": "bitcoin", "change_pct": -8.2}"""
    market_sentiment: Literal["bullish", "bearish", "neutral"]
    sources: list[dict]
