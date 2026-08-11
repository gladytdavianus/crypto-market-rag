from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel

from src.generation.llm_client import generate_structured
from src.retrieval.hybrid_retriever import get_top_movers, retrieve_context
from src.schemas import DailyReportResponse
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class _ReportNarrative(BaseModel):
    """Internal schema for the LLM call only — not exported/used elsewhere.

    Deliberately smaller than DailyReportResponse: asks the LLM only for
    the parts that genuinely need language generation (summary, sentiment
    judgment). top_movers and sources are assembled afterward from data
    we already computed/retrieved ourselves, not from the LLM.
    """

    summary: str
    market_sentiment: Literal["bullish", "bearish", "neutral"]


def _build_prompt(top_movers: list[dict], documents: list[dict]) -> str:
    movers_text = "\n".join(
        f"- {m['coin_id']}: {m['change_pct']:+.2f}%" for m in top_movers
    ) or "No price movement data available."

    context_text = "\n\n".join(
        f"[{doc['entity_type']}] {doc['title']}\n{doc['content']}" for doc in documents
    ) or "No additional context available."

    return f"""You are a crypto market analyst. Based on the data below, write a
brief market summary (2-4 sentences) and judge the overall market sentiment.

Price movements (last 24h):
{movers_text}

Relevant context:
{context_text}

Respond with a concise, factual summary. Do not invent numbers not shown above."""


def generate_daily_report(query: str = "crypto market summary today") -> DailyReportResponse:
    """Generate a full daily market report.

    Combines three things:
    1. Deterministic price movements (get_top_movers, pure SQL, no LLM)
    2. Relevant text context (retrieve_context, vector search)
    3. LLM-generated narrative (summary + sentiment) grounded in 1 and 2

    This keeps the numeric fields (top_movers) trustworthy while still
    getting a readable, synthesized summary out of the LLM.
    """
    top_movers = get_top_movers(limit=5)
    context = retrieve_context(query, top_k=5)

    prompt = _build_prompt(top_movers, context["documents"])
    raw_narrative = generate_structured(prompt, schema=_ReportNarrative)
    narrative = _ReportNarrative.model_validate(raw_narrative)

    sources = [
        {"type": doc["entity_type"], "title": doc["title"], "url": doc.get("source_url")}
        for doc in context["documents"]
    ]

    report = DailyReportResponse(
        report_date=datetime.now(timezone.utc),
        summary=narrative.summary,
        top_movers=top_movers,
        market_sentiment=narrative.market_sentiment,
        sources=sources,
    )

    logger.info(
        "daily_report_generated",
        mover_count=len(top_movers),
        sentiment=narrative.market_sentiment,
    )
    return report
