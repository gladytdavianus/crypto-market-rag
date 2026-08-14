from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel

from rag_src.generation.llm_client import generate_structured
from rag_src.retrieval.hybrid_retriever import retrieve_context
from rag_src.schemas import RAGResponse
from rag_src.utils.logger import setup_logger

logger = setup_logger(__name__)


class _AnswerNarrative(BaseModel):
    """Internal schema for the LLM call only — same pattern as
    report_generator._ReportNarrative. Only the parts that genuinely need
    language generation (answer, confidence judgment) come from the LLM;
    sources, coins_mentioned, generated_at, and query are assembled from
    data we already have, not asked from the LLM.
    """

    answer: str
    confidence: Literal["high", "medium", "low"]


def _build_prompt(
    query: str,
    documents: list[dict],
    price_data: list[dict],
    history: list[dict] | None = None,
) -> str:
    context_text = "\n\n".join(
        f"[{doc['entity_type']}] {doc['title']}\n{doc['content']}" for doc in documents
    ) or "No relevant context found."

    price_text = "\n".join(
        f"- {p['coin_id']}: ${p['price_usd']} (as of {p['price_date']})" for p in price_data
    ) or "No price data available."

    history_text = ""
    if history:
        history_lines = "\n".join(f"{h['role']}: {h['message']}" for h in history)
        history_text = f"\nPrevious conversation (for context, most recent last):\n{history_lines}\n"

    return f"""You are a crypto market analyst. Answer the user's question using
ONLY the context and price data below. If the context doesn't contain enough
information to answer confidently, say so and use "low" confidence.
{history_text}
Question: {query}

Relevant context:
{context_text}

Current price data:
{price_text}

Respond with a concise, factual answer grounded in the context above."""


def answer_query(
    query: str,
    top_k: int = 5,
    history: list[dict] | None = None,
) -> RAGResponse:
    """Answer a free-text query about the crypto market, as a RAGResponse.

    This is the function meant to be reused across interfaces (CLI, Telegram
    bot) — none of them should reimplement this orchestration themselves,
    per the project guide's "reuse penuh module, tanpa duplikasi logic"
    principle.

    `history` is optional short-term conversational context (e.g. from
    rag.chat_history, exact retrieval by session_id — not vector search).
    Each item shapes like {"role": "user"|"assistant", "message": str}.
    The CLI doesn't pass this (stateless, one query = one answer); the
    Telegram bot does, for multi-turn conversations within a session.
    """
    context = retrieve_context(query, top_k=top_k)

    prompt = _build_prompt(query, context["documents"], context["price_data"], history=history)
    raw_narrative = generate_structured(prompt, schema=_AnswerNarrative)
    narrative = _AnswerNarrative.model_validate(raw_narrative)

    sources = [
        {"type": doc["entity_type"], "title": doc["title"], "url": doc.get("source_url")}
        for doc in context["documents"]
    ]
    coins_mentioned = sorted(
        {doc["entity_id"] for doc in context["documents"] if doc["entity_type"] == "coin"}
    )

    response = RAGResponse(
        answer=narrative.answer,
        confidence=narrative.confidence,
        sources=sources,
        coins_mentioned=coins_mentioned,
        generated_at=datetime.now(UTC),
        query=query,
    )

    logger.info("query_answered", query=query, confidence=narrative.confidence)
    return response
