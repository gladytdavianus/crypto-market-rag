import psycopg

from src.retrieval.vector_search import search_similar
from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _get_latest_prices(coin_ids: list[str]) -> list[dict]:
    """Fetch the latest known price for each coin_id from fact_coin_prices.

    This is the anti-corruption layer in practice: the ONLY function in the
    entire codebase allowed to assume that a generic entity_id (for
    entity_type="coin") is the same string as the legacy fact_coin_prices/
    dim_coins coin_id. Every other module stays fully generic.

    Uses rag_user, which only has SELECT on fact_coin_prices/dim_coins —
    this can read prices but can never modify pipeline data.
    """
    if not coin_ids:
        return []

    with psycopg.connect(settings.postgres_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (coin_id)
                coin_id, price_usd, market_cap_usd, volume_24h_usd, price_date
            FROM fact_coin_prices
            WHERE coin_id = ANY(%s)
            ORDER BY coin_id, price_date DESC
            """,
            (coin_ids,),
        )
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

    return [dict(zip(columns, row)) for row in rows]


def retrieve_context(query: str, top_k: int = 5) -> dict:
    """Retrieve everything needed to answer `query`.

    Combines two sources:
    1. Relevant text chunks (vector similarity search, fully generic)
    2. For any coins mentioned in those chunks, their latest structured
       price data (the "hybrid" part — vector search + structured SQL)

    This is the single adapter point described in the project guide: the
    only place where generic entity_id meets legacy coin_id. Everything
    downstream (generation/) consumes this dict without needing to know
    that translation happened.
    """
    documents = search_similar(query, top_k=top_k)

    coin_ids = list({doc["entity_id"] for doc in documents if doc["entity_type"] == "coin"})
    price_data = _get_latest_prices(coin_ids)

    logger.info(
        "hybrid_retrieve_completed",
        query=query,
        document_count=len(documents),
        coin_count=len(coin_ids),
    )

    return {
        "documents": documents,
        "price_data": price_data,
    }
