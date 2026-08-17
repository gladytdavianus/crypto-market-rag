import psycopg

from rag_src.ingestion.base import BaseIngestionSource
from rag_src.schemas import RawDocument
from rag_src.utils.config import settings
from rag_src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CoinDescriptionSource(BaseIngestionSource):
    """Concrete ingestion source: coin descriptions, read from dim_coins.

    Descriptions are fetched from CoinGecko and stored directly in
    dim_coins.description by crypto-market-pipeline (see backfill_coin_description.py
    in that project). This source does NOT call CoinGecko itself — it only
    reads what the pipeline already fetched, staying consistent with the
    rule that crypto-market-rag never duplicates pipeline's ingestion work,
    it only reads from it (read-only, via rag_user).
    """

    def source_name(self) -> str:
        return "dim_coins_description"

    def fetch(self) -> list[RawDocument]:
        with psycopg.connect(settings.postgres_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT coin_id, name, description FROM dim_coins " "WHERE description IS NOT NULL"
            )
            rows = cur.fetchall()

        documents = [
            RawDocument(
                entity_id=coin_id,
                entity_type="coin",
                title=f"{name} — Overview",
                content=description,
                source_url=f"https://www.coingecko.com/en/coins/{coin_id}",
                published_at=None,
                metadata={},
            )
            for coin_id, name, description in rows
        ]

        logger.info("coin_description_fetch_completed", document_count=len(documents))
        return documents
