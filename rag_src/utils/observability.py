import psycopg

from rag_src.utils.config import settings
from rag_src.utils.logger import setup_logger

logger = setup_logger(__name__)


def start_ingestion_run(source_name: str) -> int:
    """Record the start of an ingestion run. Returns the run's id.

    This is the first use of rag.ingestion_runs, created back in
    vector_schema.sql (Fase 1, step 1) but unused until now — DAGs are
    what actually generate scheduled runs worth tracking.
    """
    with psycopg.connect(settings.postgres_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO rag.ingestion_runs (source_name, status) VALUES (%s, 'running') RETURNING id",
            (source_name,),
        )
        run_id = cur.fetchone()[0]
        conn.commit()

    logger.info("ingestion_run_started", source_name=source_name, run_id=run_id)
    return run_id


def finish_ingestion_run(
    run_id: int,
    status: str,
    documents_fetched: int | None = None,
    chunks_created: int | None = None,
    error_message: str | None = None,
) -> None:
    """Mark an ingestion run as finished (success or failed)."""
    with psycopg.connect(settings.postgres_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE rag.ingestion_runs
            SET finished_at = now(),
                status = %s,
                documents_fetched = %s,
                chunks_created = %s,
                error_message = %s
            WHERE id = %s
            """,
            (status, documents_fetched, chunks_created, error_message, run_id),
        )
        conn.commit()

    logger.info("ingestion_run_finished", run_id=run_id, status=status)
