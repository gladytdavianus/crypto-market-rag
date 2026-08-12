from datetime import datetime

from airflow.decorators import dag, task

from rag_src.embedding.indexer import index_documents
from rag_src.ingestion.coin_description import CoinDescriptionSource
from rag_src.ingestion.news_scraper import CryptoNewsSource
from rag_src.utils.logger import setup_logger
from rag_src.utils.observability import finish_ingestion_run, start_ingestion_run

logger = setup_logger(__name__)


@dag(
    dag_id="news_ingestion_dag",
    start_date=datetime(2026, 1, 1),
    schedule="0 */6 * * *",  # every 6 hours
    catchup=False,
    tags=["rag", "ingestion"],
)
def news_ingestion_dag():
    @task
    def ingest_news_task():
        source_name = "crypto_news_rss"
        run_id = start_ingestion_run(source_name)

        try:
            docs = CryptoNewsSource().fetch()
            chunk_count = index_documents(docs, source_name=source_name)

            finish_ingestion_run(
                run_id,
                status="success",
                documents_fetched=len(docs),
                chunks_created=chunk_count,
            )
            logger.info(
                "news_ingestion_task_completed",
                document_count=len(docs),
                chunk_count=chunk_count,
            )
            return {"document_count": len(docs), "chunk_count": chunk_count}

        except Exception as e:
            finish_ingestion_run(run_id, status="failed", error_message=str(e))
            logger.error("news_ingestion_task_failed", error=str(e))
            raise

    @task
    def ingest_coin_description_task():
        source_name = "dim_coins_description"
        run_id = start_ingestion_run(source_name)

        try:
            docs = CoinDescriptionSource().fetch()
            chunk_count = index_documents(docs, source_name=source_name)

            finish_ingestion_run(
                run_id,
                status="success",
                documents_fetched=len(docs),
                chunks_created=chunk_count,
            )
            logger.info(
                "coin_description_ingestion_task_completed",
                document_count=len(docs),
                chunk_count=chunk_count,
            )
            return {"document_count": len(docs), "chunk_count": chunk_count}

        except Exception as e:
            finish_ingestion_run(run_id, status="failed", error_message=str(e))
            logger.error("coin_description_ingestion_task_failed", error=str(e))
            raise

    # Independent sources, no dependency between them - run in parallel.
    ingest_news_task()
    ingest_coin_description_task()


dag = news_ingestion_dag()
