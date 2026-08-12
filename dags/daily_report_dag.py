import os
from datetime import datetime

from airflow.decorators import dag, task

from rag_src.generation.report_generator import generate_daily_report
from rag_src.utils.logger import setup_logger

logger = setup_logger(__name__)

REPORT_OUTPUT_DIR = "/opt/airflow/data/daily_reports"


@dag(
    dag_id="daily_report_dag",
    start_date=datetime(2026, 1, 1),
    schedule="0 7 * * *",  # every day at 07:00
    catchup=False,
    tags=["rag", "report"],
)
def daily_report_dag():
    @task
    def generate_report_task():
        try:
            report = generate_daily_report()

            output_path = f"{REPORT_OUTPUT_DIR}/{report.report_date.date()}.json"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report.model_dump_json(indent=2))

            logger.info(
                "daily_report_task_completed",
                sentiment=report.market_sentiment,
                mover_count=len(report.top_movers),
                output_path=output_path,
            )
            return output_path

        except Exception as e:
            logger.error("daily_report_task_failed", error=str(e))
            raise

    generate_report_task()


dag = daily_report_dag()
