import logging
import sys

import structlog


def setup_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Configure and return a structured logger.

    Uses structlog for consistent, machine-parseable logs across ingestion,
    embedding, retrieval, and generation. This matters most for scheduled
    runs inside dags/, where stdout/log files are the only visibility into
    what happened during an automated run.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    return structlog.get_logger(name)
