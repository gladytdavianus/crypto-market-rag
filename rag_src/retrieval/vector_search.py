import psycopg

from rag_src.embedding.embedder import embed_text
from rag_src.utils.config import settings
from rag_src.utils.logger import setup_logger

logger = setup_logger(__name__)


def _to_pgvector_literal(embedding: list[float]) -> str:
    """Format a Python list of floats as pgvector's text input format.

    pgvector accepts vectors as a string like "[0.1,0.2,0.3]". Building this
    string explicitly and casting with ::vector in SQL (rather than relying
    on psycopg's automatic type adapter) avoids version-dependent adapter
    registration issues — this works the same regardless of psycopg/pgvector
    package versions.
    """
    return "[" + ",".join(str(x) for x in embedding) + "]"


def search_similar(
    query: str,
    top_k: int = 5,
    entity_type: str | None = None,
) -> list[dict]:
    """Find chunks in rag.documents most similar in meaning to `query`.

    Fully generic: has no knowledge of "coins" or any other specific domain.
    Works purely in terms of entity_id/entity_type, the same contract
    RawDocument uses. Callers that need domain-specific data (like coin
    prices) translate the results themselves — see hybrid_retriever.py.

    `entity_type` optionally restricts the search to one kind of entity
    (e.g. "coin" or "news_article"). Leave it None to search everything.

    Returns rows as plain dicts, ordered by similarity (most similar first).
    Each dict has a "similarity" field between 0 (unrelated) and 1 (identical).
    """
    query_embedding = _to_pgvector_literal(embed_text(query))

    with psycopg.connect(settings.postgres_dsn) as conn, conn.cursor() as cur:
        base_query = """
            SELECT entity_id, entity_type, title, content, source_url,
                   published_at, metadata, source_name,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM rag.documents
        """
        params: list = [query_embedding]

        if entity_type is not None:
            base_query += " WHERE entity_type = %s"
            params.append(entity_type)

        base_query += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding, top_k])

        cur.execute(base_query, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description or []]

    results = [dict(zip(columns, row, strict=True)) for row in rows]
    logger.info("vector_search_completed", query=query, result_count=len(results))
    return results
