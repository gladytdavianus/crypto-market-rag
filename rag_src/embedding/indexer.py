import psycopg
from psycopg.types.json import Json

from rag_src.embedding.chunker import chunk_text
from rag_src.embedding.embedder import embed_text
from rag_src.schemas import RawDocument
from rag_src.utils.config import settings
from rag_src.utils.logger import setup_logger

logger = setup_logger(__name__)


def index_document(document: RawDocument, source_name: str) -> int:
    """Chunk, embed, and store a single RawDocument into rag.documents.

    This is the missing link between embedding/ (chunk + embed) and actually
    having searchable data: without this, chunk_text()/embed_text() results
    would just be computed and discarded.

    Uses INSERT ... ON CONFLICT DO UPDATE against the (entity_id, source_name,
    chunk_index) unique constraint from vector_schema.sql, so re-indexing the
    same document (e.g. ingestion re-run) updates existing chunks instead of
    creating duplicates.

    Returns the number of chunks stored.
    """
    chunks = chunk_text(document.content)

    if not chunks:
        logger.warning("index_document_empty_content", entity_id=document.entity_id)
        return 0

    with psycopg.connect(settings.postgres_dsn) as conn, conn.cursor() as cur:
        for chunk_index, chunk in enumerate(chunks):
            embedding = embed_text(chunk)

            cur.execute(
                """
                INSERT INTO rag.documents
                    (entity_id, entity_type, title, content, source_url,
                     published_at, metadata, source_name, chunk_index, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (entity_id, source_name, chunk_index)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    title = EXCLUDED.title,
                    source_url = EXCLUDED.source_url,
                    published_at = EXCLUDED.published_at,
                    metadata = EXCLUDED.metadata;
                """,
                (
                    document.entity_id,
                    document.entity_type,
                    document.title,
                    chunk,
                    document.source_url,
                    document.published_at,
                    Json(document.metadata),
                    source_name,
                    chunk_index,
                    embedding,
                ),
            )
        conn.commit()

    logger.info(
        "index_document_completed",
        entity_id=document.entity_id,
        chunk_count=len(chunks),
    )
    return len(chunks)


def index_documents(documents: list[RawDocument], source_name: str) -> int:
    """Index multiple RawDocuments. Returns total chunk count stored."""
    total_chunks = 0
    for document in documents:
        total_chunks += index_document(document, source_name=source_name)

    logger.info(
        "index_documents_completed",
        document_count=len(documents),
        total_chunk_count=total_chunks,
    )
    return total_chunks
