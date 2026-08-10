CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS rag;

CREATE TABLE IF NOT EXISTS rag.documents (
    id              BIGSERIAL PRIMARY KEY,
    entity_id       TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    title           TEXT NOT NULL,
    content         TEXT NOT NULL,
    source_url      TEXT,
    published_at    TIMESTAMPTZ,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_name     TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL DEFAULT 0,
    embedding       vector(768) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_documents_entity_chunk
        UNIQUE (entity_id, source_name, chunk_index)
);

CREATE INDEX IF NOT EXISTS ix_documents_entity_id
    ON rag.documents (entity_id);

CREATE INDEX IF NOT EXISTS ix_documents_entity_type
    ON rag.documents (entity_type);

CREATE INDEX IF NOT EXISTS ix_documents_published_at
    ON rag.documents (published_at);

CREATE INDEX IF NOT EXISTS ix_documents_embedding_hnsw
    ON rag.documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS rag.ingestion_runs (
    id              BIGSERIAL PRIMARY KEY,
    source_name     TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    documents_fetched  INTEGER,
    chunks_created     INTEGER,
    status          TEXT NOT NULL DEFAULT 'running',
    error_message   TEXT
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'rag_user') THEN
        CREATE ROLE rag_user LOGIN PASSWORD 'rag_password';
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO rag_user;
GRANT SELECT ON public.fact_coin_prices TO rag_user;
GRANT SELECT ON public.dim_coins TO rag_user;

GRANT USAGE, CREATE ON SCHEMA rag TO rag_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA rag TO rag_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA rag TO rag_user;

ALTER DEFAULT PRIVILEGES IN SCHEMA rag
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rag_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA rag
    GRANT USAGE, SELECT ON SEQUENCES TO rag_user;
