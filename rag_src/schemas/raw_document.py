from datetime import datetime

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    """Result of ingestion from any source, normalized into this shape.

    This is the generic contract every ingestion source must produce.
    It intentionally avoids domain-specific fields (e.g. no `coin_id`):
    domain-specific data belongs in `metadata`, not in the core schema.
    See `retrieval/hybrid_retriever.py` for the only place allowed to
    translate `entity_id` back into domain-specific identifiers (the
    anti-corruption layer pattern).
    """

    entity_id: str = Field(
        ...,
        description="Generic entity identifier, e.g. a coin_id, equipment_id, case_id.",
    )
    entity_type: str = Field(
        ...,
        description='Generic entity category, e.g. "coin", "equipment", "legal_case".',
    )
    title: str
    content: str
    source_url: str | None = None
    published_at: datetime | None = None
    metadata: dict = Field(
        default_factory=dict,
        description="Domain-specific fields go here, without touching the core schema.",
    )
