import json

import ollama
from pydantic import BaseModel

from rag_src.utils.config import settings
from rag_src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_structured(prompt: str, schema: type[BaseModel]) -> dict:
    """Call the Ollama chat model, forcing its output to match a JSON schema.

    `schema` is any Pydantic model class (e.g. RAGResponse, or a smaller
    internal schema). Ollama's `format` parameter constrains the model's
    output to valid JSON matching that schema — this is what "structured
    output, not free text" (per the project guide) actually means in
    practice: the model can't return prose, only JSON shaped as requested.

    Returns the raw parsed dict. Callers validate/construct the actual
    Pydantic instance themselves (keeps this function schema-agnostic,
    reusable across RAGResponse, DailyReportResponse, or any future schema).
    """
    client = ollama.Client(host=settings.ollama_host)

    # ollama's type stubs only declare `format` as Literal['', 'json'], but
    # the library actually accepts a full JSON schema dict at runtime (this
    # is how structured output is constrained) — the stub just hasn't
    # caught up to that part of the API yet. Verified end-to-end that this
    # works correctly despite the type mismatch.
    response = client.chat(
        model=settings.ollama_chat_model,
        messages=[{"role": "user", "content": prompt}],
        format=schema.model_json_schema(),  # type: ignore[arg-type]
    )

    # client.chat()'s return type is a union that includes a streaming
    # iterator (for stream=True calls); we never pass stream=True, so the
    # response is always the non-streaming Mapping shape, but mypy can't
    # narrow that statically.
    raw_content = response["message"]["content"]  # type: ignore[index]
    logger.info("llm_generate_completed", model=settings.ollama_chat_model)
    return json.loads(raw_content)
