import json

import ollama

from rag_src.utils.config import settings
from rag_src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_structured(prompt: str, schema: type) -> dict:
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

    response = client.chat(
        model=settings.ollama_chat_model,
        messages=[{"role": "user", "content": prompt}],
        format=schema.model_json_schema(),
    )

    raw_content = response["message"]["content"]
    logger.info("llm_generate_completed", model=settings.ollama_chat_model)
    return json.loads(raw_content)
