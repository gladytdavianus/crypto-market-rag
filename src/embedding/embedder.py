import ollama

from src.utils.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for a single piece of text via Ollama.

    Returns a list of 768 floats (nomic-embed-text's output dimension),
    matching the vector(768) column defined in sql/vector_schema.sql.
    """
    client = ollama.Client(host=settings.ollama_host)
    response = client.embeddings(model=settings.ollama_embedding_model, prompt=text)
    return response["embedding"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts.

    One Ollama call per text (Ollama's embeddings endpoint doesn't batch
    multiple prompts in a single request). Fine for portfolio scale; if
    this becomes a bottleneck later, parallelizing these calls is the
    first thing to try.
    """
    logger.info("embedding_batch_started", text_count=len(texts))
    embeddings = [embed_text(text) for text in texts]
    logger.info("embedding_batch_completed", text_count=len(texts))
    return embeddings
