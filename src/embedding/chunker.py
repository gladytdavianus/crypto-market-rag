import tiktoken

# cl100k_base is a general-purpose tokenizer (used by GPT-3.5/4). nomic-embed-text
# has its own tokenizer internally, but this is used purely as a consistent,
# reasonably accurate way to measure "how much text is this" for chunk sizing -
# exact token-for-token match with the embedding model's own tokenizer isn't
# necessary for that purpose.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks, sized by token count.

    Token-based (not word/character count) because it's what actually
    determines whether a chunk fits in the embedding model's input window.

    `overlap` repeats the last N tokens of one chunk at the start of the
    next, so a sentence that would otherwise get cut in half at a chunk
    boundary still has surrounding context in at least one of the chunks.

    Returns an empty list for empty/whitespace-only input, and a single
    chunk (the original text) if it's already within chunk_size.
    """
    text = text.strip()
    if not text:
        return []

    tokens = _ENCODING.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_piece = _ENCODING.decode(chunk_tokens).strip()

        if chunk_piece:
            chunks.append(chunk_piece)

        if end >= len(tokens):
            break

        start = end - overlap

    return chunks
