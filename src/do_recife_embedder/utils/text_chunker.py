import tiktoken
from semchunk import chunk

_EMBEDDING_MODEL = "text-embedding-3-large"


class TextChunker:
    """Semantic chunking via ``semchunk`` with a tiktoken token counter.

    Defaults to 512-token chunks with 20% overlap, sized for the
    ``text-embedding-3-large`` embedding model. Callers can override
    ``chunk_size``/``overlap`` to tune chunking per use case.
    """

    def __init__(self, chunk_size: int = 512, overlap: float = 0.2):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._encoding = tiktoken.encoding_for_model(_EMBEDDING_MODEL)

    def chunk_text(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return chunk(
            text,
            chunk_size=self.chunk_size,
            token_counter=self._token_counter,
            overlap=self.overlap,
        )

    def _token_counter(self, text: str) -> int:
        return len(self._encoding.encode(text))
