"""Local embedding provider (sentence-transformers). No mock fallback."""
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Adapter over a local sentence-transformers model (cosine-normalized vectors)."""

    def __init__(self, model_name: str) -> None:
        if not model_name:
            raise ValueError("model_name is required (set EMBEDDING_MODEL)")
        from sentence_transformers import SentenceTransformer  # fails loudly if missing
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())
        self._e5 = "e5" in model_name.lower()  # e5 models expect query:/passage: prefixes

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True,
                                     batch_size=32, show_progress_bar=False)
        return [[float(x) for x in v] for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([f"passage: {t}" for t in texts] if self._e5 else texts)

    def embed_query(self, question: str) -> list[float]:
        return self._encode([f"query: {question}" if self._e5 else question])[0]
