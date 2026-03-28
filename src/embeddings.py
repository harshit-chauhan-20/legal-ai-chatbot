from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class EmbeddingService:
    """PyTorch-based embeddings via sentence-transformers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)
        return _l2_normalize(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        vec = self._model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float32)[0]
        return _l2_normalize(vec.reshape(1, -1))[0]
