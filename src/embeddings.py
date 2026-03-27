from typing import List

import numpy as np
from fastembed import TextEmbedding


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class EmbeddingService:
    """ONNX-based embeddings via fastembed (smaller disk footprint than PyTorch)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        # fastembed yields normalized vectors for most models; normalize defensively.
        vectors = np.array(list(self._model.embed(texts, batch_size=batch_size)), dtype=np.float32)
        return _l2_normalize(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        vec = np.array(list(self._model.embed([query])), dtype=np.float32)[0]
        return _l2_normalize(vec.reshape(1, -1))[0]
