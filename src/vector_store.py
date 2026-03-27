import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np


class FaissVectorStore:
    """
    Persisted cosine similarity search using inner product on L2-normalized vectors.
    """

    def __init__(self, store_dir: Path, collection_name: str = "legal_doc_chunks"):
        self.store_dir = Path(store_dir)
        self.collection_name = collection_name
        self.index_path = self.store_dir / f"{collection_name}.faiss"
        self.meta_path = self.store_dir / f"{collection_name}_meta.json"
        self._index: faiss.Index | None = None
        self._chunks: List[Dict] = []
        self._dim: int | None = None
        if self.index_path.exists() and self.meta_path.exists():
            self._load()

    def _load(self) -> None:
        self._index = faiss.read_index(str(self.index_path))
        with open(self.meta_path, encoding="utf-8") as f:
            payload = json.load(f)
        self._chunks = payload["chunks"]
        self._dim = int(payload["dim"])

    def count(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    def upsert_chunks(self, chunks: List[Dict], vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        self._dim = vectors.shape[1]
        self._chunks = chunks
        index = faiss.IndexFlatIP(self._dim)
        index.add(vectors)
        self._index = index
        self.store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {"dim": self._dim, "chunks": chunks},
                f,
                ensure_ascii=False,
            )

    def search(self, query_vector: np.ndarray, top_k: int = 6) -> List[Dict]:
        if self._index is None or self._index.ntotal == 0:
            return []
        q = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self._index.search(q, min(top_k, self._index.ntotal))
        hits = []
        for rank in range(scores.shape[1]):
            idx = int(indices[0, rank])
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            sim = float(scores[0, rank])
            hits.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": sim,
                    "metadata": {"word_count": chunk.get("word_count")},
                }
            )
        return hits
