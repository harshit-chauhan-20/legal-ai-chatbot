import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from src.chunking import chunk_text
from src.embeddings import EmbeddingService
from src.logging_utils import setup_logger
from src.preprocess import clean_document
import faiss
import numpy as np

from src.vector_store import FaissVectorStore


def main() -> None:
    logger = setup_logger("ingest", settings.log_level)
    pdf_path: Path = settings.raw_pdf_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Reading and cleaning PDF: %s", pdf_path)
    cleaned_text = clean_document(pdf_path)

    logger.info("Chunking document")
    chunks = chunk_text(
        cleaned_text,
        min_words=settings.chunk_min_words,
        max_words=settings.chunk_max_words,
    )
    logger.info("Total chunks created: %s", len(chunks))

    settings.processed_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.processed_json_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    logger.info("Embedding chunks in batches")
    embedder = EmbeddingService(settings.embedding_model_name)
    vectors = embedder.embed_texts([c["text"] for c in chunks], batch_size=32)
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    faiss.normalize_L2(vectors)

    logger.info("Persisting FAISS index to %s", settings.vector_store_dir)
    store = FaissVectorStore(settings.vector_store_dir, settings.collection_name)
    store.upsert_chunks(chunks, vectors)
    logger.info("Ingestion complete. Stored chunks: %s", store.count())


if __name__ == "__main__":
    main()
