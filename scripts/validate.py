"""
End-to-end validation for the legal RAG chatbot (problem statement checks).

Steps:
1) PDF exists and preprocess + chunking produce content
2) If FAISS index exists: embedding + retrieval return hits
3) If Ollama is up: one non-streaming generation call succeeds (optional)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from config import settings
    from src.chunking import chunk_text
    from src.preprocess import clean_document

    print("1) PDF path and preprocessing...")
    if not settings.raw_pdf_path.exists():
        print(f"FAIL: PDF not found: {settings.raw_pdf_path}")
        return 1
    cleaned = clean_document(settings.raw_pdf_path)
    if len(cleaned.strip()) < 200:
        print("FAIL: Cleaned document too short after preprocessing.")
        return 1
    chunks = chunk_text(
        cleaned,
        min_words=settings.chunk_min_words,
        max_words=settings.chunk_max_words,
    )
    if not chunks:
        print("FAIL: Chunking produced zero chunks.")
        return 1
    print(f"   OK: {len(chunks)} chunks (min words sample: {chunks[0].get('word_count')})")

    print("2) Vector store + retrieval...")
    from src.embeddings import EmbeddingService
    from src.vector_store import FaissVectorStore

    store = FaissVectorStore(settings.vector_store_dir, settings.collection_name)
    if store.count() == 0:
        print("WARN: FAISS index empty. Run: python scripts/ingest.py")
        print("PARTIAL: Preprocess/chunking OK; full RAG pending ingest.")
        return 2

    embedder = EmbeddingService(settings.embedding_model_name)
    query = "Who is the contracting entity if you reside in the United States?"
    qvec = embedder.embed_query(query)
    hits = store.search(qvec, top_k=settings.retrieval_top_k)
    if not hits:
        print("FAIL: Retrieval returned no hits.")
        return 1
    top = hits[0]
    print(f"   OK: top hit {top['chunk_id']} score={top['score']:.4f}")

    print("3) Ollama LLM (optional)...")
    try:
        import ollama

        models = ollama.list()
        names = [m["model"] for m in models.get("models", [])]
        if settings.llm_model_name not in names and not any(
            settings.llm_model_name.split(":")[0] in n for n in names
        ):
            print(
                f"WARN: Model '{settings.llm_model_name}' not listed in `ollama list`. "
                f"Pull it with: ollama pull {settings.llm_model_name}"
            )
        else:
            r = ollama.chat(
                model=settings.llm_model_name,
                messages=[{"role": "user", "content": "Say OK in one word."}],
                options={"temperature": 0, "num_predict": 8},
            )
            content = (r.get("message") or {}).get("content", "")
            print(f"   OK: Ollama reply sample: {content[:80]!r}")
    except Exception as exc:
        print(f"WARN: Ollama not available (skip LLM check): {exc}")

    print("PASS: Validation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
