from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import ollama

from config import settings
from src.embeddings import EmbeddingService
from src.extractive_answer import build_extractive_answer, stream_answer_text
from src.logging_utils import setup_logger
from src.prompting import (
    SYSTEM_PROMPT,
    build_context_block,
    build_user_prompt,
    truncate_context_block,
)
from src.vector_store import FaissVectorStore

FALLBACK_RESPONSE = "I could not find this information in the provided document."

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")


def _ollama_reachable(timeout_sec: float = 1.0) -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=timeout_sec)
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _gguf_available() -> bool:
    p = Path(settings.local_gguf_path)
    return p.is_file()


class RAGPipeline:
    def __init__(self):
        self.logger = setup_logger("rag_pipeline", settings.log_level)
        self.embeddings = EmbeddingService(settings.embedding_model_name)
        self.store = FaissVectorStore(
            settings.vector_store_dir, settings.collection_name
        )

    def _sanitize_query(self, query: str) -> str:
        clean = " ".join(query.split())
        if len(clean) > settings.max_query_chars:
            clean = clean[: settings.max_query_chars]
        return clean

    def retrieve(self, query: str) -> List[Dict]:
        query_vec = self.embeddings.embed_query(query)
        hits = self.store.search(query_vec, top_k=settings.retrieval_top_k)
        self.logger.info("Retrieved %s chunks", len(hits))
        return hits

    def _is_relevant(self, hits: List[Dict]) -> bool:
        if not hits:
            return False
        return hits[0]["score"] >= settings.relevance_threshold

    def stream_answer(
        self, query: str, chat_history: List[Dict] | None = None
    ) -> Tuple[Generator[str, None, None], List[Dict]]:
        safe_query = self._sanitize_query(query)
        retrieved = self.retrieve(safe_query)

        if not self._is_relevant(retrieved):
            def empty_stream() -> Generator[str, None, None]:
                yield FALLBACK_RESPONSE

            return empty_stream(), []

        context_block, source_ids = build_context_block(retrieved)
        ctx_for_llm = truncate_context_block(
            context_block, settings.llm_max_context_chars
        )
        user_prompt = build_user_prompt(safe_query, ctx_for_llm)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if chat_history:
            messages.extend(chat_history[-4:])
        messages.append({"role": "user", "content": user_prompt})

        source_payload = [c for c in retrieved if c["chunk_id"] in set(source_ids)]

        if not settings.use_llm:
            self.logger.info("USE_LLM disabled; using extractive grounded answer")

            def extractive_only() -> Generator[str, None, None]:
                text = build_extractive_answer(safe_query, retrieved)
                yield from stream_answer_text(text)

            return extractive_only(), source_payload

        # Use LangChain with free Hugging Face model
        self.logger.info("Using LangChain model %s", settings.langchain_model_name)

        def langchain_stream() -> Generator[str, None, None]:
            try:
                from src.llm_langchain import stream_generate

                for chunk in stream_generate(SYSTEM_PROMPT, user_prompt):
                    yield chunk
            except Exception as exc:
                self.logger.warning("LangChain failed (%s); falling back to extractive", exc)
                text = build_extractive_answer(safe_query, retrieved)
                yield from stream_answer_text(text)

        return langchain_stream(), source_payload
