from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import faiss
import numpy as np

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

# Short queries that only make sense in context of a previous message
_FOLLOWUP_TRIGGERS = (
    "explain", "simplify", "elaborate", "clarify", "what does that mean",
    "in simple", "in simpler", "rephrase", "summarise", "summarize",
    "what do you mean", "tell me more", "expand", "break it down",
    "can you explain", "what is that", "what are those",
)


def _is_followup(query: str) -> bool:
    """Return True if the query looks like a contextual follow-up."""
    q = query.strip().lower()
    # Short queries (under 6 words) that start with a trigger phrase
    if len(q.split()) <= 6:
        for trigger in _FOLLOWUP_TRIGGERS:
            if q.startswith(trigger) or q == trigger:
                return True
    return False


def _rewrite_followup(query: str, chat_history: List[Dict]) -> str:
    """
    Prepend the last assistant reply to a short follow-up query so FAISS
    can find relevant chunks.
    E.g. "Explain in simple words" → "Explain in simple words: <last answer>"
    """
    if not chat_history:
        return query
    # Find last assistant message
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            last_answer = msg.get("content", "").strip()
            if last_answer and last_answer != FALLBACK_RESPONSE:
                # Use first 400 chars of last answer as retrieval anchor
                anchor = last_answer[:400]
                return f"{query}: {anchor}"
    return query


def _ollama_reachable(timeout_sec: float = 1.0) -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=timeout_sec)
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _gguf_available() -> bool:
    if settings.disable_local_gguf:
        return False
    p = Path(settings.local_gguf_path)
    return p.is_file()


def _groq_answer(messages: list, logger) -> Generator[str, None, None]:
    """Call Groq API with streaming and yield tokens."""
    import json as _json

    groq_api_key = os.getenv("GROQ_API_KEY", "")
    if not groq_api_key:
        logger.warning("GROQ_API_KEY not set; falling back to extractive")
        return

    groq_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    payload = _json.dumps({
        "model": groq_model,
        "messages": messages,
        "stream": True,
        "temperature": 0,
        "max_tokens": 512,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {groq_api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = _json.loads(data_str)
                token = chunk["choices"][0]["delta"].get("content", "")
                if token:
                    yield token
            except Exception:
                continue


class RAGPipeline:
    def __init__(self):
        self.logger = setup_logger("rag_pipeline", settings.log_level)
        self.embeddings = EmbeddingService(settings.embedding_model_name)
        self.store = FaissVectorStore(
            settings.vector_store_dir, settings.collection_name
        )
        self._ensure_faiss_from_chunks_if_needed()

    def _ensure_faiss_from_chunks_if_needed(self) -> None:
        if self.store.count() > 0:
            return
        path = settings.processed_json_path
        if not path.exists():
            self.logger.warning("FAISS empty and no chunks.json at %s", path)
            return
        self.logger.info("Rebuilding FAISS from %s (first run or incompatible index)", path)
        with open(path, encoding="utf-8") as f:
            chunks = json.load(f)
        vectors = self.embeddings.embed_texts(
            [c["text"] for c in chunks], batch_size=32
        )
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)
        self.store.upsert_chunks(chunks, vectors)

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
        chat_history = chat_history or []

        # ── Follow-up rewriting ───────────────────────────────────────────────
        retrieval_query = safe_query
        if _is_followup(safe_query):
            retrieval_query = _rewrite_followup(safe_query, chat_history)
            self.logger.info(
                "Follow-up detected. Rewritten retrieval query: %s", retrieval_query[:120]
            )

        retrieved = self.retrieve(retrieval_query)

        if not self._is_relevant(retrieved):
            # Second chance: try original query if rewrite didn't help
            if retrieval_query != safe_query:
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

        # ── Mode: Extractive / Groq Grounded (USE_LLM=false) ──────────────────
        if not settings.use_llm:
            self.logger.info("USE_LLM disabled; using Groq grounded answer")

            def groq_grounded_stream() -> Generator[str, None, None]:
                try:
                    tokens = list(_groq_answer(messages, self.logger))
                    if tokens:
                        yield from tokens
                    else:
                        text = build_extractive_answer(safe_query, retrieved)
                        yield from stream_answer_text(text)
                except Exception as exc:
                    self.logger.warning("Groq failed (%s); falling back to extractive", exc)
                    text = build_extractive_answer(safe_query, retrieved)
                    yield from stream_answer_text(text)

            return groq_grounded_stream(), source_payload

        # ── Mode: Local GGUF ──────────────────────────────────────────────────
        if _gguf_available():
            self.logger.info("Using local GGUF model at %s", settings.local_gguf_path)

            def gguf_stream() -> Generator[str, None, None]:
                try:
                    from src.llm_gguf import stream_generate
                    for chunk in stream_generate(SYSTEM_PROMPT, user_prompt):
                        yield chunk
                except Exception as exc:
                    self.logger.warning("GGUF failed (%s); falling back to extractive", exc)
                    text = build_extractive_answer(safe_query, retrieved)
                    yield from stream_answer_text(text)

            return gguf_stream(), source_payload

        # ── Mode: Ollama ──────────────────────────────────────────────────────
        if _ollama_reachable():
            self.logger.info("Using Ollama model %s", settings.llm_model_name)

            def ollama_stream() -> Generator[str, None, None]:
                try:
                    import ollama  # lazy import — not installed on Streamlit Cloud
                    stream = ollama.chat(
                        model=settings.llm_model_name,
                        messages=messages,
                        stream=True,
                        options={"temperature": 0, "num_predict": 512},
                    )
                    for chunk in stream:
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                except Exception as exc:
                    self.logger.warning("Ollama failed (%s); falling back to extractive", exc)
                    text = build_extractive_answer(safe_query, retrieved)
                    yield from stream_answer_text(text)

            return ollama_stream(), source_payload

        # ── Final fallback: extractive ────────────────────────────────────────
        self.logger.info("No LLM available; using pure extractive answer")

        def extractive_fallback() -> Generator[str, None, None]:
            text = build_extractive_answer(safe_query, retrieved)
            yield from stream_answer_text(text)

        return extractive_fallback(), source_payload
