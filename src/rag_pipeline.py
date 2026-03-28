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

# ────────────────────────────────────────────────────────────────
# FOLLOW-UP HANDLING
# ────────────────────────────────────────────────────────────────

_FOLLOWUP_TRIGGERS = (
    "explain", "simplify", "elaborate", "clarify", "what does that mean",
    "in simple", "in simpler", "rephrase", "summarise", "summarize",
    "what do you mean", "tell me more", "expand", "break it down",
    "can you explain", "what is that", "what are those",
)


def _is_followup(query: str) -> bool:
    q = query.strip().lower()
    if len(q.split()) <= 6:
        return any(q.startswith(t) for t in _FOLLOWUP_TRIGGERS)
    return False


def _rewrite_followup(query: str, chat_history: List[Dict]) -> str:
    if not chat_history:
        return query

    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            last = msg.get("content", "").strip()
            if last and last != FALLBACK_RESPONSE:
                return f"{query}: {last[:400]}"
    return query


# ────────────────────────────────────────────────────────────────
# GROQ STREAMING
# ────────────────────────────────────────────────────────────────

def _groq_answer(messages: list, logger) -> Generator[str, None, None]:
    import json as _json

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY missing")
        return

    payload = _json.dumps({
        "model": os.getenv("GROQ_MODEL", "llama3-8b-8192"),
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
            "Authorization": f"Bearer {api_key}",
        },
    )

    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = _json.loads(data)
                token = chunk["choices"][0]["delta"].get("content", "")
                if token:
                    yield token
            except:
                continue


# ────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ────────────────────────────────────────────────────────────────

class RAGPipeline:
    def __init__(self):
        self.logger = setup_logger("rag_pipeline", settings.log_level)
        self.embeddings = EmbeddingService(settings.embedding_model_name)
        self.store = FaissVectorStore(
            settings.vector_store_dir, settings.collection_name
        )
        self._ensure_faiss()

    def _ensure_faiss(self):
        if self.store.count() > 0:
            return

        path = settings.processed_json_path
        if not path.exists():
            return

        with open(path, encoding="utf-8") as f:
            chunks = json.load(f)

        vecs = self.embeddings.embed_texts([c["text"] for c in chunks])
        vecs = vecs.astype(np.float32)
        faiss.normalize_L2(vecs)

        self.store.upsert_chunks(chunks, vecs)

    def _sanitize(self, q: str) -> str:
        q = " ".join(q.split())
        return q[: settings.max_query_chars]

    def retrieve(self, query: str) -> List[Dict]:
        vec = self.embeddings.embed_query(query)
        hits = self.store.search(vec, settings.retrieval_top_k)
        return hits

    def _is_relevant(self, hits, is_follow=False):
        if not hits:
            return False
        thresh = settings.relevance_threshold
        if is_follow:
            thresh *= 0.7
        return hits[0]["score"] >= thresh

    def stream_answer(self, query, chat_history=None):
        chat_history = chat_history or []
        safe_query = self._sanitize(query)

        # 🔥 FOLLOW-UP FIX
        is_follow = _is_followup(safe_query)
        retrieval_query = (
            _rewrite_followup(safe_query, chat_history)
            if is_follow else safe_query
        )

        retrieved = self.retrieve(retrieval_query)

        if not self._is_relevant(retrieved, is_follow):
            return (iter([FALLBACK_RESPONSE]), [])

        context, ids = build_context_block(retrieved)
        context = truncate_context_block(context, settings.llm_max_context_chars)

        # 🔥 GENERATION FIX
        generation_query = safe_query
        if is_follow:
            generation_query += f"\n\nContext:\n{retrieval_query}"

        user_prompt = build_user_prompt(generation_query, context)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(chat_history[-4:])
        messages.append({"role": "user", "content": user_prompt})

        sources = [c for c in retrieved if c["chunk_id"] in set(ids)]

        # 🔥 GROQ (PRIMARY)
        def stream():
            try:
                tokens = list(_groq_answer(messages, self.logger))
                if tokens:
                    yield from tokens
                    return
            except Exception as e:
                self.logger.warning(f"Groq failed: {e}")

            # fallback
            text = build_extractive_answer(safe_query, retrieved)
            yield from stream_answer_text(text)

        return stream(), sources
