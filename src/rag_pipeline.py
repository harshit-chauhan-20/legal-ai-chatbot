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

# ── Follow-up detection ────────────────────────────────────────────────────────
_FOLLOWUP_TRIGGERS = (
    "explain", "simplify", "elaborate", "clarify",
    "what does that mean", "what does this mean",
    "in simple", "in simpler", "in short",
    "rephrase", "summarise", "summarize",
    "what do you mean", "tell me more", "expand",
    "break it down", "can you explain",
    "what is that", "what are those",
    "what is it", "what is it about", "what about this",
    "explain again", "say that again", "repeat",
    "will it", "is it", "does it", "how does it",
    "why is it", "why does it", "what happens",
    "beneficial", "advantage", "disadvantage",
    "good or bad", "pros", "cons",
    "more details", "give more", "go on",
    "and then", "what next", "so what",
)


def _is_followup(query: str) -> bool:
    """Return True if query is a short contextual follow-up."""
    q = query.lower().strip().rstrip("?.")
    words = q.split()
    if len(words) > 8:
        return False
    for trigger in _FOLLOWUP_TRIGGERS:
        if q == trigger or q.startswith(trigger):
            return True
    return False


def _rewrite_followup(query: str, chat_history: List[Dict]) -> str:
    """
    Prepend last user question + assistant answer to the follow-up query
    so FAISS can find relevant chunks.
    """
    if not chat_history:
        return query

    last_user = ""
    last_assistant = ""
    for msg in reversed(chat_history):
        if msg["role"] == "assistant" and not last_assistant:
            last_assistant = msg["content"]
        elif msg["role"] == "user" and not last_user:
            last_user = msg["content"]
        if last_user and last_assistant:
            break

    anchor = f"{last_user}\n{last_assistant}"
    return f"{query}: {anchor[:500]}"


# ── Groq streaming ─────────────────────────────────────────────────────────────
def _groq_answer(messages: list, logger) -> Generator[str, None, None]:
    import json as _json

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set; skipping Groq")
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
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
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
            except Exception:
                continue


# ── Ollama ─────────────────────────────────────────────────────────────────────
def _ollama_reachable(timeout_sec: float = 1.0) -> bool:
    try:
        urllib.request.urlopen(
            f"{OLLAMA_HOST.rstrip('/')}/api/tags", timeout=timeout_sec
        )
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _gguf_available() -> bool:
    if settings.disable_local_gguf:
        return False
    return Path(settings.local_gguf_path).is_file()


# ── Pipeline ───────────────────────────────────────────────────────────────────
class RAGPipeline:
    def __init__(self):
        self.logger = setup_logger("rag_pipeline", settings.log_level)
        self.embeddings = EmbeddingService(settings.embedding_model_name)
        self.store = FaissVectorStore(
            settings.vector_store_dir, settings.collection_name
        )
        self._ensure_faiss()

    def _ensure_faiss(self) -> None:
        if self.store.count() > 0:
            return
        path = settings.processed_json_path
        if not path.exists():
            self.logger.warning("FAISS empty and no chunks.json at %s", path)
            return
        self.logger.info("Rebuilding FAISS from %s", path)
        with open(path, encoding="utf-8") as f:
            chunks = json.load(f)
        vecs = self.embeddings.embed_texts([c["text"] for c in chunks]).astype(np.float32)
        faiss.normalize_L2(vecs)
        self.store.upsert_chunks(chunks, vecs)

    def _sanitize(self, q: str) -> str:
        return " ".join(q.split())[: settings.max_query_chars]

    def retrieve(self, query: str) -> List[Dict]:
        vec = self.embeddings.embed_query(query)
        hits = self.store.search(vec, settings.retrieval_top_k)
        self.logger.info("Retrieved %d chunks (top score=%.3f)",
                         len(hits), hits[0]["score"] if hits else 0)
        return hits

    def _is_relevant(self, hits: List[Dict]) -> bool:
        return bool(hits) and hits[0]["score"] >= settings.relevance_threshold

    def _build_messages(
        self,
        query: str,
        context: str,
        chat_history: List[Dict],
        is_followup: bool,
    ) -> list:
        """Build the messages list for the LLM."""
        if is_followup and chat_history:
            # Find last assistant answer to include as context for the follow-up
            prev = ""
            for msg in reversed(chat_history):
                if msg["role"] == "assistant":
                    prev = msg["content"]
                    break
            followup_prompt = (
                f"The user is asking a follow-up question.\n\n"
                f"Previous answer:\n{prev}\n\n"
                f"Follow-up question: {query}\n\n"
                f"Using ONLY the document context below, explain or clarify. "
                f"Do not use information outside the document.\n\n"
                f"Document context:\n{context}"
            )
            user_prompt = followup_prompt
        else:
            user_prompt = build_user_prompt(query, context)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(chat_history[-4:])
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def stream_answer(
        self, query: str, chat_history: List[Dict] | None = None
    ) -> Tuple[Generator[str, None, None], List[Dict]]:

        chat_history = chat_history or []
        safe_query = self._sanitize(query)

        # ── Follow-up rewriting for retrieval ─────────────────────────────────
        is_follow = _is_followup(safe_query)
        retrieval_query = (
            _rewrite_followup(safe_query, chat_history) if is_follow else safe_query
        )

        retrieved = self.retrieve(retrieval_query)

        # Second chance with original query if rewrite didn't help
        if not self._is_relevant(retrieved) and retrieval_query != safe_query:
            retrieved = self.retrieve(safe_query)

        # For follow-ups, lower the threshold so context is always passed to LLM
        if not self._is_relevant(retrieved) and not is_follow:
            def empty_stream() -> Generator[str, None, None]:
                yield FALLBACK_RESPONSE
            return empty_stream(), []

        if not retrieved:
            def empty_stream() -> Generator[str, None, None]:
                yield FALLBACK_RESPONSE
            return empty_stream(), []

        context_block, source_ids = build_context_block(retrieved)
        context = truncate_context_block(context_block, settings.llm_max_context_chars)
        sources = [c for c in retrieved if c["chunk_id"] in set(source_ids)]
        messages = self._build_messages(safe_query, context, chat_history, is_follow)

        # ── Mode: Extractive / Groq (USE_LLM=false) ───────────────────────────
        if not settings.use_llm:
            self.logger.info("Extractive mode: calling Groq")

            def groq_stream() -> Generator[str, None, None]:
                try:
                    tokens = list(_groq_answer(messages, self.logger))
                    if tokens:
                        yield from tokens
                        return
                except Exception as exc:
                    self.logger.warning("Groq failed (%s); falling back to extractive", exc)
                # Pure extractive fallback
                yield from stream_answer_text(
                    build_extractive_answer(safe_query, retrieved)
                )

            return groq_stream(), sources

        # ── Mode: Local GGUF ───────────────────────────────────────────────────
        if _gguf_available():
            self.logger.info("Using local GGUF at %s", settings.local_gguf_path)
            user_prompt = build_user_prompt(safe_query, context)

            def gguf_stream() -> Generator[str, None, None]:
                try:
                    from src.llm_gguf import stream_generate
                    yield from stream_generate(SYSTEM_PROMPT, user_prompt)
                except Exception as exc:
                    self.logger.warning("GGUF failed (%s); extractive fallback", exc)
                    yield from stream_answer_text(
                        build_extractive_answer(safe_query, retrieved)
                    )

            return gguf_stream(), sources

        # ── Mode: Ollama ───────────────────────────────────────────────────────
        if _ollama_reachable():
            self.logger.info("Using Ollama model %s", settings.llm_model_name)

            def ollama_stream() -> Generator[str, None, None]:
                try:
                    import ollama  # lazy — not on Streamlit Cloud
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
                    self.logger.warning("Ollama failed (%s); extractive fallback", exc)
                    yield from stream_answer_text(
                        build_extractive_answer(safe_query, retrieved)
                    )

            return ollama_stream(), sources

        # ── Final extractive fallback ──────────────────────────────────────────
        self.logger.info("No LLM available; pure extractive answer")

        def extractive_fallback() -> Generator[str, None, None]:
            yield from stream_answer_text(build_extractive_answer(safe_query, retrieved))

        return extractive_fallback(), sources
