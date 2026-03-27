"""Local GGUF inference via llama-cpp-python (Q4_K_M file on disk; no Ollama)."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Generator, Optional

from config import settings
from src.logging_utils import setup_logger

logger = setup_logger("llm_gguf", settings.log_level)

_llm_lock = threading.Lock()
_llm = None


def get_gguf_path() -> Optional[Path]:
    p = Path(settings.local_gguf_path)
    return p if p.is_file() else None


def load_llm():
    global _llm
    path = get_gguf_path()
    if not path:
        return None
    with _llm_lock:
        if _llm is None:
            from llama_cpp import Llama

            logger.info("Loading GGUF with llama-cpp-python: %s", path)
            # Match model train context (TinyLlama: 2048); avoid overflow garbage.
            _llm = Llama(
                model_path=str(path),
                n_ctx=2048,
                n_batch=256,
                n_gpu_layers=0,
                verbose=False,
            )
        return _llm


def stream_generate(system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
    llm = load_llm()
    if llm is None:
        raise RuntimeError("GGUF path missing")
    stream = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=384,
        repeat_penalty=1.12,
        top_p=0.9,
        stream=True,
    )
    for chunk in stream:
        delta = chunk["choices"][0].get("delta") or {}
        piece = delta.get("content") or ""
        if piece:
            yield piece
