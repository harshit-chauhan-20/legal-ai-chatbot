import os
from dataclasses import dataclass
from pathlib import Path

# Optional dotenv (safe for local, ignored in cloud)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    # ✅ FIXED: No Windows absolute path (breaks on Streamlit Cloud)
    raw_pdf_path: Path = Path(
        os.getenv(
            "RAW_PDF_PATH",
            str(BASE_DIR / "data" / "raw" / "AI_Training_Document.pdf"),
        )
    )

    processed_json_path: Path = BASE_DIR / "data" / "processed" / "chunks.json"
    vector_store_dir: Path = BASE_DIR / "data" / "processed" / "faiss_store"

    collection_name: str = os.getenv("VECTOR_COLLECTION_NAME", "legal_doc_chunks")

    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # ⚠️ Ollama won't work on Streamlit Cloud → keep fallback ready
    llm_model_name: str = os.getenv(
        "LLM_MODEL_NAME",
        "qwen2.5:0.5b-instruct",
    )

    chunk_min_words: int = int(os.getenv("CHUNK_MIN_WORDS", "100"))
    chunk_max_words: int = int(os.getenv("CHUNK_MAX_WORDS", "300"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.28"))
    max_query_chars: int = int(os.getenv("MAX_QUERY_CHARS", "2000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    use_llm: bool = _env_bool("USE_LLM", "true")

    # ❌ FIXED: Disable local GGUF for cloud (no local files allowed)
    disable_local_gguf: bool = _env_bool("DISABLE_LOCAL_GGUF", "true")

    local_gguf_path: Path = Path(
        os.getenv(
            "LOCAL_GGUF_PATH",
            str(BASE_DIR / "models" / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"),
        )
    )

    llm_max_context_chars: int = int(os.getenv("LLM_MAX_CONTEXT_CHARS", "2800"))


settings = Settings()


def generator_display_name() -> str:
    """What the UI should show for the text generator."""
    if settings.disable_local_gguf:
        if not settings.use_llm:
            return "Extractive only (grounded)"
        return "Cloud mode (LLM fallback / extractive)"

    if settings.local_gguf_path.is_file():
        return f"Local GGUF (Q4): {settings.local_gguf_path.name}"

    return f"Ollama: {settings.llm_model_name} (or extractive if offline)"
