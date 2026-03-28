import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv()


def _env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes")


_DEFAULT_RAW_PDF = BASE_DIR / "data" / "raw" / "AI_Training_Document.pdf"
_DEFAULT_LOCAL_GGUF = BASE_DIR / "models" / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


@dataclass(frozen=True)
class Settings:
    raw_pdf_path: Path = Path(
        os.getenv("RAW_PDF_PATH", str(_DEFAULT_RAW_PDF))
    )
    processed_json_path: Path = BASE_DIR / "data" / "processed" / "chunks.json"
    vector_store_dir: Path = BASE_DIR / "data" / "processed" / "faiss_store"
    collection_name: str = os.getenv("VECTOR_COLLECTION_NAME", "legal_doc_chunks")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "qwen2.5:0.5b-instruct")
    # Free Hugging Face model for LangChain inference
    langchain_model_name: str = os.getenv("LANGCHAIN_MODEL_NAME", "google/flan-t5-small")
    chunk_min_words: int = int(os.getenv("CHUNK_MIN_WORDS", "100"))
    chunk_max_words: int = int(os.getenv("CHUNK_MAX_WORDS", "300"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.28"))
    max_query_chars: int = int(os.getenv("MAX_QUERY_CHARS", "2000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    use_llm: bool = _env_bool("USE_LLM", "true")
    # On Streamlit Cloud, set DISABLE_LOCAL_GGUF=true (no GGUF file / no wheel).
    disable_local_gguf: bool = _env_bool("DISABLE_LOCAL_GGUF", "false")
    local_gguf_path: Path = Path(
        os.getenv("LOCAL_GGUF_PATH", str(_DEFAULT_LOCAL_GGUF))
    )
    llm_max_context_chars: int = int(os.getenv("LLM_MAX_CONTEXT_CHARS", "2800"))


settings = Settings()


def generator_display_name() -> str:
    """What the UI should show for the text generator."""
    return f"LangChain: {settings.langchain_model_name}"
