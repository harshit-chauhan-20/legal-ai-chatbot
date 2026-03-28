import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    raw_pdf_path: Path = Path(
        os.getenv(
            "RAW_PDF_PATH",
            r"c:\Users\Chars\AppData\Roaming\Cursor\User\workspaceStorage\353102621f8acf2440669663e164abc2\pdfs\c4d52795-b268-49d8-b60b-cf0aa164f31d\AI Training Document (1).pdf",
        )
    )
    processed_json_path: Path = BASE_DIR / "data" / "processed" / "chunks.json"
    vector_store_dir: Path = BASE_DIR / "data" / "processed" / "faiss_store"
    collection_name: str = os.getenv("VECTOR_COLLECTION_NAME", "legal_doc_chunks")
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    # Smallest practical instruct model on Ollama (~0.5B). Override if you use TinyLlama etc.
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "qwen2.5:0.5b-instruct")
    # Free Hugging Face model for LangChain inference
    langchain_model_name: str = os.getenv("LANGCHAIN_MODEL_NAME", "google/flan-t5-small")
    chunk_min_words: int = int(os.getenv("CHUNK_MIN_WORDS", "100"))
    chunk_max_words: int = int(os.getenv("CHUNK_MAX_WORDS", "300"))
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "6"))
    # Cosine similarity via inner product on normalized vectors; ~0.25–0.45 is typical.
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.28"))
    max_query_chars: int = int(os.getenv("MAX_QUERY_CHARS", "2000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # If false, never call Ollama — only extractive answers from chunks.
    use_llm: bool = os.getenv("USE_LLM", "true").lower() in ("1", "true", "yes")
    # Q4_K_M GGUF on disk (ctransformers). Default: D: layout used on this machine.
    # TinyLlama 1.1B Chat Q4_K_M — works with ctransformers on CPU without Ollama.
    local_gguf_path: Path = Path(
        os.getenv(
            "LOCAL_GGUF_PATH",
            r"D:\rag_legal_run\models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        )
    )
    # Keep RAG context small so tiny GGUF models stay coherent (~1–2k tokens).
    llm_max_context_chars: int = int(os.getenv("LLM_MAX_CONTEXT_CHARS", "2800"))


settings = Settings()


def generator_display_name() -> str:
    """What the UI should show for the text generator."""
    return f"LangChain: {settings.langchain_model_name}"
