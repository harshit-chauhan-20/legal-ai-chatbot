import os

import streamlit as st


def _apply_streamlit_secrets_to_env() -> None:
    """Expose Streamlit Cloud secrets as env vars so `config` picks them up."""
    try:
        if hasattr(st, "secrets"):
            for key in st.secrets:
                val = st.secrets[key]
                if isinstance(val, str):
                    os.environ.setdefault(key, val)
    except Exception:
        pass


_apply_streamlit_secrets_to_env()

from config import generator_display_name, settings
from src.logging_utils import setup_logger
from src.rag_pipeline import RAGPipeline

logger = setup_logger("streamlit_app", settings.log_level)

st.set_page_config(page_title="Legal RAG Chatbot", page_icon="⚖️", layout="wide")
st.title("Legal RAG Chatbot (Grounded Answers)")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System")
    st.write(f"**Embedding:** `{settings.embedding_model_name}`")
    st.write(
        f"**Indexed chunks:** `{st.session_state.pipeline.store.count()}`"
    )

    st.divider()

    st.subheader("🤖 Answer Mode")
    mode = st.radio(
        "Choose how answers are generated:",
        options=["Extractive (Grounded)", "Ollama (Local LLM)", "Local GGUF"],
        index=0,
        help=(
            "Extractive: pulls sentences directly from the document.\n\n"
            "Ollama: uses a locally running Ollama LLM.\n\n"
            "Local GGUF: uses a quantized model file on disk."
        ),
    )

    if mode == "Ollama (Local LLM)":
        st.info(
            "🔌 **Ollama** requires a locally running Ollama server "
            "(`ollama serve`). This will not work on Streamlit Cloud — "
            "the app will fall back to Extractive mode automatically if "
            "Ollama is unreachable.",
            icon="ℹ️",
        )
    elif mode == "Local GGUF":
        st.warning(
            "⚠️ **Local GGUF / llama-cpp-python is not supported on "
            "Streamlit Cloud.** There are no prebuilt wheels for this "
            "environment. Use Extractive mode or run the app locally "
            "to use a GGUF model.",
            icon="🚫",
        )

    # Map UI choice → env-style flags for the pipeline
    os.environ["USE_LLM"] = "false" if mode == "Extractive (Grounded)" else "true"
    os.environ["DISABLE_LOCAL_GGUF"] = "false" if mode == "Local GGUF" else "true"

    st.divider()

    st.write(f"**Active generator:** {generator_display_name()}")

    if st.button("🗑️ Reset chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **{src['chunk_id']}** (score={src['score']:.3f})"
                        f"\n\n{src['text'][:500]}..."
                    )

# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask a question based on the provided legal document...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        answer_container = st.empty()
        streaming_answer = ""

        history_for_model = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m["role"] in {"user", "assistant"}
        ]

        stream, sources = st.session_state.pipeline.stream_answer(
            query, chat_history=history_for_model
        )
        for token in stream:
            streaming_answer += token
            answer_container.markdown(streaming_answer)

        if sources:
            with st.expander("Sources used"):
                for src in sources:
                    st.markdown(
                        f"- **{src['chunk_id']}** (score={src['score']:.3f})"
                        f"\n\n{src['text'][:500]}..."
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": streaming_answer, "sources": sources}
    )
