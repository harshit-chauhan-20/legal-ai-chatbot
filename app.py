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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System")
    st.write(f"**Embedding:** `{settings.embedding_model_name}`")
    st.write(f"**Indexed chunks:** `{st.session_state.pipeline.store.count()}`")

    # ── Groq status badge ─────────────────────────────────────────────────────
    groq_key_set = bool(os.getenv("GROQ_API_KEY", ""))
    if groq_key_set:
        st.success("✅ Groq API active — LLM answers enabled", icon="🤖")
    else:
        st.warning("⚠️ GROQ_API_KEY not set — falling back to pure extractive", icon="🔑")

    st.divider()

    # ── Answer Mode selector ──────────────────────────────────────────────────
    st.subheader("🤖 Answer Mode")
    mode = st.radio(
        "Choose how answers are generated:",
        options=["Extractive (Grounded)", "Ollama (Local LLM)", "Local GGUF"],
        index=0,
        help=(
            "Extractive (Grounded): Uses Groq LLM (cloud) to generate a "
            "natural answer strictly from retrieved document chunks.\n\n"
            "Ollama (Local LLM): Runs a local Ollama model — only works "
            "when deployed on your own machine.\n\n"
            "Local GGUF: Runs a quantized GGUF model file — only works "
            "when deployed locally with llama-cpp-python installed."
        ),
    )

    if mode == "Extractive (Grounded)":
        if groq_key_set:
            st.info(
                "🤖 **Groq LLM** will generate a fluent, grounded answer "
                "using retrieved chunks from your document. "
                "No information outside the document will be used.",
                icon="✅",
            )
        else:
            st.info(
                "📄 **Pure extractive mode** — relevant sentences are pulled "
                "directly from the document without an LLM. "
                "Add `GROQ_API_KEY` to Streamlit Secrets to enable LLM answers.",
                icon="ℹ️",
            )

    elif mode == "Ollama (Local LLM)":
        st.info(
            "🖥️ **Ollama (Local LLM)** — this mode is implemented in the code "
            "and works when you run the app locally on your own machine.\n\n"
            "**To use locally:**\n"
            "1. Install Ollama from [ollama.com](https://ollama.com)\n"
            "2. Run `ollama pull qwen2.5:0.5b-instruct`\n"
            "3. Run `ollama serve`\n"
            "4. Then start the app with `streamlit run app.py`\n\n"
            "⚠️ On Streamlit Cloud, Ollama cannot run — the app will "
            "automatically fall back to extractive mode.",
            icon="ℹ️",
        )

    elif mode == "Local GGUF":
        st.warning(
            "📦 **Local GGUF** — this mode is implemented in the code "
            "and works when you run the app locally on your own machine.\n\n"
            "**To use locally:**\n"
            "1. Download a `.gguf` model file (e.g. TinyLlama Q4)\n"
            "2. Place it at `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`\n"
            "3. Install: `pip install llama-cpp-python`\n"
            "4. Then start the app with `streamlit run app.py`\n\n"
            "🚫 **Not supported on Streamlit Cloud** — `llama-cpp-python` "
            "has no prebuilt wheels for this environment.",
            icon="🚫",
        )

    # Map UI choice → env flags consumed by the pipeline
    os.environ["USE_LLM"] = "false" if mode == "Extractive (Grounded)" else "true"
    os.environ["DISABLE_LOCAL_GGUF"] = "false" if mode == "Local GGUF" else "true"

    st.divider()
    st.write(f"**Active generator:** {generator_display_name()}")

    if st.button("🗑️ Reset chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
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
