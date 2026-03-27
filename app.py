import streamlit as st

from config import generator_display_name, settings
from src.logging_utils import setup_logger
from src.rag_pipeline import RAGPipeline

logger = setup_logger("streamlit_app", settings.log_level)

st.set_page_config(page_title="Legal RAG Chatbot", page_icon="AI", layout="wide")
st.title("Legal RAG Chatbot (Grounded Answers)")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

with st.sidebar:
    st.header("System")
    st.write(f"**Generator:** {generator_display_name()}")
    st.write(f"**Embedding:** `{settings.embedding_model_name}`")
    st.write(
        f"**Indexed chunks:** `{st.session_state.pipeline.store.count()}`"
    )
    if st.button("Reset chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **{src['chunk_id']}** (score={src['score']:.3f})\n\n{src['text'][:500]}..."
                    )

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
                        f"- **{src['chunk_id']}** (score={src['score']:.3f})\n\n{src['text'][:500]}..."
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": streaming_answer, "sources": sources}
    )
