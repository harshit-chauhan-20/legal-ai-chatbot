from typing import Dict, List, Tuple


SYSTEM_PROMPT = """You are a legal assistant chatbot.
Your job is to answer strictly and only from the provided context chunks.

Rules you MUST follow:
1) Do not use external knowledge.
2) Do not infer facts not stated in context.
3) If the answer is missing, unclear, or out of scope, respond exactly:
"I could not find this information in the provided document."
4) If context has conflicting statements, explicitly mention the conflict and cite both sources.
5) Keep response concise, factual, and legally neutral.
"""


def build_context_block(retrieved_chunks: List[Dict]) -> Tuple[str, List[str]]:
    lines = []
    source_ids = []
    for chunk in retrieved_chunks:
        source_ids.append(chunk["chunk_id"])
        lines.append(f"[{chunk['chunk_id']}] {chunk['text']}")
    return "\n\n".join(lines), source_ids


def build_user_prompt(user_query: str, context_block: str) -> str:
    return f"""Context chunks:
{context_block}

User query:
{user_query}

Output format:
Answer: <grounded answer or fallback sentence>
Sources: <comma separated chunk IDs used>
"""


def truncate_context_block(context_block: str, max_chars: int) -> str:
    if len(context_block) <= max_chars:
        return context_block
    return context_block[: max_chars - 30].rstrip() + "\n[... context truncated ...]"
