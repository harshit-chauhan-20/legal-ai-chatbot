"""
Grounded answers without an LLM: pick sentences from retrieved chunks that best
overlap the query. No paraphrasing — only document text — so hallucination risk is minimal.
"""

from __future__ import annotations

import re
from typing import Dict, Generator, List, Tuple

# Minimal English stopwords to improve overlap scoring
_STOP = frozenset(
    "a an the and or but if in on at to for of as is are was were be been being "
    "it its this that these those with from by not no yes do does did will would "
    "can could should may might must shall about into than then so such what which "
    "who whom whose how when where why".split()
)


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if len(w) > 2 and w not in _STOP}


def split_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.strip() for p in parts if p.strip()]


def build_extractive_answer(
    query: str,
    chunks: List[Dict],
    max_sentences: int = 7,
    max_answer_words: int = 320,
) -> str:
    """
    Returns markdown text citing chunk ids. Uses overlap scoring; if weak overlap,
    falls back to the leading sentences of the top-retrieved chunk.
    """
    if not chunks:
        return ""

    q_tokens = _tokens(query)
    if not q_tokens:
        q_tokens = set(re.findall(r"\w+", query.lower()))

    scored: List[Tuple[float, str, str]] = []
    for ch in chunks:
        cid = ch["chunk_id"]
        base = float(ch.get("score", 0.0))
        for sent in split_sentences(ch["text"]):
            if len(sent) < 25:
                continue
            st = _tokens(sent)
            overlap = len(q_tokens & st) if q_tokens else 0
            score = overlap * 2.0 + base * 0.15 + min(len(sent), 400) / 4000.0
            scored.append((score, sent, cid))

    scored.sort(key=lambda x: x[0], reverse=True)

    seen_norm: set[str] = set()
    picked: List[Tuple[str, str]] = []
    for _score, sent, cid in scored:
        key = re.sub(r"\s+", " ", sent.lower())[:120]
        if key in seen_norm:
            continue
        seen_norm.add(key)
        picked.append((sent, cid))
        if len(picked) >= max_sentences:
            break

    if not picked:
        top = chunks[0]
        for sent in split_sentences(top["text"])[:4]:
            if len(sent) >= 20:
                picked.append((sent, top["chunk_id"]))

    # Produce a concise summary instead of raw chunk dumps.
    source_ids = sorted({cid for _sent, cid in picked})
    summary_sentences = [sent for sent, _cid in picked][:5]
    answer = " ".join(summary_sentences)
    if not answer:
        answer = "I could not find this information in the provided document."

    return f"{answer}\n\nSources: {', '.join(source_ids)}"


def stream_answer_text(text: str) -> Generator[str, None, None]:
    """Token-ish streaming for Streamlit (small chunks read better than char-by-char)."""
    if not text:
        yield ""
        return
    step = 48
    for i in range(0, len(text), step):
        yield text[i : i + step]
