import re
from typing import Dict, List


def split_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    # Simple sentence-aware splitting without heavy NLP dependencies.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", text)
    return [p.strip() for p in parts if p.strip()]


def word_count(text: str) -> int:
    return len(text.split())


def chunk_text(
    text: str,
    min_words: int = 100,
    max_words: int = 300,
) -> List[Dict]:
    sentences = split_sentences(text)
    chunks: List[Dict] = []
    buffer: List[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = word_count(sentence)
        if current_words + sentence_words > max_words and buffer:
            chunk_text_value = " ".join(buffer).strip()
            if word_count(chunk_text_value) >= min_words:
                chunks.append(chunk_text_value)
            buffer = [sentence]
            current_words = sentence_words
        else:
            buffer.append(sentence)
            current_words += sentence_words

    if buffer:
        tail = " ".join(buffer).strip()
        if chunks and word_count(tail) < min_words:
            chunks[-1] = f"{chunks[-1]} {tail}".strip()
        else:
            chunks.append(tail)

    results = []
    for idx, chunk in enumerate(chunks, start=1):
        results.append(
            {
                "chunk_id": f"chunk_{idx:04d}",
                "text": chunk,
                "word_count": word_count(chunk),
            }
        )
    return results
