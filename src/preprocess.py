import re
from collections import Counter
from pathlib import Path
from typing import List

from pypdf import PdfReader


def extract_pdf_pages(pdf_path: Path) -> List[str]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pages(raw_pages: List[str]) -> List[str]:
    cleaned = []
    line_counter: Counter[str] = Counter()
    split_pages: List[List[str]] = []

    for page in raw_pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        split_pages.append(lines)
        line_counter.update(lines)

    repeated_noise = {
        line
        for line, count in line_counter.items()
        if count >= max(3, int(0.6 * len(raw_pages))) and len(line.split()) <= 12
    }

    for lines in split_pages:
        filtered = []
        for line in lines:
            if line in repeated_noise:
                continue
            if re.fullmatch(r"page\s+\d+(\s+of\s+\d+)?", line, flags=re.I):
                continue
            filtered.append(line)
        cleaned.append(_normalize_whitespace("\n".join(filtered)))

    return cleaned


def clean_document(pdf_path: Path) -> str:
    pages = extract_pdf_pages(pdf_path)
    cleaned_pages = clean_pages(pages)
    text = "\n\n".join(page for page in cleaned_pages if page)
    return _normalize_whitespace(text)
