import re

from .loaders import Unit

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_units(units: list[Unit], size: int, overlap: int) -> list[Unit]:
    """Split each unit into ~`size`-word chunks with `overlap` words of carry-over.

    Chunks never cross a page/section boundary, so every chunk keeps an exact citation location.
    Splits prefer sentence boundaries so chunks don't end mid-thought.
    """
    chunks: list[Unit] = []
    for unit in units:
        text = re.sub(r"[ \t]+", " ", unit.text).strip()
        if not text:
            continue
        sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
        current: list[str] = []
        for sentence in sentences:
            words = sentence.split()
            if current and len(current) + len(words) > size:
                chunks.append(Unit(" ".join(current), unit.page, unit.section))
                current = current[-overlap:] if overlap else []
            current.extend(words)
            # a single giant "sentence" (tables, code) still gets split
            while len(current) > size:
                chunks.append(Unit(" ".join(current[:size]), unit.page, unit.section))
                current = current[size - overlap:]
        if current:
            chunks.append(Unit(" ".join(current), unit.page, unit.section))
    return chunks
