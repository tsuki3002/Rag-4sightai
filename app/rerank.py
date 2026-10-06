"""Cross-encoder reranking: reads the question and each candidate chunk together, which judges relevance far more
reliably than comparing two separately computed embeddings. Its score is also what decides "not in the course"."""
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from .config import ROOT, get_settings
from .store import Hit


@lru_cache
def _model() -> TextCrossEncoder:
    return TextCrossEncoder(get_settings().rerank_model, cache_dir=str(ROOT / "storage" / "models"))


def rerank(query: str, hits: list[Hit]) -> list[Hit]:
    if not hits:
        return hits
    limit = get_settings().rerank_max_words
    previews = [" ".join(h.headed_text.split()[:limit]) for h in hits]
    for h, score in zip(hits, _model().rerank(query, previews, batch_size=len(previews))):
        h.rerank_score = h.score = float(score)
    return sorted(hits, key=lambda h: h.score, reverse=True)
