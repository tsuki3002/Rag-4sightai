"""Embeddings (local fastembed) + vector store (Qdrant, embedded on disk or remote).

Each chunk is stored with two vectors: a dense one (meaning) and a sparse BM25 one (exact keywords). Hybrid search
runs both and merges the rankings with reciprocal rank fusion, so "iloc" or "F1 score" match even when the dense
embedding of a 2-word query is weak.
"""
import uuid
from dataclasses import dataclass
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from .config import ROOT, get_settings

DENSE, SPARSE = "dense", "bm25"
RRF_K = 60  # standard reciprocal-rank-fusion constant


class IndexFormatError(RuntimeError):
    pass


@dataclass
class Hit:
    text: str
    score: float                 # final relevance used for ranking (rerank score if reranked, else fused/dense)
    course_id: str
    doc_id: str
    doc_title: str
    source_path: str
    page: int | None
    section: str | None
    source_url: str | None = None
    license: str | None = None
    dense_score: float | None = None   # cosine similarity, when the chunk came up in the dense search
    rerank_score: float | None = None

    @property
    def headed_text(self) -> str:
        return _headed(self.doc_title, self.section, self.text)


def _headed(title: str, section: str | None, text: str) -> str:
    return " — ".join(x for x in (title, section) if x) + "\n" + text


def _models_dir() -> str:
    return str(ROOT / "storage" / "models")


@lru_cache
def embedder() -> TextEmbedding:
    return TextEmbedding(get_settings().embedding_model, cache_dir=_models_dir())


@lru_cache
def sparse_embedder() -> SparseTextEmbedding:
    return SparseTextEmbedding(get_settings().sparse_model, cache_dir=_models_dir())


@lru_cache
def client() -> QdrantClient:
    s = get_settings()
    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
    return QdrantClient(path=str(ROOT / s.qdrant_path))


def ensure_collection() -> None:
    s = get_settings()
    if client().collection_exists(s.collection):
        params = client().get_collection(s.collection).config.params
        if not isinstance(params.vectors, dict) or SPARSE not in (params.sparse_vectors or {}):
            raise IndexFormatError("The index was built by an older version. Rebuild it with:  "
                                   "python -m scripts.ingest --all --rebuild")
        return
    dim = len(next(iter(embedder().query_embed("dim probe"))))
    client().create_collection(
        s.collection,
        vectors_config={DENSE: models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for field in ("course_id", "doc_id"):
        client().create_payload_index(s.collection, field, models.PayloadSchemaType.KEYWORD)


def drop_collection() -> None:
    if client().collection_exists(get_settings().collection):
        client().delete_collection(get_settings().collection)


def _match(**fields) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key=k, match=models.MatchValue(value=v))
                               for k, v in fields.items()])


def doc_hash(doc_id: str) -> str | None:
    """File hash stored at last ingest, or None if the doc isn't indexed."""
    points, _ = client().scroll(get_settings().collection, scroll_filter=_match(doc_id=doc_id),
                                limit=1, with_payload=["file_hash"])
    return points[0].payload["file_hash"] if points else None


def indexed_doc_ids(course_id: str) -> set[str]:
    ids, offset = set(), None
    while True:
        points, offset = client().scroll(get_settings().collection, scroll_filter=_match(course_id=course_id),
                                         limit=512, offset=offset, with_payload=["doc_id"])
        ids.update(p.payload["doc_id"] for p in points)
        if offset is None:
            return ids


def delete_doc(doc_id: str) -> None:
    client().delete(get_settings().collection, points_selector=models.FilterSelector(filter=_match(doc_id=doc_id)))


def _sparse(vec) -> models.SparseVector:
    return models.SparseVector(indices=vec.indices.tolist(), values=vec.values.tolist())


def upsert_chunks(texts: list[str], payloads: list[dict]) -> None:
    # index each chunk with its document title + section so short or context-free chunks still match
    headed = [_headed(p["doc_title"], p.get("section"), t) for t, p in zip(texts, payloads)]
    dense = embedder().passage_embed(headed)
    sparse = sparse_embedder().passage_embed(headed)
    points = [
        models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{p['doc_id']}#{p['chunk_index']}")),
            vector={DENSE: d.tolist(), SPARSE: _sparse(sv)},
            payload={**p, "text": t},
        )
        for t, p, d, sv in zip(texts, payloads, dense, sparse)
    ]
    for i in range(0, len(points), 256):
        client().upsert(get_settings().collection, points=points[i:i + 256])


def _to_hit(p, score: float, dense_score: float | None) -> Hit:
    pl = p.payload
    return Hit(text=pl["text"], score=score, course_id=pl["course_id"], doc_id=pl["doc_id"],
               doc_title=pl["doc_title"], source_path=pl["source_path"], page=pl.get("page"),
               section=pl.get("section"), source_url=pl.get("source_url"), license=pl.get("license"),
               dense_score=dense_score)


def search(query: str, course_ids: list[str] | None, limit: int, mode: str = "hybrid") -> list[Hit]:
    """Candidate chunks for `query`, best first. mode: "dense" (meaning only) or "hybrid" (meaning + keywords)."""
    s = get_settings()
    flt = None
    if course_ids:
        flt = models.Filter(must=[models.FieldCondition(key="course_id", match=models.MatchAny(any=course_ids))])

    dense_vec = next(iter(embedder().query_embed(query))).tolist()
    dense = client().query_points(s.collection, query=dense_vec, using=DENSE, query_filter=flt,
                                  limit=limit, with_payload=True).points
    if mode == "dense":
        return [_to_hit(p, p.score, p.score) for p in dense]

    sparse_vec = _sparse(next(iter(sparse_embedder().query_embed(query))))
    sparse = client().query_points(s.collection, query=sparse_vec, using=SPARSE, query_filter=flt,
                                   limit=limit, with_payload=True).points

    # reciprocal rank fusion: rewards chunks ranked highly by either search, robust to their different scales
    fused: dict = {}
    for ranking in (dense, sparse):
        for rank, p in enumerate(ranking, start=1):
            entry = fused.setdefault(p.id, {"point": p, "rrf": 0.0, "dense": None})
            entry["rrf"] += 1 / (RRF_K + rank)
    for p in dense:
        fused[p.id]["dense"] = p.score
    ranked = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:limit]
    return [_to_hit(e["point"], e["rrf"], e["dense"]) for e in ranked]
