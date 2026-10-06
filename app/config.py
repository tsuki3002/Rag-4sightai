from functools import lru_cache
from typing import Literal
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    rewrite_model: str | None = None   # model that turns follow-ups into standalone questions (default: llm_model)
    history_messages: int = 6          # how many recent chat messages are sent with each question

    # Images: local OCR always runs; set VISION_MODEL to also get an LLM description of diagrams/charts
    vision_model: str | None = None
    vision_base_url: str | None = None  # default: llm_base_url
    vision_api_key: str | None = None   # default: llm_api_key

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    qdrant_path: str = "storage/qdrant"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    collection: str = "course_chunks"

    sparse_model: str = "Qdrant/bm25"

    chunk_size_words: int = 220
    chunk_overlap_words: int = 40

    search_mode: Literal["hybrid", "dense"] = "hybrid"  # meaning + keywords, or meaning only
    rerank_model: str | None = "Xenova/ms-marco-MiniLM-L-6-v2"  # empty = no reranking
    rerank_candidates: int = 12          # chunks fetched by search and re-scored by the reranker
    rerank_max_words: int = 128          # reranker reads this much of each chunk (cost grows fast with length)
    min_rerank_score: float = -7.0       # reranker relevance cutoff ("not in the course" below this); see eval calibration
    top_k: int = 5                       # chunks given to the LLM
    max_chunks_per_doc: int = 2          # keeps answers from leaning on a single document
    min_score: float = 0.62              # dense-similarity cutoff, only used when reranking is off

    courses_file: str = "config/courses.yaml"
    admin_token: str | None = None  # enables POST /courses/{id}/documents
    log_interactions: bool = True   # record questions, retrieved chunks and answers in storage/logs.db


class Source(BaseModel):
    """A web document to download into the course folder (see scripts/fetch_sources.py)."""
    url: str
    title: str | None = None
    license: str | None = None


class Course(BaseModel):
    id: str
    name: str
    description: str = ""
    docs_dir: str
    sources: list["Source"] = []

    @property
    def path(self) -> Path:
        return ROOT / self.docs_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_courses() -> dict[str, Course]:
    data = yaml.safe_load((ROOT / get_settings().courses_file).read_text()) or {}
    courses = {}
    for c in data.get("courses") or []:
        c["sources"] = [{"url": src} if isinstance(src, str) else src for src in c.get("sources") or []]
        courses[c["id"]] = Course(**c)
    return courses
