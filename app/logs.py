"""Interaction log + student feedback, in a small SQLite file (storage/logs.db).

Every answered question is recorded with what was retrieved and what was cited, so bad answers can be traced to
their cause (wrong chunks retrieved vs. bad generation) and turned into new eval questions.
Disable with LOG_INTERACTIONS=false.
"""
import json
import sqlite3
import time
import uuid
from contextlib import closing

from .config import ROOT, get_settings

DB = ROOT / "storage" / "logs.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    course_ids TEXT,
    question TEXT NOT NULL,
    search_query TEXT,
    history_len INTEGER,
    retrieved TEXT,          -- JSON: [{doc_title, section, score, dense_score, rerank_score}]
    answer TEXT,
    cited TEXT,              -- JSON: [doc_title, ...]
    latency_ms INTEGER,
    model TEXT,
    error TEXT,
    rating INTEGER,          -- +1 helpful / -1 not helpful
    comment TEXT
);
CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions(ts);
"""


def _connect() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.executescript(SCHEMA)
    return conn


def record(*, question: str, course_ids: list[str] | None, history_len: int, result: dict | None,
           hits: list, latency_ms: int, error: str | None = None) -> str | None:
    if not get_settings().log_interactions:
        return None
    interaction_id = uuid.uuid4().hex
    retrieved = [{"doc_title": h.doc_title, "section": h.section, "score": round(h.score, 4),
                  "dense_score": h.dense_score and round(h.dense_score, 4),
                  "rerank_score": h.rerank_score and round(h.rerank_score, 4)} for h in hits]
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO interactions (id, ts, course_ids, question, search_query, history_len, retrieved, answer,"
            " cited, latency_ms, model, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (interaction_id, time.time(), json.dumps(course_ids), question,
             result and result["search_query"], history_len, json.dumps(retrieved),
             result and result["answer"], result and json.dumps([c["doc_title"] for c in result["citations"]]),
             latency_ms, get_settings().llm_model, error),
        )
    return interaction_id


def set_feedback(interaction_id: str, rating: int, comment: str | None) -> bool:
    with closing(_connect()) as conn, conn:
        cur = conn.execute("UPDATE interactions SET rating = ?, comment = ? WHERE id = ?",
                           (rating, comment, interaction_id))
        return cur.rowcount == 1


def recent(rating: int | None = None, limit: int = 50) -> list[dict]:
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        where, args = ("WHERE rating = ?", [rating]) if rating is not None else ("", [])
        rows = conn.execute(f"SELECT * FROM interactions {where} ORDER BY ts DESC LIMIT ?", [*args, limit])
        return [dict(r) for r in rows]
