"""Index a course's documents. Re-running is cheap: unchanged files are skipped, edited files are
re-indexed, and files removed from the folder are dropped from the index."""
import hashlib
import json
import logging
from functools import cache
from pathlib import Path

from . import store
from .chunking import chunk_units
from .config import ROOT, Course, get_settings
from .loaders import SUPPORTED, load

log = logging.getLogger(__name__)


@cache
def _manifest(folder: Path) -> dict:
    """Metadata written by scripts/fetch_sources.py for downloaded web documents."""
    f = folder / "_manifest.json"
    return json.loads(f.read_text()) if f.exists() else {}


def ingest_course(course: Course, force: bool = False) -> dict:
    s = get_settings()
    store.ensure_collection()
    course.path.mkdir(parents=True, exist_ok=True)

    _manifest.cache_clear()
    stats = {"indexed": 0, "skipped": 0, "removed": 0, "chunks": 0}
    seen: set[str] = set()

    for path in sorted(p for p in course.path.rglob("*") if p.suffix.lower() in SUPPORTED):
        rel = path.relative_to(course.path).as_posix()
        doc_id = f"{course.id}/{rel}"
        seen.add(doc_id)
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()

        if not force and store.doc_hash(doc_id) == file_hash:
            stats["skipped"] += 1
            continue

        chunks = chunk_units(load(path), s.chunk_size_words, s.chunk_overlap_words)
        store.delete_doc(doc_id)
        if not chunks:
            log.warning("No text extracted from %s", rel)
            continue
        meta = _manifest(path.parent).get(path.name, {})
        payloads = [
            {"course_id": course.id, "doc_id": doc_id,
             "doc_title": meta.get("title") or path.stem.replace("_", " ").replace("-", " "),
             "source_path": path.relative_to(ROOT).as_posix(), "source_url": meta.get("url"),
             "license": meta.get("license"), "file_hash": file_hash,
             "chunk_index": i, "page": c.page, "section": c.section}
            for i, c in enumerate(chunks)
        ]
        store.upsert_chunks([c.text for c in chunks], payloads)
        stats["indexed"] += 1
        stats["chunks"] += len(chunks)
        log.info("Indexed %s (%d chunks)", rel, len(chunks))

    for stale in store.indexed_doc_ids(course.id) - seen:
        store.delete_doc(stale)
        stats["removed"] += 1
        log.info("Removed %s (file no longer present)", stale)

    return stats
