import json
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import openai
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import llm, logs, rag, store
from .config import ROOT, get_settings, load_courses
from .ingest import ingest_course
from .loaders import SUPPORTED


@asynccontextmanager
async def lifespan(_app: FastAPI):
    store.ensure_collection()  # fails fast with a clear message if the index needs rebuilding
    yield
    store.client().close()


app = FastAPI(title="Course RAG", lifespan=lifespan)
STATIC = Path(__file__).parent / "static"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20000)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    course_ids: list[str] | None = None  # None / empty = search all courses
    history: list[ChatMessage] = Field(default=[], max_length=50)  # earlier turns, oldest first


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/courses")
def courses():
    return [c.model_dump(exclude={"docs_dir"}) for c in load_courses().values()]


def _check_courses(course_ids: list[str] | None) -> None:
    known = load_courses()
    unknown = [c for c in course_ids or [] if c not in known]
    if unknown:
        raise HTTPException(404, f"Unknown course(s): {', '.join(unknown)}")


def _llm_error(e: Exception) -> tuple[int, str]:
    if isinstance(e, openai.AuthenticationError):
        return 502, "LLM API key is missing or invalid — check LLM_API_KEY in .env"
    if isinstance(e, openai.RateLimitError):
        return 429, "The LLM provider's rate limit was hit — wait a moment and try again"
    if isinstance(e, openai.APIError):
        return 502, f"LLM provider error: {e}"
    return 500, "Unexpected error while answering"


def _require_admin(token: str | None) -> None:
    expected = get_settings().admin_token
    if not expected or token != expected:
        raise HTTPException(403, "Requires a valid X-Admin-Token header (set ADMIN_TOKEN in .env)")


async def _prepare(req: AskRequest) -> rag.Prepared:
    _check_courses(req.course_ids)
    try:
        return await run_in_threadpool(rag.prepare, req.question, req.course_ids or None,
                                       [m.model_dump() for m in req.history])
    except openai.APIError as e:
        raise HTTPException(*_llm_error(e))


def _log(req: AskRequest, prep: rag.Prepared, result: dict | None, started: float, error: str | None = None):
    return logs.record(question=req.question, course_ids=req.course_ids, history_len=len(req.history),
                       result=result, hits=prep.hits, latency_ms=int((time.perf_counter() - started) * 1000),
                       error=error)


@app.post("/ask")
async def ask(req: AskRequest):
    started = time.perf_counter()
    prep = await _prepare(req)
    try:
        text = rag.NOT_FOUND if prep.messages is None else await run_in_threadpool(
            llm.chat, prep.messages, temperature=0.1)
    except openai.APIError as e:
        _log(req, prep, None, started, error=str(e))
        raise HTTPException(*_llm_error(e))
    result = rag.finish(prep, text)
    return {**result, "id": _log(req, prep, result, started)}


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """Server-sent events: {"type": "meta"} then {"type": "delta", "text"}... then {"type": "done", ...result, id}
    (or {"type": "error", "message"} if generation fails part-way)."""
    started = time.perf_counter()
    prep = await _prepare(req)

    def events():
        def sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        yield sse({"type": "meta", "search_query": prep.search_query})
        pieces = []
        try:
            for piece in rag.answer_stream(prep):
                pieces.append(piece)
                yield sse({"type": "delta", "text": piece})
        except Exception as e:
            _log(req, prep, None, started, error=str(e))
            yield sse({"type": "error", "message": _llm_error(e)[1]})
            return
        result = rag.finish(prep, "".join(pieces))
        yield sse({"type": "done", **result, "id": _log(req, prep, result, started)})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class Feedback(BaseModel):
    id: str = Field(min_length=8, max_length=64)
    rating: Literal[1, -1]
    comment: str | None = Field(default=None, max_length=2000)


@app.post("/feedback")
def feedback(fb: Feedback):
    if not logs.set_feedback(fb.id, fb.rating, fb.comment):
        raise HTTPException(404, "Unknown interaction id")
    return {"ok": True}


@app.get("/admin/interactions")
def interactions(rating: Literal[1, -1] | None = None, limit: int = 50,
                 x_admin_token: str | None = Header(None)):
    """Recent questions with what was retrieved and cited, e.g. ?rating=-1 for answers students marked unhelpful."""
    _require_admin(x_admin_token)
    return logs.recent(rating, min(limit, 500))


@app.get("/files/{path:path}")
def source_file(path: str):
    """Serve a cited source document (only files inside a registered course folder)."""
    target = (ROOT / path).resolve()
    in_course = any(target.is_relative_to(c.path.resolve()) for c in load_courses().values())
    if not in_course or not target.is_file() or target.suffix.lower() not in SUPPORTED:
        raise HTTPException(404)
    return FileResponse(target)


@app.post("/courses/{course_id}/documents")
async def upload(course_id: str, file: UploadFile = File(...), x_admin_token: str | None = Header(None)):
    """Add a document to a course and index it. Disabled unless ADMIN_TOKEN is set in the environment."""
    _require_admin(x_admin_token)
    course = load_courses().get(course_id)
    if not course:
        raise HTTPException(404, f"Unknown course: {course_id}")
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(400, f"Supported types: {', '.join(sorted(SUPPORTED))}")

    course.path.mkdir(parents=True, exist_ok=True)
    with open(course.path / name, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return await run_in_threadpool(ingest_course, course)
