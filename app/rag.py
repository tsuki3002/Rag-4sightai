import re
from dataclasses import asdict, dataclass

from . import llm, rerank, store
from .config import get_settings

NOT_FOUND = "I couldn't find this in the course materials. Try rephrasing, or ask your instructor."

SYSTEM_PROMPT = """You are a teaching assistant for an online course. Answer the student's question using ONLY \
the numbered course excerpts provided with their latest message, plus what was already established earlier in \
this conversation.

Rules:
- Cite every factual claim with the excerpt number in square brackets, e.g. [1] or [2][3]. Only cite excerpt \
numbers from the latest message; earlier turns' numbers no longer apply.
- If the excerpts don't contain the answer, say so plainly instead of guessing. You may say what related \
topics the excerpts do cover.
- Explain clearly for a student: short paragraphs, examples from the excerpts where helpful, markdown allowed.
- Some excerpts describe figures or images; use them like any other excerpt.
- Don't mention "excerpts" or "context" — refer to "the course material"."""

REWRITE_PROMPT = """Rewrite the student's latest message as a single standalone question that can be understood \
without the conversation, resolving pronouns and references like "it", "that", "the second one", "more \
examples". Keep the student's intent and wording where possible. If it is already standalone, return it \
unchanged. Output only the question, nothing else."""

NO_NEW_SOURCES = """(No course excerpts matched this message. If it can be answered from what was already said \
in this conversation, e.g. a summary, clarification or thanks, answer briefly without citations. Otherwise reply \
exactly: "{not_found}")"""


def _normalize_citations(text: str) -> str:
    """Models write citations in different styles ([[2]], [2, 5], 【1】); turn them all into [2][5]."""
    text = re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", text)
    text = re.sub(r"\[\[(\d+)\]\]", r"[\1]", text)
    return re.sub(r"\[(\d+(?:\s*[,;]\s*\d+)+)\]",
                  lambda m: "".join(f"[{n}]" for n in re.split(r"\s*[,;]\s*", m.group(1))), text)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " …"


def _recent(history: list[dict]) -> list[dict]:
    """Last N messages, trimmed. Old [n] markers are stripped so they can't be confused with this turn's sources."""
    msgs = history[-get_settings().history_messages:] if get_settings().history_messages else []
    return [{"role": m["role"], "content": _clip(re.sub(r"\[\d+\]", "", m["content"]), 1500)} for m in msgs]


def standalone_question(question: str, history: list[dict]) -> str:
    if not history:
        return question
    transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    rewritten = llm.chat(
        [{"role": "system", "content": REWRITE_PROMPT},
         {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {question}"}],
        model=get_settings().rewrite_model, temperature=0, max_tokens=1024,  # room for reasoning models
    ).strip().strip('"')
    return rewritten or question


def _format_sources(hits: list[store.Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        where = ", ".join(x for x in (h.doc_title, f"page {h.page}" if h.page else None, h.section) if x)
        text = re.sub(r"(?<![\w\])])\\?\[\d+\\?\]", "", h.text)  # stray footnote markers would look like our citations
        blocks.append(f"[{i}] ({where})\n{text}")
    return "\n\n".join(blocks)


def retrieve(query: str, course_ids: list[str] | None = None) -> list[store.Hit]:
    """The search half of the pipeline (shared by answer() and the eval harness):
    hybrid search -> rerank -> relevance cutoff -> at most N chunks per document -> top k."""
    s = get_settings()
    hits = store.search(query, course_ids, s.rerank_candidates, s.search_mode)
    if s.rerank_model:
        hits = [h for h in rerank.rerank(query, hits) if h.score >= s.min_rerank_score]
    elif max((h.dense_score or 0 for h in hits), default=0) < s.min_score:
        hits = []  # without a reranker, gate the whole query on its best dense similarity

    kept, per_doc = [], {}
    for h in hits:
        if per_doc.get(h.doc_id, 0) < s.max_chunks_per_doc:
            kept.append(h)
            per_doc[h.doc_id] = per_doc.get(h.doc_id, 0) + 1
        if len(kept) == s.top_k:
            break
    return kept


@dataclass
class Prepared:
    question: str
    search_query: str
    hits: list[store.Hit]
    messages: list[dict] | None  # None = nothing relevant and no conversation: reply NOT_FOUND without the LLM


def prepare(question: str, course_ids: list[str] | None = None, history: list[dict] | None = None) -> Prepared:
    """Everything before generation: rewrite the follow-up, retrieve, and build the LLM messages."""
    history = _recent(history or [])
    search_query = standalone_question(question, history)
    hits = retrieve(search_query, course_ids)

    if not hits and not history:
        return Prepared(question, search_query, hits, None)
    if hits:
        user_turn = f"Course excerpts:\n\n{_format_sources(hits)}\n\nStudent question: {question}"
        if search_query != question:
            user_turn += f"\n(In context, the student is asking: {search_query})"
    else:
        user_turn = f"{question}\n\n{NO_NEW_SOURCES.format(not_found=NOT_FOUND)}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": user_turn}]
    return Prepared(question, search_query, hits, messages)


def finish(prep: Prepared, text: str) -> dict:
    """Clean up the generated answer and attach the sources it actually cites."""
    text = _normalize_citations(text)
    hits = prep.hits
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    citations = [{"id": n, **asdict(hits[n - 1]), "score": round(hits[n - 1].score, 3)} for n in cited]
    return {"answer": text, "citations": citations, "search_query": prep.search_query}


def answer(question: str, course_ids: list[str] | None = None, history: list[dict] | None = None) -> dict:
    prep = prepare(question, course_ids, history)
    text = NOT_FOUND if prep.messages is None else llm.chat(prep.messages, temperature=0.1)
    return finish(prep, text)


def answer_stream(prep: Prepared):
    """Yield answer text pieces as the LLM writes them (call finish() with the joined text afterwards)."""
    if prep.messages is None:
        yield NOT_FOUND
        return
    yield from llm.stream_chat(prep.messages, temperature=0.1)
