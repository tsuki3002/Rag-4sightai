# Course RAG

Students ask questions in plain English, including follow-ups like "give me an example of that". Answers come only from the indexed course material and carry `[n]` citations that link to the source: the original web page, or the exact page or slide of a local file.

**Stack (all free):** local `fastembed` embeddings (BAAI/bge-small-en-v1.5), BM25 keyword vectors and a MiniLM reranker · Qdrant in embedded mode (files under `storage/`) · local OCR (RapidOCR) for images · any OpenAI-compatible LLM (Groq by default, or OpenRouter) · FastAPI with a small web UI.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your Groq (or OpenRouter) key in LLM_API_KEY
```

## Run

```bash
python -m scripts.fetch_sources --all    # download the web sources listed in config/courses.yaml
python -m scripts.ingest --all           # index everything in each course folder
uvicorn app.api:app --reload             # open http://localhost:8000
```

> Embedded Qdrant allows one process at a time, so stop the server before running ingest (or use the upload endpoint while it's running).

## Courses and documents

Courses live in [config/courses.yaml](config/courses.yaml). Each course has a `docs_dir` and an optional list of web `sources`.

| Course | Material included |
|---|---|
| `ai` | Wikipedia (AI, agents, A* search, knowledge representation, RL, LLMs, transformers, RAG, AI ethics), *Dive into Deep Learning* (attention, transformer, BERT, MDPs), neural network diagram |
| `ml` | Wikipedia (ML, supervised and unsupervised learning, overfitting, bias–variance, cross-validation, decision trees, random forests, SVMs, k-means, gradient descent, precision and recall), *Dive into Deep Learning* (linear and softmax regression, MLPs, generalization, GD, CNNs), 4 diagrams |
| `data-analytics` | OpenIntro *Introduction to Modern Statistics* (data, study design, EDA, regression, hypothesis testing, inference), Wikipedia (data analysis, EDA, descriptive statistics, data cleaning, visualisation, SQL, pandas, hypothesis tests), box plot diagram |

Everything above is licensed CC BY-SA or public domain, and each citation shows its license.

- **Add a web source:** append its URL to the course's `sources`, then run `fetch_sources` and `ingest`. HTML pages are saved as clean Markdown (main content only, footnote markers and reference lists removed), and PDFs and images are saved as-is. The fetcher obeys robots.txt and backs off when rate-limited. Removing a URL deletes its file, and the next ingest drops it from the index.
- **Add a local file:** put it in the course folder and run `ingest`. Supported types: `pdf`, `docx`, `pptx`, `md`, `txt`, `html`, `png`, `jpg`, `webp`, `gif`, `bmp`, `tiff`.
- **Add a course:** add an entry to `courses.yaml`, then run `fetch_sources --course <id>` and `ingest --course <id>`.
- **Upload over HTTP:** set `ADMIN_TOKEN` in `.env`, then
  `curl -X POST localhost:8000/courses/ai/documents -H "X-Admin-Token: $ADMIN_TOKEN" -F file=@notes.pdf`

Ingest is incremental: unchanged files are skipped. Use `--force` after changing chunking or embedding settings.

### Images and scanned documents

- Standalone images, pictures inside PowerPoint slides, and scanned PDF pages (pages with no extractable text) are run through **local OCR**, so any text, labels or formulas in them become searchable.
- Diagrams with no text (plots, architecture drawings) need a **vision model** to be understood. Set `VISION_MODEL=qwen/qwen3.8-27b` (Groq) to get a written description of each image at ingest time, then run `ingest --force`. Each image costs about 2k tokens, once per image.
- PowerPoint speaker notes and tables are indexed too.

## Follow-up questions

The UI sends the last `HISTORY_MESSAGES` messages (default 6) with each question. The server is stateless, so there are no sessions to store.

1. If there is history, the LLM rewrites the latest message into a standalone question ("give me examples of it" becomes "What are examples of unsupervised learning algorithms?"). That question is used for search, and the UI shows it as "Searched for: …". `REWRITE_MODEL` (default in `.env.example`: `openai/gpt-oss-20b`) can be a smaller, faster model than `LLM_MODEL`.
2. The answer is generated with the recent conversation plus fresh excerpts. Citation numbers from earlier answers are stripped so they can't be confused with this turn's sources.
3. If nothing in the course matches a follow-up ("summarise that in one line", "thanks!"), the assistant answers from the conversation alone, or says it can't find the answer.

"New chat", or switching course, clears the history.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/courses` | list courses |
| POST | `/ask` | `{"question": "...", "course_ids": ["ml"], "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` returns `{answer, citations[], search_query, id}` |
| POST | `/ask/stream` | same request; server-sent events: `meta`, then `delta` text pieces as the answer is written, then `done` with the full result (or `error`). Used by the web UI. |
| POST | `/feedback` | `{"id": "<id from the answer>", "rating": 1 or -1, "comment": "optional"}` |
| GET | `/admin/interactions?rating=-1` | recent questions with what was retrieved and cited (admin token) |
| GET | `/files/<source_path>` | open a cited local document |
| POST | `/courses/{id}/documents` | upload and index a file (admin token) |

Each citation includes `doc_title`, `page`, `section`, `source_url`, `license`, `text` (the excerpt), `score`, `dense_score` and `rerank_score`.

## How it works

1. **Load** ([app/loaders.py](app/loaders.py)): split each file into citable units (PDF page, PPTX slide, or Markdown/DOCX/HTML heading section). Images go through OCR and, if configured, a vision model ([app/vision.py](app/vision.py)). Reference-list sections are skipped.
2. **Chunk and index** ([app/chunking.py](app/chunking.py), [app/store.py](app/store.py)): cut each unit into ~220-word chunks along sentence boundaries without crossing pages or sections. Each chunk, prefixed with its document title and section, is stored with two vectors: a dense one (meaning) and a BM25 one (exact keywords).
3. **Retrieve** ([app/rag.py](app/rag.py) `retrieve`):
   1. Rewrite follow-ups into standalone questions.
   2. **Hybrid search:** meaning and keyword searches, merged by reciprocal rank fusion, filtered by course, fetching `RERANK_CANDIDATES` chunks.
   3. **Rerank** ([app/rerank.py](app/rerank.py)): a cross-encoder reads the question with each candidate's first `RERANK_MAX_WORDS` words and re-scores it. This is far more reliable than embedding similarity.
   4. **Cutoff:** drop chunks below `MIN_RERANK_SCORE`. If nothing is left, the answer is "not found" and no LLM call is made.
   5. **Diversity:** keep at most `MAX_CHUNKS_PER_DOC` chunks per document, up to `TOP_K`.
4. **Generate:** the LLM answers only from the numbered excerpts and the recent conversation, citing them as `[n]`, streamed to the browser. Citation styles are normalised, and only citations that actually appear in the answer are returned.
5. **Log** ([app/logs.py](app/logs.py)): question, rewritten query, retrieved chunks and scores, answer, latency and 👍/👎 feedback go to `storage/logs.db`.

## Measuring quality

[eval/questions.yaml](eval/questions.yaml) holds 78 student-style questions across the three courses (including bare keywords, acronyms and paraphrases), 8 off-topic questions that must be declined, and 4 multi-turn follow-ups. Each question lists the document(s) that should answer it.

```bash
python -m eval.run                                     # retrieval: hit@1, hit@k, MRR, off-topic rejection (free, ~1 min)
python -m eval.run --answers --sample 24 --delay 20    # + LLM answers & follow-ups (uses API quota)
```

Results are saved to `eval/results/<label>.json`. The output also prints a **calibration** line: the lowest top score among answerable questions and the highest among off-topic ones. `MIN_RERANK_SCORE` should sit between them.

Measured on the current material (retrieval, 78 questions):

| Setup | hit@1 | hit@5 | Off-topic declined | Gap between real and off-topic scores | Median latency |
|---|---|---|---|---|---|
| Original: meaning-based search + similarity cutoff | 88% | 96% | 100% | 0.01 (they overlap) | 13 ms |
| **Current: hybrid search + reranker + diversity** | **95%** | **100%** | **100%** | 2.4 (clean separation) | 450 ms |

The original setup's misses were short or keyword questions ("what is the F1 score?", "iloc", "frame problem") falling below the similarity cutoff. The reranker fixes these. Keyword search makes no measurable difference on this material, but it protects exact identifiers that real course documents contain (assignment numbers, codes, names, formulas). Set `SEARCH_MODE=dense` to turn it off.

**Improving over time:** run `python -m scripts.feedback` (or `GET /admin/interactions?rating=-1`) to see the answers students marked unhelpful, along with what was retrieved and cited. A wrong document in the retrieved list is a search problem; the right document with a bad answer is a generation problem. Add those questions to `eval/questions.yaml` so fixes stay fixed.

## Tuning and scaling

- `MIN_RERANK_SCORE` (-7.0): set from the calibration above (real questions ≥ -5.8, off-topic ≤ -8.2). Re-run the eval after adding a lot of new material and adjust if the gap moves.
- `RERANK_CANDIDATES` (12) × `RERANK_MAX_WORDS` (128) is the speed/quality trade-off. Reranking full chunks × 20 scored only 1 point higher but took about 1.5 s.
- `TOP_K` and `CHUNK_SIZE_WORDS`: change them, then re-index with `--force`. After an index format change (like adding keyword vectors), use `--all --rebuild`.
- `LOG_INTERACTIONS=false` turns logging off. The log contains students' questions, so treat `storage/logs.db` as personal data.
- When the data outgrows a laptop, set `QDRANT_URL` / `QDRANT_API_KEY` to point at a Qdrant server or Qdrant Cloud's free tier. No code changes are needed.
