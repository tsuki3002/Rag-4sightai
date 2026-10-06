"""Measure retrieval (and optionally answer) quality against eval/questions.yaml.

    python -m eval.run                 # retrieval only: hit@1, hit@k, MRR, off-topic rejection (free)
    python -m eval.run --answers       # also run the LLM: cited-correct-source rate, follow-ups (uses API quota)
    python -m eval.run --answers --sample 24 --delay 20   # cheaper run that fits a free tier's tokens/minute
    python -m eval.run --label baseline   # name the saved results file

Results are saved to eval/results/<label>.json so runs can be compared.
"""
import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import yaml

from app import rag, store
from app.config import get_settings

HERE = Path(__file__).parent


def matches(title: str, expect: list[str]) -> bool:
    return any(e.lower() in title.lower() for e in expect)


def first_match_rank(hits, expect) -> int | None:
    return next((i for i, h in enumerate(hits, start=1) if matches(h.doc_title, expect)), None)


def eval_retrieval(data: dict) -> dict:
    rows, latencies = [], []
    for item in data["questions"]:
        t = time.perf_counter()
        hits = rag.retrieve(item["q"], [item["course"]])
        latencies.append(time.perf_counter() - t)
        rank = first_match_rank(hits, item["expect"])
        rows.append({"q": item["q"], "course": item["course"], "rank": rank,
                     "top_score": hits[0].score if hits else None,
                     "top": [f"{h.score:.3f} {h.doc_title} | {h.section or ''}" for h in hits[:3]]})

    off = []
    for q in data["off_topic"]:
        hits = rag.retrieve(q, None)
        off.append({"q": q, "rejected": not hits, "top": [f"{h.score:.3f} {h.doc_title}" for h in hits[:2]]})

    # calibration aid: with the cutoff disabled, how do top scores of answerable vs off-topic questions compare?
    s = get_settings()
    saved = (s.min_rerank_score, s.min_score)
    s.min_rerank_score, s.min_score = float("-inf"), float("-inf")
    try:
        pos = [rag.retrieve(i["q"], [i["course"]]) for i in data["questions"]]
        neg = [rag.retrieve(q, None) for q in data["off_topic"]]
    finally:
        s.min_rerank_score, s.min_score = saved
    pos_top = sorted(h[0].score for h in pos if h)
    neg_top = sorted(h[0].score for h in neg if h)

    n = len(rows)
    ranks = [r["rank"] for r in rows]
    return {
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@k": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "off_topic_rejected": sum(o["rejected"] for o in off) / len(off),
        "latency_ms_median": round(statistics.median(latencies) * 1000),
        "calibration": {"answerable_top_scores_lowest5": [round(x, 3) for x in pos_top[:5]],
                        "off_topic_top_scores_highest5": [round(x, 3) for x in neg_top[-5:]]},
        "misses": [r for r in rows if r["rank"] is None],
        "low_rank": [r for r in rows if r["rank"] and r["rank"] > 1],
        "off_topic_leaks": [o for o in off if not o["rejected"]],
    }


def eval_answers(data: dict, delay: float, sample: int | None) -> dict:
    def cited_ok(out, expect):
        return any(matches(c["doc_title"], expect) for c in out["citations"])

    questions = data["questions"]
    if sample and sample < len(questions):  # evenly spaced, so every course and question type is represented
        questions = [questions[round(i * len(questions) / sample)] for i in range(sample)]
    rows = []
    for item in questions:
        out = rag.answer(item["q"], [item["course"]])
        rows.append({"q": item["q"], "ok": cited_ok(out, item["expect"]),
                     "not_found": out["answer"] == rag.NOT_FOUND or "couldn't find" in out["answer"].lower(),
                     "cited": [c["doc_title"] for c in out["citations"]]})
        time.sleep(delay)

    off = []
    for q in data["off_topic"]:
        out = rag.answer(q, None)
        off.append({"q": q, "declined": "couldn't find" in out["answer"].lower() and not out["citations"],
                    "answer": out["answer"][:200]})
        time.sleep(delay)

    fups = []
    for item in data.get("follow_ups", []):
        history = []
        for turn in item["turns"]:
            out = rag.answer(turn, [item["course"]], history)
            history += [{"role": "user", "content": turn}, {"role": "assistant", "content": out["answer"]}]
            time.sleep(delay)
        fups.append({"turns": item["turns"], "search_query": out["search_query"], "ok": cited_ok(out, item["expect"]),
                     "cited": [c["doc_title"] for c in out["citations"]]})

    return {
        "answer_cites_expected": sum(r["ok"] for r in rows) / len(rows),
        "answer_wrongly_not_found": sum(r["not_found"] for r in rows) / len(rows),
        "off_topic_declined": sum(o["declined"] for o in off) / len(off),
        "follow_up_cites_expected": sum(f["ok"] for f in fups) / len(fups) if fups else None,
        "answer_failures": [r for r in rows if not r["ok"]],
        "off_topic_answered": [o for o in off if not o["declined"]],
        "follow_ups": fups,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true", help="also evaluate LLM answers (uses API quota)")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between LLM calls (free-tier rate limits)")
    parser.add_argument("--sample", type=int, help="with --answers: only evaluate N evenly spaced questions")
    parser.add_argument("--label", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()

    data = yaml.safe_load((HERE / "questions.yaml").read_text())
    s = get_settings()
    config = {k: getattr(s, k) for k in ("search_mode", "rerank_model", "rerank_candidates", "rerank_max_words", "max_chunks_per_doc",
                                         "top_k", "min_score", "min_rerank_score", "llm_model") if hasattr(s, k)}
    try:
        result = {"label": args.label, "config": config, "retrieval": eval_retrieval(data)}
        if args.answers:
            result["answers"] = eval_answers(data, args.delay, args.sample)
    finally:
        store.client().close()

    out = HERE / "results" / f"{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    r = result["retrieval"]
    print(f"\n== {args.label}  {config}")
    print(f"retrieval  hit@1 {r['hit@1']:.0%}  hit@k {r['hit@k']:.0%}  MRR {r['mrr']:.2f}  "
          f"off-topic rejected {r['off_topic_rejected']:.0%}  median latency {r['latency_ms_median']} ms")
    print(f"calibration  answerable lowest top-scores {r['calibration']['answerable_top_scores_lowest5']}  "
          f"off-topic highest {r['calibration']['off_topic_top_scores_highest5']}")
    for m in r["misses"]:
        print(f"  MISS  [{m['course']}] {m['q']}\n        got: {m['top']}")
    for o in r["off_topic_leaks"]:
        print(f"  LEAK  {o['q']}  -> {o['top']}")
    if a := result.get("answers"):
        print(f"answers    cites expected {a['answer_cites_expected']:.0%}  wrongly not-found "
              f"{a['answer_wrongly_not_found']:.0%}  off-topic declined {a['off_topic_declined']:.0%}  "
              f"follow-ups {a['follow_up_cites_expected']:.0%}")
        for f in a["answer_failures"]:
            print(f"  ANSWER-FAIL  {f['q']}  cited={f['cited']}")
        for f in a["follow_ups"]:
            if not f["ok"]:
                print(f"  FOLLOW-UP-FAIL  {f['turns']} -> searched {f['search_query']!r} cited={f['cited']}")
    print(f"saved {out.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
