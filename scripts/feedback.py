"""Review logged answers, e.g. the ones students marked unhelpful, to find what to fix and what to add to
eval/questions.yaml.

    python -m scripts.feedback               # answers marked 👎
    python -m scripts.feedback --all -n 20   # the 20 most recent answers
"""
import argparse
import json
from datetime import datetime

from app import logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="show all recent answers, not only 👎")
    parser.add_argument("-n", type=int, default=20)
    args = parser.parse_args()

    rows = logs.recent(None if args.all else -1, args.n)
    if not rows:
        print("Nothing logged yet." if args.all else "No answers marked unhelpful yet.")
    for r in rows:
        when = datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M")
        rating = {1: "👍", -1: "👎"}.get(r["rating"], "·")
        print(f"\n{rating} {when}  [{', '.join(json.loads(r['course_ids'] or 'null') or ['all'])}]  {r['latency_ms']} ms")
        print(f"  Q: {r['question']}")
        if r["search_query"] and r["search_query"] != r["question"]:
            print(f"  searched: {r['search_query']}")
        for h in json.loads(r["retrieved"] or "[]"):
            print(f"    {h['score']:>7}  {h['doc_title']} | {h['section'] or ''}")
        print(f"  cited: {json.loads(r['cited'] or '[]')}")
        if r["comment"]:
            print(f"  comment: {r['comment']}")
        if r["error"]:
            print(f"  error: {r['error']}")
        print(f"  A: {(r['answer'] or '')[:300]}")


if __name__ == "__main__":
    main()
