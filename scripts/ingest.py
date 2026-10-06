"""Usage:
    python -m scripts.ingest --course ai      # one course
    python -m scripts.ingest --all            # every course in config/courses.yaml
    python -m scripts.ingest --all --force    # re-embed everything (e.g. after changing chunk size/model)
    python -m scripts.ingest --all --rebuild  # delete the index and build it from scratch (index format changes)
"""
import argparse
import logging

from app import store
from app.config import load_courses
from app.ingest import ingest_course


def main() -> None:
    parser = argparse.ArgumentParser(description="Index course documents into the vector store.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--course", action="append", help="course id (repeatable)")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-index even unchanged files")
    parser.add_argument("--rebuild", action="store_true", help="delete the whole index first (requires --all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    courses = load_courses()
    ids = list(courses) if args.all else args.course
    for cid in ids:
        if cid not in courses:
            parser.error(f"unknown course '{cid}'. Known: {', '.join(courses)}")
    if args.rebuild and not args.all:
        parser.error("--rebuild deletes every course's index, so it requires --all")
    try:
        if args.rebuild:
            store.drop_collection()
        for cid in ids:
            print(f"[{cid}] {ingest_course(courses[cid], force=args.force)}")
    finally:
        store.client().close()


if __name__ == "__main__":
    main()
