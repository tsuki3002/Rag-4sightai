"""Download the web `sources:` listed for each course in config/courses.yaml into <docs_dir>/web/.

Web pages are saved as clean Markdown (main content only), PDFs and images as-is. A _manifest.json next to
the files records each one's original URL, title and license so citations can link back to the source.

Usage:
    python -m scripts.fetch_sources --course ml
    python -m scripts.fetch_sources --all [--refresh]
then run `python -m scripts.ingest ...` to index what was downloaded.
"""
import argparse
import hashlib
import json
import logging
import re
import time
import urllib.robotparser
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
import trafilatura

from app.config import Course, Source, load_courses

log = logging.getLogger("fetch")
USER_AGENT = "CourseRAG/1.0 (educational course-assistant indexer; python-httpx)"
MANIFEST = "_manifest.json"
EXT_BY_TYPE = {"application/pdf": ".pdf", "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
               "image/gif": ".gif"}

_robots: dict[str, urllib.robotparser.RobotFileParser] = {}


def allowed_by_robots(url: str, http: httpx.Client) -> bool:
    # fetched with our own client: urllib's default agent is refused by some sites, which robotparser
    # would misread as "disallow everything"
    origin = "{0.scheme}://{0.netloc}".format(urlparse(url))
    if origin not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            resp = http.get(origin + "/robots.txt")
            if resp.status_code in (401, 403):
                rp.disallow_all = True
            elif resp.status_code >= 400:
                rp.allow_all = True
            else:
                rp.parse(resp.text.splitlines())
        except httpx.HTTPError:
            rp.allow_all = True
        _robots[origin] = rp
    return _robots[origin].can_fetch(USER_AGENT, url)


def get_with_retry(http: httpx.Client, url: str, attempts: int = 4) -> httpx.Response:
    """GET that backs off on 429/503, honouring Retry-After when the server sends it."""
    for attempt in range(attempts):
        resp = http.get(url)
        if resp.status_code not in (429, 503) or attempt == attempts - 1:
            return resp
        retry_after = resp.headers.get("retry-after", "")
        delay = min(int(retry_after), 60) if retry_after.isdigit() else 5 * 2 ** attempt
        log.info("Rate limited by %s; retrying in %ds", urlparse(url).netloc, delay)
        time.sleep(delay)
    return resp


def slug(source: Source) -> str:
    path = unquote(urlparse(source.url).path.rstrip("/")).rsplit("/", 1)[-1]
    path = re.sub(r"\.(html?|php|aspx?)$", "", path) or urlparse(source.url).netloc
    base = re.sub(r"[^A-Za-z0-9]+", "_", source.title or path).strip("_").lower()[:60]
    return f"{base}_{hashlib.sha1(source.url.encode()).hexdigest()[:6]}"


def clean_markdown(text: str) -> str:
    """Drop footnote markers like <sup>[1]</sup> / \\[1\\]: they look exactly like our [n] citations."""
    text = re.sub(r"<sup>.*?</sup>", "", text)
    return re.sub(r"(?<![\w\])])\\?\[(\d+|citation needed|note \d+)\\?\]", "", text)


def page_title(meta, text: str, fallback: str) -> str:
    title = meta.title if meta and meta.title else title_from(text, fallback)
    sites = ["Wikipedia"] + ([re.escape(meta.sitename)] if meta and meta.sitename else [])
    title = re.sub(rf"\s+[-–|]\s+({'|'.join(sites)})$", "", title)
    return title.strip()


def title_from(markdown: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", markdown, re.M)
    return m.group(1).strip() if m else fallback


def fetch_course(course: Course, http: httpx.Client, refresh: bool = False) -> dict:
    out_dir = course.path / "web"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    known = {v["url"]: k for k, v in manifest.items()}
    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    for src in course.sources:
        if src.url in known and (out_dir / known[src.url]).exists() and not refresh:
            stats["skipped"] += 1
            continue
        if not allowed_by_robots(src.url, http):
            log.warning("robots.txt disallows %s; skipping", src.url)
            stats["failed"] += 1
            continue
        try:
            resp = get_with_retry(http, src.url)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("Failed %s: %s", src.url, e)
            stats["failed"] += 1
            continue

        ctype = resp.headers.get("content-type", "").split(";")[0].strip()
        name = slug(src)
        if ctype in EXT_BY_TYPE:
            filename, title = name + EXT_BY_TYPE[ctype], src.title or name.rsplit("_", 1)[0].replace("_", " ")
            (out_dir / filename).write_bytes(resp.content)
        elif ctype in {"text/html", "application/xhtml+xml"}:
            text = trafilatura.extract(resp.text, url=src.url, output_format="markdown",
                                       include_tables=True, include_comments=False)
            if not text or len(text) < 200:
                log.warning("No main content extracted from %s; skipping", src.url)
                stats["failed"] += 1
                continue
            text = clean_markdown(text)
            title = src.title or page_title(trafilatura.extract_metadata(resp.text), text, name)
            if not text.lstrip().startswith("# "):
                text = f"# {title}\n\n{text}"
            filename = name + ".md"
            (out_dir / filename).write_text(text + "\n", encoding="utf-8")
        else:
            log.warning("Unsupported content type %r at %s; skipping", ctype, src.url)
            stats["failed"] += 1
            continue

        if src.url in known and known[src.url] != filename:
            (out_dir / known[src.url]).unlink(missing_ok=True)
            manifest.pop(known[src.url], None)
        manifest[filename] = {"url": src.url, "title": title, "license": src.license,
                              "fetched": date.today().isoformat()}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        stats["downloaded"] += 1
        log.info("Saved %s", filename)
        time.sleep(1)  # be polite to the hosts

    # sources removed from courses.yaml: delete their files so the next ingest drops them from the index
    wanted = {s.url for s in course.sources}
    for filename, entry in list(manifest.items()):
        if entry["url"] not in wanted:
            (out_dir / filename).unlink(missing_ok=True)
            del manifest[filename]
            log.info("Removed %s (no longer listed)", filename)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Download course web sources.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--course", action="append", help="course id (repeatable)")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="re-download sources that already exist")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    courses = load_courses()
    ids = list(courses) if args.all else args.course
    for cid in ids:
        if cid not in courses:
            parser.error(f"unknown course '{cid}'. Known: {', '.join(courses)}")
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60) as http:
        for cid in ids:
            print(f"[{cid}] {fetch_course(courses[cid], http, refresh=args.refresh)}")


if __name__ == "__main__":
    main()
