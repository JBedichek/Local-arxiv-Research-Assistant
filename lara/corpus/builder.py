"""Turn a goal in the reader's words into a reviewed, built corpus.

    "I need to learn single-variable calculus for an exam in three weeks"
        -> the model proposes search queries
        -> each query is searched
        -> each result is fetched, read, licence-checked and scored for relevance
        -> the reader accepts or rejects
        -> build: chunk, embed, index

The model's only job is the first step. Decomposing a goal into the queries a librarian
would type is a language problem and it is good at it; everything after that is fetching,
measuring and asking, which is code.

**The budget is a running total the reader can move, not a wall.** A cap that silently
stops a build looks identical to a build that found nothing else, so discovery reports how
much text it has accumulated and stops with a reason rather than just stopping. Disk is
checked separately, because "you have room for this" and "you asked to stop at 500 MB" are
different questions and only one of them is negotiable.

**Candidates live on disk, not in memory.** Review is a human step that may take minutes,
and holding the bytes of fifty documents — a 64 MB ceiling each — for that whole time is
gigabytes of resident memory doing nothing. Raw bytes go straight into the corpus's
content-addressed store as they arrive and the candidate carries a path and a short
preview; the build re-reads them. Rejected downloads are pruned afterwards, so the cost of
this is disk that is already transient rather than memory that is not.

**Every decision is written as it is made.** Discovery persists each candidate into the
recipe before moving to the next, so a build interrupted after forty fetches resumes with
forty fetches done. The searches are rate-limited to one every six seconds; losing them to
a crash is the most expensive mistake this module could make.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from lara.corpus import build as BUILD
from lara.corpus import fetch as FETCH
from lara.corpus import licence as LIC
from lara.corpus import search as SEARCH
from lara.corpus import validate as VALIDATE
from lara.corpus.store import Corpus, Recipe, Source, human

QUERY_SYSTEM = """You turn a reader's goal into web search queries for finding documents \
worth putting in a personal search corpus.

- Reply with a JSON array of strings and nothing else.
- Each query is what a librarian would type: specific, no punctuation tricks, no site: \
operators unless one specific site is obviously right.
- Prefer queries that surface whole documents — textbooks, manuals, lecture notes, \
specifications — over queries that surface discussion about them.
- Cover different facets of the goal rather than rephrasing one query several ways."""

#: How much of each document to keep for the review screen. Enough to recognise what it is.
PREVIEW_CHARS = 600


@dataclass
class Candidate:
    """One fetched, judged document awaiting the reader's decision."""

    source: Source
    preview: str = ""
    raw_path: Path | None = None
    verdict: VALIDATE.Verdict | None = None
    thin: bool = False
    alternates: list[str] = field(default_factory=list)

    @property
    def megabytes(self) -> float:
        return self.source.chars / 1e6


@dataclass
class Budget:
    """What the reader is willing to spend, and what has been spent."""

    text_limit: int
    text_used: int = 0
    disk_free: int = 0
    min_free: int = 5 << 30

    @property
    def remaining(self) -> int:
        return max(0, self.text_limit - self.text_used)

    @property
    def fraction(self) -> float:
        return self.text_used / self.text_limit if self.text_limit else 0.0

    @property
    def disk_low(self) -> bool:
        return bool(self.disk_free) and self.disk_free < self.min_free

    def warning(self) -> str | None:
        """The one sentence worth interrupting the reader for, or nothing."""
        if self.disk_low:
            return (f"Only {self.disk_free / 1e9:.1f} GB free on disk — "
                    f"a build should stop well before the disk does.")
        if self.fraction >= 0.9:
            return (f"{self.text_used / 1e6:.0f} MB of the "
                    f"{self.text_limit / 1e6:.0f} MB text budget used.")
        return None


async def propose_queries(cfg, goal: str, n: int = 8, model: str | None = None) -> list[str]:
    """Ask the generator for search queries. Falls back to the goal itself.

    A build must still work when no generator is running: the reader gets their own words
    as a single query rather than an error, which finds less but finds something.
    """
    from lara.serve.generate import stream_answer

    buf = ""
    try:
        async for tok in stream_answer(
            cfg, f"Goal:\n{goal}\n\nGive {n} search queries as a JSON array.",
            [], system=QUERY_SYSTEM, model=model, temperature=0.3,
            max_tokens=400, raw_user=True,
        ):
            buf += tok
    except Exception:                                  # noqa: BLE001
        return [goal]

    m = re.search(r"\[.*\]", buf, re.S)
    if not m:
        return [goal]
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [goal]
    out = [str(q).strip() for q in raw if isinstance(q, (str, int)) and str(q).strip()]
    # Deduplicate case-insensitively: models like to offer "calculus textbook pdf" and
    # "Calculus textbook PDF" as two ideas, and each costs a rate-limited search.
    seen, uniq = set(), []
    for q in out:
        if q.lower() not in seen:
            seen.add(q.lower())
            uniq.append(q)
    return uniq[:n] or [goal]


def _score_or_none(score: float) -> float | None:
    return None if score is None or math.isnan(score) else round(score, 4)


async def discover(cfg, embedder, corpus: Corpus, recipe: Recipe, *,
                   queries: list[str], per_query: int = 8, model: str | None = None,
                   max_doc_bytes: int = 64 << 20, on_event=None) -> list[Candidate]:
    """Search, fetch, judge, and record. Returns candidates for the reader to decide on.

    Everything that happens is reported through ``on_event`` rather than printed, so the
    same routine drives a terminal prompt, a progress bar or an HTTP endpoint.
    """
    def emit(kind: str, **payload):
        if on_event is not None:
            on_event({"kind": kind, **payload})

    goal = recipe.goal or " ".join(queries)
    budget = Budget(text_limit=recipe.text_budget,
                    text_used=recipe.text_bytes(),
                    disk_free=corpus.disk_free())
    if budget.disk_low:
        emit("warning", detail=budget.warning())

    for q in queries:
        if q not in recipe.queries:
            recipe.queries.append(q)

    stats = SEARCH.SearchStats()
    results = SEARCH.search_many(queries, k=per_query, stats=stats)
    emit("searched", queries=len(queries), results=len(results), cached=stats.cached,
         blocked=stats.blocked, stale=stats.stale, errors=list(stats.errors))
    if stats.was_blocked and not results:
        emit("blocked", detail="the search engine is throttling; try again shortly")
        return []

    known = {s.url for s in recipe.sources}
    known_hashes = {s.sha256 for s in recipe.sources if s.sha256}
    out: list[Candidate] = []

    for r in results:
        if r.url in known:
            emit("skipped", url=r.url, why="already considered")
            continue
        known.add(r.url)

        if budget.remaining <= 0:
            emit("budget", used=budget.text_used, limit=budget.text_limit,
                 detail="text budget reached; raise it to keep going")
            break

        doc = FETCH.fetch(r.url, max_bytes=max_doc_bytes)
        if doc.sha256 and doc.sha256 in known_hashes:
            emit("skipped", url=r.url, why="same bytes as a source already listed")
            continue
        if doc.sha256:
            known_hashes.add(doc.sha256)

        verdict = await VALIDATE.validate(cfg, embedder, goal, doc, model=model)
        lic = doc.licence or LIC.Licence(LIC.UNKNOWN, "unknown")
        src = Source(
            url=r.url, title=doc.title or r.title, sha256=doc.sha256,
            bytes_downloaded=doc.bytes_downloaded, chars=doc.chars,
            content_type=doc.content_type, licence=lic.verdict, licence_label=lic.label,
            relevance=_score_or_none(verdict.score),
            decided="pending" if verdict.relevant else "rejected",
            reason="" if verdict.relevant else verdict.as_reason(),
            found_by=r.query, added_utc=_now(),
        )

        # Keep the bytes only for documents that could still be accepted, and keep them on
        # disk rather than in hand. A rejected candidate is not worth the storage, and the
        # store is content-addressed so re-fetching one later costs nothing extra.
        raw_path = FETCH.save_raw(doc, corpus.raw_dir) if verdict.relevant else None
        preview = re.sub(r"\s+", " ", doc.text[:PREVIEW_CHARS * 2]).strip()[:PREVIEW_CHARS]

        recipe.sources.append(src)
        corpus.save(recipe)                    # persist before the next six-second search
        out.append(Candidate(source=src, preview=preview, raw_path=raw_path,
                             verdict=verdict, thin=doc.thin, alternates=doc.alternates))
        doc.raw = None                         # let the bytes go now they are on disk

        if verdict.relevant:
            budget.text_used += doc.chars
        emit("candidate", url=r.url, title=src.title, chars=doc.chars,
             licence=lic.verdict, licence_label=lic.label, relevance=src.relevance,
             relevant=verdict.relevant, reason=verdict.as_reason(), thin=doc.thin,
             alternates=doc.alternates, preview=preview,
             used=budget.text_used, limit=budget.text_limit)

        warn = budget.warning()
        if warn:
            emit("warning", detail=warn)

    return out


def decide(corpus: Corpus, recipe: Recipe, url: str, decided: str,
           reason: str = "") -> bool:
    """Record an accept or reject and persist it. Returns False if the url is unknown."""
    src = recipe.by_url(url)
    if src is None:
        return False
    src.decided = decided
    src.reason = reason
    corpus.save(recipe)
    return True


def add_file(corpus: Corpus, recipe: Recipe, path: Path,
             decided: str = "accepted") -> Source | None:
    """Take a file the reader supplied directly. Returns None if it yields no text.

    The other half of the promise: a corpus is whatever the reader wants in it, and some
    of that is never going to be findable by search — their own notes, a scanned manual, a
    paper a colleague sent them.
    """
    doc = FETCH.from_path(Path(path))
    if not doc.text.strip():
        return None
    if recipe.by_hash(doc.sha256):
        return recipe.by_hash(doc.sha256)
    lic = doc.licence or LIC.Licence(LIC.UNKNOWN, "unknown")
    src = Source(url=doc.url, title=doc.title or Path(path).name, sha256=doc.sha256,
                 bytes_downloaded=doc.bytes_downloaded, chars=doc.chars,
                 content_type=doc.content_type, licence=lic.verdict,
                 licence_label=lic.label, decided=decided, reason="supplied by the reader",
                 found_by="upload", added_utc=_now())
    FETCH.save_raw(doc, corpus.raw_dir)
    recipe.sources.append(src)
    corpus.save(recipe)
    return src


def raw_path_for(corpus: Corpus, src: Source) -> Path | None:
    """Where a source's bytes are stored, if they still are."""
    if not src.sha256:
        return None
    hits = sorted(corpus.raw_dir.glob(f"{src.sha256[:16]}.*"))
    return hits[0] if hits else None


def prune_raw(corpus: Corpus, recipe: Recipe) -> int:
    """Delete stored bytes no accepted source refers to. Returns bytes reclaimed."""
    keep = {s.sha256[:16] for s in recipe.accepted() if s.sha256}
    freed = 0
    if not corpus.raw_dir.is_dir():
        return 0
    for f in corpus.raw_dir.iterdir():
        if f.is_file() and f.stem not in keep:
            freed += f.stat().st_size
            f.unlink()
    return freed


def build(corpus: Corpus, recipe: Recipe, embedder, *,
          dim_full: int = 768, dim_trunc: int = 256, on_event=None,
          chunking: dict | None = None) -> BUILD.BuildStats:
    """Chunk, embed and index every accepted source. Safe to re-run.

    Re-running is the normal case, not a recovery path: a study corpus grows a document at
    a time, and documents are keyed by content hash while vectors are appended, so a second
    build embeds only what the first one did not.

    `chunking` carries the config.yaml `chunking.*` dict (target_chars /
    overlap_frac / never_cross_sections — the same keys lara/ingest/fulltext.py:74-76
    filters for the crawl path). M20: until now the only add_document call site passed
    no chunking kwargs, so builds ran on parse.py's hardcoded defaults and editing
    config.yaml changed nothing.
    """
    from lara.index.vectors import VectorStore

    chunk_kwargs = {k: v for k, v in (chunking or {}).items()
                    if k in ("target_chars", "overlap_frac")}
    # add_document's chunk_blocks call hardcodes never_cross_sections=True, so that
    # key is deliberately not forwarded here (unlike the crawl path, where
    # fulltext.py filters all three for parse_html).

    def emit(kind: str, **payload):
        if on_event is not None:
            on_event({"kind": kind, **payload})

    t0 = time.time()
    stats = BUILD.BuildStats()
    conn = BUILD.connect(corpus.db_path)
    store = VectorStore(corpus.fp16_path, corpus.int8_path,
                        dim_full=dim_full, dim_trunc=dim_trunc)
    try:
        for src in recipe.accepted():
            path = raw_path_for(corpus, src)
            if path is None:
                stats.skipped += 1
                emit("missing", url=src.url, title=src.title,
                     detail="stored bytes are gone; re-fetch to include it")
                continue
            doc = FETCH.from_path(path, url=src.url, content_type=src.content_type)
            # The recipe is the record of what was decided, so its licence and title win
            # over anything re-extraction infers a second time.
            doc.licence = LIC.Licence(src.licence, src.licence_label)
            doc.title = src.title or doc.title
            added = BUILD.add_document(conn, doc, doc.raw, **chunk_kwargs)
            if added:
                stats.documents += 1
                stats.chunks += added
            else:
                stats.skipped += 1
            emit("document", title=doc.title, chunks=added, url=src.url)

        stats.embedded = BUILD.embed_pending(
            conn, store, embedder,
            progress=lambda n: emit("embedded", done=n, total=stats.chunks))
        BUILD.rebuild_fts(conn)
    finally:
        conn.close()

    stats.seconds = time.time() - t0
    recipe.chunks = stats.chunks
    recipe.embedded = stats.embedded
    recipe.built_utc = _now()
    corpus.save(recipe)
    emit("built", documents=stats.documents, chunks=stats.chunks,
         embedded=stats.embedded, seconds=stats.seconds)
    return stats


def summarise(recipe: Recipe) -> str:
    """The few lines a reader needs before pressing build."""
    acc = recipe.accepted()
    if not acc:
        return "Nothing accepted yet."
    by_lic: dict[str, int] = {}
    for s in acc:
        by_lic[s.licence] = by_lic.get(s.licence, 0) + 1
    lic_bits = ", ".join(f"{n} {k}" for k, n in sorted(by_lic.items()))
    _ok, why = recipe.publishable()
    return (f"{len(acc)} source(s), {human(recipe.text_bytes())} of text "
            f"({lic_bits}).\n{why}")



def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
