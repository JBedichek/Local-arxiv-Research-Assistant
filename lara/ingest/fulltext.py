"""Full-text crawler — polite, adaptive, resumable (D2, D7).

Source order per paper, first success wins:

1. ``arxiv.org/html/{id}v{n}`` — arXiv's native LaTeXML rendering, papers from Dec 2023.
2. ``ar5iv.labs.arxiv.org/html/{id}`` — LaTeXML for older papers, same anchor scheme.
3. ``arxiv.org/pdf/{id}v{n}`` — last resort. Anchors degrade from DOM ids to page numbers,
   so highlight precision drops to page level. Tracked as ``fulltext_source='pdf'`` so the
   UI can be honest about it.

**Politeness.** D7 sets 3 req/s, which is above arXiv's published ~1 req/3 s API guidance,
so the backoff is deliberately conservative to compensate: a descriptive User-Agent with a
contact address, 60 s initial wait on 429 doubling to an hour, and — importantly —
``adaptive_throttle`` permanently halves the standing rate after three consecutive 429s
rather than retrying at a rate the server has already refused.

**Resumability.** State lives in ``papers.fulltext_status``, so the crawler is a queue
rather than a scan. Each paper is committed as it completes; killing the process loses at
most the handful in flight, and restarting picks up the remaining ``pending`` rows.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import httpx
import zstandard as zstd
from aiolimiter import AsyncLimiter

from lara.ingest.parse import ParsedPaper, parse_html
from lara.store.db import utcnow

USER_AGENT_FALLBACK = "Local-arxiv-Research-Assistant/0.1"


@dataclass
class FetchResult:
    arxiv_id: str
    status: str                 # ok | failed | unavailable
    source: str | None = None   # arxiv_html | ar5iv | pdf
    version: int | None = None
    parsed: ParsedPaper | None = None
    raw: bytes | None = None
    error: str | None = None


def raw_path(root: Path, arxiv_id: str, source: str) -> Path:
    """Shard by id prefix — 330k files in one directory is a filesystem problem."""
    safe = arxiv_id.replace("/", "_")
    shard = safe[:4]
    return root / shard / f"{safe}.{source}.html.zst"


class FullTextFetcher:
    def __init__(
        self,
        *,
        user_agent: str,
        rate_per_sec: float = 3.0,
        max_concurrency: int = 6,
        timeout_sec: float = 60.0,
        raw_root: Path | None = None,
        adaptive_throttle: bool = True,
        on_429_initial_sec: float = 60.0,
        backoff_multiplier: float = 2.0,
        backoff_max_sec: float = 3600.0,
    ) -> None:
        self.rate = rate_per_sec
        self._limiter = AsyncLimiter(max(rate_per_sec, 0.1), 1.0)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            headers={"User-Agent": user_agent or USER_AGENT_FALLBACK},
            timeout=timeout_sec,
            follow_redirects=True,
        )
        self.raw_root = raw_root
        self.adaptive = adaptive_throttle
        self.backoff_initial = on_429_initial_sec
        self.backoff_multiplier = backoff_multiplier
        self.backoff_max = backoff_max_sec
        self._consecutive_429 = 0
        self._pause_until = 0.0
        self._compressor = zstd.ZstdCompressor(level=10)
        self.stats: dict[str, int] = {"ok": 0, "failed": 0, "unavailable": 0, "429": 0}

    async def close(self) -> None:
        await self._client.aclose()

    async def _respect_pause(self) -> None:
        loop = asyncio.get_running_loop()
        while self._pause_until > loop.time():
            await asyncio.sleep(min(5.0, self._pause_until - loop.time()))

    def _note_429(self, retry_after: float | None) -> None:
        self.stats["429"] += 1
        self._consecutive_429 += 1
        wait = retry_after if retry_after else self.backoff_initial * (
            self.backoff_multiplier ** (self._consecutive_429 - 1)
        )
        wait = min(wait, self.backoff_max)
        self._pause_until = asyncio.get_running_loop().time() + wait
        if self.adaptive and self._consecutive_429 >= 3:
            # The server has refused this rate three times running. Retrying at the same
            # pace after a wait is not politeness, it is stubbornness.
            self.rate = max(self.rate / 2.0, 0.2)
            self._limiter = AsyncLimiter(self.rate, 1.0)
            self._consecutive_429 = 0

    async def _get(self, url: str) -> tuple[int, bytes | None]:
        await self._respect_pause()
        async with self._sem, self._limiter:
            try:
                resp = await self._client.get(url)
            except httpx.RequestError as exc:
                return 0, str(exc).encode()[:200]

        if resp.status_code == 429:
            ra = resp.headers.get("Retry-After")
            self._note_429(float(ra) if ra and ra.isdigit() else None)
            return 429, None
        if resp.status_code == 200:
            self._consecutive_429 = 0
            return 200, resp.content
        return resp.status_code, None

    async def fetch(self, arxiv_id: str, version: int, sources: list[str]) -> FetchResult:
        errors: list[str] = []
        for source in sources:
            if source == "arxiv_html":
                url = f"https://arxiv.org/html/{arxiv_id}v{version}"
            elif source == "ar5iv":
                url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
            elif source == "pdf":
                url = f"https://arxiv.org/pdf/{arxiv_id}v{version}"
            else:
                continue

            code, body = await self._get(url)
            if code == 429:
                # Retry this source once the pause elapses rather than burning a fallback.
                code, body = await self._get(url)
            if code != 200 or not body:
                errors.append(f"{source}:{code}")
                continue

            try:
                if source == "pdf":
                    parsed = parse_pdf(body)
                else:
                    # ar5iv serves a styled placeholder for papers it could not convert;
                    # a real paper always yields body blocks.
                    parsed = parse_html(body)
                    if not parsed.chunks:
                        errors.append(f"{source}:empty")
                        continue
            except Exception as exc:  # malformed HTML, unusual LaTeXML output
                errors.append(f"{source}:parse:{type(exc).__name__}")
                continue

            return FetchResult(arxiv_id, "ok", source, version, parsed, body)

        status = "unavailable" if all(
            e.endswith(("404", "empty")) for e in errors
        ) else "failed"
        return FetchResult(arxiv_id, status, error=",".join(errors))

    def write_raw(self, arxiv_id: str, source: str, body: bytes) -> None:
        """Keep the fetched source compressed so re-parsing never re-crawls.

        ~50 KB/paper compressed, so the whole Core corpus is ~16 GB — cheap insurance
        against a parser bug costing another multi-day crawl.
        """
        if self.raw_root is None:
            return
        path = raw_path(self.raw_root, arxiv_id, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self._compressor.compress(body))


def parse_pdf(body: bytes) -> ParsedPaper:
    """PDF fallback. Page-level anchors only — no DOM ids exist to hang precision on."""
    import pymupdf

    from lara.ingest.parse import Block, Chunk, chunk_blocks

    doc = pymupdf.open(stream=body, filetype="pdf")
    blocks: list[Block] = []
    for pageno, page in enumerate(doc, start=1):
        for i, para in enumerate(t for t in page.get_text("text").split("\n\n") if t.strip()):
            text = " ".join(para.split())
            if len(text) < 40:
                continue
            blocks.append(Block(
                anchor=f"page={pageno}", section_anchor=f"page={pageno}",
                kind="body", text=text, ordinal=len(blocks),
            ))
    title = doc.metadata.get("title") or "" if doc.metadata else ""
    doc.close()
    chunks: list[Chunk] = chunk_blocks(blocks)
    return ParsedPaper(title=title, sections=[], blocks=blocks, chunks=chunks, bib_arxiv_ids=[])


# ── persistence ────────────────────────────────────────────────────────────────────

def persist(conn: sqlite3.Connection, result: FetchResult) -> int:
    """Write one paper's parse and flip its status, in a single transaction."""
    if result.status != "ok" or result.parsed is None:
        with conn:
            conn.execute(
                "UPDATE papers SET fulltext_status=?, fulltext_attempts=fulltext_attempts+1, "
                "fulltext_fetched_utc=? WHERE arxiv_id=?",
                (result.status, utcnow(), result.arxiv_id),
            )
        return 0

    p, aid, ver = result.parsed, result.arxiv_id, result.version
    with conn:
        conn.execute("DELETE FROM chunks WHERE arxiv_id=? AND version=?", (aid, ver))
        conn.execute("DELETE FROM sections WHERE arxiv_id=? AND version=?", (aid, ver))
        conn.executemany(
            "INSERT INTO sections(arxiv_id,version,anchor,title,kind,depth,ordinal) "
            "VALUES (?,?,?,?,?,?,?)",
            [(aid, ver, s.anchor, s.title, s.kind, s.depth, s.ordinal) for s in p.sections],
        )
        cur = conn.executemany(
            "INSERT INTO chunks(arxiv_id,version,ordinal,section_anchor,anchor_start,"
            "char_start,anchor_end,char_end,kind,n_chars,text) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (aid, ver, c.ordinal, c.section_anchor, c.anchor_start, c.char_start,
                 c.anchor_end, c.char_end, c.kind, len(c.text), c.text)
                for c in p.chunks
            ],
        )
        _ = cur
        # Bibliography arXiv ids are a free citation-graph seed while S2 catches up.
        if p.bib_arxiv_ids:
            conn.executemany(
                "INSERT OR IGNORE INTO citations(src,dst) VALUES (?,?)",
                [(aid, dst) for dst in p.bib_arxiv_ids],
            )
        conn.execute(
            "UPDATE papers SET fulltext_status='ok', fulltext_source=?, fulltext_version=?, "
            "fulltext_fetched_utc=?, fulltext_attempts=fulltext_attempts+1, n_chunks=? "
            "WHERE arxiv_id=?",
            (result.source, ver, utcnow(), len(p.chunks), aid),
        )
    return len(p.chunks)


def pending_queue(conn: sqlite3.Connection, limit: int, max_attempts: int = 3) -> list[tuple[str, int]]:
    """Next papers to crawl.

    Ordered by citation count where known (D2's priority) and by recency otherwise, so the
    queue degrades gracefully to newest-first while the S2 pass is still running.
    """
    rows = conn.execute(
        "SELECT arxiv_id, COALESCE(latest_version,1) AS v FROM papers "
        "WHERE in_scope=1 AND fulltext_status IN ('pending','failed') "
        "AND fulltext_attempts < ? "
        "ORDER BY COALESCE(cited_by_count,-1) DESC, submitted_utc DESC LIMIT ?",
        (max_attempts, limit),
    ).fetchall()
    return [(r["arxiv_id"], r["v"]) for r in rows]
