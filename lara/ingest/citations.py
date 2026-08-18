"""Citation enrichment via Semantic Scholar (D6).

The bibliography scraper in :mod:`lara.ingest.parse` only sees arXiv ids that a paper
happens to print as text, and only for papers we have already crawled — which left the
graph at roughly one edge per four papers. S2 gives the real reference and citation lists.

Verified 2026-08-17 without an API key: ``POST /graph/v1/paper/batch`` accepts
``ARXIV:1706.03762`` identifiers directly. Since the arXiv id *is* our primary key, no
id-resolution step is needed — the reason S2 was chosen over OpenAlex, whose arXiv linkage
runs through DOIs that 404 for older papers.

Resumable at batch granularity via ``papers.s2_status``, like every other ingest stage.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass, field

import httpx

FIELDS = "externalIds,citationCount,referenceCount,references.externalIds,citations.externalIds"


@dataclass
class Stats:
    batches: int = 0
    papers: int = 0
    edges: int = 0
    missing: int = 0
    errors: int = 0
    retries: int = 0
    detail: dict[str, int] = field(default_factory=dict)


def pending(conn: sqlite3.Connection, limit: int) -> list[str]:
    """In-scope papers not yet enriched, most-crawled first.

    Ordering by full-text presence puts the papers a reader can actually open at the front
    of the queue, so the graph densifies where it is visible soonest.
    """
    return [
        r[0]
        for r in conn.execute(
            "SELECT arxiv_id FROM papers "
            "WHERE in_scope=1 AND deleted=0 AND s2_status='pending' "
            "ORDER BY (fulltext_status='ok') DESC, submitted_utc DESC LIMIT ?",
            (limit,),
        )
    ]


def _arxiv_of(external: dict | None) -> str | None:
    if not external:
        return None
    val = external.get("ArXiv") or external.get("arXiv") or external.get("arxiv")
    return str(val) if val else None


class S2Client:
    def __init__(self, endpoint: str, api_key: str | None = None,
                 timeout: float = 60.0, min_interval: float = 1.1) -> None:
        self.endpoint = endpoint
        self.min_interval = min_interval
        self._last = 0.0
        headers = {"User-Agent": "Local-arxiv-Research-Assistant/0.1"}
        if api_key:
            headers["x-api-key"] = api_key
        self.client = httpx.Client(headers=headers, timeout=timeout)

    def close(self) -> None:
        self.client.close()

    def batch(self, ids: list[str], stats: Stats, max_retries: int = 6) -> list[dict | None]:
        """Fetch one batch, honouring 429s.

        The unauthenticated tier is a shared pool, so 429s are routine rather than
        exceptional and are retried with escalating backoff instead of being treated as
        failures.
        """
        payload = {"ids": [f"ARXIV:{i}" for i in ids]}
        delay = 3.0
        for attempt in range(max_retries):
            gap = time.monotonic() - self._last
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)
            self._last = time.monotonic()
            try:
                resp = self.client.post(self.endpoint, params={"fields": FIELDS}, json=payload)
            except httpx.RequestError:
                stats.retries += 1
                time.sleep(min(delay, 120))
                delay *= 2
                continue

            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else []
            if resp.status_code in (429, 502, 503, 504):
                stats.retries += 1
                wait = float(resp.headers.get("Retry-After") or delay)
                time.sleep(min(wait, 120))
                delay *= 2
                continue
            if resp.status_code == 400:
                # A malformed id in the batch poisons the whole request; S2 reports which.
                stats.errors += 1
                return []
            resp.raise_for_status()
        stats.errors += 1
        return []


def write_batch(conn: sqlite3.Connection, ids: list[str], results: list[dict | None],
                stats: Stats) -> None:
    """Persist edges and mark the batch done, in one transaction.

    Every id in the request is marked — ``ok`` or ``missing`` — so a paper S2 has never
    heard of is not retried forever.
    """
    edges: list[tuple[str, str]] = []
    counts: list[tuple[int | None, int | None, str, str]] = []

    for arxiv_id, entry in zip(ids, results + [None] * (len(ids) - len(results))):
        if not entry:
            counts.append((None, None, "missing", arxiv_id))
            stats.missing += 1
            continue
        for ref in entry.get("references") or []:
            dst = _arxiv_of(ref.get("externalIds"))
            if dst and dst != arxiv_id:
                edges.append((arxiv_id, dst))
        for cit in entry.get("citations") or []:
            src = _arxiv_of(cit.get("externalIds"))
            if src and src != arxiv_id:
                edges.append((src, arxiv_id))
        counts.append((
            entry.get("citationCount"), entry.get("referenceCount"), "ok", arxiv_id,
        ))
        stats.papers += 1

    with conn:
        if edges:
            before = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
            conn.executemany("INSERT OR IGNORE INTO citations(src,dst) VALUES (?,?)", edges)
            stats.edges += conn.execute(
                "SELECT COUNT(*) FROM citations").fetchone()[0] - before
        conn.executemany(
            "UPDATE papers SET cited_by_count=?, reference_count=?, s2_status=? "
            "WHERE arxiv_id=?",
            counts,
        )
    stats.batches += 1


def run(conn: sqlite3.Connection, cfg, *, limit: int = 0, progress=None) -> Stats:
    ccfg = cfg.get_in("ingest.citations")
    key = os.environ.get(ccfg.get("api_key_env", "S2_API_KEY")) or None
    client = S2Client(
        ccfg["batch_endpoint"], api_key=key,
        min_interval=float(ccfg.get("min_interval_sec", 1.1)),
    )
    size = int(ccfg.get("batch_size", 500))
    stats = Stats()
    try:
        while True:
            if limit and stats.papers >= limit:
                break
            ids = pending(conn, size)
            if not ids:
                break
            results = client.batch(ids, stats)
            write_batch(conn, ids, results, stats)
            if progress is not None:
                progress.send(stats)
    finally:
        client.close()
    return stats
