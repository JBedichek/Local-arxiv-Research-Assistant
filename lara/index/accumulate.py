"""Making the machine's own research retrievable, and keeping it forever.

lara already stores every synthesis run and every claim it extracted — 1,645 claims across
30 runs on this machine before a line of this was written. What it does not do is let you
*find* them again: they sit in `synthesis_claims`, not in the chunk index, so the next
question starts from the raw corpus as though the last thirty runs never happened.

This promotes them into the retrieval stack, so the corpus keeps what its own
research produced. A claim becomes a chunk with `kind='claim'`
attributed to the paper it came from; a run's summaries become chunks under a pseudo-paper
named for the question. From there lara's own pipeline does the rest — the embedder picks
up anything with a null vector row, and retrieval, hydration and citation all work
unchanged.

**Why this is worth doing.** A claim is the same information as the passage it came from,
compressed by a model that had the question in front of it, with the numbers and their
conditions pulled out. Retrieving one costs a fraction of the context of retrieving the
passage, and it has already been judged relevant once. The corpus grows a layer of
pre-digested material that gets better the more research is done over it.

**Nothing is ever deleted.** Accumulation is unbounded on purpose: a claim extracted for
one question is evidence for questions nobody has asked yet, and the storage is a few
kilobytes against a 141 GB corpus.

**Idempotent.** Promotion is recorded, so running it twice does not duplicate a run's
claims — which would quietly bias retrieval toward whatever happened to be promoted twice.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass

from lara.index import backends as BK
from lara.index import embed
from lara.index import scope as SC

#: `kind` values this module writes. Retrieval can bias on these, and they are the only
#: way to tell a compressed claim from the passage it compresses.
CLAIM = "claim"
SYNTHESIS = "synthesis"
KINDS = (CLAIM, SYNTHESIS)

#: Ordinals for synthetic chunks start here, so they sort after a paper's real text and
#: never interleave with it in a reading view.
SYNTHETIC_ORDINAL_BASE = 900_000

#: Pseudo-paper id for a synthesis run. The prefix keeps them identifiable at a glance and
#: impossible to confuse with a real arXiv id.
def run_paper_id(run_id: str) -> str:
    return f"synth/{run_id}"


SCHEMA = """
CREATE TABLE IF NOT EXISTS autoresearch_promoted (
    run_id       TEXT PRIMARY KEY,
    chunks       INTEGER NOT NULL,
    promoted_utc TEXT NOT NULL
);
-- What has already been turned into a chunk. Accumulation is unbounded but *unique*:
-- the same claim extracted by two runs is one chunk, not two. Without this, popular
-- papers would accrue a duplicate for every run that read them, and retrieval would
-- rank them by how often they had been researched rather than by relevance.
CREATE TABLE IF NOT EXISTS autoresearch_fingerprints (
    fingerprint TEXT PRIMARY KEY,
    chunk_id    INTEGER NOT NULL
);
"""


def fingerprint(arxiv_id: str, text: str) -> str:
    return hashlib.sha1(f"{arxiv_id}\x00{text}".encode()).hexdigest()


@dataclass
class Promotion:
    runs: int = 0
    claim_chunks: int = 0
    synthesis_chunks: int = 0
    duplicates: int = 0
    skipped: int = 0

    @property
    def chunks(self) -> int:
        return self.claim_chunks + self.synthesis_chunks

    def to_dict(self) -> dict:
        return {"runs": self.runs, "claim_chunks": self.claim_chunks,
                "synthesis_chunks": self.synthesis_chunks, "chunks": self.chunks,
                "duplicates": self.duplicates, "skipped": self.skipped}


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def promoted_runs(conn: sqlite3.Connection) -> set[str]:
    ensure_schema(conn)
    return {r[0] for r in conn.execute("SELECT run_id FROM autoresearch_promoted")}


def _next_chunk_id(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COALESCE(MAX(chunk_id), 0) + 1 FROM chunks").fetchone()
    return int(row[0])


class _Ordinals:
    """Hands out chunk ordinals that cannot collide.

    `chunks` carries a UNIQUE index on (arxiv_id, version, ordinal). Ordinals cannot come
    from a per-run counter: two runs that both cite one paper would each hand it ordinal
    900000 and the second promotion would fail. Each paper gets its own counter, seeded
    past whatever it already holds.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._next: dict[str, int] = {}

    def take(self, arxiv_id: str) -> int:
        if arxiv_id not in self._next:
            row = self._conn.execute(
                "SELECT MAX(ordinal) FROM chunks WHERE arxiv_id=?", (arxiv_id,)).fetchone()
            highest = int(row[0]) if row and row[0] is not None else -1
            self._next[arxiv_id] = max(SYNTHETIC_ORDINAL_BASE, highest + 1)
        n = self._next[arxiv_id]
        self._next[arxiv_id] = n + 1
        return n


def _source_anchor(conn: sqlite3.Connection, chunk_id: int) -> tuple[str, int, str, int, str]:
    """Where the claim's source passage lives, so a citation lands on it.

    A claim has no location of its own. Inheriting the source's anchors means following a
    claim's citation takes you to the text it compresses, which is the only destination
    that makes sense.
    """
    row = conn.execute(
        "SELECT anchor_start, char_start, anchor_end, char_end, section_anchor "
        "FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
    if row is None:
        return ("", 0, "", 0, "")
    return (row[0] or "", int(row[1] or 0), row[2] or "", int(row[3] or 0), row[4] or "")


def promote_run(conn: sqlite3.Connection, run_id: str, *, force: bool = False) -> Promotion:
    """Turn one synthesis run into retrievable chunks. Safe to call twice."""
    ensure_schema(conn)
    out = Promotion()
    if not force and run_id in promoted_runs(conn):
        out.skipped = 1
        return out

    run = conn.execute(
        "SELECT run_id, question, tldr, thorough FROM synthesis_runs WHERE run_id=?",
        (run_id,)).fetchone()
    if run is None:
        return out
    _, question, tldr, thorough = run
    claims = conn.execute(
        "SELECT chunk_id, arxiv_id, name, claim, method, metric, value, condition "
        "FROM synthesis_claims WHERE run_id=?", (run_id,)).fetchall()

    next_id = _next_chunk_id(conn)
    ordinals = _Ordinals(conn)
    rows: list[tuple] = []
    prints: list[tuple[str, int]] = []
    seen = {r[0] for r in conn.execute("SELECT fingerprint FROM autoresearch_fingerprints")}

    def add(arxiv_id: str, kind: str, text: str, anchors: tuple) -> None:
        nonlocal next_id
        fp = fingerprint(arxiv_id, text)
        if fp in seen:
            out.duplicates += 1
            return
        seen.add(fp)
        anchor_s, char_s, anchor_e, char_e, section = anchors
        rows.append((next_id, arxiv_id, 1, ordinals.take(arxiv_id), section, anchor_s,
                     char_s, anchor_e, char_e, kind, len(text), None, text))
        prints.append((fp, next_id))
        next_id += 1

    # ── the run's own summaries, under a pseudo-paper named for the question ──────
    pid = run_paper_id(run_id)
    conn.execute(
        "INSERT OR REPLACE INTO papers (arxiv_id, latest_version, title, abstract, "
        " authors, categories, primary_category, submitted_utc, in_scope, deleted, "
        " fulltext_status, fulltext_version, n_chunks) "
        "VALUES (?,1,?,?,'autoresearch','synthesis','synthesis',?,1,0,'ok',1,?)",
        (pid, (question or "")[:400], (tldr or "")[:4000],
         time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         sum(1 for x in (tldr, thorough) if (x or "").strip())))
    before = len(rows)
    for label, body in (("tldr", tldr), ("thorough", thorough)):
        if (body or "").strip():
            add(pid, SYNTHESIS, body, (label, 0, label, len(body), label))
    out.synthesis_chunks = len(rows) - before

    # ── each claim, attributed to the paper it was drawn from ────────────────────
    before = len(rows)
    for source_chunk, arxiv_id, name, claim, method, metric, value, condition in claims:
        if not (claim or "").strip() or not arxiv_id:
            continue
        # The claim plus what makes it checkable. A number without its conditions is not a
        # result, and a summary that drops them is worse than the passage it replaced.
        parts = [f"{name}: {claim}" if name else claim]
        if method:
            parts.append(f"method: {method}")
        if metric or value:
            parts.append(f"measured: {metric} = {value}")
        if condition:
            parts.append(f"condition: {condition}")
        add(arxiv_id, CLAIM, "\n".join(p for p in parts if p),
            _source_anchor(conn, int(source_chunk or 0)))
    out.claim_chunks = len(rows) - before

    if rows:
        conn.executemany(
            "INSERT INTO chunks (chunk_id, arxiv_id, version, ordinal, section_anchor, "
            " anchor_start, char_start, anchor_end, char_end, kind, n_chars, vector_row, "
            " text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.executemany(
            "INSERT OR REPLACE INTO autoresearch_fingerprints (fingerprint, chunk_id) "
            "VALUES (?,?)", prints)
    conn.execute(
        "INSERT OR REPLACE INTO autoresearch_promoted (run_id, chunks, promoted_utc) "
        "VALUES (?,?,?)",
        (run_id, len(rows), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    conn.commit()
    out.runs = 1
    return out


def promote_all(conn: sqlite3.Connection, *, limit: int = 0) -> Promotion:
    """Promote every run not already promoted, oldest first."""
    ensure_schema(conn)
    done = promoted_runs(conn)
    ids = [r[0] for r in conn.execute(
        "SELECT run_id FROM synthesis_runs ORDER BY created_utc")]
    total = Promotion()
    for run_id in ids:
        if run_id in done:
            total.skipped += 1
            continue
        one = promote_run(conn, run_id)
        total.runs += one.runs
        total.claim_chunks += one.claim_chunks
        total.synthesis_chunks += one.synthesis_chunks
        total.duplicates += one.duplicates
        if limit and total.runs >= limit:
            break
    return total


def pending_vectors(conn: sqlite3.Connection) -> int:
    """How many chunks are waiting to be embedded."""
    return int(conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE vector_row IS NULL").fetchone()[0])


def embed_pending(state, *, batch_size: int = 128, slice_size: int = 4096,
                  limit: int = 0) -> dict:
    """Embed everything waiting, using lara's own pipeline, then rebuild the index.

    The rebuild is what makes the new vectors searchable: tier-1 is a snapshot taken at
    load time, so chunks embedded since then exist on disk and are invisible to search
    until it is rebuilt.
    """
    conn = state.conn()
    stats = embed.run(conn, state.store, state.retriever.embedder, batch_size=batch_size,
                    slice_size=slice_size, limit=limit)
    reload_index(state)
    return dict(stats)


def reload_index(state) -> int:
    """Rebuild tier-1 so newly embedded chunks become searchable. Returns the row count."""
    icfg = state.cfg.get_in("index") or {}
    state.scope = SC.Scope.load(state.cfg.get_path("disk.root"))
    resident = state.scope.rows if state.scope is not None else None
    chosen = BK.choose_backend(icfg.get("backend", "auto"))
    state.retriever.dense = BK.make_index(
        state.store.load_int8(mmap=(chosen == "faiss" or resident is not None)),
        backend=chosen, precision=icfg.get("precision", "fp16"), device=state.device,
        row_ids=resident, faiss_cfg=BK.FaissConfig(**(icfg.get("faiss") or {})))
    state.retriever.remap_vectors()
    state.retriever.refresh_row_map()
    return int(state.retriever.dense.n)
