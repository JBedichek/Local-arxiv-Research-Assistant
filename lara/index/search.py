"""Hybrid retrieval: GPU dense search + BM25, fused, then cross-encoder reranked.

Pipeline (PLAN.md §5, latency budget §7)::

    query ──embed (5ms)──┬─▶ tier 1: exact GPU matmul over MRL-256   ─▶ 200
                         │       (9.5 ms at 16M, 100% recall)
                         └─▶ BM25 over FTS5                          ─▶ 200
                                          │
                              reciprocal rank fusion
                                          │
                         tier 2: exact fp16-768 rescore from mmap    ─▶ 50
                                          │
                     cross-encoder Qwen3-Reranker-0.6B-seq-cls       ─▶ 8
                              (~33 ms for 24 pairs)

Tier 0 (the open paper and its citation neighbourhood, pinned at full precision) is a
filtered case of tier 1 rather than a separate index: restrict to a row subset and the
matmul is over a few thousand vectors instead of millions.

**Why BM25 is not optional.** Dense retrieval is systematically weak on the things arXiv
questions actually hinge on — exact model names, dataset names, symbol names, citation
keys. "GQA" and "grouped-query attention" embed close together; "GQA" and "MQA" also embed
close together, which is the problem. Lexical search gets these exactly right.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field


def fragment_for(arxiv_id: str, version: int, anchor_start: str, char_start: int,
                 anchor_end: str, char_end: int) -> str:
    """Build a citation fragment from raw columns.

    Free-standing because callers that never construct a Hit still have to produce the
    identical string. Writing it out a second time is how 29.8% of the corpus — the chunks
    whose text spans two anchors — ended up with `char_end` interpreted against the
    *opening* anchor, pointing the reader's highlight at the wrong text.
    """
    base = f"/p/{arxiv_id}v{version}#{anchor_start}:{char_start}"
    if anchor_end and anchor_end != anchor_start:
        return f"{base}-{anchor_end}:{char_end}"
    return f"{base}-{char_end}"


@dataclass
class Hit:
    chunk_id: int
    vector_row: int
    score: float
    arxiv_id: str = ""
    version: int = 0
    section_anchor: str = ""
    section_title: str = ""
    anchor_start: str = ""
    char_start: int = 0
    anchor_end: str = ""
    char_end: int = 0
    kind: str = ""
    text: str = ""
    paper_title: str = ""
    provenance: dict[str, float] = field(default_factory=dict)

    def fragment(self) -> str:
        """The citation URL the LLM emits and the reader resolves (PLAN.md §4)."""
        return fragment_for(self.arxiv_id, self.version, self.anchor_start,
                            self.char_start, self.anchor_end, self.char_end)

    def to_dict(self) -> dict:
        """The wire shape every endpoint sends for a retrieved passage.

        One definition, because the key names are load-bearing: ``section`` (not
        ``section_title``) is what agent.format_excerpts and the answer prompt read, and a
        near-miss silently drops the section from every excerpt rather than raising.
        """
        return {
            "chunk_id": self.chunk_id, "arxiv_id": self.arxiv_id, "version": self.version,
            "url": self.fragment(), "anchor": self.anchor_start,
            "char_start": self.char_start, "char_end": self.char_end,
            "anchor_end": self.anchor_end,
            "section": self.section_title or self.section_anchor,
            "kind": self.kind, "score": round(self.score, 4),
            "paper_title": self.paper_title, "text": self.text,
        }


def reciprocal_rank_fusion(
    rankings: dict[str, list[int]], k: int = 60
) -> tuple[list[int], dict[int, dict[str, float]]]:
    """Fuse ranked id lists by RRF: score = sum over lists of 1/(k + rank).

    RRF is used rather than score averaging because the two retrievers produce
    incomparable scales — cosine similarity in [-1, 1] against BM25's unbounded
    log-odds — and normalizing them into agreement requires per-corpus calibration that
    drifts as the corpus grows. Ranks are already comparable, and the k=60 damping keeps
    any single list from dominating on its top hit alone.
    """
    fused: dict[int, float] = {}
    parts: dict[int, dict[str, float]] = {}
    for source, ids in rankings.items():
        for rank, cid in enumerate(ids):
            contribution = 1.0 / (k + rank + 1)
            fused[cid] = fused.get(cid, 0.0) + contribution
            parts.setdefault(cid, {})[source] = contribution
    order = sorted(fused, key=lambda c: -fused[c])
    for cid in order:
        parts[cid]["rrf"] = fused[cid]
    return order, parts


def bm25_search(
    conn: sqlite3.Connection,
    query: str,
    k: int = 200,
    papers: list[str] | None = None,
    max_terms: int = 4,
    df_ceiling_frac: float = 0.02,
    corpus_size: int | None = None,
) -> list[int]:
    """Lexical half of the hybrid. Returns chunk_ids best-first."""
    match = plan_query(conn, query, max_terms, df_ceiling_frac, corpus_size)
    if not match:
        return []
    if papers:
        placeholders = ",".join("?" * len(papers))
        sql = (
            "SELECT c.chunk_id FROM chunks_fts f JOIN chunks c ON c.chunk_id = f.rowid "
            f"WHERE chunks_fts MATCH ? AND c.arxiv_id IN ({placeholders}) "
            "ORDER BY bm25(chunks_fts) LIMIT ?"
        )
        params: list = [match, *papers, k]
    else:
        sql = (
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?"
        )
        params = [match, k]
    try:
        return [r[0] for r in conn.execute(sql, params)]
    except sqlite3.OperationalError:
        return []


_TOKEN = re.compile(r"[0-9a-z]+")


def tokenize(query: str) -> list[str]:
    """Split exactly as FTS5's unicode61 tokenizer does: lowercase alphanumeric runs.

    Matching the tokenizer matters because term selection reads document frequencies out
    of ``chunks_vocab``; a token the tokenizer would never produce looks infinitely rare
    and poisons the ranking.
    """
    return _TOKEN.findall(query.lower())


def plan_query(
    conn: sqlite3.Connection,
    query: str,
    max_terms: int = 4,
    df_ceiling_frac: float = 0.02,
    corpus_size: int | None = None,
) -> str:
    """Build an FTS5 MATCH expression from the query's rarest terms.

    A naive ``term1 OR term2 OR ...`` over a natural-language question is pathologically
    slow — every common word drags in a posting list of hundreds of thousands of chunks
    that ``ORDER BY rank`` must then score (measured: 212–353 ms). Requiring all terms
    (``AND``) is fast but far too strict, returning 0–3 rows for ordinary questions.

    Selecting by inverse document frequency resolves both. It is also the *right* thing
    on the merits, not merely the fast one: BM25 earns its place in this hybrid precisely
    on rare discriminative tokens — model names, dataset names, acronyms like GQA vs MQA —
    which are exactly the terms this keeps. Common words contribute almost no BM25 score
    while costing nearly all of the query time.
    """
    tokens = set(tokenize(query))
    if not tokens:
        return ""
    if corpus_size is None:
        corpus_size = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] or 1

    placeholders = ",".join("?" * len(tokens))
    # chunk_df is the indexed materialization of chunks_vocab; see the schema note. Fall
    # back to the virtual table only if it has not been populated yet.
    freq: dict[str, int] = {}
    for table in ("chunk_df", "chunks_vocab"):
        try:
            freq = dict(
                conn.execute(
                    f"SELECT term, doc FROM {table} WHERE term IN ({placeholders})",
                    list(tokens),
                ).fetchall()
            )
        except sqlite3.OperationalError:
            continue
        if freq:
            break

    ceiling = max(int(corpus_size * df_ceiling_frac), 1)
    # A token absent from the vocab is genuinely absent from the corpus, so it cannot
    # match anything; drop it rather than treating df=0 as "maximally rare".
    scored = [(t, freq[t]) for t in tokens if freq.get(t, 0) > 0]
    scored.sort(key=lambda kv: kv[1])
    keep = [t for t, df in scored if df <= ceiling][:max_terms]
    if not keep:
        # Every term is common. Falling back to "the rarest few" still ORs several
        # enormous posting lists: a query of only common words measured 704 ms this way.
        # Take just the single rarest, and if even that appears in a large slice of the
        # corpus, skip BM25 entirely — a term that common carries almost no BM25 signal,
        # so the only thing the scan buys is latency. Dense retrieval covers the query.
        rarest_df = scored[0][1] if scored else 0
        if rarest_df > corpus_size * 0.05:
            return ""
        keep = [scored[0][0]]
    return " OR ".join(f'"{t}"' for t in keep)


def hydrate(conn: sqlite3.Connection, chunk_ids: list[int]) -> dict[int, Hit]:
    """Fetch chunk rows plus the titles needed to render a citation."""
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"""
        SELECT c.chunk_id, c.vector_row, c.arxiv_id, c.version, c.section_anchor,
               c.anchor_start, c.char_start, c.anchor_end, c.char_end, c.kind, c.text,
               p.title AS paper_title, s.title AS section_title
        FROM chunks c
        JOIN papers p ON p.arxiv_id = c.arxiv_id
        LEFT JOIN sections s ON s.arxiv_id = c.arxiv_id
                            AND s.version  = c.version
                            AND s.anchor   = c.section_anchor
        WHERE c.chunk_id IN ({placeholders})
        """,
        chunk_ids,
    ).fetchall()
    return {
        r["chunk_id"]: Hit(
            chunk_id=r["chunk_id"], vector_row=r["vector_row"] if r["vector_row"] is not None else -1,
            score=0.0, arxiv_id=r["arxiv_id"], version=r["version"],
            section_anchor=r["section_anchor"] or "", section_title=r["section_title"] or "",
            anchor_start=r["anchor_start"], char_start=r["char_start"],
            anchor_end=r["anchor_end"], char_end=r["char_end"],
            kind=r["kind"] or "", text=r["text"], paper_title=r["paper_title"] or "",
        )
        for r in rows
    }
