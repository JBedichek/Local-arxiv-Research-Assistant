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

import numpy as np
import torch

from lara import device as dev


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
        base = f"/p/{self.arxiv_id}v{self.version}#{self.anchor_start}:{self.char_start}"
        if self.anchor_end != self.anchor_start:
            return f"{base}-{self.anchor_end}:{self.char_end}"
        return f"{base}-{self.char_end}"


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


class DenseIndex:
    """Exact search as a GPU matmul. No ANN structure, no build step.

    ``row_ids`` makes the index **partially resident** (PLAN.md D22): only those global
    vector rows are loaded, and the class translates between its own dense local indices
    and the global row numbers every caller speaks. Without it the index covers the whole
    corpus and local index == global row, which is the original behaviour.

    Residency is what lets a large corpus run on a small machine — the rest of the corpus
    stays reachable through BM25 over FTS5 and the tier-2 fp16 mmap, both of which are
    complete regardless of what is resident here.
    """

    def __init__(self, int8_matrix: np.ndarray, device: str | None = None,
                 block: int = 1_000_000, row_ids: np.ndarray | None = None) -> None:
        self.device = dev.resolve(device)
        device = self.device
        if row_ids is not None:
            # Sorted is a precondition, not a nicety: the global -> local translation is a
            # binary search, and an unsorted array would silently return wrong rows.
            row_ids = np.ascontiguousarray(row_ids, dtype=np.int64)
            if row_ids.size and not np.all(np.diff(row_ids) > 0):
                row_ids = np.unique(row_ids)
            int8_matrix = int8_matrix[row_ids]
        self.row_ids = row_ids
        n, dim = int8_matrix.shape
        # Built block-wise straight into the destination tensor. Converting the whole
        # matrix at once needs a full float32 copy — 11.4 GB at 11M vectors — which the
        # caching allocator then holds, so cuda:0 reported 24.7 GB for 6.5 GB of resident
        # data. Blocks keep the transient at ~1 GB regardless of corpus size.
        self.matrix = torch.empty((n, dim), dtype=torch.float16, device=device)
        for start in range(0, n, block):
            stop = min(start + block, n)
            part = torch.from_numpy(int8_matrix[start:stop]).to(device).float()
            # int8 rows are quantized from unit vectors; renormalize after the cast so the
            # inner product is a true cosine.
            self.matrix[start:stop] = torch.nn.functional.normalize(part, dim=1).half()
            del part
        dev.empty_cache(device)
        self.n, self.dim = self.matrix.shape

    def vram_bytes(self) -> int:
        return self.matrix.element_size() * self.matrix.nelement()

    @property
    def resident(self) -> bool:
        return self.row_ids is not None

    def to_local(self, rows: np.ndarray) -> np.ndarray:
        """Global vector rows -> local matrix indices, dropping any not resident.

        Callers hand us global rows from SQLite (scoping to a paper, a citation
        neighbourhood, tier 0). Under residency some of those rows are not loaded, and
        they have to be dropped rather than silently indexing the wrong vector.
        """
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        if self.row_ids is None:
            return rows
        if rows.size == 0 or self.row_ids.size == 0:
            return np.empty(0, dtype=np.int64)
        pos = np.searchsorted(self.row_ids, rows)
        pos_clipped = np.minimum(pos, self.row_ids.size - 1)
        return pos[self.row_ids[pos_clipped] == rows]

    def to_global(self, local: np.ndarray) -> np.ndarray:
        """Local matrix indices -> global vector rows."""
        return local if self.row_ids is None else self.row_ids[local]

    def search(
        self, query: np.ndarray, k: int = 200, rows: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k by inner product. ``rows`` restricts the search (tier 0, graph scoping).

        Returns **global** vector rows in both cases, so residency is invisible to callers.
        """
        q = torch.from_numpy(np.ascontiguousarray(query, dtype=np.float16)).to(self.device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if rows is None:
            scores = q @ self.matrix.T
            top = torch.topk(scores, min(k, self.n), dim=1)
            return (self.to_global(top.indices[0].cpu().numpy()),
                    top.values[0].float().cpu().numpy())

        local = self.to_local(rows)
        if local.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        subset = torch.from_numpy(local).to(self.device)
        scores = q @ self.matrix[subset].T
        top = torch.topk(scores, min(k, subset.numel()), dim=1)
        picked = subset[top.indices[0]].cpu().numpy()
        return self.to_global(picked), top.values[0].float().cpu().numpy()

    def paper_scores(
        self, query: np.ndarray, rows_by_paper: dict[str, np.ndarray], top_n: int = 3
    ) -> dict[str, float]:
        """Heatmap scoring (R9): mean of each paper's top-N chunk similarities.

        Not the mean over all chunks — that dilutes, so a 100-chunk paper with one
        perfect chunk scores low, answering "is this paper *about* the query" when
        exploration needs "does it *answer* the query". Not the plain max either, which
        lets a single spurious chunk light up a node. Top-3 mean keeps max's semantics
        with resistance to one outlier.
        """
        q = torch.from_numpy(np.ascontiguousarray(query, dtype=np.float16)).to(self.device)
        if q.ndim == 2:
            q = q[0]
        out: dict[str, float] = {}
        for paper, rows in rows_by_paper.items():
            if rows.size == 0:
                continue
            local = self.to_local(rows)
            if local.size == 0:
                continue          # paper is not resident; heatmap falls back to tier 2
            subset = torch.from_numpy(local).to(self.device)
            scores = self.matrix[subset] @ q
            n = min(top_n, scores.numel())
            out[paper] = float(torch.topk(scores, n).values.mean())
        return out


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
