"""Chunk-to-chunk training pairs from citation edges (multiple-instance learning).

**Why not citation contexts.** The obvious construction — anchor on the paragraph that
contains the citation, positive on the cited paper's abstract — is measurably biased.
Across 131k extracted contexts, **90% sit in Introduction, Related Work or Background**
(55% Introduction, 33% Related Work). Training on those teaches "related-works prose ->
abstract", which is a narrow register that appears nowhere in the actual task: at serving
time a reader highlights a method paragraph or asks about an ablation, and we search body
chunks. The pairing would optimise a skill the product never exercises.

**What this does instead.** A citation A -> B is treated as a *bag-level* label: somewhere
in A there is content that relates to somewhere in B. Both sides are ordinary body chunks,
so the training distribution matches the serving distribution on both ends, and nothing
longer than a chunk is ever encoded.

The alignment (which chunk of A goes with which of B) is latent. It is pooled with
LogSumExp over all m x m candidate pairs rather than argmax, for a specific reason: argmax
routes every gradient through whichever pair the *current* model already likes best, which
amplifies its existing beliefs instead of teaching it the citation structure it lacks. Soft
pooling spreads the gradient and lets a currently-underrated pair contribute.

The supervision is sound even where the alignment is wrong, which is the property that
makes this work: the negatives are chunks from *unrelated papers*, so the model is being
asked whether some part of A belongs with some part of B more than with a random paper.
That comparison is true regardless of which internal pairing carries it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass
class Bag:
    src: str
    dst: str
    src_chunks: list[str]
    dst_chunks: list[str]


# EXISTS rather than two JOINs: the join form makes SQLite materialise 5.6M rows against
# the papers table twice and takes minutes; this returns in ~20s over the same data.
SELECT_EDGES = """
SELECT c.src, c.dst FROM citations c
WHERE EXISTS (SELECT 1 FROM papers a WHERE a.arxiv_id = c.src
              AND a.fulltext_status='ok' AND a.in_scope=1 AND a.n_chunks >= ?)
  AND EXISTS (SELECT 1 FROM papers b WHERE b.arxiv_id = c.dst
              AND b.fulltext_status='ok' AND b.in_scope=1 AND b.n_chunks >= ?)
"""


def usable_edges(conn: sqlite3.Connection, min_chunks: int = 8,
                 limit: int = 0) -> list[tuple[str, str]]:
    """Citation edges where BOTH endpoints have full text indexed."""
    sql = SELECT_EDGES + (" LIMIT ?" if limit else "")
    params = (min_chunks, min_chunks, limit) if limit else (min_chunks, min_chunks)
    return [(r["src"], r["dst"]) for r in conn.execute(sql, params)]


def load_chunk_pool(conn: sqlite3.Connection, papers: set[str],
                    per_paper: int = 24, min_chars: int = 240) -> dict[str, list[str]]:
    """Body chunks per paper, capped.

    Captions, bare equations and reference fragments are excluded: they are legitimate
    retrieval targets but terrible training anchors, since their meaning lives in
    surrounding text the model is not being shown.
    """
    if not papers:
        return {}
    out: dict[str, list[str]] = {}
    ph = ",".join("?" * len(papers))
    rows = conn.execute(
        f"""SELECT arxiv_id, text FROM chunks
            WHERE arxiv_id IN ({ph}) AND kind IN ('body', 'abstract')
              AND length(text) >= ?
            ORDER BY arxiv_id, ordinal""",
        [*papers, min_chars],
    ).fetchall()
    for r in rows:
        bucket = out.setdefault(r["arxiv_id"], [])
        if len(bucket) < per_paper:
            bucket.append(r["text"])
    return out


def build_bags(conn: sqlite3.Connection, min_chunks: int = 8, per_paper: int = 24,
               limit_edges: int = 0) -> list[Bag]:
    edges = usable_edges(conn, min_chunks=min_chunks, limit=limit_edges)
    papers = {p for e in edges for p in e}
    pool = load_chunk_pool(conn, papers, per_paper=per_paper)
    bags = []
    for src, dst in edges:
        s, d = pool.get(src), pool.get(dst)
        if s and d and len(s) >= 2 and len(d) >= 2:
            bags.append(Bag(src, dst, s, d))
    return bags


def edge_stats(conn: sqlite3.Connection, min_chunks: int = 8) -> dict[str, int]:
    total = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
    usable = conn.execute(
        SELECT_EDGES.replace("SELECT c.src, c.dst", "SELECT COUNT(*)"),
        (min_chunks, min_chunks),
    ).fetchone()[0]
    both = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE fulltext_status='ok' AND in_scope=1"
    ).fetchone()[0]
    return {"citation_edges": total, "usable_edges": usable, "papers_with_fulltext": both}
