"""Semantic paper search — "show me papers on Muon learning rate adaptation".

Two evidence sources, fused, because each covers what the other misses:

**Paper-level vectors** (title + abstract) exist for *every* in-scope paper, crawled or
not. This is the breadth half: at the time of writing only ~16% of in-scope papers have
full text, so a chunk-only search would silently hide most of arXiv while looking
perfectly healthy.

**Chunk-level evidence**, aggregated per paper as the mean of its top-3 chunk
similarities. This is the depth half, and it finds papers whose abstract never mentions
the thing you asked about — a method introduced in §4.2, an ablation, a lemma. Abstracts
are written to sell the headline result, so they are systematically bad at this.

Aggregation uses top-3 mean rather than max or overall mean for the reasons in PLAN.md §6:
the overall mean dilutes a 100-chunk paper with one perfect chunk down to nothing, and max
lets a single spurious chunk carry a paper.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PaperHit:
    arxiv_id: str
    title: str
    abstract: str
    authors: str
    categories: str
    submitted: str
    score: float
    n_chunks: int = 0
    fulltext: bool = False
    cited_by: int = 0
    best_chunk: str = ""
    best_anchor: str = ""
    version: int = 1
    evidence: dict[str, float] = field(default_factory=dict)


def aggregate_chunk_hits(
    rows: np.ndarray,
    scores: np.ndarray,
    row_to_paper: dict[int, str],
    top_n: int = 3,
) -> dict[str, float]:
    """Group chunk scores by paper and reduce with a top-N mean."""
    by_paper: dict[str, list[float]] = {}
    for row, score in zip(rows.tolist(), scores.tolist()):
        paper = row_to_paper.get(int(row))
        if paper is not None:
            by_paper.setdefault(paper, []).append(float(score))
    out: dict[str, float] = {}
    for paper, vals in by_paper.items():
        vals.sort(reverse=True)
        out[paper] = float(sum(vals[:top_n]) / min(len(vals), top_n))
    return out


def fuse(
    paper_level: dict[str, float],
    chunk_level: dict[str, float],
    *,
    w_paper: float = 1.0,
    w_chunk: float = 1.0,
) -> dict[str, tuple[float, dict[str, float]]]:
    """Combine the two signals.

    Scores are compared directly rather than by rank fusion: both come from the same
    embedding model in the same cosine space, so they are already commensurate. RRF is
    used for the dense/BM25 hybrid precisely because those two are *not*.

    A paper present in only one source keeps that source's score rather than being
    penalised — missing chunk evidence usually means the paper has not been crawled yet,
    which says nothing about relevance.
    """
    out: dict[str, tuple[float, dict[str, float]]] = {}
    for paper in set(paper_level) | set(chunk_level):
        p = paper_level.get(paper)
        c = chunk_level.get(paper)
        parts = {}
        if p is not None:
            parts["abstract"] = p
        if c is not None:
            parts["fulltext"] = c
        if p is not None and c is not None:
            score = (w_paper * p + w_chunk * c) / (w_paper + w_chunk)
        else:
            score = p if p is not None else c or 0.0
        out[paper] = (score, parts)
    return out


def hydrate_papers(
    conn: sqlite3.Connection, scored: dict[str, tuple[float, dict[str, float]]], limit: int
) -> list[PaperHit]:
    ranked = sorted(scored.items(), key=lambda kv: -kv[1][0])[:limit]
    if not ranked:
        return []
    ids = [p for p, _ in ranked]
    ph = ",".join("?" * len(ids))
    rows = {
        r["arxiv_id"]: r
        for r in conn.execute(
            f"SELECT arxiv_id, title, abstract, authors, categories, submitted_utc, "
            f"n_chunks, fulltext_status, fulltext_version, latest_version, cited_by_count "
            f"FROM papers WHERE arxiv_id IN ({ph})",
            ids,
        )
    }
    hits: list[PaperHit] = []
    for paper, (score, parts) in ranked:
        r = rows.get(paper)
        if r is None:
            continue
        hits.append(PaperHit(
            arxiv_id=paper,
            title=r["title"] or paper,
            abstract=(r["abstract"] or "")[:400],
            authors=r["authors"] or "",
            categories=r["categories"] or "",
            submitted=(r["submitted_utc"] or "")[:10],
            score=score,
            n_chunks=r["n_chunks"] or 0,
            fulltext=r["fulltext_status"] == "ok",
            cited_by=r["cited_by_count"] or 0,
            version=r["fulltext_version"] or r["latest_version"] or 1,
            evidence=parts,
        ))
    return hits
