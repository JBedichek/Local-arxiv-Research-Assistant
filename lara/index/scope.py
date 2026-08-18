"""Topic-scoped residency — keep the relevant fraction of the corpus in RAM (PLAN.md D22).

The tier-1 index costs ``n_chunks x dim_truncated`` bytes and is loaded before anything
else, which makes it the binding memory constraint on a small machine rather than the
generator everyone worries about. At 28.7 M chunks it is 7.3 GB int8, and no laptop is
going to hold that alongside a model.

So let the user say what they care about — "data selection for LLMs" — and keep only that
slice resident:

    score(paper) = max over topics of cos(paper_vector, topic_vector)
    keep         = top `fraction` by score, unioned with 1-hop citation expansion
    resident     = chunks of kept papers

**Nothing is deleted.** This is a load-time decision. Three properties make it safe rather
than lossy:

*BM25 is the safety net, and it is free.* Retrieval already fuses dense with lexical, and
FTS5 covers every chunk on disk regardless of residency. An off-topic question still
matches lexically and reciprocal rank fusion folds it back in, so the failure mode is
reduced semantic recall outside the topic, not blindness.

*The knob is cheap.* Re-scoring and re-slicing costs seconds and needs no re-embedding —
unlike ``dim_truncated``, which invalidates every vector.

*Promotion on demand.* Tier 0 pins the open paper, so following a citation out of the kept
set still works.

Two details that matter more than the fraction itself:

**max over topics, not mean.** A paper perfectly on one of three interests must not be
penalised for ignoring the other two — the mean would rank a mediocre generalist above a
perfect specialist, which is backwards for a corpus you are pruning by interest.

**Citation expansion.** A paper cited by many kept papers is worth keeping even when its
abstract does not match: "Adam: A Method for Stochastic Optimization" scores near zero
against "data selection for LLMs" and you obviously want it. Topical similarity finds what
a field talks *about*; the citation graph finds what it is built *on*. The 6.8 M-edge graph
is already there, so this costs one query.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCOPE_DIRNAME = "scope"


@dataclass
class ScopedPaper:
    arxiv_id: str
    score: float
    via: str            # "topic" | "citation"
    title: str = ""


@dataclass
class Scope:
    """A resolved keep-set. Persisted so it cannot drift between restarts."""

    topics: list[str]
    fraction: float
    expand_min_citations: int
    papers: list[ScopedPaper] = field(default_factory=list)
    rows: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    created_utc: str = ""
    corpus_papers: int = 0
    corpus_chunks: int = 0
    dim_truncated: int = 256

    # ── sizing ────────────────────────────────────────────────────────────────────

    @property
    def n_papers(self) -> int:
        return len(self.papers)

    @property
    def n_rows(self) -> int:
        return int(self.rows.size)

    def resident_bytes(self, dtype_bytes: int = 2) -> int:
        """Tier-1 cost of this keep-set. ``dtype_bytes`` is 2 for the current fp16
        residency, 1 once D19 lands."""
        return self.n_rows * self.dim_truncated * dtype_bytes

    def by_via(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.papers:
            out[p.via] = out.get(p.via, 0) + 1
        return out

    # ── persistence ───────────────────────────────────────────────────────────────

    @staticmethod
    def dir_for(data_root: Path) -> Path:
        return Path(data_root) / SCOPE_DIRNAME

    def save(self, data_root: Path) -> Path:
        d = self.dir_for(data_root)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "rows.npy", self.rows)
        (d / "scope.json").write_text(json.dumps({
            "topics": self.topics,
            "fraction": self.fraction,
            "expand_min_citations": self.expand_min_citations,
            "created_utc": self.created_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                             time.gmtime()),
            "corpus_papers": self.corpus_papers,
            "corpus_chunks": self.corpus_chunks,
            "dim_truncated": self.dim_truncated,
            "n_papers": self.n_papers,
            "n_rows": self.n_rows,
            "papers": [{"id": p.arxiv_id, "score": round(p.score, 5), "via": p.via}
                       for p in self.papers],
        }, indent=1))
        return d

    @classmethod
    def load(cls, data_root: Path) -> "Scope | None":
        d = cls.dir_for(data_root)
        meta_path, rows_path = d / "scope.json", d / "rows.npy"
        if not meta_path.exists() or not rows_path.exists():
            return None
        m = json.loads(meta_path.read_text())
        return cls(
            topics=m["topics"], fraction=m["fraction"],
            expand_min_citations=m.get("expand_min_citations", 0),
            papers=[ScopedPaper(p["id"], p["score"], p["via"]) for p in m.get("papers", [])],
            rows=np.load(rows_path),
            created_utc=m.get("created_utc", ""),
            corpus_papers=m.get("corpus_papers", 0),
            corpus_chunks=m.get("corpus_chunks", 0),
            dim_truncated=m.get("dim_truncated", 256),
        )

    @classmethod
    def clear(cls, data_root: Path) -> bool:
        d = cls.dir_for(data_root)
        removed = False
        for name in ("scope.json", "rows.npy"):
            p = d / name
            if p.exists():
                p.unlink()
                removed = True
        return removed


# ── building ──────────────────────────────────────────────────────────────────────


def score_papers(topic_vecs: np.ndarray, paper_int8: np.ndarray,
                 block: int = 200_000) -> np.ndarray:
    """Similarity of every paper to the nearest topic. Returns one score per paper row.

    **Max over topics, not mean** — see the module docstring. Blocked so a large corpus
    does not need a float32 copy of the whole paper matrix at once.
    """
    q = np.ascontiguousarray(topic_vecs, dtype=np.float32)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)

    n = paper_int8.shape[0]
    out = np.empty(n, dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        part = paper_int8[start:stop].astype(np.float32)
        part /= np.maximum(np.linalg.norm(part, axis=1, keepdims=True), 1e-9)
        out[start:stop] = (part @ q.T).max(axis=1)
    return out


def expand_by_citations(conn: sqlite3.Connection, kept: set[str],
                        min_citations: int, limit: int = 0) -> list[tuple[str, int]]:
    """Papers cited by at least ``min_citations`` of the kept set, ranked by that count.

    This is the "you want the Adam paper" case: foundational work whose abstract shares no
    vocabulary with the topic, but which the topic's papers all lean on. Restricted to
    papers that are themselves in the corpus — an edge to something we never indexed has
    no vectors to make resident.
    """
    if min_citations <= 0 or not kept:
        return []

    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _scope_seed (arxiv_id TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM _scope_seed")
    conn.executemany("INSERT OR IGNORE INTO _scope_seed VALUES (?)", ((k,) for k in kept))

    sql = """
        SELECT c.dst, COUNT(*) AS n
        FROM citations c
        JOIN _scope_seed s ON c.src = s.arxiv_id
        WHERE c.dst NOT IN (SELECT arxiv_id FROM _scope_seed)
          AND EXISTS (SELECT 1 FROM papers p WHERE p.arxiv_id = c.dst AND p.n_chunks > 0)
        GROUP BY c.dst
        HAVING n >= ?
        ORDER BY n DESC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [(r[0], int(r[1])) for r in conn.execute(sql, (min_citations,))]


def resident_rows(conn: sqlite3.Connection, kept: set[str]) -> np.ndarray:
    """Vector rows for every embedded chunk of the kept papers, sorted ascending.

    Sorted because :class:`~lara.index.search.DenseIndex` binary-searches this array to
    translate global rows to local indices.
    """
    if not kept:
        return np.empty(0, dtype=np.int64)
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _scope_keep (arxiv_id TEXT PRIMARY KEY)")
    conn.execute("DELETE FROM _scope_keep")
    conn.executemany("INSERT OR IGNORE INTO _scope_keep VALUES (?)", ((k,) for k in kept))
    rows = [r[0] for r in conn.execute(
        "SELECT c.vector_row FROM chunks c JOIN _scope_keep k ON c.arxiv_id = k.arxiv_id "
        "WHERE c.vector_row IS NOT NULL"
    )]
    return np.sort(np.asarray(rows, dtype=np.int64))


def build(conn: sqlite3.Connection, topic_vecs: np.ndarray, topics: list[str],
          paper_int8: np.ndarray, row_to_id: dict[int, str],
          fraction: float, expand_min_citations: int = 3,
          dim_truncated: int = 256) -> Scope:
    """Score, cut, expand, and collect the resident rows."""
    scores = score_papers(topic_vecs, paper_int8)

    ids = np.array([row_to_id.get(i, "") for i in range(paper_int8.shape[0])], dtype=object)
    have = ids != ""
    idx = np.nonzero(have)[0]
    order = idx[np.argsort(-scores[idx], kind="stable")]

    n_keep = max(1, int(round(len(order) * float(fraction))))
    chosen = order[:n_keep]

    papers = [ScopedPaper(ids[i], float(scores[i]), "topic") for i in chosen]
    kept = {p.arxiv_id for p in papers}

    for aid, n in expand_by_citations(conn, kept, expand_min_citations):
        papers.append(ScopedPaper(aid, float(n), "citation"))
        kept.add(aid)

    rows = resident_rows(conn, kept)
    total_chunks = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE vector_row IS NOT NULL").fetchone()[0]

    return Scope(
        topics=topics, fraction=float(fraction),
        expand_min_citations=int(expand_min_citations),
        papers=papers, rows=rows,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        corpus_papers=int(have.sum()), corpus_chunks=int(total_chunks),
        dim_truncated=int(dim_truncated),
    )


def cut_preview(conn: sqlite3.Connection, topic_vecs: np.ndarray,
                paper_int8: np.ndarray, row_to_id: dict[int, str],
                fraction: float, window: int = 8) -> dict:
    """Papers immediately either side of the cut line, with titles.

    A knob whose effect you cannot see is a knob you cannot set. A similarity-ranked cut is
    not a clean topical boundary — title+abstract embeddings are decent but the tail is
    fuzzy — so showing what is about to be dropped is what makes the fraction meaningful.
    """
    scores = score_papers(topic_vecs, paper_int8)
    ids = np.array([row_to_id.get(i, "") for i in range(paper_int8.shape[0])], dtype=object)
    idx = np.nonzero(ids != "")[0]
    order = idx[np.argsort(-scores[idx], kind="stable")]
    n_keep = max(1, int(round(len(order) * float(fraction))))

    def rows_for(sel):
        out = []
        for rank, i in sel:
            aid = ids[i]
            r = conn.execute("SELECT title FROM papers WHERE arxiv_id=?", (aid,)).fetchone()
            out.append({"rank": rank + 1, "arxiv_id": aid, "score": float(scores[i]),
                        "title": (r[0] if r else "") or ""})
        return out

    lo = max(0, n_keep - window)
    kept_edge = [(r, order[r]) for r in range(lo, n_keep)]
    dropped_edge = [(r, order[r]) for r in range(n_keep, min(len(order), n_keep + window))]
    return {
        "n_total": len(order), "n_keep": n_keep,
        "top": rows_for([(r, order[r]) for r in range(min(5, len(order)))]),
        "kept_edge": rows_for(kept_edge),
        "dropped_edge": rows_for(dropped_edge),
    }
