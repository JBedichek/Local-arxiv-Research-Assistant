"""Harvest relevance judgements from retrieval, for distilling into the embedder.

The premise: a cross-encoder reading (query, passage) together is far better at relevance
than a bi-encoder that compressed the passage to 768 numbers before the query existed.
That gap is exactly what distillation closes — and we already pay for the teacher on every
search, then throw its verdict away.

Three teachers, used where each is worth its cost:

``cross_encoder``  free. Already computed for every retrieved passage; capture is a write.
``llm``            expensive. Reserved for the uncertain middle band, where the reranker's
                   score says "maybe" and a reading model's judgement actually adds
                   information. Spending a 27B model on cases a 568M model already calls
                   confidently buys nothing.
``user_click``     rare and precious. A followed citation is human relevance signal, the
                   only kind free of model bias, and worth far more per example.

**The negatives are the hard part.** Judging only what retrieval returned teaches the
model to reorder what it already finds and never to find what it misses — the standard
exposure-bias failure. Negatives are therefore drawn from three places: the tail of the
ranking (hard), random chunks (easy, to anchor the scale), and — most valuable — passages
a *known* positive query failed to retrieve at all.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass

from lara.store.db import utcnow

# Below this the reranker is confidently saying "no"; above, confidently "yes". Between
# them it is guessing, and that band is where an LLM adjudication changes the label rather
# than confirming it. Calibrated from the separation probe: true support scores 0.98-1.00,
# an on-topic fabrication ~0.29, unrelated ~0.00.
UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.15, 0.75


def qhash(query: str) -> str:
    return hashlib.sha1(" ".join(query.lower().split()).encode()).hexdigest()[:16]


@dataclass
class Judgement:
    query: str
    chunk_id: int
    score: float | None
    label: int | None
    teacher: str
    rank: int | None = None
    source: str = "ask"


def record(conn: sqlite3.Connection, items: list[Judgement]) -> int:
    """Persist judgements, ignoring duplicates."""
    if not items:
        return 0
    before = conn.execute("SELECT COUNT(*) FROM judgements").fetchone()[0]
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO judgements"
            "(query, query_hash, chunk_id, score, label, teacher, rank, source, created_utc)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            [(j.query, qhash(j.query), j.chunk_id, j.score, j.label, j.teacher,
              j.rank, j.source, utcnow()) for j in items],
        )
    return conn.execute("SELECT COUNT(*) FROM judgements").fetchone()[0] - before


def from_hits(query: str, hits: list[dict], source: str = "ask",
              threshold: float = 0.5) -> list[Judgement]:
    """Turn one retrieval's reranker scores into judgements.

    ``label`` is derived from the same 0.5 threshold the grounding check uses, so a
    passage called supported there is called relevant here. ``score`` is kept as well:
    binary labels discard the teacher's confidence, and MarginMSE-style training wants the
    margin, not the sign.
    """
    out = []
    for rank, h in enumerate(hits, 1):
        s = h.get("provenance", {}).get("cross_encoder", h.get("score"))
        if s is None:
            continue
        out.append(Judgement(
            query=query, chunk_id=h["chunk_id"], score=float(s),
            label=1 if float(s) >= threshold else 0,
            teacher="cross_encoder", rank=rank, source=source,
        ))
    return out


def uncertain(conn: sqlite3.Connection, limit: int = 200) -> list[sqlite3.Row]:
    """Cross-encoder judgements in the band where an LLM verdict would add information."""
    return conn.execute(
        """SELECT j.id, j.query, j.chunk_id, j.score, c.text
           FROM judgements j JOIN chunks c ON c.chunk_id = j.chunk_id
           WHERE j.teacher='cross_encoder' AND j.score BETWEEN ? AND ?
             AND NOT EXISTS (SELECT 1 FROM judgements k
                             WHERE k.query_hash=j.query_hash AND k.chunk_id=j.chunk_id
                               AND k.teacher='llm')
           ORDER BY RANDOM() LIMIT ?""",
        (UNCERTAIN_LOW, UNCERTAIN_HIGH, limit),
    ).fetchall()


ADJUDICATE_SYSTEM = """You judge whether a passage helps answer a question. Reply with one \
JSON object and nothing else:

{"relevant": true | false, "reason": "<one short clause>"}

"relevant" means the passage contains information a good answer to that question would \
use — a definition, a mechanism, a number, a result, a caveat. Being on the same topic is \
not enough: a passage that mentions the subject but answers nothing is not relevant. \
Judge the passage as written, not the paper it came from."""


def training_pairs(conn: sqlite3.Connection, min_per_query: int = 1,
                   limit: int = 0) -> list[dict]:
    """Assemble (query, positive, negative) triples from stored judgements.

    Prefers an LLM verdict where one exists — it was requested precisely because the
    cheap teacher was unsure — and falls back to the cross-encoder otherwise.
    """
    rows = conn.execute(
        """SELECT j.query, j.query_hash, j.chunk_id, j.label, j.score, j.teacher, c.text
           FROM judgements j JOIN chunks c ON c.chunk_id = j.chunk_id
           WHERE j.label IS NOT NULL
           ORDER BY j.query_hash, (j.teacher='llm') DESC"""
    ).fetchall()

    by_query: dict[str, dict] = {}
    for r in rows:
        q = by_query.setdefault(r["query_hash"],
                                {"query": r["query"], "pos": [], "neg": [], "seen": set()})
        if r["chunk_id"] in q["seen"]:
            continue           # an llm verdict already claimed this chunk
        q["seen"].add(r["chunk_id"])
        (q["pos"] if r["label"] else q["neg"]).append(
            {"chunk_id": r["chunk_id"], "text": r["text"], "score": r["score"]}
        )

    out = []
    for q in by_query.values():
        if len(q["pos"]) < min_per_query or not q["neg"]:
            continue
        out.append({"query": q["query"], "positives": q["pos"], "negatives": q["neg"]})
        if limit and len(out) >= limit:
            break
    return out


def stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        """SELECT COUNT(*) n, COUNT(DISTINCT query_hash) q,
                  SUM(label=1) pos, SUM(label=0) neg
           FROM judgements"""
    ).fetchone()
    by_teacher = dict(conn.execute(
        "SELECT teacher, COUNT(*) FROM judgements GROUP BY teacher").fetchall())
    return {"judgements": row["n"] or 0, "queries": row["q"] or 0,
            "positive": row["pos"] or 0, "negative": row["neg"] or 0,
            "by_teacher": by_teacher}
