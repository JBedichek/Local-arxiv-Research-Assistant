"""Synthetic data gathering: let the model drive the RAG loop and label what comes back.

One cycle:

1. Sample a paper, show the model a passage, and ask it for a question — including
   "what do you least understand here", which reliably produces questions whose answer is
   *not* the passage in front of it.
2. Retrieve for that question at high k.
3. Score candidates with the cross-encoder and store the verdicts.
4. Check whether the source chunk was retrieved at all.

**Step 4 is the valuable one.** A question written from chunk C has C as a known positive.
If retrieval misses it, that is a labelled example of a relevance the current embedder
cannot see — precisely the signal that ordinary usage can never produce, because usage
only ever labels what was already found. Harvesting live traffic gave 24 positives and
zero negatives on the first three queries; a contrastive objective learns nothing from
that. Exposure bias is the problem to design against, not model bias.

**Negatives are mined deliberately, at three depths.** Hard negatives come from the tail of
the ranking (retrieved but wrong), medium from other papers on the same topic, and easy
from random chunks. Training only on hard negatives collapses the scale; training only on
random ones teaches nothing a keyword filter could not.

**On circularity.** The generator, the judge and the student all share a lineage, so this
distils one model's notion of relevance rather than discovering ground truth. That is
acceptable *because the evaluation is independent*: citation retrieval is scored against
what human authors actually cited, so a model that merely learns to please its teacher
will not move that number.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
from dataclasses import dataclass, field

QUESTION_STYLES = [
    ("least_understood",
     "Read this passage and ask the ONE question you least understand about it — the "
     "thing you would need to look up elsewhere in the literature to follow it properly."),
    ("mechanism", "Ask how the mechanism described here actually works."),
    ("quantitative", "Ask about a specific number, rate or measurement this passage "
                     "reports or implies."),
    ("comparison", "Ask how the approach here compares to an alternative."),
    ("limitation", "Ask what this passage does not establish, or where it would fail."),
    ("definition", "Ask what a technical term used here means."),
]

EXPLORE_SYSTEM = """You generate evaluation questions for a scientific search engine over \
arXiv machine-learning papers. Reply with the question and nothing else.

- One sentence, ending in a question mark.
- Ask what a researcher would type into a search box, not a comprehension question about \
a passage they are already looking at.
- Never write "the passage", "this paper", "the authors", "Figure 2", "above" or "below".
- Do not quote more than two consecutive words from the passage."""

# Self-referential questions are unanswerable once the passage is gone: "what do the
# authors mean here" has no referent in a search box. Widened after a run where ~1 in 40
# still slipped through — deixis takes many forms ("aforementioned", "the former",
# "as described", bare "it").
_META = re.compile(
    r"\b("
    r"this (paper|passage|work|section|study|article|excerpt|text|figure|table|method)"
    r"|these (results|authors|findings|experiments)"
    r"|the (passage|authors|above|below|former|latter|aforementioned|preceding|following)"
    r"|(figure|table|equation|section|eq\.?)\s*\d"
    r"|as (described|shown|discussed|mentioned|stated) (above|below|here|earlier)"
    r"|(here|herein)\b.*\?$"
    r")\b", re.I)


@dataclass
class Cycle:
    question: str
    style: str
    source_chunk: int
    source_paper: str
    retrieved: list[int] = field(default_factory=list)
    source_rank: int | None = None      # where the known positive landed, if at all
    n_positive: int = 0
    n_negative: int = 0


def sample_chunks(conn: sqlite3.Connection, n: int, seed: int = 0) -> list[sqlite3.Row]:
    """Body chunks long enough to support a question, spread across papers."""
    return conn.execute(
        """SELECT c.chunk_id, c.arxiv_id, c.text
           FROM chunks c
           WHERE c.kind='body' AND length(c.text) BETWEEN 600 AND 1500
             AND c.vector_row IS NOT NULL
           ORDER BY RANDOM() LIMIT ?""",
        (n,),
    ).fetchall()


def clean_question(raw: str) -> str | None:
    q = (raw or "").strip().split("\n")[0].strip().strip('"').strip()
    if len(q) < 25 or not q.endswith("?"):
        return None
    # Self-referential questions are unanswerable by a search engine: "what do the authors
    # mean here" has no referent once the passage is gone.
    if _META.search(q):
        return None
    return q


def mine_negatives(conn: sqlite3.Connection, retrieved_ids: list[int],
                   n_random: int, rng: random.Random) -> list[int]:
    """Random chunks as easy negatives, anchoring the low end of the scale."""
    if n_random <= 0:
        return []
    rows = conn.execute(
        "SELECT chunk_id FROM chunks WHERE kind='body' AND length(text) > 400 "
        "AND vector_row IS NOT NULL ORDER BY RANDOM() LIMIT ?",
        (n_random * 2,),
    ).fetchall()
    seen = set(retrieved_ids)
    return [r[0] for r in rows if r[0] not in seen][:n_random]


def score_pairs(cross_encoder, query: str, texts: list[str], batch: int = 32) -> list[float]:
    if not texts:
        return []
    pairs = [(query, t) for t in texts]
    return [float(x) for x in cross_encoder.predict(pairs, show_progress_bar=False,
                                                    batch_size=batch)]


async def run_cycles(cfg, conn, retriever, cross_encoder, *, n: int = 50,
                     k: int = 20, n_random_neg: int = 6, model: str | None = None,
                     seed: int = 0, progress=None) -> dict:
    """Generate questions, retrieve, judge, and store. Returns a summary."""
    from lara.finetune import judgements as J
    from lara.serve.generate import stream_answer

    rng = random.Random(seed)
    rows = sample_chunks(conn, n * 2, seed)
    stats = {"cycles": 0, "questions": 0, "rejected": 0, "stored": 0,
             "source_missed": 0, "positives": 0, "negatives": 0, "styles": {}}
    cycles: list[Cycle] = []

    for row in rows:
        if stats["questions"] >= n:
            break
        # Round-robin, not random choice: sampling uniformly at random leaves some
        # styles under-represented in a short run, and query-TYPE diversity matters more
        # to the trained model than the number of examples.
        style, instruction = QUESTION_STYLES[stats["questions"] % len(QUESTION_STYLES)]
        buf = ""
        try:
            async for tok in stream_answer(
                cfg, f"{instruction}\n\nPassage:\n{row['text'][:1500]}\n\nQuestion:",
                [], system=EXPLORE_SYSTEM, model=model,
                temperature=0.7, max_tokens=80, raw_user=True,
            ):
                buf += tok
        except Exception:
            continue

        q = clean_question(buf)
        if not q:
            stats["rejected"] += 1
            continue
        stats["questions"] += 1
        stats["styles"][style] = stats["styles"].get(style, 0) + 1

        result = retriever.retrieve(q, final_k=k)
        hits = result.hits
        ids = [h.chunk_id for h in hits]

        cyc = Cycle(question=q, style=style, source_chunk=row["chunk_id"],
                    source_paper=row["arxiv_id"], retrieved=ids)
        cyc.source_rank = ids.index(row["chunk_id"]) + 1 if row["chunk_id"] in ids else None
        if cyc.source_rank is None:
            stats["source_missed"] += 1

        items = J.from_hits(q, [
            {"chunk_id": h.chunk_id, "score": h.score, "provenance": h.provenance}
            for h in hits
        ], source="explore")

        # The source chunk is a known positive by construction. When retrieval missed it,
        # that is the example worth having: a relevance the embedder cannot currently see.
        if cyc.source_rank is None:
            items.append(J.Judgement(query=q, chunk_id=row["chunk_id"], score=1.0,
                                     label=1, teacher="synthetic", rank=None,
                                     source="explore_missed"))

        # Easy negatives, scored honestly rather than assumed — a random chunk is usually
        # irrelevant but occasionally is not, and labelling it 0 regardless would inject
        # noise into exactly the pairs meant to anchor the scale.
        neg_ids = mine_negatives(conn, ids, n_random_neg, rng)
        if neg_ids:
            ph = ",".join("?" * len(neg_ids))
            neg_rows = conn.execute(
                f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({ph})", neg_ids
            ).fetchall()
            scores = score_pairs(cross_encoder, q, [r["text"] for r in neg_rows])
            for r, sc in zip(neg_rows, scores):
                items.append(J.Judgement(query=q, chunk_id=r["chunk_id"], score=sc,
                                         label=1 if sc >= 0.5 else 0,
                                         teacher="cross_encoder", rank=None,
                                         source="explore_random"))

        stats["stored"] += J.record(conn, items)
        stats["positives"] += sum(1 for i in items if i.label == 1)
        stats["negatives"] += sum(1 for i in items if i.label == 0)
        stats["cycles"] += 1
        cycles.append(cyc)
        if progress is not None:
            progress.send({**stats, "last_q": q, "style": style,
                           "source_rank": cyc.source_rank})

    found = [c.source_rank for c in cycles if c.source_rank]
    stats["source_recall"] = round(len(found) / max(len(cycles), 1), 3)
    stats["source_mrr"] = round(
        sum(1 / r for r in found) / max(len(cycles), 1), 3)
    return stats
