"""Evaluation harness. Runs before and after any fine-tune (PLAN.md §5.2).

Two tasks, deliberately different in kind:

**citation retrieval** — given the paragraph in A that cites B, rank B against a pool of
distractors. This is the task the fine-tune optimises, so it should improve; if it does
not, the training did nothing.

**paraphrase retrieval** — given a paper's abstract, retrieve that paper's own body
chunks against a pool of other papers' chunks. The fine-tune does *not* optimise this,
which is the point: it is the canary for catastrophic forgetting. A model that wins on
citations while losing here has traded general retrieval for a narrow skill, and would
make the product worse while the headline metric improved.

Both are label-free — the corpus supplies its own ground truth — so the eval set costs
nothing to build and can be regenerated whenever the corpus grows.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np


@dataclass
class Task:
    name: str
    queries: list[str]
    positives: list[str]          # gold document text, aligned with queries
    pool: list[str]               # distractor documents


def _held_out_srcs(conn: sqlite3.Connection, frac: float, seed: int) -> set[str]:
    """Split by CITING PAPER, never by pair.

    Splitting pairs at random leaks: two contexts from the same paper share phrasing and
    subject matter, so a model can score well on the test half by memorising the train
    half of the very same document.
    """
    rng = np.random.default_rng(seed)
    srcs = [r[0] for r in conn.execute("SELECT DISTINCT src FROM citation_contexts")]
    srcs.sort()
    idx = rng.permutation(len(srcs))[: max(1, int(len(srcs) * frac))]
    return {srcs[i] for i in idx}


def build_citation_task(conn: sqlite3.Connection, n: int = 800, pool_size: int = 2000,
                        frac_heldout: float = 0.2, seed: int = 0) -> Task:
    held = _held_out_srcs(conn, frac_heldout, seed)
    if not held:
        return Task("citation", [], [], [])
    ph = ",".join("?" * len(held))
    rows = conn.execute(
        f"""SELECT c.context, p.title, p.abstract
            FROM citation_contexts c JOIN papers p ON p.arxiv_id = c.dst
            WHERE c.src IN ({ph}) AND c.n_refs <= 2
              AND p.abstract IS NOT NULL AND length(p.abstract) > 200
            ORDER BY c.id LIMIT ?""",
        [*held, n],
    ).fetchall()
    queries = [r["context"] for r in rows]
    positives = [f"{r['title']}\n{r['abstract']}" for r in rows]

    pool = [
        f"{r['title']}\n{r['abstract']}"
        for r in conn.execute(
            "SELECT title, abstract FROM papers WHERE in_scope=1 AND abstract IS NOT NULL "
            "AND length(abstract) > 200 ORDER BY arxiv_id LIMIT ?", (pool_size,)
        )
    ]
    return Task("citation", queries, positives, pool)


def build_paraphrase_task(conn: sqlite3.Connection, n: int = 800,
                          pool_size: int = 2000, seed: int = 0) -> Task:
    rows = conn.execute(
        """SELECT p.abstract, c.text
           FROM papers p JOIN chunks c ON c.arxiv_id = p.arxiv_id
           WHERE p.fulltext_status='ok' AND c.kind='body' AND length(c.text) > 400
             AND p.abstract IS NOT NULL AND length(p.abstract) > 200
           GROUP BY p.arxiv_id ORDER BY p.arxiv_id LIMIT ?""",
        (n,),
    ).fetchall()
    queries = [r["abstract"] for r in rows]
    positives = [r["text"] for r in rows]
    pool = [
        r[0] for r in conn.execute(
            "SELECT text FROM chunks WHERE kind='body' AND length(text) > 400 "
            "ORDER BY chunk_id DESC LIMIT ?", (pool_size,)
        )
    ]
    return Task("paraphrase", queries, positives, pool)


def score_task(model, task: Task, batch_size: int = 128,
               query_prompt: str = "task: search result | query: ") -> dict[str, float]:
    """Rank each gold document against the pool. Returns MRR and Recall@k."""
    if not task.queries:
        return {"n": 0}
    docs = task.positives + task.pool
    q = model.encode(task.queries, prompt=query_prompt, batch_size=batch_size,
                     convert_to_numpy=True, normalize_embeddings=True,
                     show_progress_bar=False).astype(np.float32)
    d = model.encode(docs, batch_size=batch_size, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

    scores = q @ d.T
    gold = np.arange(len(task.queries))
    gold_scores = scores[gold, gold]
    # Rank = how many documents outscore the gold one. Ties count as ahead, which is the
    # pessimistic reading and avoids flattering a model that returns constant vectors.
    ranks = (scores > gold_scores[:, None]).sum(axis=1) + 1

    return {
        "n": len(task.queries),
        "mrr": float((1.0 / ranks).mean()),
        "r@1": float((ranks <= 1).mean()),
        "r@5": float((ranks <= 5).mean()),
        "r@10": float((ranks <= 10).mean()),
        "median_rank": float(np.median(ranks)),
    }


def collapse_stats(model, conn: sqlite3.Connection, n: int = 300) -> dict[str, float]:
    """Detect representation collapse.

    A destroyed encoder does not usually produce garbage — it produces vectors that are
    all nearly identical, so every ranking is arbitrary while the loss still looks
    finite. Mean off-diagonal cosine over unrelated texts is the direct test: a healthy
    encoder sits near 0.2-0.5 on arXiv prose, a collapsed one approaches 1.0.
    """
    rows = conn.execute(
        "SELECT text FROM chunks WHERE kind='body' AND length(text) > 400 "
        "ORDER BY chunk_id LIMIT ?", (n,)
    ).fetchall()
    if len(rows) < 16:
        return {}
    v = model.encode([r[0] for r in rows], batch_size=64, convert_to_numpy=True,
                     normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    sim = v @ v.T
    off = sim[~np.eye(len(v), dtype=bool)]
    return {"mean_cos": float(off.mean()), "p95_cos": float(np.percentile(off, 95)),
            "std_cos": float(off.std())}


def run(model, conn: sqlite3.Connection, n: int = 800, pool_size: int = 2000,
        seed: int = 0) -> dict[str, dict[str, float]]:
    return {
        "collapse": collapse_stats(model, conn),
        "citation": score_task(model, build_citation_task(conn, n, pool_size, seed=seed)),
        "paraphrase": score_task(model, build_paraphrase_task(conn, n, pool_size, seed=seed)),
    }


def format_report(before: dict, after: dict | None = None) -> str:
    lines = []
    for task in ("citation", "paraphrase"):
        b = before.get(task, {})
        if not b.get("n"):
            continue
        lines.append(f"  {task} (n={b['n']})")
        for k in ("mrr", "r@1", "r@5", "r@10"):
            if after is None:
                lines.append(f"    {k:6} {b[k]:.4f}")
            else:
                a = after[task][k]
                delta = a - b[k]
                arrow = "+" if delta > 0 else ""
                lines.append(f"    {k:6} {b[k]:.4f} -> {a:.4f}  ({arrow}{delta:.4f})")
    return "\n".join(lines)


# ── synthetic question generation ──────────────────────────────────────────────────

SYNTH_SYSTEM = """You write evaluation questions for a scientific search engine.

Given a passage from an arXiv machine-learning paper, write ONE question that the passage
answers specifically. Reply with the question and nothing else.

Constraints, all of which matter:
- The question must be answerable from this passage alone.
- Do NOT reuse the passage's distinctive wording. Paraphrase the technical terms where a
  natural synonym exists, and never quote a phrase of more than two words from it.
- Ask what a researcher would ask before they had found the passage, not a comprehension
  question written by someone already looking at it.
- No meta-references: never write "the passage", "this paper", "the authors", "Figure 2".
- One sentence."""


async def generate_questions(cfg, conn, n: int = 200, model: str | None = None,
                             seed: int = 0) -> list[dict]:
    """LLM-written questions paired with the chunk they came from.

    This is the only eval that matches the product's actual task shape — a natural
    language question against body chunks. The structural tasks use paragraphs and
    abstracts as queries, which is not what anyone types.

    **The prompt fights lexical leakage on purpose.** A generator that echoes the chunk's
    rare terms produces questions any retriever solves by keyword overlap, and the metric
    then measures vocabulary matching rather than retrieval quality — flattering every
    model equally and distinguishing none of them. Forbidding quotation and asking for
    paraphrase is what keeps the task discriminative.
    """
    import numpy as np

    from lara.serve.generate import stream_answer

    rng = np.random.default_rng(seed)
    rows = conn.execute(
        """SELECT chunk_id, arxiv_id, text FROM chunks
           WHERE kind='body' AND length(text) BETWEEN 500 AND 1400
           ORDER BY chunk_id LIMIT 40000"""
    ).fetchall()
    if not rows:
        return []
    pick = rng.permutation(len(rows))[:n]

    out = []
    for i in pick:
        r = rows[int(i)]
        buf = ""
        try:
            async for tok in stream_answer(
                cfg, f"Passage:\n{r['text'][:1400]}\n\nQuestion:", [],
                system=SYNTH_SYSTEM, model=model, temperature=0.3,
                max_tokens=90, raw_user=True,
            ):
                buf += tok
        except Exception:
            continue
        q = buf.strip().split("\n")[0].strip().strip('"')
        if len(q) > 25 and q.endswith("?"):
            out.append({"question": q, "chunk_id": r["chunk_id"],
                        "arxiv_id": r["arxiv_id"], "text": r["text"]})
    return out


def leakage_score(question: str, chunk: str) -> float:
    """Fraction of the question's content words that appear verbatim in the chunk.

    Reported alongside the metric so a suspiciously easy eval set is visible rather than
    silently inflating every score.
    """
    import re
    stop = set("what how why when where which does do is are the a an of to in on for and "
               "or with that this it its can could would should may might".split())
    qw = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in stop and len(w) > 2}
    if not qw:
        return 0.0
    cw = set(re.findall(r"[a-z0-9]+", chunk.lower()))
    return len(qw & cw) / len(qw)


def build_synthetic_task(items: list[dict], conn, pool_size: int = 2000) -> Task:
    """Questions as queries, their source chunks as gold, other chunks as distractors."""
    if not items:
        return Task("synthetic", [], [], [])
    gold_ids = {it["chunk_id"] for it in items}
    pool = [
        r[0] for r in conn.execute(
            "SELECT text FROM chunks WHERE kind='body' AND length(text) > 400 "
            "AND chunk_id NOT IN (%s) ORDER BY chunk_id DESC LIMIT ?"
            % ",".join(str(i) for i in gold_ids),
            (pool_size,),
        )
    ]
    return Task("synthetic",
                [it["question"] for it in items],
                [it["text"] for it in items],
                pool)
