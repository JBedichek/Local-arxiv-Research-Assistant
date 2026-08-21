"""Hierarchical scope — answer from the open paper before searching 29.5 M chunks.

Someone reading a paper and asking "why does this need a warmup?" almost always means
*in this paper*. Searching the whole corpus first answers a question they did not ask, and
buries the paragraph two sections down under a hundred topically similar passages from
elsewhere. So retrieval walks outward instead:

    1. paper          the open paper's own chunks
    2. neighbourhood  it plus everything it cites and everything citing it
    3. corpus         all 29.5 M chunks

Between tiers a model reads what came back, marks which passages actually help, and
decides whether to widen. Two things make that more than a heuristic.

**Confirmed passages become extra query vectors, fused rather than averaged.** This is
pseudo-relevance feedback with the assumption removed — classic PRF assumes the top-k are
relevant and is wrong about half the time; here a model has checked. The vectors are *not*
averaged into a centroid: normalised embeddings from different subtopics average to a point
close to neither, so each confirmed chunk runs its own search and the ranked lists are
combined by reciprocal rank fusion. Feedback widens recall; the question alone still
decides the final order, which is what keeps the answer from drifting toward whatever
subtopic the confirmed chunks happened to share.

**The middle tier is the one that usually holds the answer.** "How does this compare to
prior work" is answered by the papers this one cites. Escalating straight from one paper
to the whole corpus skips it.

Every decision is recorded. The rejected passages are the valuable half: they were
retrieved, so they are topically close, and then judged unhelpful — exactly the hard
negatives that ordinary usage never produces (the first real queries here yielded 24
positives and zero negatives).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TIERS = ("paper", "neighbourhood", "corpus")

# ── prompts ───────────────────────────────────────────────────────────────────────
#
# Both prompts state what the decision *causes*, not just what is being asked. A model
# told only "is this relevant?" optimises for topical similarity; told that keeping a
# passage turns it into a search key for the whole corpus, it starts weighing whether that
# key would pull in something useful — which is the actual question.

TAG_SYSTEM = """You are selecting which passages from one paper are worth building a \
search on.

For each numbered excerpt, decide whether it helps answer the question.

What your decision causes:
- KEEP: the excerpt is shown to the model that writes the answer, AND its embedding \
becomes an additional search key used to find related passages in other papers. A kept \
excerpt about the wrong subtopic will drag that wider search off course.
- DROP: the excerpt is discarded for this question, and recorded as an example of \
something that looked relevant but was not.

Keep an excerpt if it answers the question, defines a term the question uses, states the \
result or method the question is about, or gives a condition or caveat that changes the \
answer. Do not keep an excerpt merely because it shares vocabulary with the question.

Keeping nothing is a valid and useful answer: it means this paper does not address the \
question, and the search should widen.

Reply with JSON only:
{"keep": [1, 4], "why": "one short clause naming what they contribute"}"""


ESCALATE_SYSTEM = """You are deciding how widely to search for evidence.

You are told what the current tier found. Choose the next scope.

What each choice causes:
- "stop": answer from what is already retrieved. Fastest, and correct when the paper \
already contains the answer. If it does not, the answer will be thin or hedged.
- "neighbourhood": also search the papers this one cites and the papers citing it. Costs \
a few hundred milliseconds. This is the right choice for comparisons with prior work, \
"is this novel", "who else found this", and for terms this paper uses but does not define.
- "corpus": search all 29.5 million chunks from 377,000 papers. Widest and slowest, and \
the most likely to return passages that are topically similar but about a different \
setting. Choose it for general questions that this paper only happens to mention, or when \
the neighbourhood came back empty.

Prefer the narrowest scope that can answer the question. Widening costs latency and adds \
material the reader did not ask about; refusing to widen when the evidence is absent \
produces a confident answer with nothing behind it.

Reply with JSON only:
{"scope": "stop|neighbourhood|corpus", "why": "one short clause"}"""


# ── results ───────────────────────────────────────────────────────────────────────


@dataclass
class TierRun:
    tier: str
    hits: list[dict]
    ms: float
    n_candidates: int = 0
    note: str = ""


@dataclass
class Decision:
    kind: str                    # "tag" | "escalate"
    tier: str
    verdict: str
    why: str = ""
    via: str = "model"           # model | scores | config | fallback
    kept: list[int] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)


@dataclass
class Result:
    hits: list[dict]
    tiers: list[TierRun] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    feedback_ids: list[int] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def path(self) -> str:
        return " -> ".join(t.tier for t in self.tiers)


# ── sizing ────────────────────────────────────────────────────────────────────────


def paper_k(n_chunks: int, frac: float = 0.2, min_k: int = 5, max_k: int = 30) -> int:
    """How many of a paper's own chunks to consider.

    A bare proportion misbehaves at both tails: papers average 78 chunks, but a 300-chunk
    survey at 0.2 yields 60 — more than fits comfortably in one tagging call — while a
    15-chunk workshop paper yields 3, which is too few to find anything. Clamped.
    """
    if n_chunks <= 0:
        return min_k
    return max(min_k, min(max_k, int(round(n_chunks * float(frac)))))


# ── model steps ───────────────────────────────────────────────────────────────────


def to_dict(h) -> dict:
    """A Hit as the rest of the server passes it around.

    Matches the shape `/api/retrieve` emits key for key — note ``section``, not
    ``section_title``: agent.format_excerpts and the answer prompt both read ``section``,
    and a near-miss here would silently drop the section from every excerpt shown to the
    model rather than raising.
    """
    return h if isinstance(h, dict) else h.to_dict()


def _excerpts(hits: list[dict]) -> str:
    out = []
    for i, h in enumerate(hits, 1):
        head = f"{h.get('paper_title') or ''} > {h.get('section') or ''}".strip(" >")
        out.append(f"[{i}] {head}\n{(h.get('text') or '')[:900]}")
    return "\n\n".join(out)


async def tag(cfg, question: str, hits: list[dict], model: str | None,
              stream_answer) -> Decision:
    """Ask which excerpts genuinely help. Falls back to keeping the top few."""
    if not hits:
        return Decision("tag", "paper", "none", "nothing retrieved", via="scores")

    prompt = (f"Question: {question}\n\nExcerpts from the open paper:\n"
              f"{_excerpts(hits)}\n\nReply with the JSON object only.")
    from lara.serve.generate import complete_json

    d = await complete_json(cfg, prompt, system=TAG_SYSTEM, model=model, max_tokens=200)
    if d is None:
        # No verdict is not the same as "keep nothing": silently dropping every passage
        # would turn a model hiccup into an empty answer. Fall back to the reranker's own
        # top pick, which is the signal we would have used without a model at all.
        top = [1] if hits else []
        return _decision_from(hits, top, "no verdict parsed; kept the top-ranked excerpt",
                              via="fallback")
    keep = d.get("keep") or []
    why = str(d.get("why") or "")[:200]
    idx = [int(i) for i in keep if isinstance(i, (int, float, str)) and str(i).isdigit()]
    return _decision_from(hits, idx, why)


def _decision_from(hits: list[dict], one_based: list[int], why: str,
                   via: str = "model") -> Decision:
    keep_ids, drop_ids = [], []
    for i, h in enumerate(hits, 1):
        cid = h.get("chunk_id")
        if cid is None:
            continue
        (keep_ids if i in one_based else drop_ids).append(int(cid))
    verdict = "kept" if keep_ids else "none"
    return Decision("tag", "paper", verdict, why, via=via, kept=keep_ids, dropped=drop_ids)


async def escalate(cfg, question: str, tier: str, hits: list[dict], coverage: str,
                   model: str | None, stream_answer, allow: tuple[str, ...]) -> Decision:
    """Decide the next scope. Coverage short-circuits the obvious cases."""
    if coverage == "full":
        # The cheap signal is already conclusive, and it cost nothing: the excerpts answer
        # the question, so there is nothing for a wider search to add.
        return Decision("escalate", tier, "stop", "coverage is full", via="scores")
    if not allow:
        return Decision("escalate", tier, "stop", "no wider scope enabled", via="config")

    summary = (f"Question: {question}\n\nCurrent scope: {tier}\n"
               f"Coverage of what was found: {coverage}\n"
               f"Retrieved {len(hits)} excerpt(s):\n{_excerpts(hits[:6])}\n\n"
               f"Scopes available now: {', '.join(('stop',) + allow)}\n"
               "Reply with the JSON object only.")
    from lara.serve.generate import complete_json

    d = await complete_json(cfg, summary, system=ESCALATE_SYSTEM, model=model,
                            max_tokens=120)
    choice, why, via = None, "", "model"
    if d is not None:
        choice = str(d.get("scope") or "").strip().lower()
        why = str(d.get("why") or "")[:200]
    if choice not in {"stop", *allow}:
        # Widen by one step rather than stopping. A parse failure should not silently
        # convert "I could not decide" into "the paper was enough".
        choice, why, via = allow[0], why or "no verdict parsed; widened one step", "fallback"
    return Decision("escalate", tier, choice, why, via=via)


# ── recording ─────────────────────────────────────────────────────────────────────


def record(db_path, question: str, decisions: list[Decision], result_path: str) -> None:
    """Persist every decision for later training. Never raises.

    Tags go to ``judgements`` with teacher ``llm_scope`` so they join the existing
    training pipeline unchanged — and the dropped ones arrive as labelled negatives, which
    is the half that pipeline has always been short of. Tier decisions go to their own
    table, since they are about a query rather than a passage.
    """
    try:
        from lara.finetune.judgements import Judgement, qhash, record as record_j
        from lara.store import db

        conn = db.connect(db_path)
        try:
            items: list[Judgement] = []
            for d in decisions:
                # Only a real model verdict earns the llm_scope teacher. A fallback kept
                # the reranker's top pick because parsing failed; recording that as a
                # model judgement would quietly poison the training set with the very
                # signal distillation is meant to improve on.
                if d.kind != "tag" or d.via != "model":
                    continue
                for rank, cid in enumerate(d.kept):
                    items.append(Judgement(query=question, chunk_id=cid, score=1.0,
                                           label=1, teacher="llm_scope", rank=rank,
                                           source="hierarchy"))
                for rank, cid in enumerate(d.dropped):
                    items.append(Judgement(query=question, chunk_id=cid, score=0.0,
                                           label=0, teacher="llm_scope", rank=rank,
                                           source="hierarchy"))
            if items:
                record_j(conn, items)

            conn.execute(
                """CREATE TABLE IF NOT EXISTS scope_decisions (
                       id INTEGER PRIMARY KEY,
                       query TEXT NOT NULL, query_hash TEXT NOT NULL,
                       tier TEXT NOT NULL, verdict TEXT NOT NULL,
                       why TEXT, via TEXT, path TEXT,
                       n_kept INTEGER, n_dropped INTEGER,
                       created_utc TEXT DEFAULT (datetime('now')))"""
            )
            conn.executemany(
                "INSERT INTO scope_decisions "
                "(query, query_hash, tier, verdict, why, via, path, n_kept, n_dropped) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                [(question, qhash(question), d.tier, d.verdict, d.why, d.via,
                  result_path, len(d.kept), len(d.dropped)) for d in decisions],
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        # Harvesting training signal must never be able to fail a user's question.
        pass


# ── the walk ──────────────────────────────────────────────────────────────────────


def neighbourhood_of(state, arxiv_id: str, limit: int = 60) -> list[str]:
    n = state.neighbours(arxiv_id, limit=limit)
    return list(dict.fromkeys([arxiv_id, *n.get("cites", []), *n.get("cited_by", [])]))


async def walk(state, cfg, question: str, paper: str | None, *,
               selection: str | None = None, final_k: int = 8,
               model: str | None = None, stream_answer=None, on_step=None) -> Result:
    """Retrieve outward from the open paper, deciding at each boundary.

    Falls back to a plain corpus search whenever the premise does not hold — no paper
    open, the paper has no embedded chunks, or hierarchy is switched off — because a
    reader whose paper was never crawled should still get an answer.
    """
    hcfg = (cfg.get_in("retrieval.hierarchy") or {}) if hasattr(cfg, "get_in") else {}
    clock = time.perf_counter
    res = Result(hits=[])
    started = clock()

    def emit(tier: str, note: str) -> None:
        if on_step:
            on_step(tier, note)

    def plain(note: str) -> Result:
        t0 = clock()
        r = state.retriever.retrieve(question, selection=selection, final_k=final_k)
        res.hits = [to_dict(h) for h in r.hits]
        res.tiers.append(TierRun("corpus", res.hits, (clock() - t0) * 1000,
                                 r.n_candidates, note))
        res.timings["total"] = (clock() - started) * 1000
        return res

    if not hcfg.get("enabled", True) or not paper:
        return plain("no paper open" if not paper else "hierarchy disabled")

    n_chunks = 0
    try:
        row = state.conn().execute(
            "SELECT COUNT(*) FROM chunks WHERE arxiv_id=? AND vector_row IS NOT NULL",
            (paper,)).fetchone()
        n_chunks = int(row[0]) if row else 0
    except Exception:
        n_chunks = 0
    if n_chunks == 0:
        return plain(f"{paper} has no embedded chunks")

    # ── tier 1: the paper itself ──────────────────────────────────────────────
    k = paper_k(n_chunks, float(hcfg.get("paper_frac", 0.2)),
                int(hcfg.get("paper_min_k", 5)), int(hcfg.get("paper_max_k", 30)))
    emit("paper", f"searching {paper} ({k} of {n_chunks} chunks)")
    t0 = clock()
    r = state.retriever.retrieve(question, papers=[paper], selection=selection, final_k=k)
    in_paper = [to_dict(h) for h in r.hits]
    res.tiers.append(TierRun("paper", in_paper, (clock() - t0) * 1000, r.n_candidates))

    # ── tag ───────────────────────────────────────────────────────────────────
    if hcfg.get("tag_with_model", True) and stream_answer is not None and in_paper:
        emit("paper", f"checking which of {len(in_paper)} passages help")
        d = await tag(cfg, question, in_paper, model, stream_answer)
    else:
        # Without a model, the reranker's ordering is the only relevance signal there is.
        keep = [i for i, _ in enumerate(in_paper[:3], 1)]
        d = _decision_from(in_paper, keep, "model tagging off; kept top 3", via="config")
    res.decisions.append(d)
    res.feedback_ids = list(d.kept)
    confirmed = [h for h in in_paper if int(h.get("chunk_id", -1)) in set(d.kept)]
    res.hits = confirmed or in_paper[:final_k]

    # ── decide whether to widen ───────────────────────────────────────────────
    from lara.serve import agent as A

    coverage = A.score_prior(confirmed) or "partial"
    allow: list[str] = []
    if hcfg.get("allow_neighbourhood", True):
        allow.append("neighbourhood")
    if hcfg.get("allow_corpus", True):
        allow.append("corpus")

    if stream_answer is not None and allow:
        d2 = await escalate(cfg, question, "paper", confirmed or in_paper, coverage,
                            model, stream_answer, tuple(allow))
    else:
        d2 = Decision("escalate", "paper", "stop", "no model available", via="config")
    res.decisions.append(d2)

    if d2.verdict == "stop":
        res.timings["total"] = (clock() - started) * 1000
        record(state.db_path, question, res.decisions, res.path)
        return res

    # ── wider tiers, carrying confirmed passages as extra query vectors ───────
    fb = feedback_vectors(state, d.kept, int(hcfg.get("max_feedback_vectors", 4)))
    order = [t for t in ("neighbourhood", "corpus")
             if t in allow and TIERS.index(t) >= TIERS.index(d2.verdict)]

    for tier in order:
        papers = neighbourhood_of(state, paper) if tier == "neighbourhood" else None
        emit(tier, f"searching {len(papers)} papers" if papers else "searching the corpus")
        t0 = clock()
        r = state.retriever.retrieve(question, papers=papers, selection=selection,
                                     final_k=final_k, feedback=fb)
        hits = [to_dict(h) for h in r.hits]
        res.tiers.append(TierRun(tier, hits, (clock() - t0) * 1000, r.n_candidates))
        merged, _ = A.merge_hits(res.hits, hits, cap=max(final_k, len(res.hits) + final_k))
        res.hits = merged
        if tier == d2.verdict and tier != "corpus":
            cov = A.score_prior(hits) or "partial"
            d3 = await escalate(cfg, question, tier, hits, cov, model, stream_answer,
                                ("corpus",)) if stream_answer and "corpus" in allow else \
                Decision("escalate", tier, "stop", "corpus not enabled", via="config")
            res.decisions.append(d3)
            if d3.verdict == "stop":
                break

    res.hits = res.hits[:final_k]
    res.timings["total"] = (clock() - started) * 1000
    record(state.db_path, question, res.decisions, res.path)
    return res


def feedback_vectors(state, chunk_ids: list[int], cap: int = 4) -> list[np.ndarray]:
    """Full-precision vectors for confirmed chunks, best-first, capped.

    Capped because each one is another dense search and another list in the fusion; past a
    handful they stop adding recall and start flattening the ranking, since every list
    contributes the same reciprocal weight regardless of how good it was.
    """
    if not chunk_ids:
        return []
    rows = []
    conn = state.conn()
    ph = ",".join("?" * len(chunk_ids[:cap]))
    for r in conn.execute(
        f"SELECT vector_row FROM chunks WHERE chunk_id IN ({ph}) AND vector_row IS NOT NULL",
        chunk_ids[:cap],
    ):
        rows.append(int(r[0]))
    if not rows:
        return []
    try:
        state.retriever._ensure_fp16_current()
        fp16 = state.retriever.fp16
        return [np.asarray(fp16[r], dtype=np.float32) for r in rows
                if 0 <= r < state.retriever.n_vector_rows]
    except Exception:
        return []
