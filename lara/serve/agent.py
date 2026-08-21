"""Multi-round retrieval agent with three tools.

The model gets three capabilities beyond answering:

``clarify``          the question is too vague to search usefully — ask the reader first
``search``           the excerpts do not answer it — search again with a better query
``expand_context``   a chunk is unintelligible alone — pull its neighbours or whole section

**Decisions are JSON, not native tool calls.** vLLM's tool-calling needs a per-model parser
enabled at launch (``--tool-call-parser``), which would tie the agent to one generator and
break the model picker the moment someone selects a different checkpoint. A small JSON
verdict works on any instruction-tuned model and degrades to "answer directly" if parsing
fails, so a model that cannot follow the schema still produces an answer rather than an
error.

**Every round emits progress events.** A search that renders nothing for twenty seconds is
indistinguishable from a hang, and the honest fix is to show the work rather than to hide
the latency.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from lara.serve.generate import SYSTEM, stream_answer

# ── the speed <-> accuracy spectrum ────────────────────────────────────────────────
#
# One dial, because every knob here spends the same currency: latency for recall. The
# estimates are shown in the UI, since a slider that does not say what it costs invites
# people to max it out and then think the app has hung.

@dataclass
class Breadth:
    name: str
    label: str
    max_rounds: int
    k: int
    candidates: int
    expand_context: bool
    allow_clarify: bool
    budget_sec: float
    estimate: str
    decompose: bool = False


# Estimates are MEASURED end-to-end medians, not search overhead. They matter: an earlier
# version advertised "~0.5s" for Instant, which measured 5.3s, because it counted only the
# retrieval leg. A dial that understates its cost is worse than one with no numbers at all.
#
# Note the floor. Generation alone is ~3s to first token plus ~2s of streaming, so the
# bottom three settings differ far less than their names suggest — the dial buys search
# depth, and search was never the expensive part.
SPECTRUM: list[Breadth] = [
    Breadth("instant", "Instant", 1, 5, 100, False, False, 20, "~5s"),
    Breadth("fast", "Fast", 1, 8, 200, False, False, 30, "~5s"),
    Breadth("balanced", "Balanced", 2, 8, 200, True, False, 60, "~6s", decompose=True),
    Breadth("thorough", "Thorough", 3, 12, 300, True, True, 120, "~7-10s", decompose=True),
    Breadth("exhaustive", "Exhaustive", 5, 20, 400, True, True, 300, "~10-18s", decompose=True),
]
BY_NAME = {b.name: b for b in SPECTRUM}
DEFAULT = "balanced"


def resolve(name: str | None) -> Breadth:
    return BY_NAME.get((name or DEFAULT).lower(), BY_NAME[DEFAULT])


# ── the decision step ──────────────────────────────────────────────────────────────

DECIDE_SYSTEM = """You are the retrieval controller for a question-answering system over \
arXiv machine-learning papers. You do not answer the question yourself. You decide what \
the system should do next, and you reply with a single JSON object and nothing else.

Schema:
{"action": "answer" | "search" | "clarify" | "expand",
 "reason": "<one short sentence>",
 "queries": ["<search query>", ...],      // only for "search", 1-3 queries
 "question": "<what to ask the reader>",  // only for "clarify"
 "options": ["<suggestion>", ...],        // only for "clarify", 2-4 concrete refinements
 "chunk_ids": [<int>, ...]}               // only for "expand"

Choose "answer" when the excerpts contain enough to answer, even partially. Prefer it.
Choose "search" when the excerpts are on-topic but miss the specific fact asked for; the \
queries should use the paper's own terminology rather than repeating the question.
Choose "clarify" ONLY when the question is so ambiguous that any search would be a guess \
-- for example it names no method, dataset, or property, or could plausibly mean several \
unrelated things.
Choose "expand" when an excerpt is clearly a fragment whose meaning depends on \
surrounding text -- a bare equation, a sentence beginning "This shows that...", a table \
row without its caption."""


@dataclass
class Step:
    kind: str                      # search | decide | expand | clarify | answer
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)


def format_excerpts(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        where = h.get("section") or ""
        title = h.get("paper_title") or h.get("arxiv_id", "")
        lines.append(f"[{i}] (id={h.get('chunk_id')}) ({title} > {where}) {h.get('text','').strip()}")
    return "\n\n".join(lines) if lines else "(nothing retrieved yet)"


_JSON = re.compile(r"\{.*\}", re.S)


def parse_decision(text: str) -> dict[str, Any]:
    """Extract the JSON verdict, tolerating prose or fences around it.

    A model that ignores the schema should not break the request — an unparseable verdict
    means "just answer", which is the safe default.
    """
    m = _JSON.search(text or "")
    if not m:
        return {"action": "answer", "reason": "no decision returned"}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "answer", "reason": "unparseable decision"}
    if not isinstance(d, dict) or d.get("action") not in {"answer", "search", "clarify", "expand"}:
        return {"action": "answer", "reason": "unknown action"}
    return d


async def decide(cfg, question: str, hits: list[dict], breadth: Breadth,
                 rounds_done: int, model: str | None) -> dict[str, Any]:
    """One controller turn. Returns the parsed verdict."""
    remaining = breadth.max_rounds - rounds_done
    prompt = (
        f"Question: {question}\n\n"
        f"Excerpts retrieved so far:\n{format_excerpts(hits)}\n\n"
        f"Search rounds remaining: {remaining}. "
        f"{'Clarification is not available; do not choose it.' if not breadth.allow_clarify else ''} "
        f"{'Context expansion is not available; do not choose it.' if not breadth.expand_context else ''}\n"
        "Reply with the JSON object only."
    )
    from lara.serve.generate import complete

    # parse_decision does its own tolerant extraction (it accepts prose and fences around
    # the object), so this takes the raw text rather than complete_json.
    return parse_decision(await complete(cfg, prompt, system=DECIDE_SYSTEM, model=model,
                                         max_tokens=400))


# ── context expansion ──────────────────────────────────────────────────────────────

def expand_chunks(conn, chunk_ids: list[int], before: int = 2, after: int = 1,
                  limit: int = 12) -> list[dict]:
    """Fetch neighbouring chunks so a fragment can be read in context.

    Cheap by construction: chunks carry a per-paper ``ordinal``, so this is an index
    lookup with no embedding and no search. Reads more before than after, because the
    usual failure is a chunk that refers backwards ("this shows that...", "the above
    bound").
    """
    if not chunk_ids:
        return []
    out: list[dict] = []
    seen: set[int] = set()
    for cid in chunk_ids[:6]:
        row = conn.execute(
            "SELECT arxiv_id, version, ordinal FROM chunks WHERE chunk_id=?", (cid,)
        ).fetchone()
        if row is None:
            continue
        rows = conn.execute(
            """SELECT c.chunk_id, c.arxiv_id, c.version, c.ordinal, c.section_anchor,
                      c.anchor_start, c.char_start, c.anchor_end, c.char_end, c.kind, c.text,
                      p.title AS paper_title, s.title AS section_title
               FROM chunks c
               JOIN papers p ON p.arxiv_id = c.arxiv_id
               LEFT JOIN sections s ON s.arxiv_id=c.arxiv_id AND s.version=c.version
                                   AND s.anchor=c.section_anchor
               WHERE c.arxiv_id=? AND c.version=? AND c.ordinal BETWEEN ? AND ?
               ORDER BY c.ordinal""",
            (row["arxiv_id"], row["version"], row["ordinal"] - before, row["ordinal"] + after),
        ).fetchall()
        for r in rows:
            if r["chunk_id"] in seen or len(out) >= limit:
                continue
            seen.add(r["chunk_id"])
            out.append({
                "chunk_id": r["chunk_id"], "arxiv_id": r["arxiv_id"], "version": r["version"],
                "url": S.fragment_for(r["arxiv_id"], r["version"], r["anchor_start"],
                                      r["char_start"], r["anchor_end"], r["char_end"]),
                "anchor": r["anchor_start"], "char_start": r["char_start"],
                "char_end": r["char_end"], "anchor_end": r["anchor_end"],
                "section": r["section_title"] or r["section_anchor"], "kind": r["kind"],
                "score": 0.0, "paper_title": r["paper_title"], "text": r["text"],
                "via": "context",
            })
    return out


def merge_hits(existing: list[dict], new: list[dict], cap: int) -> tuple[list[dict], int]:
    """Add only chunks not already present. Returns (merged, n_added).

    Deduplicating across rounds matters: without it round 2 re-retrieves most of round 1,
    the model sees no new evidence, and the loop burns its whole budget standing still.
    """
    seen = {h.get("chunk_id") for h in existing}
    added = 0
    for h in new:
        if h.get("chunk_id") in seen:
            continue
        seen.add(h.get("chunk_id"))
        existing.append(h)
        added += 1
        if len(existing) >= cap:
            break
    return existing, added


# ── coverage assessment and grounding verification ─────────────────────────────────

ASSESS_SYSTEM = """You judge whether a set of excerpts answers a question. You do not \
answer it. Reply with one JSON object and nothing else:

{"coverage": "full" | "partial" | "none",
 "missing": "<what the excerpts do not establish, one clause; empty if full>",
 "conflict": "<what two excerpts disagree about, or empty>"}

"full"    the excerpts contain what is needed to answer directly.
"partial" they bear on the question but leave a specific part unestablished.
"none"    they do not address what was asked, even if they share its topic.

Judge only what the excerpts state. Do not use outside knowledge to fill a gap: if the \
answer requires a fact that is not present, that is "partial" or "none" however familiar \
the fact may be."""

# Reranker scores below this mean nothing retrieved is really on topic. The cross-encoder
# already scored every excerpt against the question, so an obviously empty result set can
# be called without spending an LLM turn on it.
SCORE_NONE = 0.05
SCORE_STRONG = 0.55


def score_prior(hits: list[dict]) -> str | None:
    """Cheap coverage verdict from reranker scores alone, or None if inconclusive.

    Worth doing before asking the model: the scores are already computed, they cost
    nothing to read, and at the extremes they are more reliable than a self-assessment —
    a model shown five irrelevant excerpts will often still find something to say about
    them.
    """
    scores = [h.get("score") for h in hits if isinstance(h.get("score"), (int, float))]
    if not scores:
        return None
    top = max(scores)
    if top < SCORE_NONE:
        return "none"
    if top > SCORE_STRONG and sum(s > SCORE_STRONG for s in scores) >= 3:
        return "full"
    return None


async def assess(cfg, question: str, hits: list[dict], model: str | None) -> dict[str, Any]:
    """Decide whether the excerpts answer the question, and how."""
    prior = score_prior(hits)
    if prior == "none":
        return {"coverage": "none", "missing": "nothing retrieved was on topic",
                "conflict": "", "via": "scores"}

    prompt = (
        f"Question: {question}\n\nExcerpts:\n{format_excerpts(hits)}\n\n"
        "Reply with the JSON object only."
    )
    from lara.serve.generate import complete_json

    d = await complete_json(cfg, prompt, system=ASSESS_SYSTEM, model=model, max_tokens=250)
    if d is None:
        return {"coverage": prior or "full", "missing": "", "conflict": "", "via": "default"}
    cov = d.get("coverage")
    if cov not in {"full", "partial", "none"}:
        cov = prior or "full"
    return {"coverage": cov, "missing": str(d.get("missing") or "")[:200],
            "conflict": str(d.get("conflict") or "")[:200], "via": "model"}


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_CITE = re.compile(r"\[(\d+)\]")


# Calibrated against bge-reranker-v2-m3 on deliberately mis-cited sentences:
#
#   genuinely supported claim ........... 0.98 - 1.00
#   fabricated but on-topic claim ....... ~0.29   <- the dangerous case
#   wholly unrelated claim .............. ~0.00
#
# The first threshold tried was 0.06, which caught only the unrelated case and waved
# through the invented-but-plausible one — precisely the failure the check exists to
# catch. Real support scores near 1.0, so a much higher bar costs almost no false
# positives on legitimate paraphrase.
GROUNDING_THRESHOLD = 0.5


def check_grounding(cross_encoder, answer: str, hits: list[dict],
                    threshold: float = GROUNDING_THRESHOLD) -> list[dict]:
    """Score each cited sentence against the excerpt it cites.

    The most damaging failure mode in a citing RAG system is not a missing citation but a
    confident sentence carrying a citation that does not support it: the marker makes the
    claim look checked, and a reader who trusts the format will not click through. The
    cross-encoder is already loaded and already answers exactly this question — does this
    passage support this text — so verification costs one extra batch.

    Returns one record per cited sentence, flagged when its own source scores it low.
    """
    if cross_encoder is None or not answer.strip():
        return []
    by_marker = {h.get("marker", i + 1): h for i, h in enumerate(hits)}
    pairs, meta = [], []
    for sentence in _SENT_SPLIT.split(answer.strip()):
        markers = {int(x) for x in _CITE.findall(sentence)}
        clean = _CITE.sub("", sentence).strip()
        if not markers or len(clean) < 25:
            continue
        for mk in markers:
            hit = by_marker.get(mk)
            if not hit:
                continue
            pairs.append((clean, hit.get("text", "")))
            meta.append({"sentence": clean[:220], "marker": mk,
                         "arxiv_id": hit.get("arxiv_id", "")})
    if not pairs:
        return []
    scores = cross_encoder.predict(pairs, show_progress_bar=False)
    out = []
    for rec, sc in zip(meta, scores):
        rec["support"] = round(float(sc), 4)
        rec["supported"] = float(sc) >= threshold
        out.append(rec)
    return out


# ── query decomposition ────────────────────────────────────────────────────────────

DECOMPOSE_SYSTEM = """You split a research question into independently searchable parts. \
You do not answer it. Reply with one JSON object and nothing else:

{"kind": "atomic" | "parallel" | "sequential",
 "parts": ["<self-contained sub-question>", ...],
 "reason": "<one short clause>"}

"atomic"     one thing is being asked; parts must be [].
"parallel"   several things are asked that can be looked up independently.
"sequential" a later part needs the answer to an earlier one (multi-hop).

Rules:
- Prefer "atomic". Most questions are atomic, and splitting one needlessly retrieves
  worse for every part than not splitting at all.
- At most 3 parts.
- Each part must stand alone: resolve pronouns and carry over the subject, so "and which
  is faster?" becomes "Is X faster than Y?" rather than a fragment.
- Split on what is being ASKED, not on grammar. "How does GQA reduce memory and latency?"
  is one mechanism asked about twice over and stays atomic."""

# Cheap pre-filter. Running the decomposer on every question would add a full LLM turn to
# the majority of asks that are plainly atomic, so an LLM turn is only spent when the
# surface form suggests more than one thing is being asked.
_COMPOUND = re.compile(
    r"\b(compare[sd]?|versus|vs\.?|difference between|better than|trade-?offs?"
    r"|and (?:which|what|how|why|when)|as well as|both .+ and )\b",
    re.I,
)


def looks_compound(q: str) -> bool:
    if q.count("?") > 1:
        return True
    return bool(_COMPOUND.search(q))


async def decompose(cfg, question: str, model: str | None,
                    max_parts: int = 3) -> dict[str, Any]:
    """Split a compound question, or report it atomic."""
    if not looks_compound(question):
        return {"kind": "atomic", "parts": [], "reason": "no compound markers", "via": "heuristic"}

    from lara.serve.generate import complete_json

    d = await complete_json(
        cfg, f"Question: {question}\n\nReply with the JSON object only.",
        system=DECOMPOSE_SYSTEM, model=model, max_tokens=300)
    if d is None:
        return {"kind": "atomic", "parts": [], "reason": "unparseable", "via": "default"}

    kind = d.get("kind") if d.get("kind") in {"atomic", "parallel", "sequential"} else "atomic"
    parts = [str(p).strip() for p in (d.get("parts") or []) if str(p).strip()][:max_parts]
    # A "compound" verdict with fewer than two usable parts is just an atomic question
    # with extra steps.
    if kind == "atomic" or len(parts) < 2:
        return {"kind": "atomic", "parts": [], "reason": str(d.get("reason") or "")[:120],
                "via": "model"}
    return {"kind": kind, "parts": parts, "reason": str(d.get("reason") or "")[:120],
            "via": "model"}


def merge_parts(part_hits: list[tuple[str, list[dict]]], cap: int) -> list[dict]:
    """Interleave per-part results so every part is represented near the top.

    Concatenating the parts would let one part with high absolute scores fill the context
    and starve the others — which reproduces exactly the failure decomposition exists to
    fix. Round-robin guarantees each sub-question contributes before any contributes
    twice. Each hit records the part it came from, so the prompt can group evidence and
    the model can see which half of the question it is answering.
    """
    for part, hits in part_hits:
        for h in hits:
            h.setdefault("part", part)
    merged: list[dict] = []
    seen: set[int] = set()
    for rank in range(max((len(h) for _, h in part_hits), default=0)):
        for _, hits in part_hits:
            if rank >= len(hits):
                continue
            h = hits[rank]
            cid = h.get("chunk_id")
            if cid in seen:
                continue
            seen.add(cid)
            merged.append(h)
            if len(merged) >= cap:
                return merged
    return merged


# ── the driver ────────────────────────────────────────────────────────────────────────
#
# What `/api/ask` actually runs. It lives here rather than inside the endpoint because it
# is the agent loop -- rewrite, scope, decompose, decide, assess, answer -- and an SSE
# generator is a poor place to keep three hundred lines of control flow: nothing in it can
# be read, tested or reasoned about without a live HTTP connection. The endpoint is now
# transport only, and this mirrors `synthesis.run_synthesis`: results are pushed through
# `emit(name, payload)` as they happen, and the caller decides how to frame them.


class AskRequest(BaseModel):
    query: str
    selection: str | None = None
    paper: str | None = None
    scope: str = "corpus"
    hits: list[dict] | None = None       # reuse speculative retrieval; skip re-retrieving
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1024
    breadth: str | None = None           # instant | fast | balanced | thorough | exhaustive


def vectors_for_chunks(s, chunk_ids: list[int]) -> list:
    """Full-precision vectors for chunks a previous answer cited.

    These become feedback vectors on the next retrieval — each runs its own dense search
    and joins the rank fusion, so a follow-up starts from where the last answer landed
    rather than from cold. Failure is silent: losing the seed degrades recall slightly,
    while raising would fail the question outright.
    """
    if not chunk_ids:
        return []
    try:
        from lara.serve.synthesis import vectors_for
        vecs = vectors_for(s, chunk_ids)
        return [vecs[c] for c in chunk_ids if c in vecs]
    except Exception:
        return []


def capture_hit_dicts(s, query: str, hits: list[dict], source: str) -> None:
    """Store what retrieval returned as training data. Never fails a user's question."""
    try:
        from lara.finetune import judgements as J
        from lara.store import db

        items = J.from_hits(query, hits, source=source)
        if not items:
            return
        conn = db.connect(s.db_path)
        try:
            J.record(conn, items)
        finally:
            conn.close()
    except Exception:
        pass


async def run_ask(s, cfg, req: AskRequest, *, emit) -> None:
    """One grounded answer, start to finish, reporting progress through ``emit``.

    Emits, in order: ``step`` throughout, ``hits`` whenever the excerpt set changes,
    optionally ``clarify``, then ``coverage``, a run of ``token``, ``context``, ``perf``,
    optionally ``grounding``, and finally ``done``. ``error`` replaces the tail if
    anything raises.
    """
    import asyncio
    import time

    from lara.serve import thread as TH
    from lara.serve.deps import memory_root

    started = time.time()
    breadth = resolve(req.breadth)

    def step(kind: str, detail: str, **payload) -> None:
        emit("step", {"kind": kind, "detail": detail, **payload})

    # Every scope the UI offers must be honoured here. `neighbourhood` was accepted and
    # silently treated as whole-corpus, so choosing it changed nothing and said nothing —
    # /api/retrieve resolved it correctly, which made the two endpoints disagree about
    # what the same request meant.
    restrict = None
    if req.scope == "paper" and req.paper:
        restrict = [req.paper]
    elif req.scope == "neighbourhood" and req.paper:
        n = s.neighbours(req.paper)
        restrict = [req.paper, *n["cites"], *n["cited_by"]]

    # The thread is keyed by paper, so opening another one starts a fresh conversation.
    root = memory_root()
    turns = TH.turns_for(root, req.paper)
    from lara.serve import memory as MEM
    thread_summary = MEM.get_thread_summary(root, TH.thread_id(req.paper))
    history = TH.history_block(turns, summary=thread_summary)
    # Passages the last answers cited are strong candidates for a follow-up, and cost
    # nothing to reuse: they were already retrieved and reranked.
    seed_ids = TH.prior_chunk_ids(turns)
    feedback = vectors_for_chunks(s, seed_ids) if seed_ids else []
    search_query = {"q": req.query}

    def run_search(query: str, k: int) -> list[dict]:
        result = s.retriever.retrieve(
            query, papers=restrict, selection=req.selection, final_k=k,
            feedback=feedback,
        )
        return [h.to_dict() for h in result.hits]

    cap = breadth.k * (breadth.max_rounds + 2)

    try:
        # Reference resolution happens BEFORE retrieval. "why is that?" embedded
        # verbatim matches nothing; history in the answer prompt cannot repair a
        # search that already fetched the wrong passages.
        if turns:
            rw = await TH.rewrite(cfg, req.query, turns, req.model, stream_answer)
            if rw["rewritten"]:
                search_query["q"] = rw["query"]
                step("rewrite", rw["query"], original=req.query, why=rw["why"])

        hits = req.hits

        # Hierarchical scope: with a paper open and no scope pinned, start inside that
        # paper and widen only if the model says the answer is not there. The dropdown
        # still wins when the reader sets it — "this paper" and "citation
        # neighbourhood" are instructions, not hints, and overriding them would make
        # the control a lie.
        hcfg = cfg.get_in("retrieval.hierarchy") or {}
        use_walk = (hits is None and req.paper and req.scope == "corpus"
                    and hcfg.get("enabled", True))
        if use_walk:
            from lara.serve import hierarchy as HR

            walked = await HR.walk(
                s, cfg, search_query["q"], req.paper,
                selection=req.selection, final_k=breadth.k,
                model=req.model, stream_answer=stream_answer,
                on_step=lambda tier, note: step("scope", f"{tier}: {note}"),
            )
            if walked is not None and walked.hits:
                hits = walked.hits
                path = walked.path or "corpus"
                kept = sum(len(d.kept) for d in walked.decisions if d.kind == "tag")
                step("scope", f"searched {path} · {kept} passage(s) confirmed "
                              f"in the paper", path=path)
            else:
                # The paper had nothing embedded, or the walk failed. Fall through to
                # the flat search rather than answering from nothing.
                use_walk = False

        if hits is None:
            # A compound question embeds to a single vector sitting BETWEEN its parts,
            # retrieving well for none of them. Splitting first and searching each
            # part is the fix; the parts run concurrently, so the extra cost is one
            # decomposition call rather than N retrievals.
            split = {"kind": "atomic", "parts": []}
            if breadth.decompose:
                split = await decompose(cfg, search_query["q"], req.model)

            if split["parts"]:
                step(
                    "decompose",
                    f"Split into {len(split['parts'])} parts: "
                    + " | ".join(p[:48] for p in split["parts"]),
                    # not `kind=`: that is step()'s own positional parameter
                    parts=split["parts"], split_kind=split["kind"],
                )
                per_part = max(3, breadth.k // len(split["parts"]) + 2)
                results = await asyncio.gather(*[
                    run_in_threadpool(run_search, part, per_part)
                    for part in split["parts"]
                ])
                hits = merge_parts(list(zip(split["parts"], results)), breadth.k * 2)
            else:
                step("search", f"Searching ({breadth.label})…")
                hits = await run_in_threadpool(run_search, search_query["q"], breadth.k)
        emit("hits", hits)
        # The resolved query is what should retrieve these passages, so it is the
        # honest training pair; the pronoun the reader typed is not.
        capture_hit_dicts(s, search_query["q"], hits, "ask")

        rounds = 0
        while rounds < breadth.max_rounds:
            if time.time() - started > breadth.budget_sec:
                step("budget", "Time budget reached; answering with what I have.")
                break
            # A single round needs no controller turn: there is nothing yet to judge
            # against, and the extra LLM call would double "instant" latency.
            if breadth.max_rounds == 1:
                break

            step("decide", "Assessing whether the excerpts answer the question…")
            verdict = await decide(cfg, search_query["q"], hits, breadth, rounds, req.model)
            action = verdict.get("action", "answer")

            if action == "clarify" and breadth.allow_clarify:
                emit("clarify", {
                    "question": verdict.get("question") or "Could you be more specific?",
                    "options": verdict.get("options", [])[:4],
                    "reason": verdict.get("reason", ""),
                })
                # Non-blocking by design: the reader still gets an answer from what we
                # have, with the refinements offered alongside. A model that stops to
                # ask a question and returns nothing is worse than one that guesses
                # well and offers to narrow down.
                break

            if action == "expand" and breadth.expand_context:
                ids = [int(i) for i in verdict.get("chunk_ids", [])[:6] if str(i).isdigit()]
                step("expand", f"Reading around {len(ids)} excerpt(s) for context…")
                extra = await run_in_threadpool(expand_chunks, s.conn(), ids)
                hits, added = merge_hits(hits, extra, cap)
                emit("hits", hits)
                if added == 0:
                    break
                rounds += 1
                continue

            if action == "search":
                queries = [q for q in verdict.get("queries", []) if isinstance(q, str)][:3]
                if not queries:
                    break
                step("search", f"Searching again: {'; '.join(queries)[:110]}",
                     queries=queries)
                total_added = 0
                for q in queries:
                    more = await run_in_threadpool(run_search, q, breadth.k)
                    hits, added = merge_hits(hits, more, cap)
                    total_added += added
                emit("hits", hits)
                # No new evidence means another round would ask the same question of
                # the same excerpts. Stop rather than spend the budget.
                if total_added == 0:
                    step("converged", "No new material found; answering.")
                    break
                rounds += 1
                continue

            break   # "answer"

        # Coverage gate. Naming an insufficient result set explicitly is what stops
        # the model quietly filling the gap from parametric memory, which is the most
        # common way a RAG answer goes wrong while looking right.
        step("assess", "Checking whether the excerpts answer the question…")
        verdict = await assess(cfg, search_query["q"], hits, req.model)
        coverage = verdict.get("coverage", "full")
        emit("coverage", verdict)
        label = {"full": "Answering", "partial": "Answering what is supported",
                 "none": "Summarising what was found"}[coverage]
        step("answer", f"{label} from {len(hits)} excerpts…",
             rounds=rounds, coverage=coverage)

        try:
            custom_prompt = MEM.get_prompt(root)
        except Exception:
            custom_prompt = None

        answer = ""
        prompt_parts: list = []
        usage: dict = {}
        gen_started = time.time()
        first_token_at = None
        async for token in stream_answer(
            # Deliberately req.query, not the rewrite: retrieval needed a
            # self-contained query, but the answer must address the question the
            # reader actually asked. History is in the prompt, so the model can
            # resolve the reference itself.
            cfg, req.query, hits, selection=req.selection, history=history,
            model=req.model, temperature=req.temperature, max_tokens=req.max_tokens,
            coverage=coverage, system=custom_prompt,
            prompt_parts=prompt_parts, usage_out=usage,
        ):
            if first_token_at is None:
                first_token_at = time.time()
            answer += token
            emit("token", token)

        # Context accounting and throughput, both from the generator itself rather
        # than estimated here — see generate.count_tokens.
        try:
            from lara.serve import generate as GEN

            vcfg = cfg.get_in("serving.vllm") or {}
            base_url = vcfg.get("base_url", "http://127.0.0.1:8000/v1")
            model_name = req.model or vcfg.get("default_model") or ""
            parts = list(prompt_parts)

            counts = await GEN.count_tokens(base_url, model_name,
                                            [t for _, t in parts]) if parts else None
            limit = await GEN.context_limit(base_url, model_name) or int(
                vcfg.get("max_model_len", 32768))
            emit("context", {
                "limit": limit,
                "reserved": int(req.max_tokens),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "exact": counts is not None,
                "parts": [
                    {"name": n, "chars": len(t),
                     "tokens": (counts[i] if counts else round(len(t) / 4))}
                    for i, (n, t) in enumerate(parts)
                ],
            })

            done_at = time.time()
            out_tokens = usage.get("completion_tokens")
            gen_s = done_at - (first_token_at or gen_started)
            emit("perf", {
                "ttft_ms": round(((first_token_at or done_at) - gen_started) * 1000),
                "completion_tokens": out_tokens,
                "prompt_tokens": usage.get("prompt_tokens"),
                # Measured from FIRST TOKEN, not from the request: including the
                # prefill in a decode rate makes a long prompt look like a slow model.
                "tok_per_sec": (round((out_tokens - 1) / gen_s, 1)
                                if out_tokens and out_tokens > 1 and gen_s > 0 else None),
                "generate_ms": round((done_at - gen_started) * 1000),
            })
        except Exception:
            pass

        try:
            paper_title = next(
                (h.get("paper_title", "") for h in hits
                 if req.paper and h.get("arxiv_id") == req.paper), "")
            MEM.record_question(
                root, question=req.query, answer=answer,
                arxiv_id=req.paper, title=paper_title,
                scope=req.scope, selection=req.selection,
            )
        except Exception:
            pass      # never lose a finished answer because the library write failed

        # Post-hoc grounding check: score every cited sentence against the excerpt it
        # actually cites. A citation marker makes a claim look verified, so an
        # unsupported one is worse than none at all.
        if answer.strip() and s.retriever.cross_encoder is not None:
            checks = await run_in_threadpool(
                check_grounding, s.retriever.cross_encoder, answer, hits
            )
            if checks:
                emit("grounding", checks)
    except Exception as exc:
        emit("error", str(exc)[:300])
    emit("done", {"elapsed_ms": round((time.time() - started) * 1000)})
