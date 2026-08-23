"""Deep automated research — iterative retrieval, structured extraction, consolidation.

A question like *"what is the most sample-efficient Muon-based optimiser in the
literature?"* is not answerable by one retrieval. It needs a survey: find candidates, read
them, notice what is missing, look again, and only then decide what the literature actually
says. This runs that loop.

    seed ─▶ [ retrieve ─▶ label + name + extract ─▶ expand ] xN ─▶ consolidate ─▶ answers

**Relevance alone re-finds the same cluster.** Fusing confirmed chunks back into the query
pulls *toward* what is already held, so round three looks like round two. Four pressures
push outward instead: chunks already seen are dropped before ranking, no single paper may
dominate a round, selection is diversified by maximal marginal relevance, and each round
walks the citation graph out of the confirmed papers into ones similarity never reached.

**Superlative questions invite invention.** "Most sample-efficient" has no answer unless
someone measured it, so extraction is *structured* — method, metric, value, condition —
and the consolidation prompt is required to say when the literature does not support a
ranking. A table of what each paper actually reported is an honest answer; a confident
ordering assembled from incomparable numbers is not.

**Provenance has to survive two hops.** The final answer is chunk → claim → narrative, so
every claim carries its ``chunk_id`` forward and the answers cite those ids. Without that
the grounding check has nothing to score and citations cannot be rendered.

**Termination is not left to the model alone.** Models stop early on hard questions and
loop on easy ones. The model's vote is one input; saturation is the other, and a run ends
when consecutive rounds stop finding anything new regardless of what the model wants.
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import numpy as np

# ── prompts ───────────────────────────────────────────────────────────────────────

EXTRACT_SYSTEM = """You are surveying the literature to answer a research question.

For each numbered excerpt decide whether it carries information that helps answer the \
question, and if so extract it in structured form.

What your decision causes:
- RELEVANT: the excerpt's embedding becomes a search key for the next round, its claim \
enters the evidence table the final answer is built from, and its name is shown to the \
user in the retrieval graph. A wrongly kept excerpt steers the next round off course and \
puts a false row in the table.
- NOT RELEVANT: discarded for this question and recorded as an example of something that \
looked related but carried nothing.

Be strict. Shared vocabulary is not relevance. An excerpt that merely mentions the topic \
without stating a method, result, measurement, limitation or definition is NOT relevant.

For each relevant excerpt give:
  "name"      3-6 words, what a reader would call this passage. Shown as a graph label.
  "claim"     the single most important thing it establishes, in ONE sentence, and \
SHORTER than the excerpt. Compress; do not paraphrase at length.
  "method"    the technique or system it is about, or "" if none
  "metric"    what was measured, or "" if nothing was
  "value"     the measured result including units, or "" if none
  "condition" the setting the result holds in (dataset, scale, budget), or ""

Reply with a JSON array only, one object per excerpt you judge relevant:
[{"n": 1, "name": "...", "claim": "...", "method": "...", "metric": "...", \
"value": "...", "condition": "..."}]
Return [] if none are relevant."""


CONTINUE_SYSTEM = """You are deciding whether a literature survey is complete.

You are told what the last round found and what the whole run has gathered.

**Default to continuing.** A survey that stops early is the common failure, and it is the \
expensive one: the answer then reports a fraction of the literature as though it were all \
of it. Another round costs seconds. An incomplete survey is wrong for as long as anyone \
reads it.

Stop only when you can state that no further round could change the answer. Before \
choosing to stop, ask yourself and answer honestly:
- Is every method named in the evidence actually measured, or are some only mentioned?
- Is there a comparison the evidence implies but never makes?
- Are there obvious variants, successors or competitors of the named methods that have \
not appeared at all?
- Does any claim stand without corroboration or contradiction?
- Would a specialist in this area name something the evidence has not covered?

If any of those has an answer, a gap remains: continue, and put that gap in "next_query" \
as a specific search — name the method, metric or comparison you want, not a restatement \
of the original question. A round steered at a named gap is worth far more than one that \
repeats the question.

What your decision causes:
- "continue": another retrieval round runs, steered by "next_query" and pushed toward \
material not yet seen.
- "stop": the run consolidates immediately and no further evidence can enter the answer.

Rounds already stop automatically once they stop finding anything new, so looping is not \
a risk you need to guard against. Judge only whether the evidence is complete.

Reply with JSON only:
{"decision": "continue|stop", "gap": "what is still missing, or empty if genuinely \
nothing", "next_query": "a specific query targeting that gap, or empty"}"""


THOROUGH_SYSTEM = """You are writing the full answer to a research question from an \
evidence table gathered across many papers.

Write a coherent narrative, not a list of summaries. Organise by what the reader needs to \
understand in order: what the question turns on, what each line of work established, how \
they compare, and where they disagree.

Rules:
- Cite every claim with the chunk id in square brackets, e.g. [12345]. Every factual \
sentence needs one. Ids come from the evidence table.
- Numbers only where the table has them, with their conditions attached. A result at one \
scale or dataset is not a result at another, and saying so is part of the answer.
- If the evidence does not support a ranking or a single answer, say so plainly and \
explain what would be needed to settle it. This is a valid and often correct outcome for \
"which is best" questions.
- End with a section headed "Disagreements" listing any place two sources conflict, or \
state that none were found.

Use markdown headings. Be complete: this is the long answer, and the reader chose it."""


TLDR_SYSTEM = """You are compressing a long research answer into the shortest complete one.

You are given the full answer. Produce the minimum text that actually answers the \
question — typically two to five sentences.

Rules:
- Every claim must already appear in the long answer. Introduce nothing new.
- Keep the chunk citations [12345] for whatever you assert.
- If the long answer concluded the evidence cannot settle the question, say that first \
rather than picking a winner anyway.

Write prose only, no headings."""


# ── records ───────────────────────────────────────────────────────────────────────


@dataclass
class Claim:
    chunk_id: int
    arxiv_id: str
    paper_title: str
    section: str
    name: str
    claim: str
    method: str = ""
    metric: str = ""
    value: str = ""
    condition: str = ""
    round_n: int = 0
    score: float = 0.0
    text_len: int = 0

    @property
    def compression(self) -> float:
        return len(self.claim) / max(1, self.text_len)


@dataclass
class Round:
    n: int
    query: str
    retrieved: int = 0
    fresh: int = 0
    relevant: int = 0
    new_papers: int = 0
    via: str = "dense"
    gap: str = ""
    ms: float = 0.0


@dataclass
class Run:
    run_id: str
    question: str
    rounds: list[Round] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    tldr: str = ""
    thorough: str = ""
    stopped_because: str = ""
    started: float = field(default_factory=time.time)
    ms: float = 0.0

    @property
    def papers(self) -> list[str]:
        return sorted({c.arxiv_id for c in self.claims})


# ── diversity ─────────────────────────────────────────────────────────────────────


def cap_per_paper(hits: list[dict], cap: int) -> list[dict]:
    """At most ``cap`` chunks from any one paper.

    A single long paper on the exact topic will otherwise fill an entire round, and a
    survey of one paper is not a survey.
    """
    seen: dict[str, int] = {}
    out = []
    for h in hits:
        pid = h.get("arxiv_id", "")
        if seen.get(pid, 0) >= cap:
            continue
        seen[pid] = seen.get(pid, 0) + 1
        out.append(h)
    return out


def mmr(hits: list[dict], vectors: dict[int, np.ndarray], k: int,
        lambda_: float = 0.7) -> list[dict]:
    """Maximal marginal relevance: rank by relevance minus similarity to what is chosen.

    Retrieval returns near-duplicates — the same result restated in the abstract, the
    introduction and the conclusion. Sending three copies to the extractor costs three
    calls and yields one claim.
    """
    if len(hits) <= k:
        return hits
    chosen: list[dict] = []
    pool = list(hits)
    while pool and len(chosen) < k:
        best, best_score = None, -1e9
        for h in pool:
            rel = float(h.get("score") or 0.0)
            v = vectors.get(int(h.get("chunk_id", -1)))
            pen = 0.0
            if v is not None and chosen:
                sims = [float(np.dot(v, vectors[int(c["chunk_id"])]))
                        for c in chosen if int(c["chunk_id"]) in vectors]
                pen = max(sims) if sims else 0.0
            s = lambda_ * rel - (1 - lambda_) * pen
            if s > best_score:
                best, best_score = h, s
        chosen.append(best)
        pool.remove(best)
    return chosen


def vectors_for(state, chunk_ids: list[int]) -> dict[int, np.ndarray]:
    """Unit-normalised full-precision vectors, keyed by chunk id."""
    if not chunk_ids:
        return {}
    out: dict[int, np.ndarray] = {}
    try:
        conn = state.conn()
        ph = ",".join("?" * len(chunk_ids))
        rows = {int(r[0]): int(r[1]) for r in conn.execute(
            f"SELECT chunk_id, vector_row FROM chunks "
            f"WHERE chunk_id IN ({ph}) AND vector_row IS NOT NULL", chunk_ids)}
        if not rows:
            return {}
        state.retriever._ensure_fp16_current()
        fp16 = state.retriever.fp16
        n = state.retriever.n_vector_rows
        for cid, r in rows.items():
            if 0 <= r < n:
                v = np.asarray(fp16[r], dtype=np.float32)
                nrm = float(np.linalg.norm(v))
                if nrm > 1e-12:
                    out[cid] = v / nrm
    except Exception:
        return out
    return out


# ── model steps ───────────────────────────────────────────────────────────────────


def _numbered(hits: list[dict], limit: int = 1100) -> str:
    out = []
    for i, h in enumerate(hits, 1):
        head = f"{h.get('paper_title') or h.get('arxiv_id','')} > {h.get('section') or ''}"
        out.append(f"[{i}] (id={h.get('chunk_id')}) {head.strip(' >')}\n"
                   f"{(h.get('text') or '')[:limit]}")
    return "\n\n".join(out)


async def extract(cfg, question: str, hits: list[dict], model, stream_answer,
                  round_n: int) -> tuple[list[Claim], list[int]]:
    """Label, name and structure one batch of excerpts. Returns (claims, rejected ids)."""
    if not hits:
        return [], []
    prompt = (f"Research question: {question}\n\nExcerpts:\n{_numbered(hits)}\n\n"
              "Reply with the JSON array only.")
    from lara.serve.generate import complete_json

    # No verdict is not evidence of irrelevance. Dropping the batch would silently lose a
    # round's work, so on failure nothing is recorded as rejected either.
    rows = await complete_json(cfg, prompt, system=EXTRACT_SYSTEM, shape="array",
                               model=model, max_tokens=1400, default=None)
    if rows is None:
        return [], []

    claims: list[Claim] = []
    kept_idx: set[int] = set()
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            continue
        try:
            n = int(r.get("n", 0))
        except (TypeError, ValueError):
            continue
        if not (1 <= n <= len(hits)):
            continue
        h = hits[n - 1]
        kept_idx.add(n)
        claims.append(Claim(
            chunk_id=int(h.get("chunk_id", 0)), arxiv_id=str(h.get("arxiv_id", "")),
            paper_title=str(h.get("paper_title", "")), section=str(h.get("section", "")),
            name=str(r.get("name") or "")[:80], claim=str(r.get("claim") or "")[:600],
            method=str(r.get("method") or "")[:120], metric=str(r.get("metric") or "")[:120],
            value=str(r.get("value") or "")[:120],
            condition=str(r.get("condition") or "")[:200],
            round_n=round_n, score=float(h.get("score") or 0.0),
            text_len=len(h.get("text") or ""),
        ))
    rejected = [int(h.get("chunk_id", 0)) for i, h in enumerate(hits, 1) if i not in kept_idx]
    return claims, rejected


async def should_continue(cfg, question: str, rnd: Round, run: Run, model,
                          stream_answer) -> dict:
    """Ask whether a gap remains. Saturation is handled by the caller, not here."""
    table = "\n".join(
        f"- {c.name}: {c.claim}" + (f"  [{c.metric} = {c.value}]" if c.value else "")
        for c in run.claims[-25:]
    ) or "(nothing yet)"
    prompt = (f"Research question: {question}\n\n"
              f"Last round: {rnd.relevant} relevant of {rnd.retrieved} retrieved, "
              f"{rnd.new_papers} new papers.\n"
              f"Total so far: {len(run.claims)} claims from {len(run.papers)} papers, "
              f"{len(run.rounds)} rounds.\n\nEvidence gathered:\n{table}\n\n"
              "Reply with the JSON object only.")
    from lara.serve.generate import complete_json

    d = await complete_json(cfg, prompt, system=CONTINUE_SYSTEM, model=model,
                            max_tokens=200)
    if d is None:
        return {"decision": "stop", "gap": "", "next_query": "", "via": "fallback"}
    return {"decision": "continue" if d.get("decision") == "continue" else "stop",
            "gap": str(d.get("gap") or "")[:300],
            "next_query": str(d.get("next_query") or "")[:300], "via": "model"}


def evidence_table(claims: list[Claim]) -> str:
    """The claims as the consolidation prompt sees them. Ids are load-bearing."""
    lines = []
    for c in claims:
        bits = [f"[{c.chunk_id}] {c.name} ({c.arxiv_id})", f"  claim: {c.claim}"]
        if c.method:
            bits.append(f"  method: {c.method}")
        if c.metric or c.value:
            bits.append(f"  measured: {c.metric} = {c.value}")
        if c.condition:
            bits.append(f"  condition: {c.condition}")
        lines.append("\n".join(bits))
    return "\n\n".join(lines) if lines else "(no evidence gathered)"


async def consolidate(cfg, run: Run, model, stream_answer, on_token=None) -> None:
    """Thorough first, then TLDR derived from it.

    Generated independently the two can contradict each other, and a reader who notices
    that stops trusting both. Compressing the long answer guarantees the short one asserts
    nothing the long one does not.
    """
    prompt = (f"Research question: {run.question}\n\n"
              f"Evidence table ({len(run.claims)} claims from {len(run.papers)} papers, "
              f"gathered over {len(run.rounds)} rounds):\n\n{evidence_table(run.claims)}\n\n"
              "Write the full answer.")
    buf = ""
    async for tok in stream_answer(cfg, prompt, [], system=THOROUGH_SYSTEM, model=model,
                                   temperature=0.2, max_tokens=2600, raw_user=True):
        buf += tok
        if on_token:
            on_token("thorough", tok)
    run.thorough = buf.strip()

    short = ""
    async for tok in stream_answer(
        cfg, f"Research question: {run.question}\n\nLong answer:\n{run.thorough}\n\n"
             "Write the shortest complete answer.",
        [], system=TLDR_SYSTEM, model=model, temperature=0.1, max_tokens=400,
        raw_user=True,
    ):
        short += tok
        if on_token:
            on_token("tldr", tok)
    run.tldr = short.strip()


# ── persistence ───────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS synthesis_runs (
    run_id TEXT PRIMARY KEY, question TEXT NOT NULL,
    tldr TEXT, thorough TEXT, stopped_because TEXT,
    n_rounds INTEGER, n_claims INTEGER, n_papers INTEGER, ms REAL,
    created_utc TEXT DEFAULT (datetime('now')));
CREATE TABLE IF NOT EXISTS synthesis_rounds (
    run_id TEXT NOT NULL, n INTEGER, query TEXT, retrieved INTEGER, fresh INTEGER,
    relevant INTEGER, new_papers INTEGER, via TEXT, gap TEXT, ms REAL);
CREATE TABLE IF NOT EXISTS synthesis_claims (
    run_id TEXT NOT NULL, chunk_id INTEGER, arxiv_id TEXT, paper_title TEXT,
    section TEXT, name TEXT, claim TEXT, method TEXT, metric TEXT, value TEXT,
    condition TEXT, round_n INTEGER, score REAL, text_len INTEGER);
CREATE INDEX IF NOT EXISTS idx_syn_claims_run ON synthesis_claims(run_id);
CREATE INDEX IF NOT EXISTS idx_syn_rounds_run ON synthesis_rounds(run_id);
"""


def save(db_path, run: Run, rejected: list[tuple[str, int]]) -> None:
    """Persist the run and every judgement it made. Never raises."""
    try:
        from lara.finetune.judgements import Judgement
        from lara.finetune.judgements import record as record_j
        from lara.store import db

        conn = db.connect(db_path)
        try:
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT OR REPLACE INTO synthesis_runs (run_id, question, tldr, thorough,"
                " stopped_because, n_rounds, n_claims, n_papers, ms)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (run.run_id, run.question, run.tldr, run.thorough, run.stopped_because,
                 len(run.rounds), len(run.claims), len(run.papers), run.ms))
            conn.execute("DELETE FROM synthesis_rounds WHERE run_id=?", (run.run_id,))
            conn.executemany(
                "INSERT INTO synthesis_rounds VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(run.run_id, r.n, r.query, r.retrieved, r.fresh, r.relevant,
                  r.new_papers, r.via, r.gap, r.ms) for r in run.rounds])
            conn.execute("DELETE FROM synthesis_claims WHERE run_id=?", (run.run_id,))
            conn.executemany(
                "INSERT INTO synthesis_claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(run.run_id, c.chunk_id, c.arxiv_id, c.paper_title, c.section, c.name,
                  c.claim, c.method, c.metric, c.value, c.condition, c.round_n, c.score,
                  c.text_len) for c in run.claims])

            # Same teacher as hierarchical scope: these are model relevance verdicts on
            # retrieved passages, and the rejects are the hard negatives.
            items = [Judgement(query=run.question, chunk_id=c.chunk_id, score=1.0, label=1,
                               teacher="llm_scope", rank=i, source="synthesis")
                     for i, c in enumerate(run.claims)]
            items += [Judgement(query=q, chunk_id=cid, score=0.0, label=0,
                                teacher="llm_scope", rank=i, source="synthesis")
                      for i, (q, cid) in enumerate(rejected)]
            if items:
                record_j(conn, items)
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def _no_table(exc: sqlite3.OperationalError) -> bool:
    """Whether this is "the table is not there yet" rather than a real failure.

    The synthesis tables are created by :func:`save`, so between a fresh install and the
    first completed run they legitimately do not exist. Reading is still a reasonable
    thing to do then -- the answer is "no runs" -- and it is worth distinguishing from a
    corrupt or unreachable database, which must still raise.
    """
    return "no such table" in str(exc)


def load_run(conn, run_id: str) -> dict | None:
    """One run by id, or None.

    **Read-only.** This used to open with ``executescript(SCHEMA)``, which is DDL, while
    the connection the server hands it is opened ``mode=ro``.

    That combination is not always fatal, which is what made it survive: SQLite lets the
    script through as long as every ``CREATE ... IF NOT EXISTS`` in it is a no-op, and
    raises ``attempt to write a readonly database`` the moment one table is genuinely
    absent and it would have to write. Measured: all three tables present succeeds, none
    present fails, *one of three* present fails.

    So it broke on exactly one database -- the one where no run has been saved yet, which
    is every fresh install, and precisely the state someone is in when they open deep
    research for the first time. Creating the schema is the writer's job; see :func:`save`.
    """
    try:
        r = conn.execute("SELECT * FROM synthesis_runs WHERE run_id=?",
                         (run_id,)).fetchone()
    except sqlite3.OperationalError as exc:
        if not _no_table(exc):
            raise
        return None
    if r is None:
        return None
    d = dict(r)
    d["rounds"] = [dict(x) for x in conn.execute(
        "SELECT * FROM synthesis_rounds WHERE run_id=? ORDER BY n", (run_id,))]
    d["claims"] = [dict(x) for x in conn.execute(
        "SELECT * FROM synthesis_claims WHERE run_id=? ORDER BY round_n, chunk_id",
        (run_id,))]
    return d


def delete_run(db_path, run_id: str) -> bool:
    """Remove a run and everything it recorded. Returns whether a row went away.

    Deleting for real, rather than hiding the run from the library, is what keeps the two
    views honest: the research pane's history list reads the same tables, so a run that is
    only unlisted would still be offered there and would come back on the next backfill.

    Takes a path rather than the reader's connection, exactly as :func:`save` does. The
    server holds the corpus open ``mode=ro`` so that serving pages can never corrupt 42 GB
    of index; writes open their own connection, and passing the shared one here fails with
    "attempt to write a readonly database".
    """
    from lara.store import db

    conn = db.connect(db_path)
    try:
        with conn:
            cur = conn.execute("DELETE FROM synthesis_runs WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM synthesis_rounds WHERE run_id=?", (run_id,))
            conn.execute("DELETE FROM synthesis_claims WHERE run_id=?", (run_id,))
        return cur.rowcount > 0
    except sqlite3.OperationalError as exc:
        if not _no_table(exc):
            raise
        return False
    finally:
        conn.close()


def list_runs(conn, limit: int = 50) -> list[dict]:
    """Recent runs, newest first. Read-only; empty before the first run is saved."""
    try:
        return [dict(r) for r in conn.execute(
            "SELECT run_id, question, n_rounds, n_claims, n_papers, ms, created_utc,"
            " substr(COALESCE(tldr,''),1,240) AS tldr FROM synthesis_runs"
            " ORDER BY created_utc DESC LIMIT ?", (limit,))]
    except sqlite3.OperationalError as exc:
        if not _no_table(exc):
            raise
        return []


# ── the run ───────────────────────────────────────────────────────────────────────


def round_limit_reached(rounds_done: int, max_rounds: int) -> bool:
    """Whether the run has used up its hard round budget.

    Every other stop condition is evidence-driven -- the corpus went dry, or the model
    voted to stop twice -- and a broad question satisfies none of them. "code LLMs"
    matches a large enough slice of the corpus that each round keeps turning up genuinely
    new, genuinely relevant claims, so saturation never arrives and the model never votes
    stop. The run does not hang; it makes progress forever, which is worse, because it
    looks like it is working.

    ``max_rounds <= 0`` disables the cap, for someone who has decided they want that.
    """
    return max_rounds > 0 and rounds_done >= max_rounds


def saturated(rounds: list, total_papers: int, *, window: int, min_papers: int,
              min_rounds: int) -> bool:
    """Whether the corpus has stopped yielding for this question.

    The test is marginal yield over a window, not a round of exactly zero. Requiring zero
    is a bar a 29-million-chunk corpus essentially never clears -- measured over eight
    ten-round runs, the mean round-9 still produced 8 claims from 6.8 new papers, and not
    one run ever recorded two consecutive empty rounds. A rule that cannot fire is not a
    stopping rule; the round cap was doing all the work while appearing not to.

    **Two guards, and both are load-bearing.** A run that has not reached ``min_rounds``,
    or that has found no papers at all, is not saturated -- it has not started. Without
    them the rule fires on rounds 1 and 2 of a slow question and reports "saturated: 0 new
    papers", which reads as "the corpus has nothing" and is indistinguishable from a
    genuine negative result. That is the worst failure available here: research that
    quietly returns nothing, in a second, and calls it an answer.
    """
    if len(rounds) < max(window, min_rounds):
        return False
    if not total_papers:
        return False
    return sum(r.new_papers for r in rounds[-window:]) < min_papers


async def run_synthesis(state, cfg, question: str, *, model=None, stream_answer=None,
                        emit=None, should_stop=None) -> Run:
    """Iterate retrieval until saturated, then consolidate.

    ``emit(event, payload)`` streams progress. ``should_stop()`` lets the caller cancel;
    a cancelled run still consolidates whatever it gathered rather than discarding it.
    """
    scfg = (cfg.get_in("retrieval.synthesis") or {}) if hasattr(cfg, "get_in") else {}
    per_round = int(scfg.get("per_round", 12))
    over_fetch = int(scfg.get("over_fetch", 4))
    paper_cap = int(scfg.get("cap_per_paper", 3))
    dry_limit = int(scfg.get("dry_rounds", 2))
    fb_cap = int(scfg.get("max_feedback_vectors", 6))
    expand_every = int(scfg.get("expand_every", 2))
    min_rounds = int(scfg.get("min_rounds", 4))
    stop_votes_needed = int(scfg.get("stop_votes", 2))
    max_rounds = int(scfg.get("max_rounds", 10))
    # Saturation: fewer than `sat_min_papers` new papers across `sat_window` rounds.
    # Defaults chosen against measured runs -- see the note at the check itself.
    sat_window = int(scfg.get("saturation_window", 2))
    sat_min_papers = int(scfg.get("saturation_min_new_papers", 3))
    # A floor above the ceiling would have min_rounds forcing "continue" on rounds the cap
    # will never allow. The cap still wins -- it is checked before any work -- but the
    # forced-continue votes in between are noise, so fold the floor down to meet it.
    if max_rounds > 0:
        min_rounds = min(min_rounds, max_rounds)

    run = Run(run_id=uuid.uuid4().hex[:12], question=question)
    seen_chunks: set[int] = set()
    seen_papers: set[str] = set()
    rejected: list[tuple[str, int]] = []
    dry = 0
    stop_votes = 0
    query = question
    via = "dense"

    def ev(name, payload):
        if emit:
            emit(name, payload)

    ev("start", {"run_id": run.run_id, "question": question})

    while True:
        if should_stop and should_stop():
            run.stopped_because = "cancelled by user"
            break

        # Checked before any work, so the limit is the number of rounds actually run.
        if round_limit_reached(len(run.rounds), max_rounds):
            # Say what it was still finding when the budget ran out. "Reached the limit"
            # alone does not distinguish a survey that was done from one that was cut off
            # mid-yield, and those call for opposite responses from whoever reads it.
            tail = run.rounds[-sat_window:]
            still = sum(x.new_papers for x in tail)
            run.stopped_because = (
                f"reached the {max_rounds}-round limit; still finding {still} new "
                f"paper(s) per {len(tail)} rounds"
                + (f"; last gap: {run.rounds[-1].gap}"
                   if run.rounds and run.rounds[-1].gap else ""))
            ev("limit", {"rounds": len(run.rounds), "max_rounds": max_rounds})
            break

        n = len(run.rounds) + 1
        t0 = time.perf_counter()
        ev("round", {"n": n, "query": query, "via": via, "phase": "retrieving"})

        papers_scope = None
        if via == "citations" and seen_papers:
            # Structural reach: papers the confirmed ones cite, which similarity alone
            # will not surface when they use different vocabulary for the same idea.
            nb: list[str] = []
            for pid in list(seen_papers)[:20]:
                n1 = state.neighbours(pid, limit=25)
                nb.extend(n1.get("cites", []) + n1.get("cited_by", []))
            papers_scope = [p for p in dict.fromkeys(nb) if p not in seen_papers] or None

        fb = _feedback(state, run.claims, fb_cap)
        # Off the event loop. This call takes ~500 ms and used to hold the loop for all of
        # it, so with several syntheses running concurrently nobody streamed a token while
        # any one of them retrieved -- measured at 10.7% of a six-question batch. Moving it
        # to a thread lets the others keep generating; the semaphore inside is what stops
        # the retrievals it now allows to overlap from colliding on one GPU index.
        r = await asyncio.to_thread(
            _retrieve, state.retriever, query, papers_scope, per_round * over_fetch, fb,
            int(scfg.get("concurrent_retrievals", 2)))
        from lara.serve.hierarchy import to_dict
        hits = [to_dict(h) for h in r.hits]
        rnd = Round(n=n, query=query, retrieved=len(hits), via=via)

        fresh = [h for h in hits if int(h.get("chunk_id", -1)) not in seen_chunks]
        rnd.fresh = len(fresh)
        fresh = cap_per_paper(fresh, paper_cap)
        picked = mmr(fresh, vectors_for(state, [int(h["chunk_id"]) for h in fresh]),
                     per_round)
        for h in picked:
            seen_chunks.add(int(h.get("chunk_id", -1)))

        if not picked:
            dry += 1
            rnd.ms = (time.perf_counter() - t0) * 1000
            run.rounds.append(rnd)
            ev("round_done", asdict(rnd))
            if dry >= dry_limit:
                run.stopped_because = f"{dry} rounds found nothing new"
                break
            via = "citations" if via == "dense" else "dense"
            continue

        ev("round", {"n": n, "phase": "reading", "n_chunks": len(picked)})
        claims, rej = await extract(cfg, question, picked, model, stream_answer, n)
        rejected.extend((question, cid) for cid in rej)
        before_papers = set(seen_papers)
        for c in claims:
            seen_papers.add(c.arxiv_id)
        run.claims.extend(claims)
        rnd.relevant = len(claims)
        rnd.new_papers = len(seen_papers - before_papers)
        rnd.ms = (time.perf_counter() - t0) * 1000
        run.rounds.append(rnd)

        ev("claims", {"round": n, "claims": [asdict(c) for c in claims]})
        ev("round_done", asdict(rnd))

        # Saturation overrides the model: a corpus that has stopped yielding is evidence,
        # whatever the model would prefer.
        #
        if saturated(run.rounds, len(run.papers), window=sat_window,
                     min_papers=sat_min_papers, min_rounds=min_rounds):
            found = sum(x.new_papers for x in run.rounds[-sat_window:])
            run.stopped_because = (f"saturated: {found} new paper(s) across "
                                   f"the last {sat_window} rounds")
            break
        if rnd.relevant == 0:
            dry += 1
            if dry >= dry_limit:
                run.stopped_because = f"{dry} rounds found nothing relevant"
                break
        else:
            dry = 0

        verdict = await should_continue(cfg, question, rnd, run, model, stream_answer)
        rnd.gap = verdict.get("gap", "")

        # Two independent brakes on stopping early, because that is the failure that
        # matters: a survey reporting a fraction of the literature as though it were all
        # of it. `min_rounds` ignores the vote entirely while the run is still shallow,
        # and after that a single stop vote is treated as a suggestion — it takes
        # consecutive ones to end the run. Any "continue" resets the count.
        if verdict["decision"] == "stop":
            stop_votes += 1
        else:
            stop_votes = 0
        forced = n < min_rounds
        ev("decision", {"round": n, **verdict, "stop_votes": stop_votes,
                        "needed": stop_votes_needed, "forced_continue": forced})
        if verdict["decision"] == "stop" and not forced and stop_votes >= stop_votes_needed:
            run.stopped_because = (f"model voted stop {stop_votes}x"
                                   + (f"; last gap: {rnd.gap}" if rnd.gap else ""))
            break

        query = verdict.get("next_query") or question
        via = "citations" if (n % expand_every == 0) else "dense"

    ev("consolidating", {"claims": len(run.claims), "papers": len(run.papers)})
    if run.claims and stream_answer is not None:
        try:
            await consolidate(cfg, run, model, stream_answer,
                              on_token=lambda k, t: ev("token", {"target": k, "text": t}))
        except Exception as exc:
            # The rounds are the expensive part -- ten to twenty minutes of retrieval and
            # model judgements -- and consolidation is one call at the very end. Letting
            # it take the run down with it discarded all of that and saved nothing, so the
            # work could not even be reopened from the library. The same reasoning the
            # cancellation path already applies: keep what was gathered.
            #
            # `consolidate` is also the only model call here that is not already
            # exception-swallowing: `complete` returns "" on failure, so a bad round
            # degrades quietly. That asymmetry is why this was the step that lost runs.
            run.thorough = (
                f"{len(run.claims)} claims from {len(run.papers)} papers were gathered "
                f"over {len(run.rounds)} rounds, but writing the answer failed: {exc}\n\n"
                "The evidence is saved and the run can be reopened.")
            run.tldr = f"Consolidation failed: {exc}"
            run.stopped_because = ((run.stopped_because or "") +
                                   f"; consolidation failed: {type(exc).__name__}").lstrip("; ")
            ev("error", f"consolidation failed, evidence kept: {exc}")
    elif not run.claims:
        run.thorough = "No relevant evidence was found in the corpus for this question."
        run.tldr = run.thorough

    run.ms = (time.time() - run.started) * 1000
    save(state.db_path, run, rejected)
    ev("done", {"run_id": run.run_id, "ms": run.ms, "claims": len(run.claims),
                "papers": len(run.papers), "rounds": len(run.rounds),
                "stopped_because": run.stopped_because,
                "tldr": run.tldr, "thorough": run.thorough})
    return run


@lru_cache(maxsize=8)
def _retrieval_slots(n: int) -> threading.BoundedSemaphore:
    """One bound per configured width. Process-wide on purpose: it is guarding a single
    GPU index and a single cross-encoder, which every synthesis in this process shares."""
    return threading.BoundedSemaphore(max(1, n))


def _retrieve(retriever, query: str, papers, final_k: int, feedback, concurrency: int):
    """Retrieval, run in a worker thread under a concurrency bound.

    Two at a time by default. The work is a dense matmul and a cross-encoder forward on
    one card; letting an unbounded number in would trade the loop-blocking problem for a
    VRAM one.
    """
    with _retrieval_slots(concurrency):
        return retriever.retrieve(query, papers=papers, final_k=final_k, feedback=feedback)


def _feedback(state, claims: list[Claim], cap: int) -> list[np.ndarray]:
    """Vectors of the highest-scoring confirmed chunks, capped.

    Best-first rather than most-recent: a late round should still be steered by the
    strongest evidence found, not by whatever happened to arrive last.
    """
    if not claims:
        return []
    best = sorted(claims, key=lambda c: -c.score)[:cap]
    vecs = vectors_for(state, [c.chunk_id for c in best])
    return [vecs[c.chunk_id] for c in best if c.chunk_id in vecs]
