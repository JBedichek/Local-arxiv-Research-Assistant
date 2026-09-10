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

#: Full-paper reads are capped per run (not per round or per batch): a run answering a
#: broad question could otherwise spend its whole budget reading one paper cover to
#: cover instead of surveying the literature, which is the opposite of what deep
#: research is for. Tracked on `Run.full_papers_read` -- see `resolve_full_paper`.
MAX_FULL_PAPERS_PER_RUN = 5

#: The reconstructed paper is read only inside its own isolated sub-call (see
#: `_answer_from_paper`) and never enters the round-judging context, so this can afford
#: to be generous without growing the prompt every other excerpt sits in.
FULL_PAPER_MAX_CHARS = 45_000

FULL_PAPER_TOOL = "get_full_paper"

EXTRACT_SYSTEM = f"""You are surveying the literature to answer a research question.

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
[{{"n": 1, "name": "...", "claim": "...", "method": "...", "metric": "...", \
"value": "...", "condition": "..."}}]
Return [] if none are relevant.

If a specific excerpt looks central to the question but is too thin to judge -- a table \
or configuration you'd expect is in the paper but isn't in this excerpt, or the excerpt \
implies a number it doesn't give -- you may ask that paper a targeted question instead of \
judging the excerpt from this alone. Do this rarely: reaching for the full paper on every \
excerpt defeats the point of retrieving excerpts and is slow.

To ask, put one item like this in the array instead of judgements for those excerpts:
{{"tool": "{FULL_PAPER_TOOL}", "requests": [{{"n": <excerpt number>, "query": "<a \
specific question about that paper, phrased against the research question -- not \
\\"summarize this paper\\">"}}, ...]}}
You can ask about several papers, or ask one paper more than one question, in the same \
"requests" list. Each question is answered by reading that paper alone -- you do not see \
the paper itself, only the answer, labeled by which paper and question it addresses -- \
and then you judge every excerpt in one more reply. At most {MAX_FULL_PAPERS_PER_RUN} \
distinct papers can be read in full across the whole run, so use this when it matters."""


PAPER_QA_SYSTEM = """Answer the question using only the paper text given below.

Cite where in the paper the answer comes from when it helps (a section name, a table, \
the nearby text) so the reader can tell the answer was actually found there.

If the paper does not address the question, say so plainly rather than forcing an answer \
from somewhere else in it. A clear "this paper doesn't cover that" is a correct and \
useful answer; an invented one is not."""


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
    #: arxiv_id -> full reassembled text, for papers `get_full_paper` has read this run.
    #: Doubles as the cap tracker (its length is how many distinct papers have been read
    #: in full, capped at MAX_FULL_PAPERS_PER_RUN) and as a cache, so a repeat request
    #: for a paper already read is served for free rather than counted twice.
    full_papers_read: dict[str, str] = field(default_factory=dict)
    #: (arxiv_id, query) -> the sub-call's answer, so an identical question about a
    #: paper asked twice in the same run is answered from cache rather than re-run.
    paper_qa_answers: dict[tuple[str, str], str] = field(default_factory=dict)

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


# ── full-paper tool ──────────────────────────────────────────────────────────────
#
# The extractor sees chunk-sized excerpts, and a chunk is sometimes too thin to judge:
# the table with the actual number is two sections away, or an abstract-level claim
# needs the method section to check. `get_full_paper` lets the model ask a targeted
# question of the whole paper -- not read the whole paper itself, which would blow up
# the small per-round judging context with however many thousand tokens the paper
# happens to be. The paper is read once, in an isolated sub-call (`_answer_from_paper`)
# whose only output that reaches the round is the answer text.
#
# Reassembled natively rather than via `autoresearch.citeindex.get_paper`: that module
# lives one repository over, and `lara-core` cannot import `autoresearch` at all --
# `autoresearch/synthesizer.py`'s docstring is explicit that the leaf primitive crossing
# that boundary is `Lara.aresearch`, a plain injected callable, specifically so this
# package never imports `autoresearch` or `session`. The corpus itself is reachable
# directly, the same way `vectors_for` above already reads it via `state.conn()`.


def _read_full_paper(conn, arxiv_id: str, max_chars: int) -> str:
    """Every chunk of one paper, in reading order. Mirrors autoresearch.methods.whole_paper."""
    row = conn.execute("SELECT latest_version FROM papers WHERE arxiv_id=?",
                       (arxiv_id,)).fetchone()
    version = int(row[0]) if row and row[0] else None
    q = "SELECT text FROM chunks WHERE arxiv_id=?"
    params: list = [arxiv_id]
    if version is not None:
        q += " AND version=?"
        params.append(version)
    parts = [r[0] for r in conn.execute(q + " ORDER BY ordinal", params)]
    text = "\n\n".join(p for p in parts if p)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n[truncated at {max_chars:,} of {len(text):,} characters]"
    return text


def resolve_full_paper(state, run: Run, arxiv_id: str) -> tuple[str, bool]:
    """Full text for `arxiv_id`, honouring the per-run cap. Returns (text, capped).

    A paper already read this run is served from `run.full_papers_read` and never
    counts against the cap twice. `capped` tells the caller the request was refused so
    it can say so honestly -- back to the model, in this case -- instead of failing
    silently or raising.
    """
    if arxiv_id in run.full_papers_read:
        return run.full_papers_read[arxiv_id], False
    if len(run.full_papers_read) >= MAX_FULL_PAPERS_PER_RUN:
        return "", True
    text = _read_full_paper(state.conn(), arxiv_id, FULL_PAPER_MAX_CHARS)
    run.full_papers_read[arxiv_id] = text
    return text, False


async def _answer_from_paper(cfg, model, arxiv_id: str, text: str, query: str) -> str:
    """One isolated completion: this paper's text plus one question, nothing else.

    Its own fresh conversation, not appended to the round-judging prompt -- the whole
    point is that a paper can be tens of thousands of tokens and the round prompt must
    not grow with it. Uses `complete`, the same plain single-turn primitive the rest of
    this module's JSON verdicts are built on (via `complete_json`), just without the
    JSON parsing since this call's output is prose, not a schema.
    """
    from lara.serve.generate import complete

    prompt = f"Paper ({arxiv_id}):\n\n{text}\n\nQuestion: {query}\n\nAnswer:"
    return await complete(cfg, prompt, system=PAPER_QA_SYSTEM, model=model,
                          temperature=0.0, max_tokens=600)


def _render_paper_answers(results: list[dict]) -> dict[int, str]:
    """Tool results as text to append per excerpt, labelled so several answers don't blur."""
    by_n: dict[int, list[str]] = {}
    for r in results:
        by_n.setdefault(r["n"], []).append(
            f"[tool result -- asked {r['arxiv_id']}: {r['query']}]\n{r['answer']}")
    return {n: "\n\n".join(blocks) for n, blocks in by_n.items()}


async def _dispatch_paper_requests(cfg, model, state, run: Run, hits: list[dict],
                                   requests: list[dict], round_n: int, ev) -> list[dict]:
    """Resolve a batch of {n, query} tool requests. Returns one result dict per request.

    Fetching and cap-accounting run sequentially first -- cheap, and it is what makes
    "only fetch as many as the remaining budget allows" correct when a batch asks for
    more distinct papers than the cap has left. The paper-reading sub-calls themselves
    are the slow part and are independent of each other, so those run concurrently via
    `asyncio.gather`, the same concurrency idiom `agent.py`'s query-decomposition search
    already uses for independent sub-work.
    """
    to_run: list[tuple[int, str, str, str]] = []   # (n, arxiv_id, query, paper_text)
    results: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for req in requests:
        if not isinstance(req, dict):
            continue
        try:
            n = int(req.get("n", 0))
        except (TypeError, ValueError):
            continue
        query = str(req.get("query") or "").strip()[:400]
        if not (1 <= n <= len(hits)) or not query:
            continue
        arxiv_id = str(hits[n - 1].get("arxiv_id") or "")
        if not arxiv_id:
            continue
        pair = (arxiv_id, query)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        if pair in run.paper_qa_answers:
            results.append({"n": n, "arxiv_id": arxiv_id, "query": query,
                            "answer": run.paper_qa_answers[pair]})
            continue
        text, capped = resolve_full_paper(state, run, arxiv_id)
        if ev:
            ev("full_paper", {"round": round_n, "arxiv_id": arxiv_id, "query": query,
                              "capped": capped, "n_read": len(run.full_papers_read),
                              "cap": MAX_FULL_PAPERS_PER_RUN})
        if capped:
            results.append({"n": n, "arxiv_id": arxiv_id, "query": query,
                            "answer": (f"[cap reached: {MAX_FULL_PAPERS_PER_RUN} papers "
                                       "already read in full this run; this one was not "
                                       "fetched -- judge from the excerpt above]")})
            continue
        to_run.append((n, arxiv_id, query, text))

    if to_run:
        answers = await asyncio.gather(*[
            _answer_from_paper(cfg, model, arxiv_id, text, query)
            for _n, arxiv_id, query, text in to_run
        ])
        for (n, arxiv_id, query, _text), answer in zip(to_run, answers, strict=True):
            run.paper_qa_answers[(arxiv_id, query)] = answer
            results.append({"n": n, "arxiv_id": arxiv_id, "query": query, "answer": answer})
    return results


# ── model steps ───────────────────────────────────────────────────────────────────


def _numbered(hits: list[dict], limit: int = 1100, extra: dict[int, str] | None = None) -> str:
    out = []
    extra = extra or {}
    for i, h in enumerate(hits, 1):
        head = f"{h.get('paper_title') or h.get('arxiv_id','')} > {h.get('section') or ''}"
        body = (h.get('text') or '')[:limit]
        if i in extra:
            body += f"\n\n{extra[i]}"
        out.append(f"[{i}] (id={h.get('chunk_id')}) {head.strip(' >')}\n{body}")
    return "\n\n".join(out)


async def extract(cfg, question: str, hits: list[dict], model, stream_answer,
                  round_n: int, *, state=None, run: Run | None = None,
                  ev=None) -> tuple[list[Claim], list[int]]:
    """Label, name and structure one batch of excerpts. Returns (claims, rejected ids).

    When `state` and `run` are given, the model may also call `get_full_paper` (see
    EXTRACT_SYSTEM) to ask a targeted question of a paper in full before judging its
    excerpt -- capped per run at MAX_FULL_PAPERS_PER_RUN, see `resolve_full_paper`.
    Without them (existing callers, and most tests) this is unchanged from before the
    tool existed.
    """
    if not hits:
        return [], []
    from lara.serve.generate import complete_json

    prompt = (f"Research question: {question}\n\nExcerpts:\n{_numbered(hits)}\n\n"
              "Reply with the JSON array only.")

    # No verdict is not evidence of irrelevance. Dropping the batch would silently lose a
    # round's work, so on failure nothing is recorded as rejected either.
    rows = await complete_json(cfg, prompt, system=EXTRACT_SYSTEM, shape="array",
                               model=model, max_tokens=1400, default=None)
    if rows is None:
        return [], []
    rows = rows if isinstance(rows, list) else []

    tool_rows = [r for r in rows if isinstance(r, dict) and r.get("tool") == FULL_PAPER_TOOL]
    if tool_rows and state is not None and run is not None:
        requests = [req for r in tool_rows for req in (r.get("requests") or [])]
        if requests:
            results = await _dispatch_paper_requests(cfg, model, state, run, hits,
                                                      requests, round_n, ev)
            extra = _render_paper_answers(results)
            prompt2 = (f"Research question: {question}\n\n"
                       f"Excerpts:\n{_numbered(hits, extra=extra)}\n\n"
                       "Reply with the JSON array only, judging every excerpt now -- no "
                       "more tool calls this reply.")
            rows2 = await complete_json(cfg, prompt2, system=EXTRACT_SYSTEM, shape="array",
                                        model=model, max_tokens=1400, default=None)
            if rows2 is not None:
                rows = rows2 if isinstance(rows2, list) else []

    claims: list[Claim] = []
    kept_idx: set[int] = set()
    for r in rows:
        if not isinstance(r, dict) or r.get("tool"):
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


# Used only when the generator can't be reached right now to ask its real context window
# (context_limit/count_tokens returned None -- see their docstrings in generate.py for
# why they refuse to guess). This deployment's vLLM config sets max_model_len: 32768
# (lara-core/config.yaml); reserving room for a sizeable evidence-table prompt still
# leaves several times the old 2600-token ceiling, so a fallback answer stays the rare
# exception this was supposed to be rather than routine.
_FALLBACK_THOROUGH_TOKENS = 8000

# Chat-template role markers and special tokens add a bit on top of what a plain
# /tokenize call on raw text reports, and generation should stop short of the wire
# rather than exactly on it.
_CONTEXT_SAFETY_MARGIN = 512

# However tight the real context is, leave enough room that the answer is not just
# stopped before it started.
_MIN_THOROUGH_TOKENS = 256


async def _thorough_budget(cfg, model, system: str, prompt: str) -> int:
    """Completion tokens actually free for the thorough answer.

    The synthesizer's goals are deliberately dense and need much more room than a
    hardcoded ceiling sized for a simple question -- so this asks the server what the
    model's context window really is and what the prompt actually costs, and uses
    whatever is left. Falls back to a generous fixed budget rather than a small one when
    the server can't answer right now, since a small fallback silently reintroduces the
    bug this replaces.
    """
    from lara.serve import generate as GEN

    vcfg = cfg.get_in("serving.vllm") or {}
    base_url = vcfg.get("base_url", "http://127.0.0.1:8000/v1")
    model_name = model or vcfg.get("default_model") or ""
    api_key = vcfg.get("api_key")

    limit = await GEN.context_limit(base_url, model_name, api_key=api_key)
    counts = await GEN.count_tokens(base_url, model_name, [system, prompt], api_key=api_key)
    if limit is None or counts is None:
        return _FALLBACK_THOROUGH_TOKENS
    return max(limit - sum(counts) - _CONTEXT_SAFETY_MARGIN, _MIN_THOROUGH_TOKENS)


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
    # Bounded by the model's real remaining context, not a fixed ceiling -- see
    # _thorough_budget. Truncation should only happen if the model still doesn't finish
    # within the room that's actually there.
    thorough_budget = await _thorough_budget(cfg, model, THOROUGH_SYSTEM, prompt)
    buf = ""
    async for tok in stream_answer(cfg, prompt, [], system=THOROUGH_SYSTEM, model=model,
                                   temperature=0.2, max_tokens=thorough_budget, raw_user=True):
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
-- Temporal relationships table (Zep-style knowledge graph)
CREATE TABLE IF NOT EXISTS synthesis_temporal (
    subject TEXT NOT NULL,           -- Entity ID (claim_id, run_id, deliverable_id)
    relation TEXT NOT NULL,          -- Type of relationship (before, after, during, causes, depends_on, etc.)
    object TEXT NOT NULL,            -- Target entity ID
    valid_from REAL,                 -- Unix timestamp when this relationship became valid
    valid_to REAL,                   -- Unix timestamp when this relationship ceased to be valid
    run_id TEXT,                     -- Which run recorded this relationship
    PRIMARY KEY (subject, relation, object, valid_from)
);
CREATE INDEX IF NOT EXISTS idx_syn_temporal_subject ON synthesis_temporal(subject);
CREATE INDEX IF NOT EXISTS idx_syn_temporal_object ON synthesis_temporal(object);
CREATE INDEX IF NOT EXISTS idx_syn_temporal_run ON synthesis_temporal(run_id);
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

            # Record temporal relationships between claims within this run
            _record_temporal_relationships(conn, run)

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


def _record_temporal_relationships(conn: sqlite3.Connection, run: Run) -> None:
    """Record temporal relationships between claims and other entities.
    
    Creates temporal edges in a knowledge graph that enables "what did we believe about
    X at the time of run Y" queries. Relationships include:
    - claim → claim: chronological order within a run
    - claim → paper: which paper the claim came from
    - claim → run: which run produced the claim
    - run → run: successor relationships (for replans)
    
    Args:
        conn: Database connection
        run: Completed synthesis run with claims
    """
    if not run.claims:
        return
    
    # Get the run's timestamp (created_utc is a datetime string, convert to Unix time)
    run_timestamp = _parse_utc_to_timestamp(run.created_utc)
    
    # Record temporal edges for claims in this run
    for i, claim in enumerate(run.claims):
        claim_id = f"claim:{claim.chunk_id}"
        run_id = run.run_id
        
        # Claim → Run: This claim was produced by this run
        conn.execute(
            "INSERT OR REPLACE INTO synthesis_temporal "
            "(subject, relation, object, valid_from, valid_to, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (claim_id, "produced_by", run_id, run_timestamp, None, run_id)
        )
        
        # Claim → Paper: This claim came from this paper
        paper_id = f"paper:{claim.arxiv_id}"
        conn.execute(
            "INSERT OR REPLACE INTO synthesis_temporal "
            "(subject, relation, object, valid_from, valid_to, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (claim_id, "from_paper", paper_id, run_timestamp, None, run_id)
        )
        
        # Claim → Claim: chronological order (claim i → claim i+1)
        if i > 0:
            prev_claim_id = f"claim:{run.claims[i-1].chunk_id}"
            conn.execute(
                "INSERT OR REPLACE INTO synthesis_temporal "
                "(subject, relation, object, valid_from, valid_to, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (prev_claim_id, "before", claim_id, run_timestamp, None, run_id)
            )
    
    # Run → Run: successor relationship (for replanned runs)
    # This requires tracking parent run IDs which we don't have here yet
    # For now, just record run → run (current run)
    conn.execute(
        "INSERT OR REPLACE INTO synthesis_temporal "
        "(subject, relation, object, valid_from, valid_to, run_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, "continues", run_id, run_timestamp, None, run_id)
    )


def _parse_utc_to_timestamp(utc_str: str) -> float:
    """Parse UTC datetime string to Unix timestamp.
    
    Args:
        utc_str: ISO format UTC datetime string (e.g., "2026-08-25T14:30:00Z")
                or SQLite datetime string (e.g., "2026-08-25")
                or a numeric timestamp already
        
    Returns:
        Unix timestamp as float
    """
    import time
    import datetime
    
    if not utc_str:
        return time.time()
    
    # Already a numeric timestamp
    try:
        return float(utc_str)
    except (ValueError, TypeError):
        pass
    
    # Try ISO format with Z suffix
    try:
        dt = datetime.datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except ValueError:
        pass
    
    # Try SQLite datetime format (YYYY-MM-DD)
    try:
        dt = datetime.datetime.strptime(utc_str, "%Y-%m-%d")
        return dt.timestamp()
    except ValueError:
        pass
    
    # Fallback to current time
    return time.time()


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
        claims, rej = await extract(cfg, question, picked, model, stream_answer, n,
                                    state=state, run=run, ev=ev)
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
