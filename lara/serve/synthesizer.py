"""A dependency graph of research sub-questions that grows itself, round by round.

A question that is not yet well-scoped only reveals its second and third sub-questions once
the first answer has landed, so a plan made up front cannot ask them. This grows a graph of
`LogicalGoal`s the identical way: a digest of what has landed so far, one reasoning call
via `converse.talk` offering `spawn_goal`/`refine_goal`/`finish` (plus `retrieve_facts`, and
`spawn_subsynthesis` for a top-level run), dispatch onto a live `SynthesizerState`, execute
whatever is newly `pending` concurrently through the deep-research leaf, compress when the
digest would outgrow the model's window, and stop once enough consecutive rounds propose
nothing. The deliverable is then written section by section, not one flat compression.

**The leaf is an injected async callable**, `aresearch(question, *, model, base_url,
api_key)`, returning an object with `.thorough`/`.tldr` (`citations.CitedText`-shaped) --
in production `lara.serve.synthruns`' wrapper over `lara.serve.synthesis.run_synthesis`.
This module imports no retriever or index, so it is testable with a plain fake.

**Persistence is the graph's own.** A `SynthesizerState` is written to its own file after
every change (`save`), so a crash mid-synthesis loses at most the round in flight.

**Citations survive compression by re-resolution, not by an extra field.** `compressed` is
plain text with no attached reference table; compression re-runs `citations.bind` on its own
output against whatever the still-live goals contribute plus, optionally, the corpus itself.
A bracket a compression's text keeps is never silently dropped; one its model output
actually drops is not fabricated back in either.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lara.serve import citations as C
from lara.serve import context as CX

# ── status of one logical goal ───────────────────────────────────────────────────

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
STATUSES = (PENDING, RUNNING, DONE, FAILED)

#: How many consecutive rounds may propose nothing (no `spawn_goal`/`refine_goal` call
#: that actually created a goal) before the graph is considered exhausted and final
#: compression runs. A genuine guess —
#: sized to survive one or two rounds where the model is merely being cautious, without
#: burning many rounds once the graph really has nothing left to add. Tune from measured
#: runs once there are some.
MAX_IDLE_ROUNDS = 5

#: `max_idle_rounds` a nested synthesis (`spawn_subsynthesis`) runs with — smaller than
#: the top-level `MAX_IDLE_ROUNDS` on purpose. A sub-synthesis exists to answer one
#: already-scoped-down sub-question, not to run as long an investigation as the parent;
#: wrapping up sooner is what keeps a `spawn_subsynthesis` call's wall-clock and cost
#: bounded, on top of `MAX_SUBSYNTHESES_PER_SYNTH` bounding how many of them there can be.
SUBSYNTHESIS_MAX_IDLE_ROUNDS = 3

#: How deep a chain of `refine_goal` calls may go from the nearest breadth (`spawn_goal`)
#: goal. Generous per the user's explicit instruction — refinement is meant to be able to
#: go as deep as a genuinely productive line of follow-up questions goes, and breadth is
#: what actually bounds the graph (via `MAX_IDLE_ROUNDS`; there is no separate breadth
#: cap, deliberately — see the module docstring).
MAX_REFINEMENT_DEPTH = 10

#: How many goals may sit `pending`/`running` at once. `_execute_round` dispatches every
#: `pending` goal concurrently via one `asyncio.gather`, and the bounded-budget threshold
#: (`_deliverable_threshold`) is only checked once per round, after that round's goals
#: finish -- so an uncapped round can let the digest jump straight past the threshold in
#: one step. Capping in-flight goals bounds how much a single round can grow the digest
#: before the next check.
MAX_CONCURRENT_GOALS = 3

#: How many goals one synthesis run may answer by running a whole nested synthesis
#: (`spawn_subsynthesis`) rather than one flat `aresearch` call. Low and fixed for the
#: this is meant to be rare, reserved for a
#: sub-question that is itself genuinely multi-part, not a routine escalation — and
#: unlike `spawn_goal`/`refine_goal`, each one is a whole second research loop, not one
#: retrieval call, so the cap is also what keeps one round's wall-clock and cost bounded.
MAX_SUBSYNTHESES_PER_SYNTH = 2

#: How a finished run is graded, from how its logical goals actually landed.
SUCCESS = "success"
PARTIAL = "partial"
FAILURE = "failure"
ZERO_ENGAGEMENT = "zero-engagement"

SPAWN_TOOL = "spawn_goal"
REFINE_TOOL = "refine_goal"
FINISH_TOOL = "finish"
#: See `lara.serve.facts` — reads that module's store rather than spawning new research.
#: Offered unconditionally (no `allow_x` gate): unlike subsynthesis, a retrieval call is
#: cheap and has no depth concern, so there is nothing to reserve it for.
RETRIEVE_FACTS_TOOL = "retrieve_facts"
#: Only ever offered by a top-level run, never by a nested one — see `synth_tools` and
#: `run`'s own `allow_subsynthesis` handling. This is the whole of the depth-1 limit:
#: no counter, no threaded depth parameter, just "a run started by this tool cannot
#: itself offer it again."
SPAWN_SUBSYNTHESIS_TOOL = "spawn_subsynthesis"

#: Turns one reasoning pass may take. A
#: single round here may reasonably want to open several breadth goals at once — the
#: graph widens for free in wall-clock, so width is preferred over depth — so the pass needs room for more than one tool call before it stops.
MAX_TURNS = 6

#: `reply_room`'s fallback for `_reason_round` when `max_model_len` isn't known. Same
#: reasoning as `_compress`'s `DEFAULT_COMPRESS_TOKENS`: unreachable from `run()`, which
#: always passes a real window, but kept generous rather than small.
DEFAULT_REASON_TOKENS = 8_000

#: Ceiling on one reasoning round's reply. A round's answer is a tool call (or a short
#: "nothing to add") — far smaller than a compression summary — so this stays well below
#: `MAX_COMPRESS_TOKENS`.
MAX_REASON_TOKENS = 4_000


@dataclass
class LogicalGoal:
    """One node in the graph: a question, what it depends on, and its answer once run.

    `depth` counts refinement steps from the nearest breadth goal — a `spawn_goal` goal is
    always depth 0, a `refine_goal` of it is depth 1, a refinement of that is depth 2, and
    so on, checked against `MAX_REFINEMENT_DEPTH` before a new one is created.
    """

    id: str
    text: str
    depends_on: list[str] = field(default_factory=list)
    #: The id of the goal this one deepens, or None for a breadth (`spawn_goal`) goal.
    refines: str | None = None
    depth: int = 0
    status: str = PENDING
    #: The deep-research leaf's thorough answer, once `status` is
    #: `done`. Empty otherwise.
    summary: str = ""
    #: `{citation_key: Reference.to_dict()}` for every citation this goal's own answer
    #: resolved — the same keying `citations.py`/`serve/runs.py:Run.references` already
    #: use, so a key here is a `Reference` a moment away from `citations.Reference.from_dict`.
    citations: dict = field(default_factory=dict)
    #: Set when `status` is `failed`: what `run_synthesis` raised.
    error: str = ""
    #: Set by `spawn_subsynthesis` (never by `spawn_goal`/`refine_goal`): this goal is
    #: answered by running a whole nested `run()` over its own text as the objective,
    #: not by one flat `aresearch` call — see `_execute_round`'s `_run_one`. `summary`
    #: and `citations` end up holding exactly what they would for an ordinary goal (the
    #: nested run's own deliverable and references), so nothing downstream — `_digest`,
    #: `_cluster_goals`, the final write — needs to know or care that this one is
    #: different; only `_run_one` branches on it.
    nested: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LogicalGoal":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class SynthesizerState:
    """The live graph for one `synthesize`-kind stage.

    `goals` is an ordinary `dict`, and its insertion order is relied on as a dependency
    order for `_digest`: every goal's `depends_on` may only name an id already in the
    dict (`_do_spawn`/`_do_refine` refuse anything else), so a goal is always created
    after everything it depends on — checked at creation, not re-derived from a
    topological sort.
    """

    objective: str
    goals: dict[str, LogicalGoal] = field(default_factory=dict)
    idle_rounds: int = 0
    #: Prior compression summaries, oldest first. See the module docstring for why this
    #: carries no attached citation table of its own.
    compressed: list[str] = field(default_factory=list)
    round: int = 0
    #: Count of `_reason_round` calls that came back with no tool call, no text and no
    #: error — the starved-thinking-budget failure `_reason_round`'s docstring describes.
    #: Distinct from `idle_rounds`: an idle round can be the model genuinely proposing
    #: nothing, this is the model call itself never actually answering. Observability
    #: only — does not affect `idle_rounds` or when the loop stops.
    silent_reason_rounds: int = 0
    #: Running totals of the synthesizer's own model calls — `_reason_round` and
    #: `_compress` — not the token cost of the `aresearch(...)` calls each goal spawns.
    tokens_in: int = 0
    tokens_out: int = 0
    #: How many `spawn_subsynthesis` calls this run has accepted, checked against
    #: `MAX_SUBSYNTHESES_PER_SYNTH` by `_do_spawn_subsynthesis`. A dedicated counter
    #: rather than counting `nested` goals in `self.goals` directly, because a nested
    #: goal can be cleared out of `goals` by a mid-loop compaction the same as any other
    #: `done`/`failed` goal (see `run`) — the cap has to survive that the same way
    #: the same way, by living on a counter that is only ever appended to, never rebuilt
    #: from `goals`.
    subsyntheses_spawned: int = 0

    def to_dict(self) -> dict:
        return {"objective": self.objective,
                "goals": {gid: g.to_dict() for gid, g in self.goals.items()},
                "idle_rounds": self.idle_rounds, "compressed": list(self.compressed),
                "round": self.round,
                "silent_reason_rounds": self.silent_reason_rounds,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
                "subsyntheses_spawned": self.subsyntheses_spawned}

    @classmethod
    def from_dict(cls, d: dict) -> "SynthesizerState":
        d = d or {}
        goals = {str(gid): LogicalGoal.from_dict(g)
                 for gid, g in (d.get("goals") or {}).items()}
        return cls(objective=str(d.get("objective") or ""), goals=goals,
                    idle_rounds=int(d.get("idle_rounds") or 0),
                    compressed=[str(x) for x in (d.get("compressed") or [])],
                    round=int(d.get("round") or 0),
                    silent_reason_rounds=int(d.get("silent_reason_rounds") or 0),
                    tokens_in=int(d.get("tokens_in") or 0),
                    tokens_out=int(d.get("tokens_out") or 0),
                    subsyntheses_spawned=int(d.get("subsyntheses_spawned") or 0))


# ── persistence ───────────────────────────────────────────────────────────────────
#
# Written to its own file, keyed by whatever the caller considers the run's stable identity
# across a restart, with an atomic tmp-then-replace write.

#: Where synthesizer states live, one JSON file each.
STATES = Path.home() / ".lara" / "synthesizer"


def save(state_id: str, state: SynthesizerState, *, root: Path | None = None) -> Path:
    """Write the graph down. Never raises: losing the record must not stop the round.

    Written after every change — every dispatch, every goal's completion, every
    compression — rather than once at the end, because the interesting moment to survive
    is the middle: a crash here must lose at most the round in flight.
    """
    where = root or STATES
    path = where / f"{state_id}.json"
    try:
        where.mkdir(parents=True, exist_ok=True)
        tmp = where / f"{state_id}.writing"
        tmp.write_text(json.dumps(state.to_dict(), indent=1))
        tmp.replace(path)          # atomic: a reader never sees half a file
    except Exception as exc:                                   # noqa: BLE001
        logging.getLogger(__name__).warning(
            "synthesizer: could not persist %s to %s: %s", state_id, path, exc)
    return path


def load(state_id: str, *, root: Path | None = None) -> SynthesizerState | None:
    """The graph as it was last written, or None — a fresh `SynthesizerState` starts.

    A missing file is the ordinary "nothing written yet" case and logs nothing. A file
    that exists but fails to parse is different -- it would otherwise silently discard a
    stage's whole prior graph with no trace -- so that case logs, matching `save`'s own
    warning above.
    """
    path = (root or STATES) / f"{state_id}.json"
    try:
        d = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except Exception as exc:                                   # noqa: BLE001
        logging.getLogger(__name__).warning(
            "synthesizer: could not load %s from %s: %s", state_id, path, exc)
        return None
    return SynthesizerState.from_dict(d)


# ── the digest a reasoning pass reads ────────────────────────────────────────────


def _render_goal(goal: LogicalGoal) -> str:
    """One goal, as both `_digest` (every goal, flat) and `_cluster_digest` (one
    cluster's worth) render it — the single copy of this block, so the two can never
    drift into describing the same goal two different ways."""
    head = f"### {goal.id} (depth {goal.depth}"
    if goal.refines:
        head += f", refines {goal.refines}"
    if goal.nested:
        # Provenance, not decoration: an answer built from a whole nested synthesis
        # reads differently than one literature search's worth, and a later reasoning
        # round or the deliverable writer should be able to tell which it is looking
        # at — the same instinct `deliverable.py`'s `SHIPPED WITHOUT A MEASUREMENT`
        # stamp exists for, applied to how an answer was produced rather than what it
        # claims.
        head += ", via nested synthesis"
    head += f", {goal.status})"
    block = [head, goal.text]
    if goal.status == DONE:
        block.append(f"**Answer:** {goal.summary or '(nothing)'}")
    else:
        block.append(f"**Failed:** {goal.error or 'unknown error'}")
    return "\n".join(block)


def _digest(state: SynthesizerState) -> str:
    """`objective` + prior compressions + every `done`/`failed` goal, in dependency order.

    Everything a round needs to reason from, in one block. A `failed` goal is included
    exactly like a `done` one — a visible failure, never silently dropped — so a later round
    can retry it, route around it, or ignore it.
    """
    parts = [f"Objective: {state.objective}"]
    for i, summary in enumerate(state.compressed, 1):
        parts.append(f"### earlier compression {i}\n{summary}")
    for goal in state.goals.values():
        if goal.status not in (DONE, FAILED):
            continue
        parts.append(_render_goal(goal))
    return "\n\n".join(parts)


def _cluster_goals(goals: dict[str, LogicalGoal]) -> list[list[LogicalGoal]]:
    """Group `done`/`failed` goals by breadth root, for sectioned writing.

    A `spawn_goal` (`refines is None`) opens a theme; every `refine_goal` under it
    deepens that same theme. That is already the right table of contents for a
    deliverable — no separate clustering model is needed, just a walk up `refines`.

    A goal whose `refines` chain points at an id no longer in `goals` (its parent
    landed, was folded into a mid-loop compression, and this child kept going) becomes
    the root of its own cluster rather than raising or being dropped — an orphaned
    branch still deserves its own section.

    Order is preserved twice over: clusters appear in the order their root was first
    seen, and goals within a cluster keep dependency order — both read straight off
    `goals`'s own insertion order, the same guarantee `_digest` already relies on.
    """
    live = {gid: g for gid, g in goals.items() if g.status in (DONE, FAILED)}
    root_of: dict[str, str] = {}

    def find_root(gid: str) -> str:
        if gid in root_of:
            return root_of[gid]
        g = live.get(gid)
        if g is None or g.refines is None or g.refines not in live:
            root_of[gid] = gid
            return gid
        root = find_root(g.refines)
        root_of[gid] = root
        return root

    clusters: dict[str, list[LogicalGoal]] = {}
    for gid, g in live.items():
        clusters.setdefault(find_root(gid), []).append(g)
    return list(clusters.values())


def _cluster_digest(objective: str, cluster: list[LogicalGoal]) -> str:
    """One cluster's own slice of `_digest` — the objective plus only its goals."""
    parts = [f"Objective: {objective}"]
    parts.extend(_render_goal(g) for g in cluster)
    return "\n\n".join(parts)


# ── the tools a reasoning pass is offered ────────────────────────────────────────


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": {"type": "object", "properties": properties,
                                        "required": required}}}


def synth_tools(*, allow_subsynthesis: bool = False) -> list[dict]:
    """Provider tool schemas offered to one round's reasoning pass.

    `allow_subsynthesis` omits `spawn_subsynthesis` entirely when False — the whole of
    how a nested synthesis (see `run`'s `allow_subsynthesis` parameter) is kept from
    spawning a second level of nested syntheses: it is simply never offered the tool.
    """
    tools = [
        _schema(
            SPAWN_TOOL,
            "A new, breadth-first logical goal — a genuinely different angle on the "
            "objective, not a restatement of one already answered below. Call this; "
            "describing the question in prose changes nothing, only the call does.",
            {"text": {"type": "string",
                      "description": "The sub-question to research, phrased so it "
                                     "stands alone."},
             "depends_on": {"type": "array", "items": {"type": "string"},
                            "description": "Ids of already-answered goals below whose "
                                           "answers this one should read first."}},
            ["text"]),
        _schema(
            REFINE_TOOL,
            "Deepen one specific landed answer, by its id from below. Call only when "
            "going further on that particular goal would plausibly move the objective "
            "forward — not as a default next step after every answer.",
            {"parent_id": {"type": "string",
                           "description": "Id of the goal to deepen, from below."},
             "text": {"type": "string", "description": "The deeper sub-question."}},
            ["parent_id", "text"]),
        _schema(FINISH_TOOL,
               "Nothing further would help this round — the graph as it stands already "
               "covers the objective as far as it usefully can right now.", {}, []),
        _schema(
            RETRIEVE_FACTS_TOOL,
            "Search past research for previously-distilled facts relevant right now, to "
            "reuse settled information instead of re-deriving it. Worth calling before "
            "spawning a goal that may already be answered by earlier research.",
            {"query": {"type": "string",
                      "description": "What information would help right now, as a short "
                                     "description of the topic — not a question."}},
            ["query"]),
    ]
    if allow_subsynthesis:
        tools.append(_schema(
            SPAWN_SUBSYNTHESIS_TOOL,
            "Answer a sub-question by running a whole nested research graph over it, "
            "instead of one literature search — reserve this for a sub-question that is "
            "itself genuinely composed of several distinct angles a single search would "
            "only skim (it compares several methods across several axes, or names a "
            "broad area rather than one question). A sub-question that is merely hard, "
            "or merely broad-sounding, does not qualify on its own — most sub-questions "
            "belong under spawn_goal/refine_goal, and most rounds should call this zero "
            "times. Costs several times what an ordinary goal costs: capped, slower to "
            "answer, and its own graph runs to a smaller budget than this one's.",
            {"text": {"type": "string",
                      "description": "The sub-objective the nested synthesis should "
                                     "pursue, phrased so it stands alone."},
             "depends_on": {"type": "array", "items": {"type": "string"},
                            "description": "Ids of already-answered goals below whose "
                                           "answers this one should read first."}},
            ["text"]))
    return tools


SYNTH_SYSTEM = """You are growing a dependency graph of research sub-questions toward one \
objective, one round at a time. You reply with plain text, or with exactly one tool call.

Below is the objective, any earlier compressions, and every sub-question answered or \
failed so far. Given all of it, decide:

- spawn_goal — a genuinely new angle the graph does not cover yet.
- refine_goal — go deeper on one specific landed answer, when doing so would plausibly \
move the objective forward.
- finish — nothing further would help right now.

Never propose a goal that duplicates or trivially restates one already answered below — \
if the graph already has the answer, say so by calling finish rather than asking again in \
different words.

Before phrasing the `text` for a spawn_goal or refine_goal call, decide what would actually \
settle the question: a paper, or a measurement. The only leaf here is literature search \
against a corpus of papers — it can tell you what has been published, not run anything — so \
a sub-question that is fundamentally empirical (what one setup's own hardware, its own \
untested hyperparameter combination, or its own model size actually does, when nobody has \
published a run of that specific setup) cannot be settled by asking it to dig harder. Posed \
as an ordinary question, that leaf will still answer — with hedged literature-search prose \
standing in for a number that was never measured, indistinguishable from a real literature \
answer unless you already know to distrust it. When the sub-question is like this, say so in \
its own `text`: ask explicitly for what published work can bound or proxy, and ask it to \
state plainly that the question is only actually settled by a direct measurement in the asker's \
own setup, not by more reading. That does not mean avoiding empirical \
sub-questions — "no paper measures this; here is what published work does and does not \
bound" is a genuinely useful answer for this graph to land. It means asking them honestly \
instead of as if literature alone could answer them."""

#: Appended to `SYNTH_SYSTEM` only when `spawn_subsynthesis` is actually being offered
#: this round (`allow_subsynthesis=True`, top-level runs only — see `run`). Kept out of
#: the base prompt, the same shape `BOUNDED_CONTEXT_NUDGE` uses just below, so a nested
#: run's own system prompt never describes a tool it was not given.
SUBSYNTHESIS_NUDGE = """

A further tool, spawn_subsynthesis, answers a sub-question by running a whole nested \
research graph over it rather than one literature search — reserve it for a \
sub-question that is itself genuinely multi-part, not merely hard or broad-sounding; \
its own schema states the rest. Most rounds should never need it."""

#: Appended to `SYNTH_SYSTEM` only for a reasoning round running under `deliverable_tokens`
#: (bounded mode) — see `run`. Unbounded rounds are unaffected: their own objective text
#: already says "go deep rather than broad" and this must not contradict that.
BOUNDED_CONTEXT_NUDGE = """

This run is operating under a constrained context budget. Favor fewer, narrower, more \
targeted sub-questions over broad multi-part ones, so the graph finishes usefully within \
the smaller window instead of exhausting it on partial goals."""

#: Bounded retries for the compression call, a transport glitch or a dead engine (`EngineDeadError`)
#: is retried before giving up, so one bad call does not throw away everything the round
#: established. The truncation fallback in `_compress` is what used to fire on the very
#: first failure — this is the "reasonable bounded retries" the module docstring's "never
#: dropped" promise assumes exists.
COMPRESS_RETRIES = 2

#: `reply_room`'s fallback when it has no window to size against (`max_model_len` of 0 or
#: less) -- unreachable from `run()`, which always passes a real one, but kept generous
#: rather than small per that function's own contract: a small fallback here silently
#: reintroduces the bug this budget replaces.
DEFAULT_COMPRESS_TOKENS = 16_000

#: Ceiling on a compaction reply even when the window has room to spare. `_compress` now
#: only shrinks working memory mid-loop (`run`'s own digest, so the *next* reasoning
#: round still fits); the deliverable a person reads is written by `_write_deliverable`
#: below, section by section, and each section gets this same cap via `_compose` — see
#: its own docstring for why one shared ceiling is still the right number for both jobs.
MAX_COMPRESS_TOKENS = 32_768

COMPRESS_SYSTEM = """You are compacting a research graph's working memory so the next \
round still fits in its window — this is not the deliverable a person reads, it is what \
the *next reasoning pass* is given in place of everything below, so it must still carry \
every finding forward, not merely gesture at them.

You are given the objective and every sub-question answered or failed below. Write the \
most thorough account the material actually supports, organized by theme or \
sub-question. For each one, give the specific finding, the evidence and reasoning behind \
it, and the numbers, method names, and caveats that were reported — not a one-line \
paraphrase of them. Naming what disagrees, what remains unresolved, and what did not \
work (and why) matters as much as naming what succeeded.

Use the room you are given. Write a shorter account only because the material itself ran \
out, never because a summary is easier to produce than a full one — collapsing several \
detailed findings into one abstract sentence apiece is exactly the failure mode this \
exists to avoid.

Rules:
- Keep every citation bracket exactly as it appears below, e.g. [12345] or \
[12345, 67890]. Do not invent a citation that was not already there, and do not drop one \
that is still load-bearing for a claim you keep.
- If something failed, say so plainly rather than omitting it — a gap silently dropped is \
worse than a gap named.
- Use headings to organize by theme or sub-question; the material below almost always \
comes in enough distinct pieces to earn them."""


#: What writes one section of the actual deliverable — a genuinely different job from
#: `COMPRESS_SYSTEM`'s (a section only ever has to cover its own theme's few
#: sub-questions, not the whole graph, so it can afford to be this generous without
#: repeating `COMPRESS_SYSTEM`'s "use the room you are given" plea against a much bigger
#: ask). See `_write_deliverable`'s docstring for why the graph is split into sections at
#: all rather than compressed flat, once, the way this module used to.
SECTION_SYSTEM = """You are writing one section of a larger research report — the \
material below covers one theme's own sub-questions, not the whole objective.

Write the most thorough account this material supports: the specific finding from each \
sub-question, the evidence and reasoning behind it, and the numbers, method names and \
caveats that were reported — not a one-line paraphrase of them. State what disagrees, \
what remains unresolved, and what did not work as plainly as what succeeded.

A later pass places this section among the others and writes its own transitions around \
it — do not write your own heading, opening, or "in conclusion"; start on the finding \
and end when the material does.

Rules:
- Keep every citation bracket exactly as it appears below, e.g. [12345] or \
[12345, 67890]. Never invent one; never drop one still load-bearing for a claim you keep.
- If something failed, say so plainly rather than omitting it."""


def _section_system(n_goals: int) -> str:
    """`SECTION_SYSTEM`, anchored to how many sub-questions this one section covers.

    A soft "be thorough" instruction is not enough on its own — models reliably
    under-shoot it when handed a pile of source material and no number to write against.
    A section covers a handful of goals rather than a whole graph, so the anchor can
    afford to be concrete: real, per-finding depth, not a token count to hit.
    """
    return SECTION_SYSTEM + (
        f"\n\nThis section covers {n_goals} sub-question{'s' if n_goals != 1 else ''}. "
        "Give each one that reached a real finding its own paragraph of real depth — "
        "not one compressed sentence standing in for it.")


#: The tool a stitching pass calls — see `_stitch`. Its arguments are the *only* channel
#: that call's output can reach the caller through, which is what makes "never rewrites a
#: section" an enforced property rather than a hoped-for one: nothing here ever reads
#: free-form text back out of that call and treats it as a section's replacement.
STITCH_TOOL = "organize_report"

STITCH_SYSTEM = """You are organizing a finished report out of sections someone else \
already wrote. Each one below is complete and final — you may not rewrite, shorten, \
paraphrase, or add facts to any of them. Your only job is structure:

- Decide what order they read best in.
- Write a short transition (1-3 sentences) to place immediately before each one, \
connecting it to what came before — it orients the reader in the argument; it does not \
restate the section's own content, which follows immediately after it.
- Optionally, one short opening paragraph framing the whole report, and one short \
closing paragraph tying it together.

Call organize_report exactly once, naming every section id below in "order"."""


def _stitch_schema(section_ids: list[str]) -> dict:
    return _schema(
        STITCH_TOOL,
        "Order the finished sections and write short transitions between them. Every "
        "section's own text is inserted exactly as given — this call only decides the "
        "structure around it, never its content.",
        {"order": {"type": "array", "items": {"type": "string"},
                   "description": "Every section id below, in reading order: "
                                  + ", ".join(section_ids)},
         "intro": {"type": "string",
                   "description": "One short opening paragraph for the report as a "
                                  "whole, or empty."},
         "transitions": {"type": "object",
                         "description": "Section id -> a short (1-3 sentence) "
                                        "transition placed immediately before that "
                                        "section. Omit a section here for no "
                                        "transition.",
                         "additionalProperties": {"type": "string"}},
         "closing": {"type": "string",
                     "description": "One short closing paragraph, or empty."}},
        ["order"])


# ── ids ───────────────────────────────────────────────────────────────────────────


def _slug(text: str, fallback: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:40]
    return out.rstrip("-") or fallback


def _new_id(text: str, existing: dict) -> str:
    base = _slug(text, f"goal-{len(existing) + 1}")
    if base not in existing:
        return base
    n = 2
    while f"{base}-{n}" in existing:
        n += 1
    return f"{base}-{n}"


# ── dispatch ──────────────────────────────────────────────────────────────────────


def _in_flight_count(state: SynthesizerState) -> int:
    return sum(1 for g in state.goals.values() if g.status in (PENDING, RUNNING))


def _do_spawn(state: SynthesizerState, args: dict) -> tuple[bool, str]:
    if _in_flight_count(state) >= MAX_CONCURRENT_GOALS:
        return False, (f"already at the cap of {MAX_CONCURRENT_GOALS} in-flight goal(s) "
                       "(pending or running) — wait for one of those to finish before "
                       "spawning more; nothing was added")
    text = str((args or {}).get("text") or "").strip()
    if not text:
        return False, f"{SPAWN_TOOL} needs a non-empty text — nothing was added"
    deps = [str(d) for d in ((args or {}).get("depends_on") or []) if str(d).strip()]
    unknown = [d for d in deps if d not in state.goals]
    if unknown:
        return False, (f"depends_on names {', '.join(unknown)!r}, which "
                       f"{'is' if len(unknown) == 1 else 'are'} not a goal here — "
                       "nothing was added")
    gid = _new_id(text, state.goals)
    state.goals[gid] = LogicalGoal(id=gid, text=text, depends_on=deps, refines=None,
                                   depth=0, status=PENDING)
    return True, f"spawned {gid!r}: {text}"


def _do_spawn_subsynthesis(state: SynthesizerState, args: dict) -> tuple[bool, str]:
    """Same shape as `_do_spawn` — a breadth goal, `depends_on` and the in-flight cap
    checked identically — plus the `MAX_SUBSYNTHESES_PER_SYNTH` cap and `nested=True`,
    which is the only thing that later tells `_execute_round` to run this one through a
    whole nested `run()` instead of `aresearch`.
    """
    if state.subsyntheses_spawned >= MAX_SUBSYNTHESES_PER_SYNTH:
        return False, (f"already at the cap of {MAX_SUBSYNTHESES_PER_SYNTH} "
                       f"sub-synthes{'is' if MAX_SUBSYNTHESES_PER_SYNTH == 1 else 'es'} "
                       "for this run — use spawn_goal for anything further; nothing was "
                       "added")
    if _in_flight_count(state) >= MAX_CONCURRENT_GOALS:
        return False, (f"already at the cap of {MAX_CONCURRENT_GOALS} in-flight goal(s) "
                       "(pending or running) — wait for one of those to finish before "
                       "spawning more; nothing was added")
    text = str((args or {}).get("text") or "").strip()
    if not text:
        return False, f"{SPAWN_SUBSYNTHESIS_TOOL} needs a non-empty text — nothing was added"
    deps = [str(d) for d in ((args or {}).get("depends_on") or []) if str(d).strip()]
    unknown = [d for d in deps if d not in state.goals]
    if unknown:
        return False, (f"depends_on names {', '.join(unknown)!r}, which "
                       f"{'is' if len(unknown) == 1 else 'are'} not a goal here — "
                       "nothing was added")
    gid = _new_id(text, state.goals)
    state.goals[gid] = LogicalGoal(id=gid, text=text, depends_on=deps, refines=None,
                                   depth=0, status=PENDING, nested=True)
    state.subsyntheses_spawned += 1
    return True, f"spawned sub-synthesis {gid!r}: {text}"


def _do_refine(state: SynthesizerState, args: dict) -> tuple[bool, str]:
    if _in_flight_count(state) >= MAX_CONCURRENT_GOALS:
        return False, (f"already at the cap of {MAX_CONCURRENT_GOALS} in-flight goal(s) "
                       "(pending or running) — refining doesn't get around the cap either, "
                       "wait for one of those to finish before adding more; nothing was "
                       "added")
    parent_id = str((args or {}).get("parent_id") or "").strip()
    parent = state.goals.get(parent_id)
    if parent is None:
        return False, f"no goal {parent_id!r} here — nothing was added"
    # A refusal with a reason, rather a hard crash or a silently-ignored call.
    if parent.depth >= MAX_REFINEMENT_DEPTH:
        return False, (f"{parent_id!r} sits at refinement depth {parent.depth}, at the "
                       f"limit of {MAX_REFINEMENT_DEPTH} — refine something shallower, "
                       "or spawn a new breadth goal instead; nothing was added")
    text = str((args or {}).get("text") or "").strip()
    if not text:
        return False, f"{REFINE_TOOL} needs a non-empty text — nothing was added"
    gid = _new_id(text, state.goals)
    state.goals[gid] = LogicalGoal(id=gid, text=text, depends_on=[parent_id],
                                   refines=parent_id, depth=parent.depth + 1,
                                   status=PENDING)
    return True, f"refined {parent_id!r} as {gid!r}: {text}"


def _do_retrieve_facts(embed, args: dict) -> str:
    """Embeds the model's query and returns the nearest stored facts as plain text — see
    `facts.search_facts`/`facts.format_facts`. Never raises: an embedder that is missing
    or fails is a fact the model should see (retrieval unavailable), not a crash."""
    from lara.serve import facts as FA

    query = str((args or {}).get("query") or "").strip()
    if not query:
        return "no query given — nothing retrieved"
    if embed is None:
        return "fact retrieval is unavailable in this run"
    try:
        embedding = embed(query)
    except Exception:                                          # noqa: BLE001
        embedding = []
    if not embedding:
        return "fact retrieval is unavailable in this run"
    hits = FA.search_facts(embedding)
    return FA.format_facts(hits) or "no matching past facts found"


async def _reason_round(state: SynthesizerState, *, base_url: str, model: str,
                        max_model_len: int, api_key: str = "",
                        max_turns: int = MAX_TURNS,
                        deliverable_tokens: int | None = None,
                        allow_subsynthesis: bool = False, embed=None) -> bool:
    """One reasoning pass. Mutates `state` in place. Returns whether real progress was
    made — a goal was actually created, not merely a tool called and refused.

    Never raises: a reasoning pass that errored is a round with
    nothing proposed, which is exactly what an idle round already means, not a reason to
    take the stage down with it.

    Same fix `_compress` got for the same reason: this call's own prompt (`_digest`) can
    be just as large, and used to run with thinking on and the flat 8,000-token default,
    so a round could burn its whole budget on hidden reasoning and come back with no tool
    call and no text — indistinguishable from a genuinely idle round, and logged nowhere.
    `enable_thinking=False` and a real `max_tokens` (via `reply_room`) fix the starvation;
    capturing the `Reply` and logging that exact empty/no-error shape (and bumping
    `state.silent_reason_rounds`) is what makes a recurrence visible instead of silent.

    `deliverable_tokens`, when not None, appends `BOUNDED_CONTEXT_NUDGE` to the system
    prompt for this call only — see `run`. `allow_subsynthesis` — default False, so every
    existing caller that does not pass it keeps today's exact behavior — offers
    `spawn_subsynthesis` and its `SUBSYNTHESIS_NUDGE`; `run` is the only caller that ever
    passes True, and only for a top-level run (see its own `allow_subsynthesis`).

    `embed`, when given, is a plain `str -> list[float]` callable (see `facts.embedder_fn`)
    backing `retrieve_facts` — this module's own dependency-injection shape, so it still
    needs no import of `lara`. `None` (the default) makes the tool report itself
    unavailable rather than erroring.
    """
    from lara.serve import converse as CV

    progressed = False

    def dispatch(name: str, args: dict) -> str:
        nonlocal progressed
        if name == SPAWN_TOOL:
            ok, msg = _do_spawn(state, args)
            progressed = progressed or ok
            return msg
        if name == REFINE_TOOL:
            ok, msg = _do_refine(state, args)
            progressed = progressed or ok
            return msg
        if name == FINISH_TOOL:
            return "noted — nothing further this round"
        if name == SPAWN_SUBSYNTHESIS_TOOL and allow_subsynthesis:
            ok, msg = _do_spawn_subsynthesis(state, args)
            progressed = progressed or ok
            return msg
        if name == RETRIEVE_FACTS_TOOL:
            # Not counted toward `progressed`: a retrieval is not itself a graph goal.
            return _do_retrieve_facts(embed, args)
        return f"{name} failed: unknown tool"

    system = (SYNTH_SYSTEM
             + (SUBSYNTHESIS_NUDGE if allow_subsynthesis else "")
             + (BOUNDED_CONTEXT_NUDGE if deliverable_tokens is not None else ""))
    digest = _digest(state)
    room = CX.reply_room(max_model_len, system, digest, stage="synthesizer.reason",
                         default=DEFAULT_REASON_TOKENS, cap=MAX_REASON_TOKENS)
    try:
        reply = await CV.talk(base_url, model, CV.opening(system, digest),
                              tools=synth_tools(allow_subsynthesis=allow_subsynthesis),
                              dispatch=dispatch, max_turns=max_turns,
                              api_key=api_key, tool_choice="auto", max_tokens=room,
                              enable_thinking=False)
        state.tokens_in += int(getattr(reply, "tokens_in", 0) or 0)
        state.tokens_out += int(getattr(reply, "tokens_out", 0) or 0)
        error = getattr(reply, "error", "") or ""
        text = (getattr(reply, "text", "") or "").strip()
        tool_calls = getattr(reply, "tool_calls", 0)
        if not error and not text and not tool_calls:
            state.silent_reason_rounds += 1
            logging.getLogger(__name__).warning(
                "synthesizer: reason round %d returned nothing -- no tool call, no "
                "text, no error (stopped_because=%r, digest=%d chars, max_tokens=%d)",
                state.round, getattr(reply, "stopped_because", ""), len(digest), room)
        elif error:
            logging.getLogger(__name__).warning(
                "synthesizer: reason round %d failed: %s", state.round, error)
    except Exception:                                          # noqa: BLE001
        pass
    state.idle_rounds = 0 if progressed else state.idle_rounds + 1
    return progressed


# ── executing a round's pending goals concurrently ───────────────────────────────

#: The sentinel `lara.serve.synthesis.run_synthesis` appends to its own `stopped_because`
#: when the one call it does not already swallow — final consolidation — raises (see that
#: function's `except Exception as exc` around the `consolidate()` call). That path never
#: re-raises: it keeps whatever evidence the run gathered and bakes the failure into
#: `thorough`/`tldr` as prose instead ("N claims ... but writing the answer failed: exc"),
#: so `aresearch` returns normally and `_run_one`'s own `except` below never fires. Checking
#: this marker — deterministic, not the exception's arbitrary free text — is how `_run_one`
#: tells that case apart from a real answer.
_CONSOLIDATION_FAILED_MARKER = "consolidation failed"


async def _execute_round(state: SynthesizerState, *, aresearch, model: str,
                         base_url: str, api_key: str = "", on_change=None,
                         max_model_len: int = 0, conn=None,
                         deliverable_tokens: int | None = None) -> None:
    """Run every currently-`pending` goal through `aresearch`, concurrently — or, for a
    goal `spawn_subsynthesis` created (`goal.nested`), through a whole nested `run()`.

    Goals proposed in the same round cannot depend on each other — `_do_spawn`/
    `_do_refine` only accept `depends_on`/`parent_id` naming a goal already `done` or
    `failed` — so there is no ordering to respect within one batch; `asyncio.gather`
    covers it directly. A nested
    goal is no exception: it runs concurrently with everything else in the batch, its
    own inner rounds notwithstanding.

    Each goal's own failure is caught inside `_run_one` and recorded on that goal alone
    `on_change()` — no arguments, the caller's closure already holds `state` — is called
    the moment *that* goal finishes, not once at the end of the batch: a mid-batch crash
    must only lose the goals still in flight.

    `max_model_len`/`conn`/`deliverable_tokens` only matter for a nested goal — a plain
    one never reaches the branch that reads them, so an existing caller with no nested
    goals in play (every test that predates `spawn_subsynthesis`) is unaffected by their
    defaults.
    """
    pending = [g for g in state.goals.values() if g.status == PENDING]
    if not pending:
        return
    for g in pending:
        g.status = RUNNING
    if on_change is not None:
        on_change()

    async def _run_one(goal: LogicalGoal) -> None:
        try:
            if goal.nested:
                # The recursive case: a whole second `run()`, over the sub-objective
                # this goal's own text names, at a tighter round budget than the parent
                # (`SUBSYNTHESIS_MAX_IDLE_ROUNDS`) and never offered `spawn_subsynthesis`
                # itself (`allow_subsynthesis=False`) — the entire depth-1 limit lives
                # in that one argument. Its deliverable and references land on this
                # goal exactly as an ordinary goal's `aresearch` answer would, so
                # nothing downstream needs to know this one took a different path.
                nested_state = SynthesizerState(objective=goal.text)
                nested_result = await run(
                    nested_state, base_url=base_url, model=model, api_key=api_key,
                    max_model_len=max_model_len, aresearch=aresearch, conn=conn,
                    max_idle_rounds=SUBSYNTHESIS_MAX_IDLE_ROUNDS,
                    deliverable_tokens=deliverable_tokens, allow_subsynthesis=False)
                if nested_result.total_done == 0:
                    goal.status = FAILED
                    goal.error = (
                        "nested synthesis established nothing "
                        f"({nested_result.total_failed} sub-question(s) failed)")[:400]
                else:
                    goal.status = DONE
                    goal.summary = nested_result.deliverable
                    goal.citations = dict(nested_result.references)
                return
            research = await aresearch(goal.text, model=model, base_url=base_url,
                                       api_key=api_key)
            thorough = getattr(research, "thorough", None)
            text = str(getattr(thorough, "text", "") or "").strip()
            refs = dict(getattr(thorough, "references", {}) or {})
            if not text:
                # A thorough answer that came back empty still leaves the tldr, the same
                # fallback `context.fitted` already prefers a complete short answer over
                # nothing from a long one that did not materialise.
                tldr = getattr(research, "tldr", None)
                text = str(getattr(tldr, "text", "") or "").strip()
                refs = dict(getattr(tldr, "references", {}) or refs)
            stopped_because = str(getattr(research, "stopped_because", "") or "")
            if _CONSOLIDATION_FAILED_MARKER in stopped_because:
                # `run_synthesis` caught its own consolidation failure and returned a
                # normal-looking result with the failure narrated in `text` — a real
                # answer never landed. Land this goal `failed`, not `done`.
                goal.status = FAILED
                goal.error = (text or stopped_because)[:400]
            else:
                goal.status = DONE
                goal.summary = text
                goal.citations = {k: (v.to_dict() if hasattr(v, "to_dict") else v)
                                  for k, v in refs.items()}
        except Exception as exc:                                # noqa: BLE001
            goal.status = FAILED
            goal.error = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            if on_change is not None:
                on_change()

    await asyncio.gather(*(_run_one(g) for g in pending), return_exceptions=True)


# ── compression ───────────────────────────────────────────────────────────────────

#: Characters a genuinely finished piece of prose can end on: sentence punctuation, a
#: closing quote/bracket/paren, or a closing code fence. Chosen against a real case whose
#: compression call stopped with
#: `finish_reason="stop"` after only 831 of its 2,000-token budget, text ending mid-word
#: ("...which purportedly decom") -- not cut off by the cap `stopped_because == "length"`
#: watches for, and not empty, so nothing existing caught it.
_COMPLETE_ENDINGS = ".!?)]}'\"`"


def _looks_complete(text: str) -> bool:
    """Whether `text` reads as finished rather than cut off mid-word by a premature stop.

    `stopped_because == "length"` catches a reply cut off by hitting its token cap. It
    cannot catch a reply the model itself chose to end (`finish_reason == "stop"`) too
    early -- a real, observed failure mode distinct from both that and an empty reply,
    where the API reports a perfectly ordinary voluntary stop and the only remaining
    signal is the text itself. `COMPRESS_SYSTEM` asks for prose, and prose that actually
    finished ends on sentence punctuation or a closing quote/bracket/fence; one a
    premature stop cut off does not.
    """
    t = text.rstrip()
    return bool(t) and t[-1] in _COMPLETE_ENDINGS


#: The two `truncated_because` messages `_compress` produces both start with one of
#: these. `verdict_for` checks against them so it can tell "kept a real, if incomplete,
#: summary" apart from "every attempt failed outright and the raw digest was used
#: instead" -- two very different claims that used to be reported with the same
#: hardcoded "fell back to an uncompressed digest" wording regardless of which had
#: actually happened.
_TRUNCATION_PREFIXES = ("compression hit its token cap", "compression stopped (\"stop\")")


def _is_truncation_reason(because: str) -> bool:
    """Whether `because` names one of `_compress`'s two truncation shapes, not a hard
    failure (exception or HTTP error) that fell back to the raw digest instead."""
    return because.startswith(_TRUNCATION_PREFIXES)


async def _compose(source: str, *, system: str, base_url: str, model: str,
                   max_model_len: int, api_key: str = "", cap: int, stage: str,
                   retries: int = COMPRESS_RETRIES
                   ) -> tuple[str, bool, str, int, int]:
    """One retried, truncation-checked write from `source` — the core `_compress` (mid-
    loop working-memory compaction) and `_write_deliverable`'s per-section calls (the
    actual deliverable) both run, sharing every failure mode both have to handle: a dead
    replica, a reply that silently reasoned its whole budget away, a reply cut off by
    `room`, a reply the model itself stopped too early. See `_compress`'s docstring —
    unchanged below, just no longer specific to "the final deliverable" — for why each of
    those needed fixing and what evidence showed it.

    Returns `(text, degraded, degraded_because, tokens_in, tokens_out)`. Never raises.
    Never returns an empty `text` — a caller that gets nothing back after every retry
    still gets `source` itself, clipped and labeled, rather than silence.
    """
    from lara.serve import converse as CV

    room = CX.reply_room(max_model_len, system, source, stage=stage,
                         default=DEFAULT_COMPRESS_TOKENS, cap=cap)
    text = ""
    degraded_because = ""
    truncated_because = ""
    tokens_in = tokens_out = 0
    for _attempt in range(1 + max(0, retries)):
        truncated_because = ""
        try:
            reply = await CV.talk(base_url, model, CV.opening(system, source),
                                  max_turns=1, max_tokens=room, api_key=api_key,
                                  enable_thinking=False)
            tokens_in += int(getattr(reply, "tokens_in", 0) or 0)
            tokens_out += int(getattr(reply, "tokens_out", 0) or 0)
            text = (reply.text or "").strip()
            degraded_because = getattr(reply, "error", "") or ""
            if text and getattr(reply, "stopped_because", "") == "length":
                # Non-empty, but cut off by `room` rather than the model choosing to
                # stop — not a complete answer. Keep the text (better than nothing);
                # the degraded signal below is what tells a caller it is incomplete.
                truncated_because = (f"{_TRUNCATION_PREFIXES[0]} ({room} tokens) "
                                     "and was cut off before it finished")
            elif (text and getattr(reply, "stopped_because", "") == "stop"
                  and not _looks_complete(text)):
                # Not cut off by the cap -- the API reported an ordinary voluntary
                # stop -- but the text itself does not read as finished. Same
                # keep-the-text, flag-it-anyway handling as the length case above; see
                # `_looks_complete`. Scoped to the literal `"stop"` value (not "anything
                # that isn't length") because that is the only other value `talk()`
                # actually produces here — confirmed against this deployment's own
                # vLLM `/metrics`, which has never once reported `"abort"` or `"error"`.
                sent = int(getattr(reply, "tokens_out", 0) or 0)
                truncated_because = (
                    f"{_TRUNCATION_PREFIXES[1]} after {sent} of {room} allotted "
                    "tokens, but the text does not end on a finished sentence — a premature "
                    "stop, not a genuine one")
        except Exception as exc:                                   # noqa: BLE001
            text = ""
            degraded_because = f"{type(exc).__name__}: {exc}"
        if text and not truncated_because:
            # Genuinely complete -- nothing left to retry for.
            break
        # Empty or truncated: retry rather than accepting it, up to `retries` --
        # previously any non-empty `text` broke out of this loop immediately, so a
        # truncated-but-non-empty reply (the two cases just above) never actually
        # consumed a retry despite the retry budget existing for exactly this.
    if text:
        degraded_because = truncated_because
    degraded = not text or bool(truncated_because)
    if not text:
        # Never let a broken call erase what was established: fall back to the
        # uncompressed source itself rather than losing the material to silence — but
        # only once every retry above has actually failed.
        text = CX.clipped(source, 8_000, what="the uncompressed digest")
        # Last attempt's reason, not every attempt's: with this few retries the last
        # failure is the one that actually gave up.
        logging.getLogger(__name__).warning(
            "synthesizer: %s fell back to an uncompressed digest after %d attempt(s): %s",
            stage, 1 + max(0, retries), degraded_because or "no reason reported")
    elif truncated_because:
        logging.getLogger(__name__).warning("synthesizer: %s", truncated_because)

    return text, degraded, degraded_because, tokens_in, tokens_out


async def _compress(state: SynthesizerState, *, base_url: str, model: str,
                    max_model_len: int, api_key: str = "", conn=None,
                    deliverable_tokens: int | None = None
                    ) -> tuple[str, dict[str, C.Reference], bool, str]:
    """One dense compaction of the graph's working memory — `run()`'s mid-loop safety
    valve, so the *next* reasoning round's own digest still fits its window. **Not** the
    deliverable a person reads; see `_write_deliverable` for that.

    `known` is built from the citations the still-live goals carry; `conn`, when given,
    lets `citations.bind` hydrate a paper citation straight from the corpus by chunk id
    even when the goal that first surfaced it has since been compressed away — see the
    module docstring for why this is how a citation survives more than one compression
    round rather than a fifth `SynthesizerState` field.

    The actual write — retried, checked for truncation both by the cap and by the model
    stopping on its own too early, falling back to a labeled raw digest only once every
    retry has failed — is `_compose`; see its docstring, and the evidence that shaped it
    (`EngineDeadError` with no retry, 26/26 real compressions logging "no reason
    reported", a run shipping `success` over prose cut off mid-word), none of which changed by moving here.

    Returns `(summary, references, degraded, degraded_because)`. `deliverable_tokens`,
    when given, replaces `MAX_COMPRESS_TOKENS` as the `cap` passed to `_compose`. `None`
    (the default) is today's exact behavior.
    """
    known: dict[str, C.Reference] = {}
    for g in state.goals.values():
        if g.status not in (DONE, FAILED):
            continue
        for k, d in (g.citations or {}).items():
            ref = C.Reference.from_dict(d)
            if ref is not None:
                known[k] = ref

    source = _digest(state)
    summary, degraded, degraded_because, tokens_in, tokens_out = await _compose(
        source, system=COMPRESS_SYSTEM, base_url=base_url, model=model,
        max_model_len=max_model_len, api_key=api_key,
        cap=(deliverable_tokens or MAX_COMPRESS_TOKENS), stage="synthesizer.compress")
    state.tokens_in += tokens_in
    state.tokens_out += tokens_out

    cited = C.bind(summary, known=known, conn=conn)
    return summary, cited.references, degraded, degraded_because


async def _stitch(sections: list[tuple[str, str, str]], objective: str, *,
                  base_url: str, model: str, max_model_len: int, api_key: str = ""
                  ) -> tuple[str, int, int]:
    """Order `sections` and write transitions between them, without ever touching a
    section's own text.

    `sections` is `(id, title, text)`, already in a sensible fallback order — the order
    each cluster's root goal was created in, the same dependency order `_digest` already
    relies on. The model's only output channel is `organize_report`'s own arguments
    (`order`, `intro`, `transitions`, `closing`) — never free-form prose read back as a
    section's replacement — so assembly happens here, in Python, from the untouched
    strings the caller already holds. A section's own text physically cannot be rewritten
    by this call, whatever the model tries to do.

    Degrades to the given order with no transitions on any failure: an unreachable
    replica, a malformed call, or an `order` that drops or invents ids. The `order`
    reconstruction below (every known id present, nothing else) already produces that
    exact fallback when `captured` comes back empty, so there is no separate failure
    branch to keep in sync with it — a caller always gets every section back, organized
    or not.
    """
    from lara.serve import converse as CV

    ids = [sid for sid, _title, _text in sections]
    by_id = {sid: (title, text) for sid, title, text in sections}
    listing = "\n\n".join(f"### {sid}\n{title}\n\n{text}" for sid, title, text in sections)
    prompt = f"Objective: {objective}\n\n{listing}"
    room = CX.reply_room(max_model_len, STITCH_SYSTEM, prompt, stage="synthesizer.stitch",
                         default=1_500, cap=4_000)

    captured: dict = {}

    def dispatch(name: str, args: dict) -> str:
        if name == STITCH_TOOL:
            captured.update(args or {})
        return "recorded"

    tokens_in = tokens_out = 0
    try:
        reply = await CV.talk(
            base_url, model, CV.opening(STITCH_SYSTEM, prompt),
            tools=[_stitch_schema(ids)], dispatch=dispatch, max_turns=1,
            max_tokens=room, api_key=api_key,
            tool_choice={"type": "function", "function": {"name": STITCH_TOOL}},
            enable_thinking=False)
        tokens_in = int(getattr(reply, "tokens_in", 0) or 0)
        tokens_out = int(getattr(reply, "tokens_out", 0) or 0)
    except Exception:                                          # noqa: BLE001
        pass

    order = [sid for sid in (captured.get("order") or []) if sid in by_id]
    order += [sid for sid in ids if sid not in order]      # a dropped id still ships
    transitions = captured.get("transitions")
    transitions = transitions if isinstance(transitions, dict) else {}

    parts = []
    intro = str(captured.get("intro") or "").strip()
    if intro:
        parts.append(intro)
    for sid in order:
        title, text = by_id[sid]
        trans = str(transitions.get(sid) or "").strip()
        if trans:
            parts.append(trans)
        parts.append(text)
    closing = str(captured.get("closing") or "").strip()
    if closing:
        parts.append(closing)
    return "\n\n".join(parts), tokens_in, tokens_out


#: Prepended to the deliverable when `_write_deliverable` had to fall back or accept a
#: truncated section anywhere in it — a reader-facing analog of `deliverable.py`'s
#: `SHIPPED WITHOUT A MEASUREMENT` stamp: the caveat belongs where the text is actually
#: read, not only in `verdict_because`, or a report thin because its budget was thin
#: reads identically to one thin because the literature was.
def _pressure_note(because: str) -> str:
    detail = f" ({because})" if because else ""
    return ("> **Written under budget pressure.** At least one section of this report "
            f"could not be written in full and fell back to a shorter or rougher pass"
            f"{detail}. Read brevity below as a budget artifact, not as a signal that "
            "little was found.")


async def _write_deliverable(state: SynthesizerState, *, base_url: str, model: str,
                             max_model_len: int, api_key: str = "", conn=None,
                             deliverable_tokens: int | None = None
                             ) -> tuple[str, dict[str, C.Reference], bool, str]:
    """Write the actual deliverable — the thing a person reads.

    **Why this is not one call over the whole graph.** That used to be `_compress`'s job
    too, and production data killed it: real graphs (86 saved states checked) commonly
    reach 30-52 goals and a 400,000+ character digest, and `synth_context_tokens` was
    smaller than the replica's real window on top of that, so the one call writing the
    "comprehensive report" over all of it was frequently left 2,000-6,000 tokens of room
    — 28.5% of 151 logged `synthesizer.compress` calls landed at or near the 2,000-token
    floor. One real run's own verdict recorded it plainly: "the final compression call
    ... hit its token cap (2000 tokens) and was cut off before it finished" over a
    121,000-token digest. No system prompt fixes a >50x compression ratio asked of one
    model call.

    **The fix uses structure the graph already has.** Every `spawn_goal` opens a theme
    and every `refine_goal` under it deepens the same one (`_cluster_goals`) — so each
    theme gets its own `_compose` call, writing only its own few goals, at its own full
    budget, not a budget shared with everything else in the graph. A 45-goal graph split
    into ~10 clusters turns "compress 125,000 tokens into 2,000" into "compress ~12,000
    tokens into 2,000-3,000, ten times" — a completely different, much gentler task.

    **Sections are then organized, not merged.** `_stitch` orders them and writes
    transitions between them without ever altering a section's own text (see its
    docstring) — the user asked for exactly this: concatenate intelligently, add
    transitions, decide structure, but never touch what a section itself already says.
    Skipped entirely when there is only one section; nothing to organize.

    Prior mid-loop compressions (`state.compressed`, populated only in unbounded mode —
    see `run`) are folded in as one further cluster rather than left untouched: unlike a
    freshly-written section, a mid-loop compression is already lossy working-memory
    prose, so one more `_compose` pass over it (this time actually meant to read well) is
    not a second loss the way re-writing a fresh section would be.

    Citations are resolved once, over the fully assembled text, from every citation any
    still-live goal carries — the same re-resolution-not-a-fifth-field approach the
    module docstring describes for `_compress`, applied here instead since this is the
    call whose output is the one a citation actually needs to survive in.

    Returns `(text, references, degraded, degraded_because)` — the same four-tuple shape
    `_compress` returns, so `run()`'s final-step call site barely changes. `degraded` is
    True if *any* section (including the folded-in prior compression) had to fall back or
    accept a truncated reply; `degraded_because` is that section's own reason.
    """
    known: dict[str, C.Reference] = {}
    for g in state.goals.values():
        if g.status not in (DONE, FAILED):
            continue
        for k, d in (g.citations or {}).items():
            ref = C.Reference.from_dict(d)
            if ref is not None:
                known[k] = ref

    cap = deliverable_tokens or MAX_COMPRESS_TOKENS
    degraded_any = False
    degraded_because = ""
    total_in = total_out = 0
    sections: list[tuple[str, str, str]] = []

    if state.compressed:
        prior_source = (f"Objective: {state.objective}\n\n"
                        + "\n\n".join(f"### earlier compression {i}\n{c}"
                                      for i, c in enumerate(state.compressed, 1)))
        text, degraded, why, tin, tout = await _compose(
            prior_source, system=_section_system(len(state.compressed)),
            base_url=base_url, model=model, max_model_len=max_model_len,
            api_key=api_key, cap=cap, stage="synthesizer.section")
        total_in += tin
        total_out += tout
        degraded_any = degraded_any or degraded
        degraded_because = why or degraded_because
        sections.append(("prior", "Established earlier in this run", text))

    for cluster in _cluster_goals(state.goals):
        root = cluster[0]
        source = _cluster_digest(state.objective, cluster)
        text, degraded, why, tin, tout = await _compose(
            source, system=_section_system(len(cluster)), base_url=base_url,
            model=model, max_model_len=max_model_len, api_key=api_key, cap=cap,
            stage="synthesizer.section")
        total_in += tin
        total_out += tout
        degraded_any = degraded_any or degraded
        degraded_because = why or degraded_because
        sections.append((f"section-{root.id}", root.text, text))

    state.tokens_in += total_in
    state.tokens_out += total_out

    if not sections:
        return "", {}, False, ""
    if len(sections) == 1:
        final = sections[0][2]
    else:
        final, stitch_in, stitch_out = await _stitch(
            sections, state.objective, base_url=base_url, model=model,
            max_model_len=max_model_len, api_key=api_key)
        state.tokens_in += stitch_in
        state.tokens_out += stitch_out

    cited = C.bind(final, known=known, conn=conn)
    return cited.text, cited.references, degraded_any, degraded_because


#: Ceiling for `answer_from_deliverable`'s reply. Well below `MAX_COMPRESS_TOKENS`: the
#: whole point of this pass is a targeted answer, not another full report, and a cap this
#: size is already generous for a paragraph or a short list — see `FINAL_ANSWER_SYSTEM`.
MAX_FINAL_ANSWER_TOKENS = 4_000

FINAL_ANSWER_SYSTEM = """You are answering one specific question about a finished \
research report — not writing another report, answering the question actually asked.

You are given the report below and a question or instruction about it. Answer only \
that, using only what the report actually says — do not introduce a claim the report \
does not make. Write as long as the answer genuinely needs and no longer; most answers \
here are a paragraph or a short list, not another full report.

Rules:
- Keep every citation bracket you carry over exactly as it appears in the report, e.g. \
[12345] or [12345, 67890]. Never invent one that is not already there.
- If the report does not actually settle what is asked, say so plainly rather than \
answering from outside it — that is itself the correct answer.
- No preamble, no restating the question, no heading of your own — a heading is added \
around this separately."""


async def answer_from_deliverable(deliverable: str, prompt: str, *, base_url: str,
                                  model: str, max_model_len: int, api_key: str = ""
                                  ) -> tuple[str, bool, str, int, int]:
    """One targeted answer to `prompt`, using the already-written `deliverable` as its
    only source — for when the full sectioned report (see `_write_deliverable`) is more
    than a reader wants and what they actually want is one question answered from it: a
    specific claim it makes, or a synthesis question like "what are the top 5 methods
    mentioned here that could be implemented".

    Not private (`answer_from_deliverable`, not `_answer_from_deliverable`): `run`'s own
    `final_compression_prompt` calls this at the end of a fresh run, and the
    post-hoc compress route calls it again, later, over a
    deliverable that already finished — the same operation, on the same kind of text,
    from two different callers, so it needed a real public name rather than an
    underscore-prefixed one reached into from outside the module.

    Reuses `_compose` — the same retried, truncation-checked write every other call in
    this module makes — rather than a bespoke call, so a dead replica or a starved reply
    here fails exactly as legibly as everywhere else. Returns the same five-tuple
    `_compose` does: `(answer, degraded, degraded_because, tokens_in, tokens_out)`.
    """
    source = f"Report:\n\n{deliverable}\n\nQuestion: {prompt}"
    return await _compose(source, system=FINAL_ANSWER_SYSTEM, base_url=base_url,
                          model=model, max_model_len=max_model_len, api_key=api_key,
                          cap=MAX_FINAL_ANSWER_TOKENS, stage="synthesizer.final_answer")


#: Separates a compressed answer from the full report it was drawn from, in the text a
#: `final_compression_prompt` pass (or a later `POST .../compress` call) produces — see
#: `wrap_with_answer`/`strip_prior_answer`. Distinctive enough (a heading no ordinary
#: report section would coincidentally produce) that `strip_prior_answer` can find it
#: reliably with a plain substring search, no markup parser required.
FULL_REPORT_MARKER = "\n\n---\n\n## Full report\n\n"


def wrap_with_answer(prompt: str, answer: str, full_report: str) -> str:
    """`full_report`, with `answer` (to `prompt`) placed above it under its own heading —
    the one assembly both `run`'s `final_compression_prompt` and a later `POST
    .../compress` call produce, so the two ever agree on the shape a compressed
    deliverable takes."""
    return f"## {prompt}\n\n{answer}{FULL_REPORT_MARKER}{full_report}"


def strip_prior_answer(deliverable: str) -> str:
    """The original full report inside `deliverable`, discarding any compressed-answer
    header a prior pass already added.

    What makes re-compressing safe to call any number of times: without this, a second
    `POST .../compress` call (a different prompt, say) would answer from a deliverable
    that already *is* a previous answer plus the report — each call's source material
    quietly shrinking to whatever the last call decided mattered, and a third call
    shrinking further still. `rfind`, not `find`: two prior compressions nest as
    `answer-2, MARKER, answer-1, MARKER, original`, and the *last* marker is the one
    with the real original after it — the true original is recovered in one call
    regardless of how many times this deliverable was compressed before, not one layer
    per call. Returns `deliverable` unchanged when it carries no such header (an
    ordinary deliverable, or one that never went through this pass), which is why this
    is safe to call unconditionally rather than only when the caller already knows a
    header is there.
    """
    idx = deliverable.rfind(FULL_REPORT_MARKER)
    if idx == -1:
        return deliverable
    return deliverable[idx + len(FULL_REPORT_MARKER):]


def _deliverable_threshold(max_model_len: int, deliverable_tokens: int) -> int:
    """Character trigger for bounded mode: `context_tokens - deliverable_tokens`.

    Converted to characters the same way `CX.budget_for` does — `CX.CHARS_PER_TOKEN`,
    floored at 8,000 the same way, so a `deliverable_tokens` configured very close to
    `max_model_len` cannot produce a degenerate near-zero threshold. No `WINDOW_FRACTION`
    here: `deliverable_tokens` already *is* the reserve, unlike the unbounded case where
    `WINDOW_FRACTION` stands in for one.
    """
    return max(8_000, int((max_model_len - deliverable_tokens) * CX.CHARS_PER_TOKEN))


# ── the driver ────────────────────────────────────────────────────────────────────


@dataclass
class SynthesisResult:
    """What one stage's worth of synthesis produced."""

    deliverable: str = ""
    #: `{citation_key: Reference.to_dict()}`, ready for `Run.append("deliverable", ...)`.
    references: dict = field(default_factory=dict)
    rounds: int = 0
    total_done: int = 0
    total_failed: int = 0
    #: True when any compression (mid-loop or final) exhausted its retries and fell back
    #: to an uncompressed digest slice instead of a genuine LLM summary — see `_compress`.
    #: `verdict_for` reads this to keep a run from grading `success` over a deliverable
    #: that is, in the case that matters most (the *final* compression), not actually
    #: compressed.
    degraded: bool = False
    #: The real reason the most recent degraded compression fell back — `_compress`'s
    #: `degraded_because` (`reply.error`, or the exception on the rare path `talk()`
    #: itself raised). Empty when `degraded` is False.
    degraded_because: str = ""
    #: How many reasoning rounds came back with no tool call, no text and no error — see
    #: `SynthesizerState.silent_reason_rounds`. Observability only: does not affect
    #: `rounds`, `degraded`, or `verdict_for`.
    silent_reason_rounds: int = 0
    #: `state.tokens_in`/`state.tokens_out` as they stood at the end of the run — the
    #: synthesizer's own reasoning-round and compression calls only, not the token cost of
    #: each goal's `aresearch(...)` deep-research call.
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict:
        return {"deliverable": self.deliverable, "references": self.references,
                "rounds": self.rounds, "total_done": self.total_done,
                "total_failed": self.total_failed, "degraded": self.degraded,
                "degraded_because": self.degraded_because,
                "silent_reason_rounds": self.silent_reason_rounds,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out}


def verdict_for(result: SynthesisResult):
    """This run's verdict, from how its logical goals actually landed.

    A run is as good as the goals it answered, not merely as good as "it returned".
    """
    total = result.total_done + result.total_failed
    if total == 0:
        return {"kind": ZERO_ENGAGEMENT,
                "because": "no logical goal was ever pursued"}
    # A degraded compression is a claim about the *deliverable*, not the goals — folded
    # into the `because` on every branch below so it is never lost behind a count that
    # still reads "all N answered", the exact silent-degradation this exists to refuse.
    #
    # Not every `degraded` compression fell back to the raw digest — `_compress` also
    # sets it when a real, LLM-written summary came back truncated (cut off by its token
    # cap, or by a premature voluntary stop) and was kept anyway because a partial
    # summary beats none. Reporting both cases as "fell back to an uncompressed digest"
    # is false for the truncated-but-real case: a live run answered 48/48 goals and
    # shipped a coherent multi-page deliverable that happened to be cut off mid-sentence
    # partway through, and the verdict claimed a raw-digest fallback that never
    # happened. `_is_truncation_reason` tells the two apart from `degraded_because`.
    degraded_note = ""
    if result.degraded:
        reason = result.degraded_because
        if reason and _is_truncation_reason(reason):
            degraded_note = (" — this stage's compression came back a real summary but "
                              f"truncated before it finished: {reason}")
        else:
            degraded_note = (" — the compression call failed after retries and fell "
                              "back to an uncompressed digest instead of a genuine "
                              "summary" + (f" ({reason})" if reason else ""))
    if result.total_failed == 0:
        if result.degraded:
            return {"kind": PARTIAL,
                    "because": f"all {result.total_done} logical goal(s) answered"
                               f"{degraded_note}"}
        return {"kind": SUCCESS,
                "because": f"all {result.total_done} logical goal(s) answered"}
    if result.total_done == 0:
        return {"kind": FAILURE,
                "because": f"none of the {result.total_failed} logical goal(s) "
                           f"attempted succeeded{degraded_note}"}
    return {"kind": PARTIAL,
            "because": f"{result.total_done} of {total} logical goal(s) succeeded"
                       f"{degraded_note}"}


async def run(state: SynthesizerState, *, base_url: str, model: str, api_key: str = "",
              max_model_len: int, aresearch, on_change=None, conn=None,
              max_idle_rounds: int = MAX_IDLE_ROUNDS,
              deliverable_tokens: int | None = None,
              allow_subsynthesis: bool = False,
              final_compression_prompt: str = "", embed=None) -> SynthesisResult:
    """Drive `state` to exhaustion, then write it into the deliverable, section by
    section — see `_write_deliverable`.

    One round is exactly the six steps the module docstring and the design spec agree
    on: digest, one reasoning call (dispatch happens inside it), execute whatever is now
    `pending` concurrently, check the *next* round's digest against the context budget
    and compact working memory (`_compress`) if it would not fit, then check
    `idle_rounds` against `max_idle_rounds`.

    `deliverable_tokens`, when given, changes two things, not just the final write's cap:

    - The mid-loop trigger becomes `_deliverable_threshold(max_model_len,
      deliverable_tokens)` — `context_tokens - deliverable_tokens`, in tokens — instead of
      `CX.budget_for(max_model_len)` (half the window). Crossing it stops the loop outright
      (no mid-loop `_compress()`, no further spawning/refining) rather than compacting and
      continuing, so the graph never grows past the point where the final write can still
      fit inside `deliverable_tokens`.
    - `_reason_round` gets a nudge (`BOUNDED_CONTEXT_NUDGE`) toward narrower goals.

    `None` (the default) is today's exact mid-loop-compact-and-continue behavior,
    unchanged — `deliverable_tokens` still overrides `MAX_COMPRESS_TOKENS` as the cap on
    every section `_write_deliverable` writes either way.

    `aresearch(question, *, model, base_url, api_key)` is the injected deep-research leaf
    — in production `synthruns`' wrapper over `run_synthesis`, returning an object with `.thorough`/`.tldr`
    (`citations.CitedText`-shaped: `.text`, `.references`). `on_change()` is called after
    every mutation to `state` (dispatch, each goal's completion, each compression) so a
    caller can persist immediately — see `save` above.

    `allow_subsynthesis` — default False, so every existing caller keeps today's exact
    behavior — offers `spawn_subsynthesis` this run's reasoning rounds, letting one of
    its own goals be answered by a whole nested `run()` instead of one `aresearch` call
    (`_execute_round`'s `_run_one`, `goal.nested`). Always False for that nested call:
    depth beyond one level is never reached because a nested run is never itself given
    this as True, not because anything counts or checks a depth.

    `final_compression_prompt`, when non-empty, adds one more step after the sectioned
    report is written: `answer_from_deliverable` answers that prompt — a specific
    question, or a synthesis instruction like "what are the top 5 methods mentioned here
    that could be implemented" — using the report as its only source. The answer is
    placed above the full report, under its own heading, never in place of it: the
    report a person actually gets is shorter to read first but nothing it said is gone.
    Empty (the default) skips this entirely — every existing caller is unaffected.

    `embed`, passed straight through to every `_reason_round` — see that function's own
    docstring — backs the `retrieve_facts` tool. `None` (the default) is today's exact
    behavior: the tool is still offered, but reports itself unavailable if called.
    """
    total_done = 0
    total_failed = 0
    #: Set once any compression this run performs — mid-loop or final — exhausts its
    #: retries and falls back to an uncompressed digest slice. Carried onto the result
    #: so `verdict_for` can grade the run honestly instead of reporting `success` over a
    #: deliverable that fell back to raw text.
    degraded = False
    #: The reason the most recent degraded compression gave up — see `_compress`.
    degraded_because = ""

    def changed() -> None:
        if on_change is not None:
            on_change()

    while state.idle_rounds < max_idle_rounds:
        state.round += 1
        await _reason_round(state, base_url=base_url, model=model,
                            max_model_len=max_model_len, api_key=api_key,
                            deliverable_tokens=deliverable_tokens,
                            allow_subsynthesis=allow_subsynthesis, embed=embed)
        changed()

        pending_ids = {g.id for g in state.goals.values() if g.status == PENDING}
        await _execute_round(state, aresearch=aresearch, model=model, base_url=base_url,
                             api_key=api_key, on_change=changed,
                             max_model_len=max_model_len, conn=conn,
                             deliverable_tokens=deliverable_tokens)
        for gid in pending_ids:
            g = state.goals.get(gid)
            if g is None:
                continue
            if g.status == DONE:
                total_done += 1
            elif g.status == FAILED:
                total_failed += 1

        if deliverable_tokens is not None:
            # Bounded mode: the reserve is the trigger, and crossing it stops the loop
            # outright — no mid-loop compress-and-continue, just fall through to the one
            # final `_compress()` below (already capped at `deliverable_tokens`).
            if len(_digest(state)) > _deliverable_threshold(max_model_len,
                                                             deliverable_tokens):
                break
            continue

        budget = CX.budget_for(max_model_len)
        if len(_digest(state)) > budget:
            summary, _refs, round_degraded, round_because = await _compress(
                state, base_url=base_url, model=model, max_model_len=max_model_len,
                api_key=api_key, conn=conn, deliverable_tokens=deliverable_tokens)
            degraded = degraded or round_degraded
            if round_degraded:
                degraded_because = round_because
            state.compressed.append(summary)
            # Never re-answer something already `done` or `failed` — only what is still
            # live survives a compression round.
            state.goals = {gid: g for gid, g in state.goals.items()
                           if g.status in (PENDING, RUNNING)}
            changed()

    if not state.goals and not state.compressed:
        # Nothing was ever established — a real, honest outcome, and one worth stating
        # plainly rather than spending a model call writing up an empty graph.
        final_summary = f"No sub-questions were established toward: {state.objective}"
        final_refs: dict[str, C.Reference] = {}
    else:
        final_summary, final_refs, final_degraded, final_because = await _write_deliverable(
            state, base_url=base_url, model=model, max_model_len=max_model_len,
            api_key=api_key, conn=conn, deliverable_tokens=deliverable_tokens)
        degraded = degraded or final_degraded
        if final_degraded:
            degraded_because = final_because
        if final_compression_prompt.strip() and final_summary:
            prompt = final_compression_prompt.strip()
            answer, ans_degraded, ans_because, ans_tin, ans_tout = (
                await answer_from_deliverable(
                    final_summary, prompt, base_url=base_url, model=model,
                    max_model_len=max_model_len, api_key=api_key))
            state.tokens_in += ans_tin
            state.tokens_out += ans_tout
            degraded = degraded or ans_degraded
            if ans_degraded:
                degraded_because = ans_because
            final_summary = wrap_with_answer(prompt, answer, final_summary)
        if final_degraded and final_summary:
            final_summary = _pressure_note(final_because) + "\n\n" + final_summary
        elif degraded and final_summary:
            # Only the mid-loop compaction degraded, not this run's own final write —
            # still worth the same caveat: the prose it compacted is what a section of
            # this deliverable was built from.
            final_summary = _pressure_note(degraded_because) + "\n\n" + final_summary
    changed()

    return SynthesisResult(
        deliverable=final_summary,
        references={k: v.to_dict() for k, v in final_refs.items()},
        rounds=state.round, total_done=total_done, total_failed=total_failed,
        degraded=degraded, degraded_because=degraded_because,
        silent_reason_rounds=state.silent_reason_rounds,
        tokens_in=state.tokens_in, tokens_out=state.tokens_out)
