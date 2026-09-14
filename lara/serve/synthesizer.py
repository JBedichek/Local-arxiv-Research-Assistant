"""A dependency graph of research sub-questions that grows itself, round by round.

Ported from autoresearch's own `synthesizer.py` -- same shape (a live graph of
`LogicalGoal`s, one reasoning round at a time: digest what has landed, decide
spawn_goal/refine_goal/finish, execute whatever is newly pending concurrently
through the deep-research leaf, compress when the digest outgrows the model's
window, stop once enough consecutive rounds propose nothing) -- but stripped
to what lara-core already has, not autoresearch's:

- No `propose_experiment` tool. That one is about autoresearch's own approve-
  and-run experiment pipeline; this module only ever grows a literature-search
  graph.
- No native tool-calling transport. autoresearch's `_reason_round` drives a
  multi-turn `converse.talk(...)` loop (up to `MAX_TURNS` tool calls per
  round, dispatched as they arrive). lara-core's `generate.py` only offers
  `complete`/`complete_json` -- one completion in, one parsed reply out, no
  tool-call protocol -- so a round here asks for exactly one JSON action
  object per call instead of a turn loop, and therefore makes at most one
  `spawn_goal`/`refine_goal` decision per round rather than autoresearch's
  "as many as the model wants before it stops."
- No `citations.py` citation-binding. Each goal's leaf call is lara-core's own
  `run_synthesis`, whose `Run.papers` is already a flat, globally-unique list
  of arXiv ids -- so a goal's provenance is just that list, and the final
  report is asked to name arXiv ids inline rather than resolve a bracket-
  citation key against a shared reference table.
- No `context.py`/`budgets.py` window-sizing machinery. Reply sizes are fixed
  constants and the mid-loop compression trigger is a plain character-count
  heuristic against the caller-supplied `max_model_len`, not a live
  `/tokenize` round trip.
- No bounded "deliverable_tokens" mode, no compression truncation-detection/
  retry ladder. One compression attempt; an empty or failed reply falls back
  to a clipped raw digest rather than losing the round's work, and that is
  the whole fallback story here.

What's identical in spirit: the goal dataclass shape, the digest format, the
spawn/refine validation rules (concurrency cap, refinement depth cap, unknown-
dependency rejection), concurrent execution of a round's pending goals via
`asyncio.gather`, idle-round-counted termination, and a final compression into
one deliverable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lara.serve.generate import complete, complete_json

# ── status of one logical goal ───────────────────────────────────────────────────

PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
STATUSES = (PENDING, RUNNING, DONE, FAILED)

#: How many consecutive rounds may propose nothing before the graph is
#: considered exhausted and final compression runs. Same tunable, same value,
#: as autoresearch's own MAX_IDLE_ROUNDS.
MAX_IDLE_ROUNDS = 5

#: How deep a chain of refine_goal calls may go from the nearest spawn_goal.
MAX_REFINEMENT_DEPTH = 10

#: How many goals may sit pending/running at once -- bounds how far a single
#: round (and, here, a single reasoning decision) can grow the digest before
#: the next mid-loop budget check.
MAX_CONCURRENT_GOALS = 3

SPAWN_TOOL = "spawn_goal"
REFINE_TOOL = "refine_goal"
FINISH_TOOL = "finish"

#: Ceiling on one reasoning round's reply -- a single JSON action object, not
#: prose, so this stays small.
MAX_REASON_TOKENS = 1_000

#: Ceiling on the compression reply.
MAX_COMPRESS_TOKENS = 4_000

#: A generic English-text estimate (matches autoresearch/context.py's own
#: value) -- used only for the mid-loop compression trigger below, not for
#: sizing a reply against a live token count.
CHARS_PER_TOKEN = 3.8

#: Mid-loop compression fires once the digest passes this fraction of the
#: model's context window (in estimated characters) -- half the window,
#: same as autoresearch's own CX.budget_for.
WINDOW_FRACTION = 0.5


def _char_budget(max_model_len: int) -> int:
    return max(8_000, int(max_model_len * WINDOW_FRACTION * CHARS_PER_TOKEN))


@dataclass
class LogicalGoal:
    """One node in the graph: a question, what it depends on, and its answer
    once run. `depth` counts refinement steps from the nearest spawn_goal."""

    id: str
    text: str
    depends_on: list[str] = field(default_factory=list)
    #: The id of the goal this one deepens, or None for a spawn_goal goal.
    refines: str | None = None
    depth: int = 0
    status: str = PENDING
    #: run_synthesis's Run.thorough (or Run.tldr if thorough came back empty),
    #: once status is done. Empty otherwise.
    summary: str = ""
    #: arXiv ids run_synthesis's Run.papers reported for this goal's answer.
    papers: list[str] = field(default_factory=list)
    #: Set when status is failed: what run_synthesis raised, or its own
    #: stopped_because if consolidation failed internally.
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> LogicalGoal:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


@dataclass
class SynthesizerState:
    """The live graph for one synthesis run.

    `goals` is an ordinary dict, and insertion order is relied on as a
    dependency order for `_digest`: `_do_spawn`/`_do_refine` only accept a
    `depends_on`/`parent_id` naming a goal already in the dict, so a goal is
    always created after everything it depends on.
    """

    objective: str
    goals: dict[str, LogicalGoal] = field(default_factory=dict)
    idle_rounds: int = 0
    #: Prior compression summaries, oldest first.
    compressed: list[str] = field(default_factory=list)
    round: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict:
        return {"objective": self.objective,
                "goals": {gid: g.to_dict() for gid, g in self.goals.items()},
                "idle_rounds": self.idle_rounds, "compressed": list(self.compressed),
                "round": self.round, "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out}

    @classmethod
    def from_dict(cls, d: dict) -> SynthesizerState:
        d = d or {}
        goals = {str(gid): LogicalGoal.from_dict(g)
                 for gid, g in (d.get("goals") or {}).items()}
        return cls(objective=str(d.get("objective") or ""), goals=goals,
                    idle_rounds=int(d.get("idle_rounds") or 0),
                    compressed=[str(x) for x in (d.get("compressed") or [])],
                    round=int(d.get("round") or 0),
                    tokens_in=int(d.get("tokens_in") or 0),
                    tokens_out=int(d.get("tokens_out") or 0))


# ── persistence ───────────────────────────────────────────────────────────────────

#: Where synthesizer states live, one JSON file each.
STATES = Path.home() / ".lara" / "synthesizer"


def save(state_id: str, state: SynthesizerState, *, root: Path | None = None) -> Path:
    """Write the graph down. Never raises: losing the record must not stop the round.

    Written after every change, not once at the end -- the interesting moment
    to survive is the middle, so a crash here loses at most the round in flight.
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
    """The graph as it was last written, or None -- a fresh SynthesizerState starts."""
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


def _digest(state: SynthesizerState) -> str:
    """objective + prior compressions + every done/failed goal, in dependency order."""
    parts = [f"Objective: {state.objective}"]
    for i, summary in enumerate(state.compressed, 1):
        parts.append(f"### earlier compression {i}\n{summary}")
    for goal in state.goals.values():
        if goal.status not in (DONE, FAILED):
            continue
        head = f"### {goal.id} (depth {goal.depth}"
        if goal.refines:
            head += f", refines {goal.refines}"
        head += f", {goal.status})"
        block = [head, goal.text]
        if goal.status == DONE:
            papers = f" [papers: {', '.join(goal.papers)}]" if goal.papers else ""
            block.append(f"**Answer:**{papers} {goal.summary or '(nothing)'}")
        else:
            block.append(f"**Failed:** {goal.error or 'unknown error'}")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


# ── the prompts ───────────────────────────────────────────────────────────────────

REASON_SYSTEM = """You are growing a dependency graph of research sub-questions toward one \
objective, one round at a time.

Below is the objective, any earlier compressions, and every sub-question answered or \
failed so far. Given all of it, decide exactly one of:

- spawn_goal — a genuinely new angle the graph does not cover yet.
- refine_goal — go deeper on one specific landed answer, when doing so would plausibly \
move the objective forward.
- finish — nothing further would help right now.

Never propose a goal that duplicates or trivially restates one already answered below — \
if the graph already has the answer, choose finish rather than asking again in different \
words.

The only leaf here is literature search against a corpus of papers — it can tell you what \
has been published, not run anything. A sub-question that is fundamentally empirical (what \
some untested setup actually does, when nobody has published a run of it) cannot be settled \
by asking it to dig harder; if you still spawn or refine such a question, phrase its `text` \
to ask explicitly for what published work can bound or proxy, and to state plainly that the \
question is only actually settled by a direct measurement, not by more reading.

Reply with exactly one JSON object and nothing else:
{"tool": "spawn_goal" | "refine_goal" | "finish", "args": {...}}

spawn_goal's args: {"text": "<the sub-question, phrased so it stands alone>", \
"depends_on": ["<ids of already-answered goals below whose answers this one should read \
first>"]}
refine_goal's args: {"parent_id": "<id of the goal to deepen, from below>", "text": "<the \
deeper sub-question>"}
finish's args: {}"""

COMPRESS_SYSTEM = """You are writing the final deliverable for a research graph: a \
comprehensive, detailed report on everything it established, not a condensed summary of it.

You are given the objective and every sub-question answered or failed below, each with the \
arXiv ids of the papers its answer drew on. Write the most thorough report the material \
actually supports, organized by theme or sub-question. For each one, give the specific \
finding, the evidence and reasoning behind it, and the numbers, method names, and caveats \
that were reported — not a one-line paraphrase of them. Naming what disagrees, what remains \
unresolved, and what did not work (and why) matters as much as naming what succeeded.

Use the room you are given. Write a shorter report only because the material itself ran \
out, never because a summary is easier to produce than a report.

Rules:
- When you state a finding, name the arXiv id(s) it came from, e.g. (arXiv:1706.03762). Do \
not invent an id that was not already given to you.
- If something failed, say so plainly rather than omitting it.
- Use headings to organize by theme or sub-question."""


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
    if parent.depth >= MAX_REFINEMENT_DEPTH:
        return False, (f"{parent_id!r} sits at refinement depth {parent.depth}, at the "
                       f"limit of {MAX_REFINEMENT_DEPTH} — refine something shallower, "
                       "or spawn a new goal instead; nothing was added")
    text = str((args or {}).get("text") or "").strip()
    if not text:
        return False, f"{REFINE_TOOL} needs a non-empty text — nothing was added"
    gid = _new_id(text, state.goals)
    state.goals[gid] = LogicalGoal(id=gid, text=text, depends_on=[parent_id],
                                   refines=parent_id, depth=parent.depth + 1,
                                   status=PENDING)
    return True, f"refined {parent_id!r} as {gid!r}: {text}"


async def _reason_round(state: SynthesizerState, *, cfg, model: str | None = None) -> bool:
    """One reasoning pass: one complete_json call, one JSON action object, at
    most one goal created. Mutates `state` in place. Returns whether real
    progress was made -- a goal actually created, not merely an action chosen
    and refused.

    Never raises: a reasoning pass that errored or came back unparseable is a
    round with nothing proposed, which is exactly what an idle round already
    means, not a reason to take the whole run down with it.
    """
    digest = _digest(state)
    prompt = f"{digest}\n\n---\nDecide your one action now."
    try:
        result = await complete_json(cfg, prompt, system=REASON_SYSTEM, model=model,
                                     max_tokens=MAX_REASON_TOKENS, default={})
    except Exception:                                            # noqa: BLE001
        result = {}
    result = result if isinstance(result, dict) else {}
    tool = str(result.get("tool") or "").strip()
    args = result.get("args") or {}

    progressed = False
    if tool == SPAWN_TOOL:
        ok, _msg = _do_spawn(state, args)
        progressed = ok
    elif tool == REFINE_TOOL:
        ok, _msg = _do_refine(state, args)
        progressed = ok
    # FINISH_TOOL, an unrecognized tool, or an unparseable reply all leave
    # progressed False -- an idle round either way.

    state.idle_rounds = 0 if progressed else state.idle_rounds + 1
    return progressed


# ── executing a round's pending goals concurrently ───────────────────────────────

#: What run_synthesis appends to its own stopped_because when consolidation
#: raises internally -- see that function's `except Exception as exc` around
#: its `consolidate()` call. It never re-raises: the run still returns
#: normally with whatever evidence it gathered, so this marker is how a
#: caller tells "answer landed" apart from "evidence gathered, but the
#: written answer failed."
_CONSOLIDATION_FAILED_MARKER = "consolidation failed"


async def _execute_round(state: SynthesizerState, *, run_synthesis, app_state, cfg,
                         model: str | None = None, on_change=None) -> None:
    """Run every currently-pending goal through run_synthesis, concurrently.

    Goals proposed in the same round cannot depend on each other -- `_do_spawn`/
    `_do_refine` only accept a `depends_on`/`parent_id` naming a goal already
    `done` or `failed` -- so there is no ordering to respect within one batch.

    Each goal's own failure is caught inside `_run_one` and recorded on that
    goal alone. `on_change()` is called the moment that goal finishes, not
    once at the end of the batch, so a mid-batch crash only loses the goals
    still in flight.
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
            run = await run_synthesis(app_state, cfg, goal.text, model=model)
            text = str(getattr(run, "thorough", "") or "").strip()
            if not text:
                text = str(getattr(run, "tldr", "") or "").strip()
            papers = list(getattr(run, "papers", []) or [])
            stopped_because = str(getattr(run, "stopped_because", "") or "")
            if _CONSOLIDATION_FAILED_MARKER in stopped_because:
                goal.status = FAILED
                goal.error = (text or stopped_because)[:400]
            else:
                goal.status = DONE
                goal.summary = text
                goal.papers = papers
        except Exception as exc:                                # noqa: BLE001
            goal.status = FAILED
            goal.error = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            if on_change is not None:
                on_change()

    await asyncio.gather(*(_run_one(g) for g in pending), return_exceptions=True)


# ── compression ───────────────────────────────────────────────────────────────────


async def _compress(state: SynthesizerState, *, cfg, model: str | None = None) -> str:
    """One dense report of everything in `goals` (done + failed) plus prior
    compressions. Falls back to a clipped raw digest if the call fails or
    comes back empty, rather than losing the round's work -- the one fallback
    autoresearch's own retry ladder exists to make rare; this keeps just the
    fallback, not the ladder."""
    source = _digest(state)
    # complete_json is for structured replies; a report is prose, so this
    # calls the plain-text primitive instead.
    try:
        summary = (await complete(cfg, source, system=COMPRESS_SYSTEM, model=model,
                                  max_tokens=MAX_COMPRESS_TOKENS)).strip()
    except Exception as exc:                                     # noqa: BLE001
        logging.getLogger(__name__).warning("synthesizer: compression failed: %s", exc)
        summary = ""
    if not summary:
        summary = source[:8_000]
        logging.getLogger(__name__).warning(
            "synthesizer: compression came back empty, falling back to an "
            "uncompressed digest slice")
    return summary


# ── the driver ────────────────────────────────────────────────────────────────────


@dataclass
class SynthesisResult:
    """What one run of the graph produced."""

    deliverable: str = ""
    rounds: int = 0
    total_done: int = 0
    total_failed: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict:
        return {"deliverable": self.deliverable, "rounds": self.rounds,
                "total_done": self.total_done, "total_failed": self.total_failed,
                "tokens_in": self.tokens_in, "tokens_out": self.tokens_out}


async def run(state: SynthesizerState, *, app_state, cfg, model: str | None = None,
              max_model_len: int, run_synthesis=None, on_change=None,
              max_idle_rounds: int = MAX_IDLE_ROUNDS) -> SynthesisResult:
    """Drive `state` to exhaustion, then compress it once more into the deliverable.

    One round: one reasoning call (at most one goal created -- see
    `_reason_round`'s docstring for why this differs from autoresearch's own
    multi-tool-call round), execute whatever is now pending concurrently,
    check the digest against a plain character budget and compress if it
    would not fit, then check `idle_rounds` against `max_idle_rounds`.

    `run_synthesis`, when not given, defaults to
    `lara.serve.synthesis.run_synthesis` -- the deep-research leaf every
    pending goal is answered through. `app_state` is lara's own `AppState`
    (embedder, retriever, index) that leaf needs; `cfg` is the app config
    both it and the reasoning/compression calls here read from. Named
    `app_state` rather than `state` (autoresearch's own parameter name for
    this) to avoid colliding with `state: SynthesizerState`, the graph this
    function drives -- lara-core's own convention already reserves `state`
    for `AppState`.

    `on_change()` is called after every mutation to `state` (a goal created,
    each goal's completion, each compression) so a caller can persist
    immediately via `save`.
    """
    if run_synthesis is None:
        from lara.serve.synthesis import run_synthesis as run_synthesis

    total_done = 0
    total_failed = 0

    def changed() -> None:
        if on_change is not None:
            on_change()

    while state.idle_rounds < max_idle_rounds:
        state.round += 1
        await _reason_round(state, cfg=cfg, model=model)
        changed()

        pending_ids = {g.id for g in state.goals.values() if g.status == PENDING}
        await _execute_round(state, run_synthesis=run_synthesis, app_state=app_state,
                             cfg=cfg, model=model, on_change=changed)
        for gid in pending_ids:
            g = state.goals.get(gid)
            if g is None:
                continue
            if g.status == DONE:
                total_done += 1
            elif g.status == FAILED:
                total_failed += 1

        if len(_digest(state)) > _char_budget(max_model_len):
            summary = await _compress(state, cfg=cfg, model=model)
            state.compressed.append(summary)
            # Never re-answer something already done or failed -- only what
            # is still live survives a compression round.
            state.goals = {gid: g for gid, g in state.goals.items()
                           if g.status in (PENDING, RUNNING)}
            changed()

    if not state.goals and not state.compressed:
        final_summary = f"No sub-questions were established toward: {state.objective}"
    else:
        final_summary = await _compress(state, cfg=cfg, model=model)
    changed()

    return SynthesisResult(deliverable=final_summary, rounds=state.round,
                           total_done=total_done, total_failed=total_failed,
                           tokens_in=state.tokens_in, tokens_out=state.tokens_out)
