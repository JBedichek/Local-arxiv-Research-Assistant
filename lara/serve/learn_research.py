"""Learn's course-map and lesson research, driven by the same agentic synthesis engine as
the Synthesize tab -- see `synthesizer.run`'s `deliverable_mode`.

Lives here, in `lara.serve`, and not in `lara.learn`, because it needs
`lara.serve.synthesizer`/`synthruns` directly to drive a run, and `lara.learn` never
imports `lara.serve` (see `lara.learn.passages.CorpusRetriever`'s own docstring: every
serve-side capability learn code needs is injected as a plain callable, the same shape
`Llm`/`CorpusRetriever`/`embed_fn` already are). This module is that boundary: it builds
the injected `synth` callable `lara.learn.graph.build_from_research` and
`lara.learn.research.build_lesson` take, closing over the `app_state`/`cfg` neither of
those modules may reference.

`on_change` is where a run's progress is bridged into the Profile tab: it calls
`lara.learn.trace.emit` directly (serve importing learn's trace module is the normal,
allowed direction -- routes/learn.py already imports `lara.learn.pipeline` the same way).
`trace.emit` finds its tracer through a contextvar, not an import-time reference, so this
works regardless of which package the call happens to live in -- see trace.py's own
docstring -- as long as it runs inside the task tree `pipeline.build_concept`/`map_course`
started with `trace.start`, which it always does: `synth` is only ever awaited from
directly inside those two call trees.
"""
from __future__ import annotations

import copy
import json

from lara.learn import trace as TR
from lara.serve import synthesizer as SY
from lara.serve import synthruns as SR

#: Both of Learn's own `synthesizer.run` calls (course-mapping, one lesson) are scoped to
#: an already-narrow objective -- a course's stated goal, or one concept within it -- the
#: same situation `SY.SUBSYNTHESIS_MAX_IDLE_ROUNDS` exists for, not the wide-open
#: standalone Synthesize tab `SY.MAX_IDLE_ROUNDS` (5) is tuned for. Confirmed by the user
#: after watching a real course-mapping run take 17+ minutes at the default: same value,
#: same reasoning as that existing precedent.
LEARN_MAX_IDLE_ROUNDS = SY.SUBSYNTHESIS_MAX_IDLE_ROUNDS

#: The hard round-count safety net (`SY.run`'s `max_rounds`) -- not a normal exit, only
#: meant to catch the pathological case actually observed: a graph that keeps legitimately
#: finding one more goal to propose every round, for an objective broad enough that it
#: never goes idle and the model never calls `finish`. `SY.run`'s own `finish`/idle exits
#: (see `FINISH_STREAK_TO_STOP`/`LEARN_MAX_IDLE_ROUNDS` above) are the exits meant to fire
#: in ordinary operation; these two numbers exist only to bound the worst case, sized
#: comfortably above what a normal run should ever need. Course-mapping's objective ("map
#: this whole course") is legitimately broader than one lesson's ("teach this one
#: concept"), hence the different ceiling for each. Unmeasured starting guesses, the same
#: honest way `SY.MAX_IDLE_ROUNDS`'s own comment describes itself -- tune from measured
#: runs once there are some.
LEARN_MAX_ROUNDS_TOPIC_GRAPH = 20
LEARN_MAX_ROUNDS_LESSON = 12


async def _run(app_state, objective: str = "", *, deliverable_mode: str,
               model: str | None, max_rounds: int | None,
               state: SY.SynthesizerState | None = None,
               state_id: str = "") -> SY.SynthesisResult:
    """`state`/`state_id` are what let a course-mapping or lesson run be resumed later
    with a learner's feedback (see `topic_graph_revise`/`lesson_revise` below) instead of
    researching from scratch: a fresh call (`topic_graph_synth`/`lesson_synth`) builds its
    own `SynthesizerState(objective=objective)` and leaves `state` `None`; a revision call
    loads the one this same `state_id` was last saved under (`SY.load`), folds its
    feedback into `state.objective` itself, and passes it straight in here. Either way,
    `state_id`, when given, is saved (`SY.save`) after every change the run makes --
    `synthruns.py`'s own persistence, reused here rather than duplicated -- so a later
    revision always resumes from the exact graph this run left behind, not a stale copy."""
    from lara.serve import facts as FA

    g = await SR.generator(app_state, model)
    embedder = getattr(app_state.retriever, "embedder", None)
    embed = FA.embedder_fn(embedder) if embedder is not None else None
    state = state or SY.SynthesizerState(objective=objective)
    spawned: set[str] = set()
    landed: set[str] = set()

    def on_change() -> None:
        # Fires after every mutation to `state` -- dispatch, a goal landing, a
        # compression -- so a goal is announced once, spawned, and once more, landed;
        # `spawned`/`landed` are what keep each of those to exactly one trace event
        # despite `on_change` re-walking the whole graph every time it fires. On a
        # resumed (revision) run this also re-announces every goal a prior run already
        # landed -- correct, not a duplicate: this run's own trace file is fresh (see
        # trace.py), so the full graph belongs in it too, not just what changed.
        for gid, goal in state.goals.items():
            if gid not in spawned:
                spawned.add(gid)
                TR.emit("goal_spawned", id=gid, text=goal.text, refines=goal.refines,
                       depth=goal.depth)
            if goal.status in (SY.DONE, SY.FAILED) and gid not in landed:
                landed.add(gid)
                TR.emit("goal_done" if goal.status == SY.DONE else "goal_failed",
                       id=gid, text=goal.text, citations=len(goal.citations or {}),
                       error=goal.error)
        TR.emit("round", round=state.round, tokens_in=state.tokens_in,
               tokens_out=state.tokens_out, compressions=len(state.compressed))
        if state_id:
            SY.save(state_id, state)

    section_seq = 0

    def on_section(label: str, text: str) -> None:
        # Fires once per section as `_write_deliverable` actually writes it (see that
        # function's own docstring on why this is real per-section content, not a token
        # stream) -- real, growing visibility into the deliverable itself, which
        # `on_change` above never carries: `on_change` only ever sees goal/round
        # bookkeeping, not the topic-graph JSON or lesson prose those goals get written
        # into.
        nonlocal section_seq
        section_seq += 1
        TR.emit("deliverable_section", index=section_seq, label=label, text=text)

    aresearch = SR.leaf(app_state, g.cfg)
    result = await SY.run(
        state, base_url=g.base_url, model=g.model, api_key=g.api_key,
        max_model_len=g.window, aresearch=aresearch, on_change=on_change,
        conn=app_state.conn(), embed=embed, deliverable_mode=deliverable_mode,
        max_idle_rounds=LEARN_MAX_IDLE_ROUNDS, max_rounds=max_rounds,
        on_section=on_section)
    TR.emit(f"{deliverable_mode}_written", rounds=result.rounds,
           goals_done=result.total_done, goals_failed=result.total_failed,
           degraded=result.degraded, degraded_because=result.degraded_because,
           tokens_in=result.tokens_in, tokens_out=result.tokens_out,
           exit_reason=result.exit_reason)
    return result


def _state_id(kind: str, course_id: str, cid: str = "") -> str:
    """`synthesizer.save`/`load`'s own key, namespaced away from the standalone
    Synthesize tab's run ids (`synthruns.py`'s `uuid.uuid4().hex[:12]` -- no colon, so
    `"course:"`/`"lesson:"` can never collide with one) and, for a lesson, from every
    other concept's own graph."""
    return f"{kind}:{course_id}:{cid}" if cid else f"{kind}:{course_id}"


def _parse_topic_graph(result: SY.SynthesisResult) -> dict:
    try:
        data = json.loads(result.deliverable or "{}")
    except ValueError:
        data = {}
    subjects = data.get("subjects") if isinstance(data, dict) else None
    return {"subjects": subjects if isinstance(subjects, list) else [],
            "references": result.references, "degraded": result.degraded,
            "tokens_in": result.tokens_in, "tokens_out": result.tokens_out,
            "rounds": result.rounds}


def _parse_lesson(result: SY.SynthesisResult) -> dict:
    return {"deliverable": result.deliverable, "references": result.references,
            "degraded": result.degraded, "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out, "rounds": result.rounds}


def _apply_feedback(state: SY.SynthesizerState, feedback: str) -> None:
    """Folds a learner's revision request into the persisted graph's own objective --
    every following reasoning round, and every section `_write_deliverable` writes, reads
    `state.objective` first (see `synthesizer._digest`/`_cluster_digest`), so this is the
    one place a revision needs to touch to be seen everywhere the original objective was.
    The model is free to decide, round by round, whether the feedback needs new research
    (`spawn_goal`/`refine_goal`) or only a different write of what is already there.

    Then gives the revision its own fresh round budget -- `idle_rounds`/`finish_streak`/
    `round` all reset -- the same way a `spawn_subsynthesis` goal gets its own budget from
    its parent. Without this, a graph that already reached its first exit (`idle_rounds`
    at `max_idle_rounds`, or two `finish` calls in a row) would find `synthesizer.run`'s
    own loop condition already false and never run a single further round -- `goals` and
    `compressed` are the only state a revision is meant to inherit, not how far the
    original run had already run down its own budget."""
    state.objective = (
        f"{state.objective}\n\nThe learner reviewed the version above and asked for this "
        f"to change before it is finalized:\n{feedback.strip()}\n\nAddress it -- spawn new "
        "research only if the feedback actually needs new evidence; otherwise leave the "
        "graph as it stands and just change what gets written from it.")
    state.finish_streak = 0
    state.idle_rounds = 0
    state.round = 0


def topic_graph_synth(app_state, course_id: str, *, model: str | None = None):
    """The `synth` capability `lara.learn.graph.build_from_research` takes: one call is
    one full course-level synthesis run, its deliverable already the concept-map JSON
    (`deliverable_mode="topic_graph"`) -- caller sets `trace.set_phase` before awaiting.
    `course_id` is only ever used to key this run's persisted graph (`_state_id`) for a
    later `topic_graph_revise` -- never sent to the model."""
    state_id = _state_id("course", course_id)

    async def synth(objective: str) -> dict:
        result = await _run(app_state, objective, deliverable_mode="topic_graph",
                            model=model, max_rounds=LEARN_MAX_ROUNDS_TOPIC_GRAPH,
                            state_id=state_id)
        return _parse_topic_graph(result)
    return synth


def topic_graph_revise(app_state, course_id: str, *, model: str | None = None):
    """The revision counterpart to `topic_graph_synth`: resumes the persisted Phase 1
    graph this same `course_id` last saved (see `_state_id`) instead of researching from
    scratch, folding the learner's feedback into its objective (`_apply_feedback`) first.

    Raises `ValueError` if nothing was persisted to resume from -- a course mapped before
    this pipeline existed, or by the old blind `graph.build` path, which never drives a
    `synthesizer.run` at all and so never has a graph here to resume."""
    state_id = _state_id("course", course_id)

    async def revise(feedback: str) -> dict:
        state = SY.load(state_id)
        if state is None:
            raise ValueError("no prior plan is on record for this course to revise from")
        _apply_feedback(state, feedback)
        result = await _run(app_state, deliverable_mode="topic_graph", model=model,
                            max_rounds=LEARN_MAX_ROUNDS_TOPIC_GRAPH, state=state,
                            state_id=state_id)
        return _parse_topic_graph(result)
    return revise


def lesson_synth(app_state, course_id: str, cid: str, *, model: str | None = None):
    """The `synth` capability `lara.learn.research.build_lesson` takes: one call is one
    full concept-scoped synthesis run, its deliverable already the lesson prose
    (`deliverable_mode="lesson"`) -- caller sets `trace.set_phase` before awaiting.
    `course_id`/`cid` are only ever used to key this run's persisted graph (`_state_id`)
    for a later `lesson_revise` -- never sent to the model."""
    state_id = _state_id("lesson", course_id, cid)

    async def synth(objective: str) -> dict:
        result = await _run(app_state, objective, deliverable_mode="lesson", model=model,
                            max_rounds=LEARN_MAX_ROUNDS_LESSON, state_id=state_id)
        return _parse_lesson(result)
    return synth


def lesson_revise(app_state, course_id: str, cid: str, *, model: str | None = None):
    """The revision counterpart to `lesson_synth` -- see `topic_graph_revise`'s own
    docstring; same mechanism, scoped to one concept's Phase 2 graph. Raises `ValueError`
    if this concept's lesson was written by the legacy claims.py + depth.deepen() pipeline
    (no `lesson_synth` was ever involved, so nothing was ever persisted to resume)."""
    state_id = _state_id("lesson", course_id, cid)

    async def revise(feedback: str) -> dict:
        state = SY.load(state_id)
        if state is None:
            raise ValueError("no prior lesson is on record for this concept to revise from")
        _apply_feedback(state, feedback)
        result = await _run(app_state, deliverable_mode="lesson", model=model,
                            max_rounds=LEARN_MAX_ROUNDS_LESSON, state=state,
                            state_id=state_id)
        return _parse_lesson(result)
    return revise


def expand_synth(app_state, *, model: str | None = None):
    """The `synth` capability `lara.learn.expand.expand` takes for its on-demand search
    fallback -- one research leaf (`synthruns.leaf`'s `aresearch`), not a whole synthesis
    graph: a highlighted passage plus a specific follow-up is inherently a single
    targeted question, not a multi-goal course/lesson objective the way `topic_graph_synth`/
    `lesson_synth`'s full `synthesizer.run` calls are. `aresearch`'s bound `.thorough`
    citations are `Reference` dataclasses, not the plain dicts `SynthesisResult.references`
    already is (see that class's own comment) -- converted here with `.to_dict()` so what
    crosses back into `lara.learn` is the same plain-dict shape every other `synth`
    capability already returns."""
    async def synth(focus: str) -> dict:
        g = await SR.generator(app_state, model)
        aresearch = SR.leaf(app_state, g.cfg)
        TR.set_phase("expand_research")
        try:
            result = await aresearch(focus, model=g.model, base_url=g.base_url,
                                     api_key=g.api_key)
        except SR.NoEvidence as e:
            TR.emit("expand_research", found=False, error=str(e))
            return {"text": "", "references": {}}
        bound = result.thorough
        refs = {k: v.to_dict() for k, v in (bound.references or {}).items()}
        TR.emit("expand_research", found=True, references=len(refs))
        return {"text": bound.text or "", "references": refs}
    return synth


# ── single deep-research leaves with their own budgets ───────────────────────────
#
# Two Learn calls are one question each, not a multi-goal graph, and each wants a budget the
# global `retrieval.synthesis` config is not tuned for: mapping a whole subject once (wide,
# deliberately thorough -- a broad question never saturates, so its round budget is what ends
# it) and grounding one diagnostic question about one concept (narrow, a few rounds). Both run
# `synthruns.leaf` on a copy of the config with those knobs overridden for that call only.

#: For the subject decomposition (`lara.learn.decompose`): starting values to tune against
#: measured runs, not settled numbers.
DECOMPOSE_SYNTHESIS = {"max_rounds": 24, "min_rounds": 8, "stop_votes": 3, "per_round": 20,
                       "over_fetch": 5, "cap_per_paper": 2, "max_feedback_vectors": 10,
                       "saturation_window": 3, "dry_rounds": 3, "expand_every": 2}
#: For one diagnostic question (`lara.learn.diagnostic`).
PROBE_SYNTHESIS = {"max_rounds": 3, "min_rounds": 2, "stop_votes": 1, "per_round": 8,
                   "saturation_window": 2, "dry_rounds": 1}


def with_synthesis(cfg, overrides: dict):
    """A copy of `cfg` whose `retrieval.synthesis` knobs are overridden by `overrides` --
    `run_synthesis` reads them from the config it is handed, per call, so nothing else sees
    the change."""
    new = copy.copy(cfg)
    retrieval = dict(cfg.get_in("retrieval") or {})
    retrieval["synthesis"] = {**(retrieval.get("synthesis") or {}), **overrides}
    new["retrieval"] = retrieval
    return new


def _leaf_research(app_state, overrides: dict, phase: str, *, model: str | None = None):
    async def research(question: str) -> dict:
        g = await SR.generator(app_state, model)
        aresearch = SR.leaf(app_state, with_synthesis(g.cfg, overrides))
        TR.emit(f"{phase}_start", question=question)
        try:
            result = await aresearch(question, model=g.model, base_url=g.base_url,
                                     api_key=g.api_key)
        except SR.NoEvidence as e:
            TR.emit(f"{phase}_done", found=False, error=str(e))
            return {"text": "", "references": {}, "stopped_because": str(e)}
        bound = result.thorough
        refs = {k: v.to_dict() for k, v in (bound.references or {}).items()}
        TR.emit(f"{phase}_done", found=True, references=len(refs),
                stopped_because=result.stopped_because)
        return {"text": bound.text or "", "references": refs,
                "stopped_because": result.stopped_because or ""}
    return research


def decompose_research(app_state, *, model: str | None = None):
    """The `research` capability `lara.learn.decompose.decompose` takes: one thorough
    deep-research call that maps a whole subject, from its foundations to its frontier."""
    return _leaf_research(app_state, DECOMPOSE_SYNTHESIS, "decompose", model=model)


def probe_research(app_state, *, model: str | None = None):
    """The `research` capability `lara.learn.diagnostic.prepare` takes: one short
    deep-research call that grounds one diagnostic question about one concept."""
    return _leaf_research(app_state, PROBE_SYNTHESIS, "probe", model=model)
