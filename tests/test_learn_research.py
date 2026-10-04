"""lara.serve.learn_research: the boundary that drives Learn's course-map and lesson
research through the same synthesizer.run() the Synthesize tab uses -- see that module's
own docstring for why it lives in lara.serve rather than lara.learn.

Existing pipeline-level tests (test_learn_trace.py) fake the `synth` callable itself and
never exercise this module's own internals, so these are the first tests of `_run`'s
actual wiring: the round cap, the on_section -> trace bridge, and the persist/resume
mechanism a plan/lesson revision uses."""
from __future__ import annotations

import asyncio
import types

import pytest

from lara.learn import trace as TR
from lara.serve import learn_research as LR
from lara.serve import synthesizer as SY
from lara.serve import synthruns as SR


def run(c):
    return asyncio.run(c)


def _fake_state(**overrides):
    kw = dict(cfg=None, base_url="http://x", api_key="", model="m", window=100_000)
    kw.update(overrides)
    return types.SimpleNamespace(**kw)


def _patch_generator_and_leaf(monkeypatch, app_state):
    async def generator(state, model=None):
        assert state is app_state
        return _fake_state()
    monkeypatch.setattr(SR, "generator", generator)
    monkeypatch.setattr(SR, "leaf", lambda state, cfg: object())
    monkeypatch.setattr(app_state, "conn", lambda: None, raising=False)


@pytest.fixture(autouse=True)
def _states_dir(tmp_path, monkeypatch):
    # Every test below drives `_run` with a real course/cid, which now always persists
    # (see learn_research._state_id) -- redirected here so no test writes to the real
    # ~/.lara/synthesizer, the same convention test_synthruns.py already uses.
    monkeypatch.setattr(SY, "STATES", tmp_path / "synthesizer")


def test_topic_graph_synth_caps_idle_rounds_at_the_learn_specific_value(monkeypatch):
    app_state = types.SimpleNamespace(retriever=None)
    _patch_generator_and_leaf(monkeypatch, app_state)
    seen = {}

    async def fake_run(state, **kw):
        seen.update(kw)
        return SY.SynthesisResult(deliverable='{"concepts": []}', references={})
    monkeypatch.setattr(SY, "run", fake_run)

    run(LR.topic_graph_synth(app_state, "course-1")("map this course"))

    # Same value as SY.SUBSYNTHESIS_MAX_IDLE_ROUNDS -- an already-scoped-down objective,
    # not the wide-open standalone Synthesize default (SY.MAX_IDLE_ROUNDS).
    assert seen["max_idle_rounds"] == LR.LEARN_MAX_IDLE_ROUNDS == SY.SUBSYNTHESIS_MAX_IDLE_ROUNDS
    assert seen["max_idle_rounds"] < SY.MAX_IDLE_ROUNDS


def test_lesson_synth_also_caps_idle_rounds(monkeypatch):
    app_state = types.SimpleNamespace(retriever=None)
    _patch_generator_and_leaf(monkeypatch, app_state)
    seen = {}

    async def fake_run(state, **kw):
        seen.update(kw)
        return SY.SynthesisResult(deliverable="lesson text", references={})
    monkeypatch.setattr(SY, "run", fake_run)

    run(LR.lesson_synth(app_state, "course-1", "c1")("teach this concept"))

    assert seen["max_idle_rounds"] == LR.LEARN_MAX_IDLE_ROUNDS


def test_topic_graph_synth_passes_its_own_hard_round_cap(monkeypatch):
    app_state = types.SimpleNamespace(retriever=None)
    _patch_generator_and_leaf(monkeypatch, app_state)
    seen = {}

    async def fake_run(state, **kw):
        seen.update(kw)
        return SY.SynthesisResult(deliverable='{"concepts": []}', references={})
    monkeypatch.setattr(SY, "run", fake_run)

    run(LR.topic_graph_synth(app_state, "course-1")("map this course"))

    assert seen["max_rounds"] == LR.LEARN_MAX_ROUNDS_TOPIC_GRAPH


def test_lesson_synth_passes_its_own_smaller_hard_round_cap(monkeypatch):
    app_state = types.SimpleNamespace(retriever=None)
    _patch_generator_and_leaf(monkeypatch, app_state)
    seen = {}

    async def fake_run(state, **kw):
        seen.update(kw)
        return SY.SynthesisResult(deliverable="lesson text", references={})
    monkeypatch.setattr(SY, "run", fake_run)

    run(LR.lesson_synth(app_state, "course-1", "c1")("teach this concept"))

    assert seen["max_rounds"] == LR.LEARN_MAX_ROUNDS_LESSON
    assert LR.LEARN_MAX_ROUNDS_LESSON < LR.LEARN_MAX_ROUNDS_TOPIC_GRAPH, \
        "one lesson's objective is narrower than mapping the whole course"


def test_written_trace_event_carries_the_runs_exit_reason(tmp_path, monkeypatch):
    app_state = types.SimpleNamespace(retriever=None)
    _patch_generator_and_leaf(monkeypatch, app_state)

    async def fake_run(state, **kw):
        return SY.SynthesisResult(deliverable='{"concepts": []}', references={},
                                  rounds=5, exit_reason="max_rounds")
    monkeypatch.setattr(SY, "run", fake_run)

    path = tmp_path / "map.trace.jsonl"
    TR.start(path)
    run(LR.topic_graph_synth(app_state, "course-1")("map this course"))
    TR.stop()

    rows = [r for r in TR.read(path) if r["type"] == "topic_graph_written"]
    assert len(rows) == 1
    assert rows[0]["exit_reason"] == "max_rounds"


def test_on_section_bridges_into_deliverable_section_trace_events(tmp_path, monkeypatch):
    app_state = types.SimpleNamespace(retriever=None)
    _patch_generator_and_leaf(monkeypatch, app_state)

    async def fake_run(state, *, on_section=None, **kw):
        # Three sections landing, in order -- exactly what _write_deliverable's own
        # sequential cluster loop does; on_section must fire once per section, not once
        # per token, and must carry that section's real text.
        on_section("prior", "established earlier")
        on_section("theme a", "section text a")
        on_section("theme b", "section text b")
        return SY.SynthesisResult(deliverable='{"concepts": []}', references={})
    monkeypatch.setattr(SY, "run", fake_run)

    path = tmp_path / "map.trace.jsonl"
    TR.start(path)
    run(LR.topic_graph_synth(app_state, "course-1")("map this course"))
    TR.stop()

    rows = [r for r in TR.read(path) if r["type"] == "deliverable_section"]
    assert [r["index"] for r in rows] == [1, 2, 3]
    assert [r["label"] for r in rows] == ["prior", "theme a", "theme b"]
    assert [r["text"] for r in rows] == [
        "established earlier", "section text a", "section text b"]


def test_expand_synth_does_not_take_a_round_cap_it_has_no_graph_to_apply_it_to(
        monkeypatch):
    """expand_synth drives one research leaf (synthruns.leaf's aresearch), not a
    synthesizer.run() graph -- there is no idle-round loop here to cap."""
    app_state = types.SimpleNamespace()

    async def generator(state, model=None):
        return _fake_state()
    monkeypatch.setattr(SR, "generator", generator)

    async def aresearch(focus, *, model, base_url, api_key):
        return types.SimpleNamespace(
            thorough=types.SimpleNamespace(text="answer", references={}))
    monkeypatch.setattr(SR, "leaf", lambda state, cfg: aresearch)

    out = run(LR.expand_synth(app_state)("what does X mean"))
    assert out == {"text": "answer", "references": {}}


# ── persisting a run, and resuming it from feedback ─────────────────────────────────


def test_a_fresh_synth_run_persists_its_graph_under_the_course_id(monkeypatch):
    app_state = types.SimpleNamespace(retriever=None)
    _patch_generator_and_leaf(monkeypatch, app_state)

    async def fake_run(state, *, on_change=None, **kw):
        state.goals["g1"] = SY.LogicalGoal(id="g1", text="q", status=SY.DONE, summary="a")
        if on_change is not None:
            on_change()
        return SY.SynthesisResult(deliverable='{"concepts": []}', references={})
    monkeypatch.setattr(SY, "run", fake_run)

    run(LR.topic_graph_synth(app_state, "course-1")("map this course"))

    saved = SY.load("course:course-1")
    assert saved is not None and "g1" in saved.goals and saved.objective == "map this course"


def test_topic_graph_revise_raises_when_nothing_was_persisted(monkeypatch):
    app_state = types.SimpleNamespace(retriever=None)
    with pytest.raises(ValueError, match="no prior plan"):
        run(LR.topic_graph_revise(app_state, "never-mapped")("shorten it"))


def test_topic_graph_revise_resumes_the_persisted_graph_with_feedback_folded_in(
        monkeypatch):
    app_state = types.SimpleNamespace(retriever=None)
    _patch_generator_and_leaf(monkeypatch, app_state)
    SY.save("course:course-1", SY.SynthesizerState(
        objective="map this course",
        goals={"g1": SY.LogicalGoal(id="g1", text="q", status=SY.DONE, summary="a")},
        idle_rounds=SY.MAX_IDLE_ROUNDS, finish_streak=SY.FINISH_STREAK_TO_STOP, round=17))
    seen = {}

    async def fake_run(state, **kw):
        seen["state"] = state
        seen.update(kw)
        return SY.SynthesisResult(deliverable='{"concepts": []}', references={})
    monkeypatch.setattr(SY, "run", fake_run)

    run(LR.topic_graph_revise(app_state, "course-1")("add a concept on X"))

    resumed = seen["state"]
    # The prior graph's own research survived -- a revision resumes, it does not restart.
    assert "g1" in resumed.goals
    assert "add a concept on X" in resumed.objective
    assert "map this course" in resumed.objective
    # And it was given its own fresh budget -- otherwise synthesizer.run's own loop
    # condition (finish_streak < FINISH_STREAK_TO_STOP and idle_rounds < max_idle_rounds
    # and round < max_rounds) would already be false and never run another round.
    assert resumed.idle_rounds == 0 and resumed.finish_streak == 0 and resumed.round == 0
    assert seen["max_rounds"] == LR.LEARN_MAX_ROUNDS_TOPIC_GRAPH


def test_lesson_revise_raises_when_nothing_was_persisted(monkeypatch):
    app_state = types.SimpleNamespace(retriever=None)
    with pytest.raises(ValueError, match="no prior lesson"):
        run(LR.lesson_revise(app_state, "course-1", "c1")("go deeper"))


def test_lesson_revise_resumes_its_own_concepts_graph_not_a_different_ones(monkeypatch):
    app_state = types.SimpleNamespace(retriever=None)
    _patch_generator_and_leaf(monkeypatch, app_state)
    SY.save("lesson:course-1:c1", SY.SynthesizerState(objective="teach c1"))
    SY.save("lesson:course-1:c2", SY.SynthesizerState(objective="teach c2"))
    seen = {}

    async def fake_run(state, **kw):
        seen["state"] = state
        return SY.SynthesisResult(deliverable="lesson text", references={})
    monkeypatch.setattr(SY, "run", fake_run)

    run(LR.lesson_revise(app_state, "course-1", "c1")("go deeper on the math"))

    assert "teach c1" in seen["state"].objective
    assert "teach c2" not in seen["state"].objective
