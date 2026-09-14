"""Tests for lara.serve.synthesizer -- the goal-graph research loop.

Pure state-machine pieces (spawn/refine validation, digest formatting, id
slugging, the char budget) get direct tests, no I/O. The async driver
functions (_reason_round, _execute_round, run) are exercised with fake
complete_json/complete/run_synthesis callables driven by asyncio.run, per this
repo's convention (see pyproject.toml: no pytest-asyncio, coroutine tests
drive asyncio.run directly) -- no real model, no real retrieval.
"""

from __future__ import annotations

import asyncio
import types

from lara.serve import synthesizer as S

# ── ids ───────────────────────────────────────────────────────────────────────────

def test_slug_lowercases_and_hyphenates():
    assert S._slug("What is the Muon optimizer?", "fallback") == "what-is-the-muon-optimizer"


def test_slug_falls_back_on_empty():
    assert S._slug("", "fallback-id") == "fallback-id"
    assert S._slug("???", "fallback-id") == "fallback-id"


def test_new_id_deduplicates():
    existing = {"same-text": S.LogicalGoal(id="same-text", text="x")}
    assert S._new_id("same text", existing) == "same-text-2"


# ── spawn/refine validation ──────────────────────────────────────────────────────

def test_do_spawn_adds_a_pending_goal():
    state = S.SynthesizerState(objective="obj")
    ok, msg = S._do_spawn(state, {"text": "sub-question"})
    assert ok
    assert len(state.goals) == 1
    gid = next(iter(state.goals))
    assert state.goals[gid].status == S.PENDING
    assert state.goals[gid].depth == 0
    assert "spawned" in msg


def test_do_spawn_rejects_empty_text():
    state = S.SynthesizerState(objective="obj")
    ok, msg = S._do_spawn(state, {"text": "   "})
    assert not ok
    assert not state.goals


def test_do_spawn_rejects_unknown_dependency():
    state = S.SynthesizerState(objective="obj")
    ok, msg = S._do_spawn(state, {"text": "q", "depends_on": ["nope"]})
    assert not ok
    assert "nope" in msg
    assert not state.goals


def test_do_spawn_respects_concurrency_cap():
    state = S.SynthesizerState(objective="obj")
    for i in range(S.MAX_CONCURRENT_GOALS):
        ok, _ = S._do_spawn(state, {"text": f"q{i}"})
        assert ok
    ok, msg = S._do_spawn(state, {"text": "one too many"})
    assert not ok
    assert "cap" in msg
    assert len(state.goals) == S.MAX_CONCURRENT_GOALS


def test_do_refine_deepens_a_done_goal():
    state = S.SynthesizerState(objective="obj")
    S._do_spawn(state, {"text": "parent"})
    gid = next(iter(state.goals))
    state.goals[gid].status = S.DONE
    ok, msg = S._do_refine(state, {"parent_id": gid, "text": "deeper"})
    assert ok
    child = [g for g in state.goals.values() if g.refines == gid][0]
    assert child.depth == 1
    assert child.depends_on == [gid]


def test_do_refine_rejects_unknown_parent():
    state = S.SynthesizerState(objective="obj")
    ok, msg = S._do_refine(state, {"parent_id": "ghost", "text": "deeper"})
    assert not ok
    assert "ghost" in msg


def test_do_refine_respects_depth_cap():
    state = S.SynthesizerState(objective="obj")
    S._do_spawn(state, {"text": "root"})
    gid = next(iter(state.goals))
    state.goals[gid].status = S.DONE
    state.goals[gid].depth = S.MAX_REFINEMENT_DEPTH
    ok, msg = S._do_refine(state, {"parent_id": gid, "text": "too deep"})
    assert not ok
    assert "limit" in msg


# ── digest ────────────────────────────────────────────────────────────────────────

def test_digest_includes_objective_papers_and_answer():
    state = S.SynthesizerState(objective="the objective")
    S._do_spawn(state, {"text": "sub-question"})
    gid = next(iter(state.goals))
    state.goals[gid].status = S.DONE
    state.goals[gid].summary = "the finding"
    state.goals[gid].papers = ["1706.03762", "2005.14165"]

    digest = S._digest(state)
    assert "the objective" in digest
    assert "sub-question" in digest
    assert "the finding" in digest
    assert "1706.03762" in digest and "2005.14165" in digest


def test_digest_reports_failed_goals_with_their_error():
    state = S.SynthesizerState(objective="obj")
    S._do_spawn(state, {"text": "sub-question"})
    gid = next(iter(state.goals))
    state.goals[gid].status = S.FAILED
    state.goals[gid].error = "boom"

    digest = S._digest(state)
    assert "**Failed:** boom" in digest


def test_digest_skips_pending_and_running_goals():
    state = S.SynthesizerState(objective="obj")
    S._do_spawn(state, {"text": "still pending"})
    digest = S._digest(state)
    assert "still pending" not in digest


def test_digest_includes_prior_compressions_in_order():
    state = S.SynthesizerState(objective="obj", compressed=["first", "second"])
    digest = S._digest(state)
    assert digest.index("first") < digest.index("second")


# ── char budget ───────────────────────────────────────────────────────────────────

def test_char_budget_scales_with_model_len_and_has_a_floor():
    assert S._char_budget(0) == 8_000
    assert S._char_budget(100_000) > S._char_budget(10_000)


# ── persistence ───────────────────────────────────────────────────────────────────

def test_save_and_load_round_trip(tmp_path):
    state = S.SynthesizerState(objective="obj")
    S._do_spawn(state, {"text": "q"})
    S.save("state-1", state, root=tmp_path)

    loaded = S.load("state-1", root=tmp_path)
    assert loaded is not None
    assert loaded.objective == "obj"
    assert list(loaded.goals) == list(state.goals)


def test_load_missing_state_returns_none(tmp_path):
    assert S.load("nope", root=tmp_path) is None


# ── _reason_round (fake complete_json) ───────────────────────────────────────────

def test_reason_round_spawns_a_goal_and_resets_idle_rounds(monkeypatch):
    state = S.SynthesizerState(objective="obj")
    state.idle_rounds = 3

    async def fake_complete_json(cfg, prompt, *, system, model=None, max_tokens=None,
                                 default=None):
        return {"tool": "spawn_goal", "args": {"text": "new sub-question"}}

    monkeypatch.setattr(S, "complete_json", fake_complete_json)
    progressed = asyncio.run(S._reason_round(state, cfg=object()))
    assert progressed is True
    assert state.idle_rounds == 0
    assert len(state.goals) == 1


def test_reason_round_finish_leaves_graph_unchanged_and_counts_idle(monkeypatch):
    state = S.SynthesizerState(objective="obj")

    async def fake_complete_json(cfg, prompt, *, system, model=None, max_tokens=None,
                                 default=None):
        return {"tool": "finish", "args": {}}

    monkeypatch.setattr(S, "complete_json", fake_complete_json)
    progressed = asyncio.run(S._reason_round(state, cfg=object()))
    assert progressed is False
    assert state.idle_rounds == 1
    assert not state.goals


def test_reason_round_survives_a_call_that_raises(monkeypatch):
    state = S.SynthesizerState(objective="obj")

    async def fake_complete_json(*a, **k):
        raise RuntimeError("engine dead")

    monkeypatch.setattr(S, "complete_json", fake_complete_json)
    progressed = asyncio.run(S._reason_round(state, cfg=object()))
    assert progressed is False
    assert state.idle_rounds == 1


# ── _execute_round (fake run_synthesis) ──────────────────────────────────────────

def _fake_run(thorough="", tldr="", papers=None, stopped_because=""):
    return types.SimpleNamespace(thorough=thorough, tldr=tldr, papers=papers or [],
                                 stopped_because=stopped_because)


def test_execute_round_lands_a_done_goal():
    state = S.SynthesizerState(objective="obj")
    S._do_spawn(state, {"text": "q"})
    gid = next(iter(state.goals))

    async def fake_run_synthesis(app_state, cfg, question, *, model=None):
        return _fake_run(thorough="the answer", papers=["1706.03762"])

    asyncio.run(S._execute_round(state, run_synthesis=fake_run_synthesis,
                                 app_state=object(), cfg=object()))
    assert state.goals[gid].status == S.DONE
    assert state.goals[gid].summary == "the answer"
    assert state.goals[gid].papers == ["1706.03762"]


def test_execute_round_falls_back_to_tldr_when_thorough_is_empty():
    state = S.SynthesizerState(objective="obj")
    S._do_spawn(state, {"text": "q"})
    gid = next(iter(state.goals))

    async def fake_run_synthesis(app_state, cfg, question, *, model=None):
        return _fake_run(thorough="", tldr="short answer")

    asyncio.run(S._execute_round(state, run_synthesis=fake_run_synthesis,
                                 app_state=object(), cfg=object()))
    assert state.goals[gid].status == S.DONE
    assert state.goals[gid].summary == "short answer"


def test_execute_round_marks_consolidation_failure_as_failed():
    state = S.SynthesizerState(objective="obj")
    S._do_spawn(state, {"text": "q"})
    gid = next(iter(state.goals))

    async def fake_run_synthesis(app_state, cfg, question, *, model=None):
        return _fake_run(thorough="evidence but no answer",
                         stopped_because="consolidation failed: KeyError")

    asyncio.run(S._execute_round(state, run_synthesis=fake_run_synthesis,
                                 app_state=object(), cfg=object()))
    assert state.goals[gid].status == S.FAILED


def test_execute_round_marks_an_exception_as_failed():
    state = S.SynthesizerState(objective="obj")
    S._do_spawn(state, {"text": "q"})
    gid = next(iter(state.goals))

    async def fake_run_synthesis(app_state, cfg, question, *, model=None):
        raise RuntimeError("retrieval offline")

    asyncio.run(S._execute_round(state, run_synthesis=fake_run_synthesis,
                                 app_state=object(), cfg=object()))
    assert state.goals[gid].status == S.FAILED
    assert "retrieval offline" in state.goals[gid].error


# ── run (full loop, fully faked) ─────────────────────────────────────────────────

def test_run_spawns_once_then_finishes_and_compresses(monkeypatch):
    """Scripted reasoning: round 1 spawns a goal, every round after says
    finish -- checks the loop actually stops on idle_rounds and produces a
    non-empty deliverable from the one goal that landed."""
    calls = {"n": 0}

    async def fake_complete_json(cfg, prompt, *, system, model=None, max_tokens=None,
                                 default=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"tool": "spawn_goal", "args": {"text": "only question"}}
        return {"tool": "finish", "args": {}}

    async def fake_complete(cfg, prompt, *, system, model=None, max_tokens=None):
        return "Compressed report citing (arXiv:1706.03762)."

    async def fake_run_synthesis(app_state, cfg, question, *, model=None):
        return _fake_run(thorough="the finding", papers=["1706.03762"])

    monkeypatch.setattr(S, "complete_json", fake_complete_json)
    monkeypatch.setattr(S, "complete", fake_complete)

    state = S.SynthesizerState(objective="test objective")
    result = asyncio.run(S.run(state, app_state=object(), cfg=object(),
                               max_model_len=8192, run_synthesis=fake_run_synthesis,
                               max_idle_rounds=2))

    assert result.total_done == 1
    assert result.total_failed == 0
    assert "1706.03762" in result.deliverable
    assert result.rounds >= 2  # the spawning round plus at least one idle round


def test_run_reports_nothing_established_when_graph_stays_empty(monkeypatch):
    async def fake_complete_json(cfg, prompt, *, system, model=None, max_tokens=None,
                                 default=None):
        return {"tool": "finish", "args": {}}

    monkeypatch.setattr(S, "complete_json", fake_complete_json)

    async def unused_run_synthesis(*a, **k):
        raise AssertionError("should never be called -- nothing was ever spawned")

    state = S.SynthesizerState(objective="unreachable objective")
    result = asyncio.run(S.run(state, app_state=object(), cfg=object(),
                               max_model_len=8192, run_synthesis=unused_run_synthesis,
                               max_idle_rounds=2))
    assert result.total_done == 0
    assert "unreachable objective" in result.deliverable
