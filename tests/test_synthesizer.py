"""The synthesizer: an adaptively-grown graph of research sub-questions within one stage."""

from __future__ import annotations

import asyncio
import inspect
import types

import pytest

from lara.serve import citations as C
from lara.serve import context as CX
from lara.serve import converse
from lara.serve import synthesizer as SY


def _goal(gid, status=SY.DONE, **kw):
    kw.setdefault("text", gid)
    return SY.LogicalGoal(id=gid, status=status, **kw)


def _fake_talk_calling(name, args):
    """Same shape `test_campaign.py` uses: a `converse.talk` that calls `dispatch` once
    with a scripted tool call and returns."""
    async def talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        got = dispatch(name, args)
        if inspect.isawaitable(got):
            got = await got
        return types.SimpleNamespace(text="")
    return talk


def _fake_talk_text(text):
    async def talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        return types.SimpleNamespace(text=text)
    return talk


def _fake_talk_silent():
    async def talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        return types.SimpleNamespace(text="")
    return talk


def _research(text, refs=None, stopped_because=""):
    return types.SimpleNamespace(
        thorough=types.SimpleNamespace(text=text, references=refs or {}),
        tldr=types.SimpleNamespace(text="", references={}),
        stopped_because=stopped_because)


# ── the digest ────────────────────────────────────────────────────────────────────


def test_digest_carries_the_objective_prior_compressions_and_done_and_failed_goals():
    state = SY.SynthesizerState(
        objective="find the best optimizer",
        compressed=["earlier finding: Muon beats Adam at small batch"],
        goals={
            "a": _goal("a", status=SY.DONE, text="how does Muon scale",
                      summary="it scales linearly to 1B params"),
            "b": _goal("b", status=SY.FAILED, text="what about SOAP",
                      error="the corpus has nothing on SOAP"),
            "c": _goal("c", status=SY.PENDING, text="not yet run"),
        })
    text = SY._digest(state)
    assert "find the best optimizer" in text
    assert "Muon beats Adam at small batch" in text
    assert "it scales linearly to 1B params" in text
    assert "the corpus has nothing on SOAP" in text
    assert "not yet run" not in text, "a still-pending goal has no answer to show yet"


def test_digest_orders_the_objective_before_everything_else():
    state = SY.SynthesizerState(objective="OBJ", goals={"a": _goal("a", summary="A")})
    text = SY._digest(state)
    assert text.index("OBJ") < text.index("A")


# ── dispatch: spawn_goal / refine_goal / finish ─────────────────────────────────


def test_spawn_goal_creates_a_pending_goal_and_resets_idle_rounds():
    state = SY.SynthesizerState(objective="obj", idle_rounds=3)
    ok, msg = SY._do_spawn(state, {"text": "a new angle", "depends_on": []})
    assert ok is True
    assert len(state.goals) == 1
    goal = next(iter(state.goals.values()))
    assert goal.status == SY.PENDING and goal.depth == 0 and goal.refines is None
    assert "a new angle" in msg


def test_spawn_goal_refuses_an_unknown_dependency():
    state = SY.SynthesizerState(objective="obj")
    ok, msg = SY._do_spawn(state, {"text": "q", "depends_on": ["nope"]})
    assert ok is False
    assert "nope" in msg and not state.goals


def test_spawn_goal_refuses_empty_text():
    state = SY.SynthesizerState(objective="obj")
    ok, msg = SY._do_spawn(state, {"text": "  "})
    assert ok is False and not state.goals


def test_refine_goal_deepens_a_specific_landed_goal():
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, depth=2)})
    ok, msg = SY._do_refine(state, {"parent_id": "a", "text": "go deeper"})
    assert ok is True
    child = next(g for g in state.goals.values() if g.id != "a")
    assert child.depth == 3 and child.refines == "a" and child.depends_on == ["a"]
    assert "go deeper" in msg


def test_refine_goal_refuses_an_unknown_parent():
    state = SY.SynthesizerState(objective="obj")
    ok, msg = SY._do_refine(state, {"parent_id": "ghost", "text": "q"})
    assert ok is False and "ghost" in msg and not state.goals


def test_refine_goal_is_refused_past_the_max_refinement_depth():
    state = SY.SynthesizerState(
        objective="obj",
        goals={"a": _goal("a", status=SY.DONE, depth=SY.MAX_REFINEMENT_DEPTH)})
    ok, msg = SY._do_refine(state, {"parent_id": "a", "text": "one too many"})
    assert ok is False
    assert str(SY.MAX_REFINEMENT_DEPTH) in msg
    assert len(state.goals) == 1, "nothing was added past the cap"


def _in_flight_state(n):
    """`n` goals already `pending`/`running` (alternating, so both statuses count)."""
    statuses = [SY.PENDING, SY.RUNNING]
    goals = {f"g{i}": _goal(f"g{i}", status=statuses[i % 2]) for i in range(n)}
    return SY.SynthesizerState(objective="obj", goals=goals)


def test_spawn_goal_is_refused_at_the_concurrent_goal_cap():
    state = _in_flight_state(SY.MAX_CONCURRENT_GOALS)
    ok, msg = SY._do_spawn(state, {"text": "one too many", "depends_on": []})
    assert ok is False
    assert str(SY.MAX_CONCURRENT_GOALS) in msg
    assert len(state.goals) == SY.MAX_CONCURRENT_GOALS, "nothing was added past the cap"


def test_spawn_goal_is_allowed_again_once_an_in_flight_goal_lands():
    state = _in_flight_state(SY.MAX_CONCURRENT_GOALS - 1)
    ok, msg = SY._do_spawn(state, {"text": "still room", "depends_on": []})
    assert ok is True
    assert len(state.goals) == SY.MAX_CONCURRENT_GOALS

    # One of the in-flight goals finishes -- the cap is on concurrently in-flight goals,
    # not a lifetime total, so a new one can be spawned again.
    next(iter(state.goals.values())).status = SY.DONE
    ok, msg = SY._do_spawn(state, {"text": "room again", "depends_on": []})
    assert ok is True
    assert len(state.goals) == SY.MAX_CONCURRENT_GOALS + 1


def test_refine_goal_is_refused_at_the_concurrent_goal_cap():
    state = _in_flight_state(SY.MAX_CONCURRENT_GOALS)
    state.goals["parent"] = _goal("parent", status=SY.DONE)
    ok, msg = SY._do_refine(state, {"parent_id": "parent", "text": "one too many"})
    assert ok is False
    assert str(SY.MAX_CONCURRENT_GOALS) in msg
    assert len(state.goals) == SY.MAX_CONCURRENT_GOALS + 1, "nothing was added past the cap"


def test_refine_goal_is_allowed_again_once_an_in_flight_goal_lands():
    state = _in_flight_state(SY.MAX_CONCURRENT_GOALS - 1)
    state.goals["parent"] = _goal("parent", status=SY.DONE)
    ok, msg = SY._do_refine(state, {"parent_id": "parent", "text": "still room"})
    assert ok is True

    in_flight = [g for g in state.goals.values() if g.status in (SY.PENDING, SY.RUNNING)]
    assert len(in_flight) == SY.MAX_CONCURRENT_GOALS
    in_flight[0].status = SY.FAILED
    ok, msg = SY._do_refine(state, {"parent_id": "parent", "text": "room again"})
    assert ok is True


# ── SYNTH_SYSTEM: literature vs. empirical sub-questions ───────────────────────
#
# Confirmed on a real run (chain-synth-hparam01, a hyperparameter-tuning-procedure
# synthesis): a spawned goal asked to "extract the ACTUAL MEASURED optimal-Muon-LR-vs-WIDTH
# data points" and got back a literature-search non-result ("the evidence table does not
# contain explicit tables of...") presented with the same confidence as a real answer. The
# only leaf here (`Lara.aresearch`) is literature search, so a sub-question this graph poses
# as if digging harder would produce a number nobody has published gets an honest-sounding
# but empty answer instead of the honest "this needs a measurement" one.


def test_the_reasoning_prompt_names_the_leaf_as_literature_search_only():
    assert "literature search" in SY.SYNTH_SYSTEM
    assert "not run anything" in SY.SYNTH_SYSTEM


def test_the_reasoning_prompt_tells_the_model_to_tell_empirical_from_literature_questions():
    assert "fundamentally empirical" in SY.SYNTH_SYSTEM
    assert "nobody has" in SY.SYNTH_SYSTEM.lower() or "nobody has" in SY.SYNTH_SYSTEM


def test_the_reasoning_prompt_says_an_empirical_goal_s_own_text_must_say_so():
    assert "state plainly" in SY.SYNTH_SYSTEM
    assert "direct measurement in the asker's own setup" in SY.SYNTH_SYSTEM
    assert "what published work can bound or proxy" in SY.SYNTH_SYSTEM


def test_the_reasoning_prompt_does_not_tell_the_model_to_avoid_empirical_questions():
    # The fix is honest framing, not scope-narrowing: "go measure X" must still be a
    # welcome answer for a goal to land.
    assert "does not mean avoiding empirical" in SY.SYNTH_SYSTEM
    assert "genuinely useful" in SY.SYNTH_SYSTEM


# ── the reasoning round: dispatch + idle-round bookkeeping ─────────────────────


def test_a_spawn_call_resets_idle_rounds_to_zero(monkeypatch):
    monkeypatch.setattr(converse, "talk", _fake_talk_calling(
        SY.SPAWN_TOOL, {"text": "new angle", "depends_on": []}))
    state = SY.SynthesizerState(objective="obj", idle_rounds=4)
    progressed = asyncio.run(
        SY._reason_round(state, base_url="x", model="m", max_model_len=100_000))
    assert progressed is True
    assert state.idle_rounds == 0
    assert len(state.goals) == 1


def test_finish_or_no_call_increments_idle_rounds(monkeypatch):
    monkeypatch.setattr(converse, "talk", _fake_talk_calling(SY.FINISH_TOOL, {}))
    state = SY.SynthesizerState(objective="obj", idle_rounds=1)
    progressed = asyncio.run(
        SY._reason_round(state, base_url="x", model="m", max_model_len=100_000))
    assert progressed is False and state.idle_rounds == 2

    monkeypatch.setattr(converse, "talk", _fake_talk_silent())
    progressed = asyncio.run(
        SY._reason_round(state, base_url="x", model="m", max_model_len=100_000))
    assert progressed is False and state.idle_rounds == 3


def test_a_refused_refinement_does_not_reset_idle_rounds(monkeypatch):
    """A refusal is not progress: the cap exists precisely so a call that cannot create
    anything must not read as though it did."""
    state = SY.SynthesizerState(
        objective="obj", idle_rounds=2,
        goals={"a": _goal("a", status=SY.DONE, depth=SY.MAX_REFINEMENT_DEPTH)})
    monkeypatch.setattr(converse, "talk", _fake_talk_calling(
        SY.REFINE_TOOL, {"parent_id": "a", "text": "too deep"}))
    progressed = asyncio.run(
        SY._reason_round(state, base_url="x", model="m", max_model_len=100_000))
    assert progressed is False
    assert state.idle_rounds == 3
    assert len(state.goals) == 1, "the refused refinement was never added"


def test_reason_round_never_raises_when_the_model_call_fails(monkeypatch):
    async def broken(*a, **kw):
        raise RuntimeError("replica unreachable")
    monkeypatch.setattr(converse, "talk", broken)
    state = SY.SynthesizerState(objective="obj")
    progressed = asyncio.run(
        SY._reason_round(state, base_url="x", model="m", max_model_len=100_000))
    assert progressed is False and state.idle_rounds == 1


# ── the reasoning round: fixed for the same "succeeded with nothing" shape PR #26
# fixed in `_compress` -- thinking left on, flat 8,000-token default, a 500,000+
# character digest, so the round can burn its whole budget on hidden reasoning and come
# back with no tool call and no text. Previously not even logged.


def test_reason_round_disables_thinking(monkeypatch):
    captured = {}

    async def talk(base_url, model, messages, *, tools=None, dispatch=None, max_turns=6,
                   api_key="", tool_choice="auto", max_tokens=8_000,
                   enable_thinking=None):
        captured["enable_thinking"] = enable_thinking
        return converse.Reply(text="", tool_calls=0)
    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj")

    asyncio.run(SY._reason_round(state, base_url="x", model="m", max_model_len=100_000))

    assert captured["enable_thinking"] is False


def test_reason_round_sizes_max_tokens_against_the_real_window_not_the_flat_default(
        monkeypatch):
    captured = {}

    async def talk(base_url, model, messages, *, tools=None, dispatch=None, max_turns=6,
                   api_key="", tool_choice="auto", max_tokens=8_000,
                   enable_thinking=None):
        captured["max_tokens"] = max_tokens
        return converse.Reply(text="", tool_calls=0)
    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj")

    # A window wide enough that a flat 8,000-token default would not reflect it.
    asyncio.run(SY._reason_round(state, base_url="x", model="m", max_model_len=262_144))

    assert captured["max_tokens"] != 8_000
    assert captured["max_tokens"] <= SY.MAX_REASON_TOKENS


def test_reason_round_logs_and_counts_a_reply_with_no_tool_call_no_text_no_error(
        monkeypatch, caplog):
    """The exact PR #26 shape, replayed: `talk()` "succeeds" with empty `.text`, no tool
    call, and no `.error` -- previously indistinguishable from a genuinely idle round and
    logged nowhere at all."""
    async def talk(base_url, model, messages, *, tools=None, dispatch=None, max_turns=6,
                   api_key="", tool_choice="auto", max_tokens=8_000,
                   enable_thinking=None):
        return converse.Reply(text="", tool_calls=0, error="",
                              stopped_because="length")
    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj", round=1)

    with caplog.at_level("WARNING", logger="lara.serve.synthesizer"):
        progressed = asyncio.run(
            SY._reason_round(state, base_url="x", model="m", max_model_len=100_000))

    assert progressed is False
    assert state.silent_reason_rounds == 1
    assert any("returned nothing" in r.message for r in caplog.records)


def test_reason_round_does_not_log_or_count_a_genuine_finish_call(monkeypatch, caplog):
    """A real tool call (even `finish`, which never sets `progressed`) is not the silent
    failure this exists to catch -- only a reply with no tool call at all is."""
    async def talk(base_url, model, messages, *, tools=None, dispatch=None, max_turns=6,
                   api_key="", tool_choice="auto", max_tokens=8_000,
                   enable_thinking=None):
        dispatch(SY.FINISH_TOOL, {})
        return converse.Reply(text="", tool_calls=1, tools_used=[SY.FINISH_TOOL],
                              error="")
    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj", round=1)

    with caplog.at_level("WARNING", logger="lara.serve.synthesizer"):
        asyncio.run(
            SY._reason_round(state, base_url="x", model="m", max_model_len=100_000))

    assert state.silent_reason_rounds == 0
    assert not any("returned nothing" in r.message for r in caplog.records)


# ── executing a round's goals concurrently ──────────────────────────────────────


def test_a_round_s_pending_goals_run_concurrently_not_sequentially():
    """Both `aresearch` calls block on the same event until *both* have started. A
    sequential implementation would deadlock on the first call — nothing would ever
    schedule the second — and time out inside it, which `_run_one`'s own try/except
    turns into a `failed` goal rather than a hang; a truly concurrent one lets both
    proceed and finish `done`."""
    entered: list[str] = []
    both_entered = asyncio.Event()

    async def aresearch(question, *, model, base_url, api_key):
        entered.append(question)
        if len(entered) == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=1.0)
        return _research(f"answer to {question}")

    state = SY.SynthesizerState(objective="obj", goals={
        "a": _goal("a", status=SY.PENDING, text="q-a"),
        "b": _goal("b", status=SY.PENDING, text="q-b")})

    asyncio.run(asyncio.wait_for(
        SY._execute_round(state, aresearch=aresearch, model="m", base_url="x"),
        timeout=2.0))

    assert state.goals["a"].status == SY.DONE
    assert state.goals["b"].status == SY.DONE
    assert state.goals["a"].summary == "answer to q-a"
    assert state.goals["b"].summary == "answer to q-b"


def test_one_goal_s_run_synthesis_failure_does_not_abort_the_round():
    async def aresearch(question, *, model, base_url, api_key):
        if question == "q-bad":
            raise RuntimeError("the corpus is unreachable")
        return _research(f"answer to {question}")

    state = SY.SynthesizerState(objective="obj", goals={
        "good": _goal("good", status=SY.PENDING, text="q-good"),
        "bad": _goal("bad", status=SY.PENDING, text="q-bad")})

    asyncio.run(SY._execute_round(state, aresearch=aresearch, model="m", base_url="x"))

    assert state.goals["good"].status == SY.DONE
    assert state.goals["good"].summary == "answer to q-good"
    assert state.goals["bad"].status == SY.FAILED
    assert "the corpus is unreachable" in state.goals["bad"].error


def test_a_goal_lands_failed_when_aresearch_bakes_a_failure_into_a_normal_result(
        monkeypatch):
    """Regression: a real run's `run_synthesis` hit `EngineDeadError` during final
    consolidation, caught it internally (see `lara/serve/synthesis.py`'s own
    `except Exception as exc` around `consolidate()`), and returned a normal-looking
    `Research` with the failure narrated in `.thorough.text` — "N claims ... but writing
    the answer failed: exc" — instead of raising. `_run_one`'s `except` never fired, so
    the goal landed `done` with a failure narrated only in prose, indistinguishable from a
    real answer to anything reading `status` alone. 4 of 47 goals in that real run landed
    this way. The fix reads `research.stopped_because`, which `run_synthesis`
    deterministically suffixes with "; consolidation failed: <ExceptionType>" on exactly
    this path."""
    async def aresearch(question, *, model, base_url, api_key):
        return _research(
            "12 claims from 4 papers were gathered over 5 rounds, but writing the "
            "answer failed: EngineDeadError\n\nThe evidence is saved and the run can "
            "be reopened.",
            stopped_because="saturated: 3 new paper(s); consolidation failed: "
                            "EngineDeadError")

    state = SY.SynthesizerState(objective="obj", goals={
        "a": _goal("a", status=SY.PENDING, text="q-a")})

    asyncio.run(SY._execute_round(state, aresearch=aresearch, model="m", base_url="x"))

    goal = state.goals["a"]
    assert goal.status == SY.FAILED, (
        "a baked-in consolidation failure must not land as done")
    assert goal.summary == ""
    assert "writing the answer failed" in goal.error
    assert goal.citations == {}


def test_a_goal_with_a_genuine_answer_still_lands_done_when_stopped_because_is_benign():
    """The marker check must not misfire on an ordinary, successful stop reason."""
    async def aresearch(question, *, model, base_url, api_key):
        return _research("a real finding [1]",
                         stopped_because="saturated: 3 new paper(s) across 2 rounds")

    state = SY.SynthesizerState(objective="obj", goals={
        "a": _goal("a", status=SY.PENDING, text="q-a")})

    asyncio.run(SY._execute_round(state, aresearch=aresearch, model="m", base_url="x"))

    goal = state.goals["a"]
    assert goal.status == SY.DONE
    assert goal.summary == "a real finding [1]"


def test_a_goal_s_completion_is_persisted_immediately_not_batched():
    """`on_change` must fire the moment each goal finishes, so a mid-batch crash loses
    only what is still in flight."""
    release_slow = asyncio.Event()
    snapshots: list[int] = []

    async def aresearch(question, *, model, base_url, api_key):
        if question == "slow":
            await release_slow.wait()
        return _research(f"answer to {question}")

    def on_change():
        snapshots.append(sum(1 for g in state.goals.values()
                             if g.status in (SY.DONE, SY.FAILED)))

    state = SY.SynthesizerState(objective="obj", goals={
        "s": _goal("s", status=SY.PENDING, text="slow"),
        "f": _goal("f", status=SY.PENDING, text="fast")})

    async def go():
        task = asyncio.ensure_future(SY._execute_round(
            state, aresearch=aresearch, model="m", base_url="x", on_change=on_change))
        # Let the fast goal finish and persist while the slow one is still in flight.
        for _ in range(200):
            if state.goals["f"].status == SY.DONE:
                break
            await asyncio.sleep(0.005)
        assert state.goals["s"].status == SY.RUNNING, "the slow goal must not be done yet"
        release_slow.set()
        await task

    asyncio.run(go())
    assert 1 in snapshots, "a persist happened while only one goal had finished"
    assert snapshots[-1] == 2


def test_an_empty_batch_does_nothing():
    state = SY.SynthesizerState(objective="obj")
    asyncio.run(SY._execute_round(state, aresearch=None, model="m", base_url="x"))
    assert state.goals == {}


# ── compression: citations survive the round trip ──────────────────────────────


def test_compression_preserves_every_citation_key_still_cited_in_its_output(monkeypatch):
    ref_a = C.paper_ref(chunk_id=111, arxiv_id="1111.11111", paper_title="Paper A",
                        text="passage a", claim="claim a")
    ref_b = C.paper_ref(chunk_id=222, arxiv_id="2222.22222", paper_title="Paper B",
                        text="passage b", claim="claim b")
    state = SY.SynthesizerState(objective="obj", goals={
        "a": _goal("a", status=SY.DONE, summary="finding a [111]",
                  citations={"111": ref_a.to_dict()}),
        "b": _goal("b", status=SY.DONE, summary="finding b [222]",
                  citations={"222": ref_b.to_dict()})})

    monkeypatch.setattr(converse, "talk", _fake_talk_text(
        "A combines both findings [111, 222]."))

    summary, refs, degraded, because = asyncio.run(
        SY._compress(state, base_url="x", model="m", max_model_len=100_000))

    assert summary == "A combines both findings [111, 222]."
    assert set(refs) == {"111", "222"}
    assert refs["111"].paper_title == "Paper A"
    assert refs["222"].paper_title == "Paper B"
    assert degraded is False
    assert because == ""


def test_compression_never_invents_a_citation_the_output_did_not_keep(monkeypatch):
    ref_a = C.paper_ref(chunk_id=111, arxiv_id="1111.11111", paper_title="Paper A")
    state = SY.SynthesizerState(objective="obj", goals={
        "a": _goal("a", status=SY.DONE, summary="finding a [111]",
                  citations={"111": ref_a.to_dict()})})
    # The compression's own output dropped the citation entirely.
    monkeypatch.setattr(converse, "talk", _fake_talk_text("A finding, uncited."))

    _summary, refs, _degraded, _because = asyncio.run(
        SY._compress(state, base_url="x", model="m", max_model_len=100_000))
    assert refs == {}


def test_compression_falls_back_to_the_digest_only_after_retries_are_exhausted(monkeypatch):
    calls = []

    async def broken(*a, **kw):
        calls.append(1)
        raise RuntimeError("replica unreachable")
    monkeypatch.setattr(converse, "talk", broken)
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})
    summary, refs, degraded, because = asyncio.run(
        SY._compress(state, base_url="x", model="m", max_model_len=100_000))
    assert "finding a" in summary
    assert degraded is True
    assert because == "RuntimeError: replica unreachable"
    # Retried, not given up on the first failure — `COMPRESS_RETRIES` extra attempts
    # after the first, matching `edits.PROPOSAL_RETRIES`'s shape.
    assert len(calls) == 1 + SY.COMPRESS_RETRIES


def test_compression_retries_and_recovers_from_a_transient_failure(monkeypatch):
    """A call that fails once and then succeeds is not degraded — the retry is the fix."""
    attempts = {"n": 0}

    async def flaky(base_url, model, messages, *, max_turns=1, max_tokens=8_000,
                    api_key="", enable_thinking=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("EngineDeadError: worker crashed")
        return types.SimpleNamespace(text="a real compressed summary")
    monkeypatch.setattr(converse, "talk", flaky)
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})
    summary, _refs, degraded, because = asyncio.run(
        SY._compress(state, base_url="x", model="m", max_model_len=100_000))
    assert summary == "a real compressed summary"
    assert degraded is False
    assert because == ""
    assert attempts["n"] == 2


def test_compression_surfaces_talks_own_error_instead_of_dropping_it(monkeypatch):
    """`talk()` never raises: a real failure comes back as `Reply(text="", error=...)`.

    Reading only `.text` (the old behaviour) throws the real reason away and reports
    only that compression "failed after retries", with no way to tell a CUDA-OOM
    apart from a malformed request short of reading the model server's raw logs.
    """
    calls = []

    async def dead_replica(base_url, model, messages, *, max_turns=1, max_tokens=8_000,
                           api_key="", enable_thinking=None):
        calls.append(1)
        return converse.Reply(text="", error="CUDA out of memory: worker crashed")
    monkeypatch.setattr(converse, "talk", dead_replica)
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})

    summary, _refs, degraded, because = asyncio.run(
        SY._compress(state, base_url="x", model="m", max_model_len=100_000))

    assert degraded is True
    assert "finding a" in summary          # the fallback digest still fired
    assert because == "CUDA out of memory: worker crashed"
    assert len(calls) == 1 + SY.COMPRESS_RETRIES


def test_compression_success_is_unaffected_by_the_error_plumbing(monkeypatch):
    monkeypatch.setattr(converse, "talk", _fake_talk_text("a genuine summary"))
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})

    summary, _refs, degraded, because = asyncio.run(
        SY._compress(state, base_url="x", model="m", max_model_len=100_000))

    assert summary == "a genuine summary"
    assert degraded is False
    assert because == ""


# ── clustering: the table of contents a sectioned deliverable writes from ─────────


def test_cluster_goals_groups_a_breadth_goal_with_its_own_refinements():
    goals = {
        "a": _goal("a", status=SY.DONE, refines=None, depth=0),
        "a-1": _goal("a-1", status=SY.DONE, refines="a", depth=1),
        "b": _goal("b", status=SY.DONE, refines=None, depth=0),
    }
    clusters = SY._cluster_goals(goals)
    ids_by_cluster = [{g.id for g in c} for c in clusters]
    assert {"a", "a-1"} in ids_by_cluster
    assert {"b"} in ids_by_cluster
    assert len(clusters) == 2


def test_cluster_goals_treats_an_orphaned_refinement_as_its_own_root():
    """The parent finished, was folded into a mid-loop compaction and dropped from
    `goals`, but the child it was refined into is still live. It must become the root of
    its own cluster, not raise or vanish."""
    goals = {"child": _goal("child", status=SY.DONE, refines="ghost-parent", depth=1)}
    clusters = SY._cluster_goals(goals)
    assert [g.id for g in clusters[0]] == ["child"]


def test_cluster_goals_ignores_pending_and_running_goals():
    goals = {"a": _goal("a", status=SY.DONE), "b": _goal("b", status=SY.PENDING),
             "c": _goal("c", status=SY.RUNNING)}
    clusters = SY._cluster_goals(goals)
    assert len(clusters) == 1
    assert [g.id for g in clusters[0]] == ["a"]


def test_cluster_goals_preserves_creation_order_within_and_across_clusters():
    goals = {
        "a": _goal("a", status=SY.DONE, refines=None, depth=0),
        "b": _goal("b", status=SY.DONE, refines=None, depth=0),
        "a-1": _goal("a-1", status=SY.DONE, refines="a", depth=1),
    }
    clusters = SY._cluster_goals(goals)
    assert [g.id for g in clusters[0]] == ["a", "a-1"]
    assert [g.id for g in clusters[1]] == ["b"]


# ── stitching: organizing finished sections without ever rewriting them ───────────


def test_stitch_never_alters_a_section_s_own_text_while_still_organizing_it(monkeypatch):
    """The actual ask this was built for: transitions and ordering are fine, rewriting a
    section's own prose is not — and the tool schema makes that a structural guarantee,
    not a hoped-for one, since `order`/`intro`/`transitions`/`closing` are the only
    channel the model's output can reach the caller through."""
    async def talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        got = dispatch(SY.STITCH_TOOL, {
            "order": ["section-b", "section-a"],
            "intro": "An overview.",
            "transitions": {"section-a": "Turning to the first finding:"},
            "closing": "In short, both matter.",
        })
        if inspect.isawaitable(got):
            await got
        return types.SimpleNamespace(text="")
    monkeypatch.setattr(converse, "talk", talk)

    sections = [("section-a", "Q1", "The first section's exact finding, verbatim."),
               ("section-b", "Q2", "The second section's exact finding, verbatim.")]
    final, _tin, _tout = asyncio.run(SY._stitch(
        sections, "obj", base_url="x", model="m", max_model_len=100_000))

    assert "The first section's exact finding, verbatim." in final
    assert "The second section's exact finding, verbatim." in final
    # Reordered to b, then a, per the model's `order`.
    assert (final.index("The second section's exact finding, verbatim.")
            < final.index("The first section's exact finding, verbatim."))
    assert "An overview." in final
    assert "Turning to the first finding:" in final
    assert "In short, both matter." in final


def test_stitch_falls_back_to_every_section_in_order_when_the_call_fails(monkeypatch):
    async def broken(*a, **kw):
        raise RuntimeError("replica unreachable")
    monkeypatch.setattr(converse, "talk", broken)

    sections = [("section-a", "Q1", "finding a"), ("section-b", "Q2", "finding b")]
    final, tin, tout = asyncio.run(SY._stitch(
        sections, "obj", base_url="x", model="m", max_model_len=100_000))

    assert "finding a" in final and "finding b" in final
    assert final.index("finding a") < final.index("finding b")
    assert tin == 0 and tout == 0


def test_stitch_ships_every_section_even_when_the_model_drops_one_from_order(monkeypatch):
    async def talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        got = dispatch(SY.STITCH_TOOL, {"order": ["section-a"]})   # section-b missing
        if inspect.isawaitable(got):
            await got
        return types.SimpleNamespace(text="")
    monkeypatch.setattr(converse, "talk", talk)

    sections = [("section-a", "Q1", "finding a"), ("section-b", "Q2", "finding b")]
    final, _tin, _tout = asyncio.run(SY._stitch(
        sections, "obj", base_url="x", model="m", max_model_len=100_000))
    assert "finding a" in final and "finding b" in final


def test_stitch_forces_the_organize_report_tool_choice(monkeypatch):
    captured = {}

    async def talk(base_url, model, messages, *, tools=None, dispatch=None,
                   tool_choice=None, **kw):
        captured["tool_choice"] = tool_choice
        captured["tool_names"] = [t["function"]["name"] for t in (tools or [])]
        return types.SimpleNamespace(text="")
    monkeypatch.setattr(converse, "talk", talk)

    asyncio.run(SY._stitch([("a", "T1", "x"), ("b", "T2", "y")], "obj",
                          base_url="x", model="m", max_model_len=100_000))

    assert captured["tool_choice"] == {"type": "function",
                                       "function": {"name": SY.STITCH_TOOL}}
    assert captured["tool_names"] == [SY.STITCH_TOOL]


# ── writing the actual deliverable: sectioned, then organized ─────────────────────


def test_write_deliverable_writes_one_section_per_breadth_cluster_then_stitches(
        monkeypatch):
    calls = []

    async def talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        calls.append(bool(tools))
        if tools:
            got = dispatch(SY.STITCH_TOOL, {"order": ["section-a", "section-b"]})
            if inspect.isawaitable(got):
                await got
            return types.SimpleNamespace(text="")
        idx = sum(1 for c in calls if not c)
        return types.SimpleNamespace(text=f"section text {idx}")
    monkeypatch.setattr(converse, "talk", talk)

    state = SY.SynthesizerState(objective="obj", goals={
        "a": _goal("a", status=SY.DONE, refines=None, summary="finding a"),
        "b": _goal("b", status=SY.DONE, refines=None, summary="finding b")})

    final, _refs, degraded, _because = asyncio.run(SY._write_deliverable(
        state, base_url="x", model="m", max_model_len=100_000))

    assert calls == [False, False, True], "two independent sections, then one stitch"
    assert degraded is False
    assert "section text 1" in final and "section text 2" in final


def test_write_deliverable_skips_stitching_for_a_single_section(monkeypatch):
    calls = {"n": 0}

    async def talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        calls["n"] += 1
        return types.SimpleNamespace(text="the only section")
    monkeypatch.setattr(converse, "talk", talk)

    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})
    final, _refs, _degraded, _because = asyncio.run(SY._write_deliverable(
        state, base_url="x", model="m", max_model_len=100_000))

    assert calls["n"] == 1, "nothing to organize with only one section"
    assert final == "the only section"


def test_write_deliverable_is_degraded_when_any_section_falls_back(monkeypatch):
    async def broken(*a, **kw):
        raise RuntimeError("replica unreachable")
    monkeypatch.setattr(converse, "talk", broken)

    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})
    _final, _refs, degraded, because = asyncio.run(SY._write_deliverable(
        state, base_url="x", model="m", max_model_len=100_000))
    assert degraded is True
    assert because == "RuntimeError: replica unreachable"


def test_write_deliverable_folds_a_prior_mid_loop_compaction_into_its_own_section(
        monkeypatch):
    calls = {"n": 0}

    async def talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        calls["n"] += 1
        return types.SimpleNamespace(text="polished prior work")
    monkeypatch.setattr(converse, "talk", talk)

    state = SY.SynthesizerState(objective="obj", compressed=["an earlier compaction"])
    final, _refs, _degraded, _because = asyncio.run(SY._write_deliverable(
        state, base_url="x", model="m", max_model_len=100_000))

    assert calls["n"] == 1, "one section for the prior compaction; nothing to stitch alone"
    assert final == "polished prior work"


def test_write_deliverable_preserves_citations_from_every_cluster(monkeypatch):
    ref_a = C.paper_ref(chunk_id=111, arxiv_id="1111.11111", paper_title="Paper A")
    ref_b = C.paper_ref(chunk_id=222, arxiv_id="2222.22222", paper_title="Paper B")

    async def talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        if tools:
            got = dispatch(SY.STITCH_TOOL, {"order": ["section-a", "section-b"]})
            if inspect.isawaitable(got):
                await got
            return types.SimpleNamespace(text="")
        content = messages[-1]["content"]
        text = "finding a [111]" if "finding a" in content else "finding b [222]"
        return types.SimpleNamespace(text=text)
    monkeypatch.setattr(converse, "talk", talk)

    state = SY.SynthesizerState(objective="obj", goals={
        "a": _goal("a", status=SY.DONE, summary="finding a",
                  citations={"111": ref_a.to_dict()}),
        "b": _goal("b", status=SY.DONE, summary="finding b",
                  citations={"222": ref_b.to_dict()})})

    _final, refs, _degraded, _because = asyncio.run(SY._write_deliverable(
        state, base_url="x", model="m", max_model_len=100_000))
    assert set(refs) == {"111", "222"}


def test_write_deliverable_of_an_empty_graph_returns_nothing():
    state = SY.SynthesizerState(objective="obj")
    final, refs, degraded, because = asyncio.run(SY._write_deliverable(
        state, base_url="x", model="m", max_model_len=100_000))
    assert final == "" and refs == {} and degraded is False and because == ""


# ── the optional final-compression pass: one targeted answer over the finished report ──


def test_answer_from_deliverable_asks_only_the_question_over_the_report(monkeypatch):
    captured = {}

    async def talk(base_url, model, messages, *, max_turns=1, max_tokens=8_000,
                   api_key="", enable_thinking=None):
        captured["system"] = messages[0]["content"]
        captured["source"] = messages[1]["content"]
        return types.SimpleNamespace(text="the top 5 methods are A, B, C, D, E.")
    monkeypatch.setattr(converse, "talk", talk)

    answer, degraded, because, tin, tout = asyncio.run(SY.answer_from_deliverable(
        "a long report about optimizers [111]",
        "what are the top 5 methods mentioned here that could be implemented",
        base_url="x", model="m", max_model_len=100_000))

    assert answer == "the top 5 methods are A, B, C, D, E."
    assert degraded is False and because == ""
    assert captured["system"] == SY.FINAL_ANSWER_SYSTEM
    assert "a long report about optimizers [111]" in captured["source"]
    assert "top 5 methods" in captured["source"]


def test_answer_from_deliverable_degrades_on_a_dead_replica(monkeypatch):
    async def broken(*a, **kw):
        raise RuntimeError("replica unreachable")
    monkeypatch.setattr(converse, "talk", broken)

    answer, degraded, because, _tin, _tout = asyncio.run(SY.answer_from_deliverable(
        "a report", "a question", base_url="x", model="m", max_model_len=100_000))
    assert degraded is True
    assert because == "RuntimeError: replica unreachable"


def test_wrap_with_answer_places_the_answer_above_the_report_under_its_own_heading():
    wrapped = SY.wrap_with_answer("what matters here?", "the answer", "the full report")
    assert wrapped.startswith("## what matters here?\n\nthe answer")
    assert wrapped.endswith("## Full report\n\nthe full report")
    assert wrapped.index("the answer") < wrapped.index("the full report")


def test_strip_prior_answer_recovers_the_original_report():
    wrapped = SY.wrap_with_answer("q", "a", "the true original report")
    assert SY.strip_prior_answer(wrapped) == "the true original report"


def test_strip_prior_answer_is_a_no_op_on_a_deliverable_with_no_prior_answer():
    plain = "an ordinary deliverable with no compressed-answer header at all"
    assert SY.strip_prior_answer(plain) == plain


def test_strip_prior_answer_recovers_the_true_original_through_any_number_of_layers():
    """Compressing twice — a different prompt the second time — must answer from the
    same original report both times, not from the first answer plus the report, and not
    one layer at a time either: the true original comes back in a single call."""
    once = SY.wrap_with_answer("first question", "first answer", "the true report")
    twice = SY.wrap_with_answer("second question", "second answer", once)
    assert SY.strip_prior_answer(twice) == "the true report"


def test_run_places_the_compressed_answer_above_the_full_report(monkeypatch):
    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None,
                            **kw):
        if tools:
            return types.SimpleNamespace(text="")
        content = messages[-1]["content"]
        if content.startswith("Report:\n\n"):
            return types.SimpleNamespace(text="the short targeted answer")
        return types.SimpleNamespace(text="the full sectioned report")
    monkeypatch.setattr(converse, "talk", scripted_talk)

    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})
    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000,
        aresearch=None, max_idle_rounds=1,
        final_compression_prompt="what are the top methods here?"))

    assert result.deliverable.startswith("## what are the top methods here?")
    assert "the short targeted answer" in result.deliverable
    assert "## Full report" in result.deliverable
    assert "the full sectioned report" in result.deliverable
    # The targeted answer reads before the full report, not after it.
    assert (result.deliverable.index("the short targeted answer")
            < result.deliverable.index("the full sectioned report"))


def test_run_skips_the_compression_pass_when_no_prompt_is_given(monkeypatch):
    monkeypatch.setattr(converse, "talk", _fake_talk_text("the compressed answer"))
    state = SY.SynthesizerState(
        objective="obj", goals={"a": _goal("a", status=SY.DONE, summary="finding a")})
    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000, aresearch=None,
        max_idle_rounds=1))
    assert "## Full report" not in result.deliverable
    assert result.deliverable == "the compressed answer"


def test_run_final_compression_prompt_defaults_to_off():
    import inspect
    sig = inspect.signature(SY.run)
    assert sig.parameters["final_compression_prompt"].default == ""


def test_run_marks_the_result_degraded_when_the_compression_pass_fails(monkeypatch):
    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None,
                            **kw):
        content = messages[-1]["content"]
        if content.startswith("Report:\n\n"):
            raise RuntimeError("replica unreachable")
        return types.SimpleNamespace(text="the full report")
    monkeypatch.setattr(converse, "talk", scripted_talk)

    state = SY.SynthesizerState(
        objective="obj", goals={"a": _goal("a", status=SY.DONE, summary="finding a")})
    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000, aresearch=None,
        max_idle_rounds=1, final_compression_prompt="a question"))

    assert result.degraded is True
    assert result.deliverable.startswith("> **Written under budget pressure.**")


# ── the pressure note: budget starvation said in the deliverable, not just the verdict ─


def test_run_prefixes_a_pressure_note_when_the_final_write_degrades(monkeypatch):
    calls = {"n": 0}

    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None,
                            **kw):
        names = {t["function"]["name"] for t in (tools or [])}
        if SY.SPAWN_TOOL in names:
            i = calls["n"]
            calls["n"] += 1
            got = (dispatch(SY.SPAWN_TOOL, {"text": "the sub-question", "depends_on": []})
                  if i == 0 else "noted")
            if inspect.isawaitable(got):
                await got
            return types.SimpleNamespace(text="")
        raise RuntimeError("replica unreachable")
    monkeypatch.setattr(converse, "talk", scripted_talk)

    async def aresearch(question, *, model, base_url, api_key):
        return _research(f"answer to {question}")

    state = SY.SynthesizerState(objective="find X")
    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000,
        aresearch=aresearch, max_idle_rounds=1))

    assert result.degraded is True
    assert result.deliverable.startswith("> **Written under budget pressure.**")
    assert "replica unreachable" in result.deliverable


def test_run_deliverable_carries_no_pressure_note_on_a_clean_run(monkeypatch):
    monkeypatch.setattr(converse, "talk", _fake_talk_text("the compressed answer"))

    async def unused_aresearch(*a, **kw):
        raise AssertionError("no goal should ever have been pending")

    state = SY.SynthesizerState(
        objective="obj", goals={"a": _goal("a", status=SY.DONE, summary="finding a")})
    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000,
        aresearch=unused_aresearch, max_idle_rounds=1))

    assert result.degraded is False
    assert not result.deliverable.startswith("> **Written under budget pressure.**")


# ── compression: the model call itself, fixed for "succeeded with nothing" ────────
#
# 26/26 real compressions in one night logged "no reason reported" -- `talk()` succeeded,
# `.text` and `.error` both empty. Live evidence: this deployment's Qwen3 replicas answer
# with a `reasoning` block separate from `content`, and the old call never disabled it nor
# sized `max_tokens` to the real digest -- so reasoning about a 500,000+ character digest
# plausibly burned the flat 8,000-token default before `content` was ever written.


def test_compression_disables_thinking(monkeypatch):
    captured = {}

    async def talk(base_url, model, messages, *, max_turns=1, max_tokens=8_000,
                   api_key="", enable_thinking=None):
        captured["enable_thinking"] = enable_thinking
        return types.SimpleNamespace(text="a summary")
    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})

    asyncio.run(SY._compress(state, base_url="x", model="m", max_model_len=100_000))

    assert captured["enable_thinking"] is False


def test_compression_sizes_max_tokens_against_the_real_window_not_the_flat_default(
        monkeypatch):
    captured = {}

    async def talk(base_url, model, messages, *, max_turns=1, max_tokens=8_000,
                   api_key="", enable_thinking=None):
        captured["max_tokens"] = max_tokens
        return types.SimpleNamespace(text="a summary")
    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})

    # A window wide enough that the old flat 8,000 would have left most of it unused.
    asyncio.run(SY._compress(state, base_url="x", model="m", max_model_len=262_144))

    assert captured["max_tokens"] > 8_000
    assert captured["max_tokens"] <= SY.MAX_COMPRESS_TOKENS


def test_compression_floors_max_tokens_when_the_window_is_tiny(monkeypatch):
    captured = {}

    async def talk(base_url, model, messages, *, max_turns=1, max_tokens=8_000,
                   api_key="", enable_thinking=None):
        captured["max_tokens"] = max_tokens
        return types.SimpleNamespace(text="a summary")
    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj", goals={
        "a": _goal("a", status=SY.DONE, summary="finding a " * 500)})

    asyncio.run(SY._compress(state, base_url="x", model="m", max_model_len=1))

    # `reply_room`'s floor: a request sized at or below zero is refused outright, which
    # is worse than one that asks for something small but answerable.
    assert captured["max_tokens"] >= 2_000


def test_compression_passes_deliverable_tokens_as_the_reply_room_cap(monkeypatch):
    captured = {}
    real_reply_room = CX.reply_room

    def reply_room(window, *parts, **kw):
        captured["cap"] = kw.get("cap")
        return real_reply_room(window, *parts, **kw)
    monkeypatch.setattr(CX, "reply_room", reply_room)
    monkeypatch.setattr(converse, "talk", _fake_talk_text("a summary"))
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})

    asyncio.run(SY._compress(state, base_url="x", model="m", max_model_len=262_144,
                             deliverable_tokens=24_000))

    assert captured["cap"] == 24_000


def test_compression_without_deliverable_tokens_still_caps_at_max_compress_tokens(
        monkeypatch):
    captured = {}
    real_reply_room = CX.reply_room

    def reply_room(window, *parts, **kw):
        captured["cap"] = kw.get("cap")
        return real_reply_room(window, *parts, **kw)
    monkeypatch.setattr(CX, "reply_room", reply_room)
    monkeypatch.setattr(converse, "talk", _fake_talk_text("a summary"))
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})

    asyncio.run(SY._compress(state, base_url="x", model="m", max_model_len=262_144))

    assert captured["cap"] == SY.MAX_COMPRESS_TOKENS


def test_a_reasoning_heavy_reply_now_resolves_once_thinking_is_off(monkeypatch):
    """What this test can and cannot prove.

    It cannot simulate a real replica truncating `content` because a `reasoning` block
    ate the completion budget — that is a fact about the live server under a real prompt,
    not something a mock can honestly stand in for. What it proves: given a fake `talk`
    built to behave the way the live one was observed to (empty `.text`, no `.error`,
    whenever `enable_thinking` was not explicitly turned off), `_compress`'s call now
    resolves to real content, because it now turns it off. The live re-verification
    against a real running chain's digest is what proves the actual server-side failure
    is fixed, not this test.
    """
    async def talk(base_url, model, messages, *, max_turns=1, max_tokens=8_000,
                   api_key="", enable_thinking=None):
        if enable_thinking is not False:
            return types.SimpleNamespace(text="", error="")   # the 26/26 failure, replayed
        return types.SimpleNamespace(text="a dense real summary")
    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})

    summary, _refs, degraded, _because = asyncio.run(
        SY._compress(state, base_url="x", model="m", max_model_len=100_000))

    assert summary == "a dense real summary"
    assert degraded is False


# ── compression: a non-empty reply can still be cut off ────────────────────────────
#
# `room` is capped at `MAX_COMPRESS_TOKENS` even when the window has more to give, so a
# genuinely thorough summary can need more. Previously only `summary`'s emptiness was
# checked, so a reply cut off mid-sentence by hitting the cap (`stopped_because ==
# "length"`) still shipped as `degraded=False` -- a truncated deliverable with no signal
# that it was incomplete, and a citation bracket cut mid-way silently failing to bind.


def test_compression_marks_a_truncated_reply_degraded_even_though_it_has_text(
        monkeypatch):
    async def talk(base_url, model, messages, *, max_turns=1, max_tokens=8_000,
                   api_key="", enable_thinking=None):
        return converse.Reply(text="a summary cut off mid-sent", stopped_because="length")
    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})

    summary, _refs, degraded, because = asyncio.run(
        SY._compress(state, base_url="x", model="m", max_model_len=100_000))

    # The truncated text is still used -- better than nothing -- but the signal must
    # reflect that it is incomplete.
    assert summary == "a summary cut off mid-sent"
    assert degraded is True
    assert "cap" in because or "cut off" in because


def test_compression_does_not_mark_a_complete_reply_degraded(monkeypatch):
    async def talk(base_url, model, messages, *, max_turns=1, max_tokens=8_000,
                   api_key="", enable_thinking=None):
        # Ends on terminal punctuation -- genuinely complete, not just non-empty. See
        # `test_compression_marks_a_premature_stop_degraded` for the same `stopped_because`
        # with text that is *not* actually finished.
        return converse.Reply(text="a complete summary.", stopped_because="stop")
    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})

    summary, _refs, degraded, because = asyncio.run(
        SY._compress(state, base_url="x", model="m", max_model_len=100_000))

    assert summary == "a complete summary."
    assert degraded is False
    assert because == ""


def test_compression_marks_a_premature_stop_degraded(monkeypatch):
    """A real run (`chain-ee61afaa--synthesize-iteration-1`) shipped `success` over a

    deliverable cut off mid-word -- `stopped_because == "stop"`, not `"length"`, confirmed
    against the run's own event log (831 of a 2,000-token budget used) and this
    deployment's vLLM `/metrics` (only `"stop"`/`"length"` ever reported, never `"abort"`
    or `"error"`, across 48,680 completions). Text with no terminal punctuation and a
    voluntary stop is exactly that shape: the old `stopped_because == "length"` check
    alone cannot see it.
    """
    async def talk(base_url, model, messages, *, max_turns=1, max_tokens=8_000,
                   api_key="", enable_thinking=None):
        return converse.Reply(text="a summary that stops mid-sent", stopped_because="stop",
                              tokens_out=831)
    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})

    summary, _refs, degraded, because = asyncio.run(
        SY._compress(state, base_url="x", model="m", max_model_len=100_000))

    # The truncated text is still used -- better than nothing -- but the signal must
    # reflect that it is incomplete, same as the `"length"` case above.
    assert summary == "a summary that stops mid-sent"
    assert degraded is True
    assert "premature" in because or "stop" in because


def test_compression_retries_a_truncated_reply_instead_of_accepting_it_immediately(
        monkeypatch):
    """A reply flagged truncated (by either check above) used to break out of the retry
    loop anyway, because the loop only ever checked whether `summary` was non-empty --
    so `COMPRESS_RETRIES` never actually fired for the two failure modes it exists to
    catch, only for a fully empty reply or an exception. A retry that then comes back
    complete must be the one that is kept."""
    calls = {"n": 0}

    async def talk(base_url, model, messages, *, max_turns=1, max_tokens=8_000,
                   api_key="", enable_thinking=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return converse.Reply(text="cut off mid-sent", stopped_because="length")
        return converse.Reply(text="a complete summary.", stopped_because="stop")
    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})

    summary, _refs, degraded, because = asyncio.run(
        SY._compress(state, base_url="x", model="m", max_model_len=100_000))

    assert calls["n"] == 2
    assert summary == "a complete summary."
    assert degraded is False
    assert because == ""


def test_compression_keeps_the_last_truncated_reply_once_every_retry_stays_truncated(
        monkeypatch):
    """Every attempt truncated: the retry budget is still spent (matching the exception
    and empty-reply cases), and the last attempt's text is what is kept, not the
    first's -- the last failure is the one that actually gave up."""
    calls = {"n": 0}

    async def talk(base_url, model, messages, *, max_turns=1, max_tokens=8_000,
                   api_key="", enable_thinking=None):
        calls["n"] += 1
        return converse.Reply(text=f"cut off mid-sent (attempt {calls['n']})",
                              stopped_because="length")
    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})

    summary, _refs, degraded, because = asyncio.run(
        SY._compress(state, base_url="x", model="m", max_model_len=100_000))

    assert calls["n"] == 1 + SY.COMPRESS_RETRIES
    assert summary == f"cut off mid-sent (attempt {1 + SY.COMPRESS_RETRIES})"
    assert degraded is True
    assert "cap" in because


# ── persistence: save/load round-trip ───────────────────────────────────────────


def test_state_round_trips_through_save_and_load_including_citations(tmp_path):
    ref = C.paper_ref(chunk_id=42, arxiv_id="4242.4242", paper_title="Paper")
    state = SY.SynthesizerState(
        objective="obj", round=3, idle_rounds=2, compressed=["earlier summary"],
        goals={"a": _goal("a", status=SY.DONE, text="q", summary="a found",
                          citations={"42": ref.to_dict()}, depth=1, refines="root")})
    SY.save("chain1--stage1", state, root=tmp_path)
    back = SY.load("chain1--stage1", root=tmp_path)

    assert back is not None
    assert back.objective == "obj" and back.round == 3 and back.idle_rounds == 2
    assert back.compressed == ["earlier summary"]
    assert back.goals["a"].summary == "a found"
    assert back.goals["a"].depth == 1 and back.goals["a"].refines == "root"
    assert back.goals["a"].citations["42"]["paper_title"] == "Paper"


def test_load_of_a_missing_state_returns_none(tmp_path):
    assert SY.load("nope", root=tmp_path) is None


def test_load_of_a_missing_state_does_not_log(tmp_path, caplog):
    """A missing file is the ordinary case -- no prior stage has run yet -- and must not
    read like the corrupted-file case below."""
    with caplog.at_level("WARNING", logger="lara.serve.synthesizer"):
        assert SY.load("nope", root=tmp_path) is None
    assert not caplog.records


def test_load_of_a_corrupted_state_logs_a_warning_and_returns_none(tmp_path, caplog):
    """Unlike a missing file, a file that exists but fails to parse would otherwise
    silently discard a stage's whole prior graph with no trace -- match `save`'s own
    warning on the write side."""
    (tmp_path / "chain1--stage1.json").write_text("{not valid json")

    with caplog.at_level("WARNING", logger="lara.serve.synthesizer"):
        result = SY.load("chain1--stage1", root=tmp_path)

    assert result is None
    assert any("chain1--stage1" in r.message for r in caplog.records)


def test_save_never_raises_when_the_directory_cannot_be_written(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("i am a file, not a directory")
    # Never raises: losing the record must not stop the round.
    SY.save("s", SY.SynthesizerState(objective="o"), root=blocked)


# ── verdict ──────────────────────────────────────────────────────────────────────


def test_verdict_is_zero_engagement_when_nothing_was_ever_pursued():
    result = SY.SynthesisResult(total_done=0, total_failed=0)
    assert SY.verdict_for(result)["kind"] == SY.ZERO_ENGAGEMENT


def test_verdict_is_success_when_nothing_failed():
    result = SY.SynthesisResult(total_done=3, total_failed=0)
    assert SY.verdict_for(result)["kind"] == SY.SUCCESS


def test_verdict_is_failure_when_nothing_succeeded():
    result = SY.SynthesisResult(total_done=0, total_failed=2)
    assert SY.verdict_for(result)["kind"] == SY.FAILURE


def test_verdict_is_partial_when_some_succeeded_and_some_failed():
    result = SY.SynthesisResult(total_done=2, total_failed=1)
    assert SY.verdict_for(result)["kind"] == SY.PARTIAL


def test_verdict_downgrades_to_partial_when_every_goal_succeeded_but_compression_degraded():
    """Every goal answered would otherwise grade `success` — but if the final compression
    fell back to a raw digest slice instead of a genuine summary, reporting unqualified
    success is the exact silent degradation this exists to refuse."""
    result = SY.SynthesisResult(total_done=3, total_failed=0, degraded=True)
    verdict = SY.verdict_for(result)
    assert verdict["kind"] == SY.PARTIAL
    assert "fell back" in verdict["because"]


def test_verdict_does_not_claim_a_digest_fallback_for_a_truncated_but_real_summary():
    """A live run (`chain-llm-classifiers-for-verbal-fluency--synthesize-iteration-1`)
    answered all 48 of its logical goals and its final compression came back a real,
    coherent summary that was simply cut off mid-sentence -- not the raw-digest
    fallback, which only happens when every attempt returns nothing at all. The verdict
    nonetheless claimed "fell back to an uncompressed digest", which never happened;
    `degraded_because` here already names a truncation (`_compress`'s
    `_TRUNCATION_PREFIXES`), and the verdict must say so instead."""
    result = SY.SynthesisResult(
        total_done=48, total_failed=0, degraded=True,
        degraded_because='compression stopped ("stop") after 2000 of 2000 allotted '
                         'tokens, but the text does not end on a finished sentence — a '
                         'premature stop, not a genuine one')
    verdict = SY.verdict_for(result)
    assert verdict["kind"] == SY.PARTIAL
    assert "fell back" not in verdict["because"]
    assert "premature stop" in verdict["because"]


def test_verdict_still_reports_a_genuine_digest_fallback_as_such():
    """The other side of the fix above: a hard failure (every attempt raised, or
    returned nothing) really did fall back to the raw digest, and the verdict should
    keep saying so -- only the truncated-but-real case must stop being described that
    way."""
    result = SY.SynthesisResult(total_done=5, total_failed=0, degraded=True,
                                degraded_because="RuntimeError: EngineDeadError")
    verdict = SY.verdict_for(result)
    assert verdict["kind"] == SY.PARTIAL
    assert "fell back" in verdict["because"]
    assert "EngineDeadError" in verdict["because"]


# ── the whole loop ────────────────────────────────────────────────────────────────


def test_run_stops_after_max_idle_rounds_and_compresses_nothing_established(monkeypatch):
    monkeypatch.setattr(converse, "talk", _fake_talk_silent())
    state = SY.SynthesizerState(objective="an objective nobody engaged with")

    async def unused_aresearch(*a, **kw):
        raise AssertionError("no goal should ever have been pending")

    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000,
        aresearch=unused_aresearch, max_idle_rounds=3))

    assert result.rounds == 3
    assert result.total_done == 0 and result.total_failed == 0
    assert "an objective nobody engaged with" in result.deliverable


def test_run_spawns_executes_and_terminates(monkeypatch):
    """A model that spawns one goal on round one, then goes idle, drives the whole loop
    end to end: dispatch, concurrent execution, termination, final compression."""
    calls = {"n": 0}

    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        i = calls["n"]
        calls["n"] += 1
        if i == 0:
            got = dispatch(SY.SPAWN_TOOL, {"text": "the sub-question", "depends_on": []})
        else:
            got = "noted"
        if inspect.isawaitable(got):
            await got
        return types.SimpleNamespace(text="the compressed answer [999]")

    monkeypatch.setattr(converse, "talk", scripted_talk)

    async def aresearch(question, *, model, base_url, api_key):
        ref = C.paper_ref(chunk_id=999, arxiv_id="9999.9999", paper_title="P")
        return _research(f"answer to {question} [999]", refs={"999": ref})

    state = SY.SynthesizerState(objective="find X")
    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000,
        aresearch=aresearch, max_idle_rounds=2))

    assert result.total_done == 1 and result.total_failed == 0
    assert "999" in result.references


def test_run_never_exceeds_the_concurrent_goal_cap_across_rounds(monkeypatch):
    """The actual regression this fix targets: a round's tool-calling conversation can
    call spawn_goal several times before yielding control back (bounded only by
    MAX_TURNS, not by how many goals are already in flight), and `_execute_round` then
    fans all of those out concurrently. Script a model that keeps trying to spawn more
    goals than the cap allows every round, and confirm the in-flight count -- goals whose
    status is pending/running -- never exceeds `MAX_CONCURRENT_GOALS` at any point across
    the whole run, while the run still completes and gets through more goals than the cap
    by spreading them across rounds."""
    attempted = [f"sub-question {i}" for i in range(12)]

    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None,
                            **kw):
        if dispatch is None:
            return types.SimpleNamespace(text="the compressed answer")
        # One reasoning round tries several spawn_goal calls in a row, same as a real
        # tool-calling conversation would before MAX_TURNS runs out.
        for _ in range(4):
            if not attempted:
                break
            got = dispatch(SY.SPAWN_TOOL, {"text": attempted.pop(0), "depends_on": []})
            if inspect.isawaitable(got):
                await got
        return types.SimpleNamespace(text="")

    monkeypatch.setattr(converse, "talk", scripted_talk)

    async def aresearch(question, *, model, base_url, api_key):
        return _research(f"answer to {question}")

    state = SY.SynthesizerState(objective="find X")
    max_in_flight = {"n": 0}

    def on_change():
        max_in_flight["n"] = max(max_in_flight["n"], SY._in_flight_count(state))

    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000,
        aresearch=aresearch, on_change=on_change, max_idle_rounds=5))

    assert max_in_flight["n"] <= SY.MAX_CONCURRENT_GOALS
    # The cap throttled spawning (12 attempted, 4 per round with only 3 landing each
    # round) but did not wedge the run -- goals still landed across several rounds.
    assert result.total_done == 9 and result.total_failed == 0
    assert result.rounds >= 3


def test_run_threads_deliverable_tokens_through_to_the_final_compression(monkeypatch):
    captured = {}
    real_reply_room = CX.reply_room

    def reply_room(window, *parts, **kw):
        captured["cap"] = kw.get("cap")
        return real_reply_room(window, *parts, **kw)
    monkeypatch.setattr(CX, "reply_room", reply_room)

    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        return types.SimpleNamespace(text="the compressed answer")
    monkeypatch.setattr(converse, "talk", scripted_talk)

    async def unused_aresearch(*a, **kw):
        raise AssertionError("no goal should ever have been pending")

    state = SY.SynthesizerState(objective="obj",
                                goals={"a": _goal("a", status=SY.DONE, summary="finding a")})
    asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000,
        aresearch=unused_aresearch, max_idle_rounds=1, deliverable_tokens=24_000))

    assert captured["cap"] == 24_000


def test_run_reports_partial_not_success_when_the_final_compression_call_fails(
        monkeypatch):
    """Every goal answers, but the final compression call itself never recovers even
    after retries. The run must not silently claim `success` over a deliverable that is
    actually the raw-digest fallback — `SynthesisResult.degraded` carries that, and
    `verdict_for` grades the run `partial`, naming the fallback in `because`."""

    calls = {"reason": 0, "compress": 0}

    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None,
                            **kw):
        if dispatch is not None:
            i = calls["reason"]
            calls["reason"] += 1
            got = (dispatch(SY.SPAWN_TOOL, {"text": "the sub-question", "depends_on": []})
                  if i == 0 else "noted")
            if inspect.isawaitable(got):
                await got
            return types.SimpleNamespace(text="")
        # The compression call — no `dispatch` is ever passed to it (see `_compress`).
        calls["compress"] += 1
        raise RuntimeError("EngineDeadError: worker crashed")

    monkeypatch.setattr(converse, "talk", scripted_talk)

    async def aresearch(question, *, model, base_url, api_key):
        return _research(f"answer to {question}")

    state = SY.SynthesizerState(objective="find X")
    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000,
        aresearch=aresearch, max_idle_rounds=1))

    assert result.total_done == 1 and result.total_failed == 0
    assert result.degraded is True
    assert result.degraded_because == "RuntimeError: EngineDeadError: worker crashed"
    # The final compression retried, not gave up on the first failure.
    assert calls["compress"] == 1 + SY.COMPRESS_RETRIES

    verdict = SY.verdict_for(result)
    assert verdict["kind"] == SY.PARTIAL
    assert "fell back" in verdict["because"]
    assert "EngineDeadError" in verdict["because"]


def test_run_surfaces_talks_own_error_reason_not_just_that_it_failed(monkeypatch):
    """The production shape: `talk()` never raises, it returns `Reply(error=...)`. That
    real reason must reach `SynthesisResult.degraded_because` and `verdict_for`'s note,
    not just a bare "failed after retries" with no way to tell what actually happened."""
    calls = {"reason": 0}

    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None,
                            **kw):
        if dispatch is not None:
            i = calls["reason"]
            calls["reason"] += 1
            got = (dispatch(SY.SPAWN_TOOL, {"text": "the sub-question", "depends_on": []})
                  if i == 0 else "noted")
            if inspect.isawaitable(got):
                await got
            return types.SimpleNamespace(text="")
        return converse.Reply(text="", error="CUDA out of memory: worker crashed")

    monkeypatch.setattr(converse, "talk", scripted_talk)

    async def aresearch(question, *, model, base_url, api_key):
        return _research(f"answer to {question}")

    state = SY.SynthesizerState(objective="find X")
    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000,
        aresearch=aresearch, max_idle_rounds=1))

    assert result.degraded is True
    assert result.degraded_because == "CUDA out of memory: worker crashed"
    verdict = SY.verdict_for(result)
    assert "CUDA out of memory" in verdict["because"]


def test_run_persists_after_every_change(monkeypatch):
    changes = []

    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        got = dispatch(SY.SPAWN_TOOL, {"text": "q", "depends_on": []}) \
            if not changes else "noted"
        if inspect.isawaitable(got):
            await got
        return types.SimpleNamespace(text="final [1]")

    monkeypatch.setattr(converse, "talk", scripted_talk)

    async def aresearch(question, *, model, base_url, api_key):
        ref = C.paper_ref(chunk_id=1, arxiv_id="1.1", paper_title="P")
        return _research("found [1]", refs={"1": ref})

    state = SY.SynthesizerState(objective="obj")
    asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000, aresearch=aresearch,
        on_change=lambda: changes.append(state.round), max_idle_rounds=2))
    assert changes, "on_change was never called"


# ── context-budget-triggered compression, inside the whole loop ────────────────


def test_a_tiny_budget_triggers_compression_mid_run(monkeypatch):
    calls = {"n": 0}

    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        i = calls["n"]
        calls["n"] += 1
        if i == 0:
            got = dispatch(SY.SPAWN_TOOL, {"text": "a long research question", "depends_on": []})
            if inspect.isawaitable(got):
                await got
            return types.SimpleNamespace(text="")
        # Every later call is either the compression call or an idle round; text answers
        # both shapes plausibly.
        return types.SimpleNamespace(text="compressed [1]")

    monkeypatch.setattr(converse, "talk", scripted_talk)

    async def aresearch(question, *, model, base_url, api_key):
        ref = C.paper_ref(chunk_id=1, arxiv_id="1.1", paper_title="P")
        # `CX.budget_for` never returns less than its 8,000-character floor regardless
        # of `max_model_len` — this has to genuinely outgrow that floor, not merely rely
        # on a tiny window, to actually exercise the compression path.
        return _research("a very long finding " * 500 + "[1]", refs={"1": ref})

    state = SY.SynthesizerState(objective="obj")
    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=1, aresearch=aresearch,
        max_idle_rounds=2))
    # A `max_model_len` of 1 forces `CX.budget_for` to its floor (8,000 characters),
    # which the long finding above already exceeds — compression must have fired
    # before the loop ended, so no `done` goal from round one is still sitting in
    # `state.goals` uncompressed.
    assert all(g.status in (SY.PENDING, SY.RUNNING) for g in state.goals.values())
    assert result.total_done == 1


def test_unbounded_run_still_compresses_mid_loop_and_continues(monkeypatch):
    """Regression guard: with no `deliverable_tokens`, the old mid-loop
    compress-and-continue path must fire exactly as it did before this change — a
    mid-loop `_compress()` plus a distinct final one, not one final-only call."""
    calls = {"reason": 0, "compress": 0}

    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        if dispatch is not None:
            i = calls["reason"]
            calls["reason"] += 1
            got = (dispatch(SY.SPAWN_TOOL, {"text": "a long research question",
                                            "depends_on": []})
                  if i == 0 else "noted")
            if inspect.isawaitable(got):
                await got
            return types.SimpleNamespace(text="")
        calls["compress"] += 1
        return types.SimpleNamespace(text="compressed [1]")

    monkeypatch.setattr(converse, "talk", scripted_talk)

    async def aresearch(question, *, model, base_url, api_key):
        ref = C.paper_ref(chunk_id=1, arxiv_id="1.1", paper_title="P")
        return _research("a very long finding " * 500 + "[1]", refs={"1": ref})

    state = SY.SynthesizerState(objective="obj")
    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=1, aresearch=aresearch,
        max_idle_rounds=2))

    assert calls["compress"] >= 2, "mid-loop compress plus a distinct final compress"
    assert result.total_done == 1


def test_deliverable_threshold_is_the_reserve_not_half_the_window():
    # 100,000 tokens window, 24,000 reserved -> 76,000 tokens of headroom, not half the
    # window and not `WINDOW_FRACTION`-scaled.
    assert SY._deliverable_threshold(100_000, 24_000) == int(76_000 * CX.CHARS_PER_TOKEN)


def test_deliverable_threshold_floors_at_eight_thousand_characters():
    # `deliverable_tokens` configured almost equal to `max_model_len` must not produce a
    # degenerate near-zero threshold -- same floor `CX.budget_for` itself uses.
    assert SY._deliverable_threshold(100_000, 99_995) == 8_000


def test_run_stops_without_mid_loop_compress_once_the_bounded_threshold_is_crossed(
        monkeypatch):
    """`deliverable_tokens` bounded mode: once the digest crosses
    `context_tokens - deliverable_tokens`, `run()` must stop spawning/refining and fall
    through to the single final `_compress()` -- never a mid-loop compress-and-continue."""
    calls = {"reason": 0, "compress": 0}

    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        if dispatch is not None:
            i = calls["reason"]
            calls["reason"] += 1
            got = (dispatch(SY.SPAWN_TOOL, {"text": "a question", "depends_on": []})
                  if i == 0 else "noted")
            if inspect.isawaitable(got):
                await got
            return types.SimpleNamespace(text="")
        calls["compress"] += 1
        return types.SimpleNamespace(text="the compressed answer")

    monkeypatch.setattr(converse, "talk", scripted_talk)

    async def aresearch(question, *, model, base_url, api_key):
        # Long enough to push the digest past the bounded threshold's 8,000-character
        # floor (`max_model_len - deliverable_tokens` == 5 tokens here) in one round.
        return _research("a very long finding " * 500 + "[1]")

    state = SY.SynthesizerState(objective="obj")
    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000, aresearch=aresearch,
        max_idle_rounds=5, deliverable_tokens=99_995))

    assert calls["reason"] == 1, "the loop must stop before a second reasoning round"
    assert calls["compress"] == 1, "exactly one compress call, at the end"
    assert result.total_done == 1


def test_reason_round_adds_the_bounded_context_nudge_when_deliverable_tokens_is_set(
        monkeypatch):
    captured = {}

    async def talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        captured["system"] = messages[0]["content"]
        return types.SimpleNamespace(text="")

    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj")
    asyncio.run(SY._reason_round(state, base_url="x", model="m", max_model_len=100_000,
                                 deliverable_tokens=24_000))
    assert SY.BOUNDED_CONTEXT_NUDGE.strip() in captured["system"]


def test_reason_round_omits_the_bounded_context_nudge_when_deliverable_tokens_is_none(
        monkeypatch):
    captured = {}

    async def talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        captured["system"] = messages[0]["content"]
        return types.SimpleNamespace(text="")

    monkeypatch.setattr(converse, "talk", talk)
    state = SY.SynthesizerState(objective="obj")
    asyncio.run(SY._reason_round(state, base_url="x", model="m", max_model_len=100_000))
    assert SY.BOUNDED_CONTEXT_NUDGE.strip() not in captured["system"]
    assert captured["system"] == SY.SYNTH_SYSTEM

def test_synthesis_no_longer_offers_experiment_proposals():
    """Synthesis is research over sources only: no tool, and no prompt guidance, for
    proposing experiments."""
    for allow in (False, True):
        names = {t["function"]["name"] for t in SY.synth_tools(allow_subsynthesis=allow)}
        assert not any("experiment" in n for n in names)
    assert "propose_experiment" not in SY.SYNTH_SYSTEM


def test_a_saved_state_from_before_the_removal_still_loads(tmp_path):
    """32 saved graphs carry an `experiments` list; loading one must ignore it, not fail,
    and saving it back must drop it."""
    old = {"objective": "obj", "experiments": [{"title": "old proposal"}], "round": 3}
    state = SY.SynthesizerState.from_dict(old)
    assert state.objective == "obj" and state.round == 3
    assert "experiments" not in state.to_dict()


def test_synth_tools_includes_retrieve_facts():
    names = {t["function"]["name"] for t in SY.synth_tools()}
    assert SY.RETRIEVE_FACTS_TOOL in names
    tool = next(t["function"] for t in SY.synth_tools()
               if t["function"]["name"] == SY.RETRIEVE_FACTS_TOOL)
    assert tool["parameters"]["required"] == ["query"]


def test_retrieve_facts_tool_is_dispatched_with_the_injected_embedder(tmp_path, monkeypatch):
    from lara.serve import facts as FA

    monkeypatch.setattr(FA, "FACTS_STORE", tmp_path / "facts.jsonl")
    monkeypatch.setattr(converse, "talk", _fake_talk_calling(
        SY.RETRIEVE_FACTS_TOOL, {"query": "retrieval architecture"}))
    FA._append_jsonl(FA.FACTS_STORE, {"id": "a", "run_id": "r1", "goal": "past goal",
                                      "fact": "a stored fact", "tag": "design tradeoff",
                                      "embedding": [1.0, 0.0]})
    state = SY.SynthesizerState(objective="obj")
    # No assertion beyond "did not raise" -- dispatch's return value is what a real
    # model sees, covered directly below via `_do_retrieve_facts`.
    asyncio.run(SY._reason_round(
        state, base_url="x", model="m", max_model_len=100_000,
        embed=lambda text: [1.0, 0.0]))


def test_do_retrieve_facts_returns_matching_facts_formatted(tmp_path, monkeypatch):
    from lara.serve import facts as FA

    monkeypatch.setattr(FA, "FACTS_STORE", tmp_path / "facts.jsonl")
    FA._append_jsonl(FA.FACTS_STORE, {"id": "a", "run_id": "r1", "goal": "past goal",
                                      "fact": "a stored fact", "tag": "design tradeoff",
                                      "embedding": [1.0, 0.0]})
    result = SY._do_retrieve_facts(lambda text: [1.0, 0.0], {"query": "anything"})
    assert "design tradeoff" in result
    assert "a stored fact" in result


def test_do_retrieve_facts_without_an_embed_callable_reports_unavailable():
    assert "unavailable" in SY._do_retrieve_facts(None, {"query": "anything"})


def test_do_retrieve_facts_without_a_query_asks_for_one():
    assert "no query" in SY._do_retrieve_facts(lambda t: [1.0], {})

# ── dispatch: spawn_subsynthesis — answering a goal via a whole nested synthesis ───


def test_synth_tools_omits_spawn_subsynthesis_by_default():
    """The depth-1 limit's other half (`_execute_round`'s nested `run()` call always
    passes `allow_subsynthesis=False`) is enforced *here*: a run never offered the tool
    cannot call it, no matter what it tries."""
    names = {t["function"]["name"] for t in SY.synth_tools()}
    assert SY.SPAWN_SUBSYNTHESIS_TOOL not in names


def test_synth_tools_includes_spawn_subsynthesis_when_allowed():
    names = {t["function"]["name"] for t in SY.synth_tools(allow_subsynthesis=True)}
    assert SY.SPAWN_SUBSYNTHESIS_TOOL in names


def test_reason_round_omits_the_subsynthesis_nudge_by_default(monkeypatch):
    captured = {}

    async def talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        captured["system"] = messages[0]["content"]
        captured["tool_names"] = {t["function"]["name"] for t in (tools or [])}
        return types.SimpleNamespace(text="")
    monkeypatch.setattr(converse, "talk", talk)

    state = SY.SynthesizerState(objective="obj")
    asyncio.run(SY._reason_round(state, base_url="x", model="m", max_model_len=100_000))

    assert SY.SUBSYNTHESIS_NUDGE.strip() not in captured["system"]
    assert SY.SPAWN_SUBSYNTHESIS_TOOL not in captured["tool_names"]


def test_reason_round_adds_the_subsynthesis_nudge_when_allowed(monkeypatch):
    captured = {}

    async def talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        captured["system"] = messages[0]["content"]
        captured["tool_names"] = {t["function"]["name"] for t in (tools or [])}
        return types.SimpleNamespace(text="")
    monkeypatch.setattr(converse, "talk", talk)

    state = SY.SynthesizerState(objective="obj")
    asyncio.run(SY._reason_round(state, base_url="x", model="m", max_model_len=100_000,
                                 allow_subsynthesis=True))

    assert SY.SUBSYNTHESIS_NUDGE.strip() in captured["system"]
    assert SY.SPAWN_SUBSYNTHESIS_TOOL in captured["tool_names"]


def test_do_spawn_subsynthesis_creates_a_nested_pending_goal():
    state = SY.SynthesizerState(objective="obj")
    ok, msg = SY._do_spawn_subsynthesis(state, {"text": "the sub-objective"})
    assert ok is True
    assert "spawned" in msg
    (goal,) = state.goals.values()
    assert goal.nested is True
    assert goal.status == SY.PENDING
    assert goal.text == "the sub-objective"
    assert state.subsyntheses_spawned == 1


def test_do_spawn_subsynthesis_refuses_empty_text():
    state = SY.SynthesizerState(objective="obj")
    ok, msg = SY._do_spawn_subsynthesis(state, {"text": "  "})
    assert ok is False
    assert not state.goals
    assert state.subsyntheses_spawned == 0


def test_do_spawn_subsynthesis_refuses_an_unknown_dependency():
    state = SY.SynthesizerState(objective="obj")
    ok, msg = SY._do_spawn_subsynthesis(state, {"text": "q", "depends_on": ["ghost"]})
    assert ok is False
    assert "ghost" in msg
    assert not state.goals


def test_do_spawn_subsynthesis_enforces_its_own_cap():
    state = SY.SynthesizerState(objective="obj")
    for i in range(SY.MAX_SUBSYNTHESES_PER_SYNTH):
        ok, _msg = SY._do_spawn_subsynthesis(state, {"text": f"q{i}"})
        assert ok is True
        # Land each one before spawning the next, so the in-flight cap (shared with
        # spawn_goal) never masks the subsynthesis-specific one being tested here.
        list(state.goals.values())[-1].status = SY.DONE

    ok, msg = SY._do_spawn_subsynthesis(state, {"text": "one too many"})
    assert ok is False
    assert str(SY.MAX_SUBSYNTHESES_PER_SYNTH) in msg
    assert len(state.goals) == SY.MAX_SUBSYNTHESES_PER_SYNTH


def test_do_spawn_subsynthesis_respects_the_concurrent_goal_cap():
    state = SY.SynthesizerState(objective="obj", goals={
        f"g{i}": _goal(f"g{i}", status=SY.PENDING) for i in range(SY.MAX_CONCURRENT_GOALS)})
    ok, msg = SY._do_spawn_subsynthesis(state, {"text": "q"})
    assert ok is False
    assert str(SY.MAX_CONCURRENT_GOALS) in msg


def test_spawn_subsynthesis_tool_is_dispatched_from_a_reasoning_round_when_allowed(
        monkeypatch):
    monkeypatch.setattr(converse, "talk", _fake_talk_calling(
        SY.SPAWN_SUBSYNTHESIS_TOOL, {"text": "the sub-objective"}))
    state = SY.SynthesizerState(objective="obj")
    asyncio.run(SY._reason_round(state, base_url="x", model="m", max_model_len=100_000,
                                 allow_subsynthesis=True))
    (goal,) = state.goals.values()
    assert goal.nested is True


def test_spawn_subsynthesis_is_refused_as_unknown_when_not_allowed(monkeypatch):
    """The dispatch-level half of the depth-1 guard: even a call that somehow reaches
    this name (a stale client, a model that ignores the offered tool list) is refused
    rather than honored, matching every other unknown-tool call."""
    monkeypatch.setattr(converse, "talk", _fake_talk_calling(
        SY.SPAWN_SUBSYNTHESIS_TOOL, {"text": "the sub-objective"}))
    state = SY.SynthesizerState(objective="obj")
    asyncio.run(SY._reason_round(state, base_url="x", model="m", max_model_len=100_000))
    assert not state.goals


def test_render_goal_names_a_nested_answer_s_provenance():
    goal = _goal("g", status=SY.DONE, summary="finding", nested=True)
    assert "via nested synthesis" in SY._render_goal(goal)


def test_render_goal_says_nothing_extra_for_an_ordinary_goal():
    goal = _goal("g", status=SY.DONE, summary="finding", nested=False)
    assert "via nested synthesis" not in SY._render_goal(goal)


def test_run_answers_a_subsynthesis_goal_by_recursing_and_never_offers_it_a_second_level(
        monkeypatch):
    """The actual end-to-end shape: a top-level round spawns a sub-synthesis, the nested
    `run()` it triggers spawns and answers one ordinary goal of its own, and the nested
    deliverable and citations land on the parent goal exactly as an ordinary answer
    would. `nested_tool_names` is the depth-1 guard's own witness: the nested run's
    reasoning rounds are never even offered `spawn_subsynthesis`.
    """
    ref = C.paper_ref(chunk_id=1, arxiv_id="1.1", paper_title="P")
    calls = {"top": 0, "nested": 0}
    nested_tool_names: list[set] = []

    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None,
                            **kw):
        names = {t["function"]["name"] for t in (tools or [])}
        if SY.SPAWN_SUBSYNTHESIS_TOOL in names:
            i = calls["top"]
            calls["top"] += 1
            if i == 0:
                got = dispatch(SY.SPAWN_SUBSYNTHESIS_TOOL,
                              {"text": "nested objective", "depends_on": []})
                if inspect.isawaitable(got):
                    await got
            return types.SimpleNamespace(text="")
        if SY.SPAWN_TOOL in names:
            nested_tool_names.append(names)
            j = calls["nested"]
            calls["nested"] += 1
            if j == 0:
                got = dispatch(SY.SPAWN_TOOL,
                              {"text": "inner sub-question", "depends_on": []})
                if inspect.isawaitable(got):
                    await got
            return types.SimpleNamespace(text="")
        # A section/compress write, nested or top-level — same fake for both, since
        # both are exercised here and neither needs to be told apart for this test.
        return types.SimpleNamespace(text="nested finding [1]")
    monkeypatch.setattr(converse, "talk", scripted_talk)

    async def aresearch(question, *, model, base_url, api_key):
        return _research(f"answer to {question} [1]", refs={"1": ref})

    state = SY.SynthesizerState(objective="top objective")
    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000, aresearch=aresearch,
        max_idle_rounds=2, allow_subsynthesis=True))

    (parent_goal,) = state.goals.values()
    assert parent_goal.nested is True
    assert parent_goal.status == SY.DONE
    assert "nested finding" in parent_goal.summary
    assert "1" in parent_goal.citations
    assert result.total_done == 1
    assert "1" in result.references
    assert nested_tool_names, "the nested run never actually reasoned"
    assert all(SY.SPAWN_SUBSYNTHESIS_TOOL not in names for names in nested_tool_names)


def test_run_marks_a_subsynthesis_goal_failed_when_the_nested_run_establishes_nothing(
        monkeypatch):
    calls = {"top": 0}

    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None,
                            **kw):
        names = {t["function"]["name"] for t in (tools or [])}
        if SY.SPAWN_SUBSYNTHESIS_TOOL in names:
            i = calls["top"]
            calls["top"] += 1
            if i == 0:
                got = dispatch(SY.SPAWN_SUBSYNTHESIS_TOOL,
                              {"text": "nested objective", "depends_on": []})
                if inspect.isawaitable(got):
                    await got
            return types.SimpleNamespace(text="")
        # The nested run's own reasoning rounds: never spawn anything at all.
        return types.SimpleNamespace(text="")
    monkeypatch.setattr(converse, "talk", scripted_talk)

    async def unused_aresearch(*a, **kw):
        raise AssertionError("the nested graph never spawned a goal to research")

    state = SY.SynthesizerState(objective="top objective")
    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000,
        aresearch=unused_aresearch, max_idle_rounds=1, allow_subsynthesis=True))

    (parent_goal,) = state.goals.values()
    assert parent_goal.status == SY.FAILED
    assert "nested synthesis established nothing" in parent_goal.error
    assert result.total_failed == 1 and result.total_done == 0


# ── token accounting: the synthesizer's own reasoning-round and compression calls ──


def test_reason_round_accumulates_a_reply_s_tokens_into_state(monkeypatch):
    async def talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        return types.SimpleNamespace(text="", tokens_in=120, tokens_out=45)
    monkeypatch.setattr(converse, "talk", talk)

    state = SY.SynthesizerState(objective="obj")
    asyncio.run(SY._reason_round(state, base_url="x", model="m", max_model_len=100_000))
    assert state.tokens_in == 120 and state.tokens_out == 45

    # A second round adds, it does not replace.
    asyncio.run(SY._reason_round(state, base_url="x", model="m", max_model_len=100_000))
    assert state.tokens_in == 240 and state.tokens_out == 90


def test_compress_accumulates_a_reply_s_tokens_into_state(monkeypatch):
    async def talk(base_url, model, messages, *, max_turns=1, max_tokens=8_000, **kw):
        # Ends on terminal punctuation -- genuinely complete, so this resolves in one
        # call rather than retrying; see the premature-stop tests for that case.
        return types.SimpleNamespace(text="a summary.", tokens_in=300, tokens_out=80,
                                     error="", stopped_because="stop")
    monkeypatch.setattr(converse, "talk", talk)

    state = SY.SynthesizerState(
        objective="obj", goals={"a": _goal("a", status=SY.DONE, summary="finding a")})
    asyncio.run(SY._compress(state, base_url="x", model="m", max_model_len=100_000))
    assert state.tokens_in == 300 and state.tokens_out == 80


def test_run_accumulates_tokens_across_rounds_and_final_compression(monkeypatch):
    """The whole loop: several reasoning-round calls plus one final compression call,
    each carrying its own `reply.tokens_in`/`tokens_out` -- the running total on `state`
    (and the copy `run()` puts on its `SynthesisResult`) must be the sum of every one of
    them, not just the last call's numbers left over from a totals bug."""
    sent_in: list[int] = []
    sent_out: list[int] = []
    calls = {"n": 0}

    async def scripted_talk(base_url, model, messages, *, tools=None, dispatch=None, **kw):
        i = calls["n"]
        calls["n"] += 1
        if dispatch is not None:
            if i == 0:
                got = dispatch(SY.SPAWN_TOOL,
                               {"text": "the sub-question", "depends_on": []})
                if inspect.isawaitable(got):
                    await got
            reply = types.SimpleNamespace(text="", tokens_in=40 + i, tokens_out=5 + i)
        else:
            # The final compression call: `_compress` never passes `dispatch`.
            reply = types.SimpleNamespace(text="the compressed answer [999]",
                                          tokens_in=300, tokens_out=80)
        sent_in.append(reply.tokens_in)
        sent_out.append(reply.tokens_out)
        return reply

    monkeypatch.setattr(converse, "talk", scripted_talk)

    async def aresearch(question, *, model, base_url, api_key):
        ref = C.paper_ref(chunk_id=999, arxiv_id="9999.9999", paper_title="P")
        return _research(f"answer to {question} [999]", refs={"999": ref})

    state = SY.SynthesizerState(objective="find X")
    result = asyncio.run(SY.run(
        state, base_url="x", model="m", max_model_len=100_000,
        aresearch=aresearch, max_idle_rounds=2))

    assert len(sent_in) >= 2, "the script never exercised more than one talk() call"
    assert state.tokens_in == sum(sent_in)
    assert state.tokens_out == sum(sent_out)
    assert result.tokens_in == state.tokens_in
    assert result.tokens_out == state.tokens_out


def test_state_tokens_round_trip_through_to_dict_and_from_dict():
    state = SY.SynthesizerState(objective="obj", tokens_in=123, tokens_out=45)
    back = SY.SynthesizerState.from_dict(state.to_dict())
    assert back.tokens_in == 123 and back.tokens_out == 45


def test_state_tokens_round_trip_through_save_and_load(tmp_path):
    """Must survive a server restart mid-run, same as every other field here."""
    state = SY.SynthesizerState(objective="obj", tokens_in=123, tokens_out=45)
    SY.save("chain1--stage1", state, root=tmp_path)
    back = SY.load("chain1--stage1", root=tmp_path)
    assert back is not None
    assert back.tokens_in == 123 and back.tokens_out == 45


def test_synthesis_result_to_dict_includes_tokens():
    result = SY.SynthesisResult(tokens_in=10, tokens_out=20)
    d = result.to_dict()
    assert d["tokens_in"] == 10 and d["tokens_out"] == 20
