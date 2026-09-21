"""Tests for lara.serve.synthruns and the /api/synthesizer handlers -- run records, the live
feed, the finishing steps, and the after-the-fact actions."""
from __future__ import annotations

import asyncio
import json
import types

import pytest

from lara.serve import deliverable as DL
from lara.serve import facts as FA
from lara.serve import interest as IN
from lara.serve import synthesizer as SY
from lara.serve import synthruns as SR
from lara.serve.routes import synthesizer as RT


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setattr(SY, "STATES", tmp_path / "synthesizer")
    monkeypatch.setattr(SR, "RUNS", tmp_path / "synthesizer" / "runs")
    monkeypatch.setattr(IN, "PROFILE_PATH", tmp_path / "profile.json")
    monkeypatch.setattr(IN, "GOAL_EMBEDDINGS_STORE", tmp_path / "goals.jsonl")
    monkeypatch.setattr(IN, "SUGGESTIONS_STORE", tmp_path / "suggestions.jsonl")
    monkeypatch.setattr(FA, "FACTS_STORE", tmp_path / "facts.jsonl")
    for d in (SR._ACTIVE, SR._TASKS, SR._GRAPHS):
        d.clear()


def body(resp):
    return json.loads(resp.body)


def frames(text_frames):
    out = []
    for f in text_frames:
        name = f.split("\n")[0].removeprefix("event: ")
        out.append((name, json.loads(f.split("\n")[1].removeprefix("data: "))))
    return out


async def collect(run_id):
    return frames([f async for f in SR.stream(run_id)])


def state_with(goal_status=SY.DONE):
    st = SY.SynthesizerState(objective="obj")
    st.goals["g1"] = SY.LogicalGoal(id="g1", text="q", status=goal_status, summary="a [1]")
    return st


# ── records ──


def test_a_record_round_trips_and_lists_newest_first_without_deliverables():
    a, b = SR.new_record("first"), SR.new_record("second")
    a["created"], b["created"] = 1.0, 2.0
    a["deliverable"] = "text"
    SR.save_record(a), SR.save_record(b)
    listed = SR.list_records()
    assert [r["goal"] for r in listed] == ["second", "first"]
    assert "deliverable" not in listed[0] and listed[1]["has_deliverable"] is True
    assert SR.load_record(a["id"])["deliverable"] == "text"


def test_a_run_left_running_by_a_dead_server_reads_as_interrupted():
    rec = SR.new_record("g")
    SR.save_record(rec)
    assert SR.load_record(rec["id"])["status"] == SR.INTERRUPTED
    assert SR.load_record(rec["id"])["ended"]


def test_a_live_run_is_not_marked_interrupted():
    rec = SR.new_record("g")
    SR.save_record(rec)
    SR._ACTIVE[rec["id"]] = SR.Feed()
    assert SR.load_record(rec["id"])["status"] == SR.RUNNING


def test_delete_removes_record_and_graph_but_refuses_a_live_run():
    rec = SR.new_record("g")
    rec["status"] = SR.DONE
    SR.save_record(rec)
    SY.save(rec["id"], state_with())
    SR._ACTIVE[rec["id"]] = SR.Feed()
    assert SR.delete_record(rec["id"]) is False
    SR._ACTIVE.clear()
    assert SR.delete_record(rec["id"]) is True
    assert SR.load_record(rec["id"]) is None and SY.load(rec["id"]) is None
    assert SR.delete_record(rec["id"]) is False


# ── the feed ──


def test_a_finished_run_streams_a_snapshot_then_ends():
    rec = SR.new_record("g")
    rec["status"] = SR.DONE
    SR.save_record(rec)
    SY.save(rec["id"], state_with())
    got = asyncio.run(collect(rec["id"]))
    assert [n for n, _ in got] == ["snapshot", "end"]
    assert got[0][1]["graph"]["goals"]["g1"]["summary"] == "a [1]"
    assert got[1][1]["because"] == SR.DONE


def test_an_unknown_run_ends_at_once():
    assert asyncio.run(collect("nope")) == [("end", {"because": "no such run"})]


def test_a_live_run_streams_its_events_after_the_snapshot_until_it_closes():
    async def go():
        rec = SR.new_record("g")
        SR.save_record(rec)
        feed = SR._ACTIVE[rec["id"]] = SR.Feed()
        feed.emit("goal.new", {"id": "early"})
        task = asyncio.create_task(collect(rec["id"]))
        await asyncio.sleep(0)
        feed.emit("goal.new", {"id": "late"})
        await asyncio.sleep(0)
        rec["status"] = SR.DONE
        SR.save_record(rec)
        feed.close()
        return await task
    got = asyncio.run(go())
    names = [n for n, _ in got]
    assert names[0] == "snapshot" and names[-1] == "end"
    # An event emitted before the watcher arrived is in the snapshot's era, not replayed.
    assert [p["id"] for n, p in got if n == "goal.new"] == ["late"]


def test_persist_emits_only_what_changed_and_saves_graph_and_record():
    rec, feed, st = SR.new_record("g"), SR.Feed(), SY.SynthesizerState(objective="obj")
    persist = SR._diff_emitter(st, feed, rec)
    st.round = 1
    st.goals["g1"] = SY.LogicalGoal(id="g1", text="q", status=SY.RUNNING)
    persist()
    persist()
    st.goals["g1"].status, st.goals["g1"].summary = SY.DONE, "s"
    st.compressed.append("folded")
    persist()
    assert [n for n, _ in feed.events] == ["goal.new", "round", "goal.update", "compression"]
    assert SY.load(rec["id"]).goals["g1"].status == SY.DONE
    assert SR.load_record(rec["id"], root=SR.RUNS)["rounds"] == 1


# ── finishing steps ──


def _patch_finishing(monkeypatch, *, condense=None, followups=None, distill=None):
    async def default_condense(cfg, text, *, level, **kw):
        return (f"{level} version", ["1"], {"1": {"key": "1"}})

    async def default_followups(cfg, plan, text, **kw):
        return ["a", "b", "c", "d", "e"]

    async def default_distill(*a, **kw):
        return []

    monkeypatch.setattr(DL, "condense", condense or default_condense)
    monkeypatch.setattr(IN, "recommend_followups", followups or default_followups)
    monkeypatch.setattr(FA, "distill_facts", distill or default_distill)


def _finish(rec, feed):
    return asyncio.run(SR._finish_up(rec, feed, cfg={}, model="m", window=1000, conn=None,
                                     embedder=None, references={}))


def test_finishing_writes_both_versions_and_the_followups(monkeypatch):
    _patch_finishing(monkeypatch)
    rec, feed = {**SR.new_record("g"), "deliverable": "full [1]"}, SR.Feed()
    _finish(rec, feed)
    assert rec["deliverable_medium"] == "medium version" and rec["deliverable_short"] == "short version"
    assert rec["followups"] == ["a", "b", "c", "d", "e"]
    assert {"deliverable_medium", "deliverable_short", "followups"} <= {n for n, _ in feed.events}


def test_a_failing_step_is_an_event_and_does_not_stop_the_others(monkeypatch):
    async def bad_condense(cfg, text, *, level, **kw):
        if level == "medium":
            raise RuntimeError("boom")
        return ("short version", [], {})
    _patch_finishing(monkeypatch, condense=bad_condense)
    rec, feed = {**SR.new_record("g"), "deliverable": "full"}, SR.Feed()
    _finish(rec, feed)
    names = [n for n, _ in feed.events]
    assert "deliverable_medium.failed" in names and rec["deliverable_short"] == "short version"
    assert rec["followups"]


def test_a_failing_followup_step_is_an_event(monkeypatch):
    async def bad(*a, **kw):
        raise RuntimeError("no model")
    _patch_finishing(monkeypatch, followups=bad)
    rec, feed = {**SR.new_record("g"), "deliverable": "full"}, SR.Feed()
    _finish(rec, feed)
    assert "followups.failed" in [n for n, _ in feed.events] and rec["followups"] == []


# ── running ──


def _app_state():
    return types.SimpleNamespace(retriever=types.SimpleNamespace(embedder=None),
                                 conn=lambda: None, cfg={})


def _fake_generator(monkeypatch):
    async def generator(app_state, model=None):
        return types.SimpleNamespace(cfg={}, base_url="http://x/v1", api_key="k", model="m",
                                     window=10_000)
    monkeypatch.setattr(SR, "generator", generator)


def test_a_run_records_its_deliverable_verdict_and_versions(monkeypatch):
    _fake_generator(monkeypatch)
    _patch_finishing(monkeypatch)
    seen = {}

    async def fake_run(state, **kw):
        seen.update(kw)
        state.goals["g1"] = SY.LogicalGoal(id="g1", text="q", status=SY.DONE, summary="s")
        kw["on_change"]()
        return SY.SynthesisResult(deliverable="Report [1].", references={"1": {"key": "1"}},
                                  rounds=2, total_done=1, tokens_in=5, tokens_out=6)

    async def go():
        rec = SR.start(_app_state(), "the goal", driver=lambda r, f, a: SR._drive(r, f, a, run=fake_run),
                       options={"allow_subsynthesis": False, "deliverable_tokens": 900,
                                "final_compression_prompt": "top 3?"})
        await SR._TASKS[rec["id"]]
        return rec
    rec = asyncio.run(go())
    stored = SR.load_record(rec["id"])
    assert stored["status"] == SR.DONE and stored["deliverable"] == "Report [1]."
    assert stored["verdict"]["kind"] == SY.SUCCESS and stored["tokens_out"] == 6
    assert stored["deliverable_short"] == "short version" and stored["followups"]
    assert seen["allow_subsynthesis"] is False and seen["deliverable_tokens"] == 900
    assert seen["final_compression_prompt"] == "top 3?" and seen["model"] == "m"
    assert not SR._ACTIVE and not SR._GRAPHS


def test_a_run_that_raises_is_failed_with_the_reason():
    async def boom(rec, feed, app_state):
        raise RuntimeError("no generator")

    async def go():
        rec = SR.start(_app_state(), "g", driver=boom)
        await SR._TASKS[rec["id"]]
        return rec
    rec = asyncio.run(go())
    stored = SR.load_record(rec["id"])
    assert stored["status"] == SR.FAILED and "no generator" in stored["error"]
    assert not SR._ACTIVE


def test_cancelling_a_run_marks_it_cancelled_and_frees_it():
    async def slow(rec, feed, app_state):
        await asyncio.sleep(60)

    async def go():
        rec = SR.start(_app_state(), "g", driver=slow)
        await asyncio.sleep(0)
        assert SR.cancel(rec["id"]) is True
        await asyncio.gather(SR._TASKS.get(rec["id"]) or asyncio.sleep(0), return_exceptions=True)
        await asyncio.sleep(0)
        return rec
    rec = asyncio.run(go())
    assert SR.load_record(rec["id"])["status"] == SR.CANCELLED
    assert SR.cancel(rec["id"]) is False


# ── after the fact ──


def test_compress_answers_from_the_original_report_each_time():
    calls = []

    async def answer(full, prompt, **kw):
        calls.append(full)
        return (f"answer to {prompt}", False, "", 3, 4)

    rec = {**SR.new_record("g"), "deliverable": "REPORT", "tokens_in": 1, "tokens_out": 1}
    asyncio.run(SR.compress(rec, "first?", base_url="x", model="m", api_key="", max_model_len=1000,
                            answer=answer))
    out = asyncio.run(SR.compress(rec, "second?", base_url="x", model="m", api_key="",
                                  max_model_len=1000, answer=answer))
    assert calls == ["REPORT", "REPORT"]
    assert "answer to second?" in out["deliverable"] and "answer to first?" not in out["deliverable"]
    assert rec["tokens_in"] == 7 and SR.load_record(rec["id"], root=SR.RUNS)["deliverable"] == out["deliverable"]


def test_refresh_logs_the_old_set_as_shown_and_avoids_it(monkeypatch):
    seen = {}

    async def followups(cfg, plan, text, **kw):
        seen.update(kw)
        return ["n1", "n2", "n3", "n4", "n5"]
    monkeypatch.setattr(IN, "recommend_followups", followups)
    rec = {**SR.new_record("g"), "deliverable": "D", "followups": ["o1", "o2"]}
    fresh = asyncio.run(SR.refresh_followups(rec, cfg={}, model="m", window=100))
    assert fresh == rec["followups"] == ["n1", "n2", "n3", "n4", "n5"]
    assert seen["avoid"] == ["o1", "o2"] and seen["learn"] is False
    assert IN.was_shown(rec["id"])


# ── handlers ──


def _with_state(monkeypatch, retriever=object()):
    monkeypatch.setattr(RT, "require_state", lambda: types.SimpleNamespace(retriever=retriever))


def test_start_needs_a_goal_and_a_loaded_retriever(monkeypatch):
    async def go():
        _with_state(monkeypatch)
        assert (await RT.start(RT.StartRequest(goal="  "))).status_code == 400
        _with_state(monkeypatch, retriever=None)
        assert (await RT.start(RT.StartRequest(goal="g"))).status_code == 503
    asyncio.run(go())


def test_a_followup_started_as_suggested_records_the_click(monkeypatch):
    _with_state(monkeypatch)
    clicked = []
    monkeypatch.setattr(IN, "record_clicked", lambda run_id, text: clicked.append((run_id, text)))
    monkeypatch.setattr(SR, "start", lambda state, goal, options=None, parent="": {"id": "new", "parent": parent})
    parent = {**SR.new_record("p"), "status": SR.DONE, "followups": ["ask this"]}
    SR.save_record(parent)

    async def go():
        as_is = await RT.start(RT.StartRequest(goal="ask this", parent=parent["id"]))
        edited = await RT.start(RT.StartRequest(goal="ask this, edited", parent=parent["id"]))
        missing = await RT.start(RT.StartRequest(goal="x", parent="nope"))
        return as_is, edited, missing
    as_is, edited, missing = asyncio.run(go())
    assert as_is.status_code == edited.status_code == 201 and missing.status_code == 404
    assert clicked == [(parent["id"], "ask this")]


def test_compress_and_refresh_refuse_a_live_or_empty_run():
    live = SR.new_record("g")
    SR.save_record(live)
    SR._ACTIVE[live["id"]] = SR.Feed()
    empty = {**SR.new_record("e"), "status": SR.DONE}
    SR.save_record(empty)

    async def go():
        return [(await RT.compress(live["id"], RT.PromptRequest(prompt="q"))).status_code,
                (await RT.refresh_followups(live["id"])).status_code,
                (await RT.compress(empty["id"], RT.PromptRequest(prompt="q"))).status_code,
                (await RT.compress("nope", RT.PromptRequest(prompt="q"))).status_code]
    assert asyncio.run(go()) == [409, 409, 400, 404]


def test_one_returns_record_and_graph_and_404s_when_unknown():
    rec = {**SR.new_record("g"), "status": SR.DONE}
    SR.save_record(rec)
    SY.save(rec["id"], state_with())
    ok = RT.one(rec["id"])
    assert body(ok)["graph"]["goals"]["g1"]["id"] == "g1" and body(ok)["run"]["goal"] == "g"
    assert RT.one("nope").status_code == 404


def test_delete_and_cancel_handlers():
    done = {**SR.new_record("g"), "status": SR.DONE}
    SR.save_record(done)
    assert RT.delete(done["id"]).status_code == 200 and RT.delete(done["id"]).status_code == 404
    assert RT.cancel("nope").status_code == 409
