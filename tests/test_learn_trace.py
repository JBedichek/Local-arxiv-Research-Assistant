import asyncio

import pytest

from lara.learn import pipeline as PL
from lara.learn import scope as SC
from lara.learn import store
from lara.learn import trace as TR
from learn_helpers import corpus, model


def run(c):
    return asyncio.run(c)


def test_emit_is_a_no_op_when_nothing_is_tracing():
    TR.emit("search", query="x")  # must not raise


def test_start_writes_events_with_sequence_numbers_and_phase(tmp_path):
    path = tmp_path / "c1.trace.jsonl"
    TR.start(path)
    TR.set_phase("claims")
    TR.emit("search", query="warmup")
    TR.set_phase("outline")
    TR.emit("llm_call", purpose="learn_outline", prompt="p", response="r")
    TR.stop()
    rows = TR.read(path)
    assert [r["seq"] for r in rows] == [1, 2]
    assert rows[0]["phase"] == "claims" and rows[0]["type"] == "search"
    assert rows[0]["query"] == "warmup"
    assert rows[1]["phase"] == "outline" and rows[1]["purpose"] == "learn_outline"


def test_starting_a_new_tracer_truncates_the_file(tmp_path):
    path = tmp_path / "c1.trace.jsonl"
    TR.start(path)
    TR.emit("search", query="a")
    TR.emit("search", query="b")
    TR.stop()
    TR.start(path)
    TR.emit("search", query="fresh")
    TR.stop()
    rows = TR.read(path)
    assert [r["query"] for r in rows] == ["fresh"], "a rebuild's trace replaces the last one"


def test_stop_clears_the_current_tracer():
    from pathlib import Path

    TR.start(Path("/tmp/does-not-matter.trace.jsonl"))
    TR.stop()
    TR.emit("search", query="after stop")  # must not raise


def test_read_pages_by_seq_cursor(tmp_path):
    path = tmp_path / "c1.trace.jsonl"
    TR.start(path)
    for i in range(5):
        TR.emit("search", query=str(i))
    TR.stop()
    first = TR.read(path, since=0, limit=2)
    assert [r["seq"] for r in first] == [1, 2]
    rest = TR.read(path, since=first[-1]["seq"])
    assert [r["seq"] for r in rest] == [3, 4, 5]


def test_read_of_a_missing_file_is_empty(tmp_path):
    assert TR.read(tmp_path / "nope.trace.jsonl") == []


def test_malformed_lines_are_skipped_not_fatal(tmp_path):
    path = tmp_path / "c1.trace.jsonl"
    path.write_text('{"seq": 1, "type": "search"}\nnot json\n{"seq": 2, "type": "search"}\n')
    rows = TR.read(path)
    assert [r["seq"] for r in rows] == [1, 2]


def test_concurrent_tasks_each_keep_their_own_phase(tmp_path):
    """The isolation `claims.build`'s per-facet rounds and `depth.py`'s per-section
    research/write rely on: `asyncio.gather` copies the context into each child task, so one
    task's `set_phase` never leaks into a sibling's events, even though the Tracer object
    itself is shared."""
    path = tmp_path / "c1.trace.jsonl"
    TR.start(path)

    async def section(label: str) -> None:
        TR.set_phase(f"facet: {label}")
        await asyncio.sleep(0)          # yields, so the two tasks genuinely interleave
        TR.emit("llm_call", purpose="learn_claims", prompt=label, response="")

    async def go():
        await asyncio.gather(section("A"), section("B"))
    run(go())
    TR.stop()
    rows = {r["prompt"]: r["phase"] for r in TR.read(path)}
    assert rows == {"A": "facet: A", "B": "facet: B"}


# ── a real build, traced end to end ─────────────────────────────────────────────────

@pytest.fixture
def _root(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path / "courses")
    PL._building.clear()


def test_a_real_build_writes_a_trace_covering_every_stage(_root):
    m = model()
    course = run(PL.map_course(m, corpus(), run(SC.begin(m, "learn pretraining"))))
    run(PL.build_concept(m, corpus(), course, "c1"))
    rows = TR.read(store.trace_path(course["id"], "c1"))
    assert rows and rows[0]["type"] == "build_start" and rows[0]["variant"] == "standard"
    phases = {r["phase"] for r in rows}
    # "facet: Warmup (round 1)" (the claims stage's own single-facet fallback, no facets
    # reply scripted -- see model()'s docstring), "outline" and "research:"/"write:" (the
    # standard lesson's own outline-driven research and writing -- DP.STANDARD_PAGES),
    # "topics", "quiz" and "visuals:" -- every stage `pipeline.STAGES` runs is represented.
    assert any(p.startswith("facet:") for p in phases)
    assert any(p.startswith("outline") for p in phases)
    assert (any(p.startswith("research:") for p in phases)
            or any(p.startswith("write:") for p in phases))
    assert "topics" in phases and "quiz" in phases
    assert any(p.startswith("visuals:") for p in phases)
    types = {r["type"] for r in rows}
    assert {"llm_call", "search", "coverage_probe"} <= types


def test_a_forced_rebuild_retraces_from_the_claims_stage(_root):
    m = model()
    course = run(PL.map_course(m, corpus(), run(SC.begin(m, "learn pretraining"))))
    run(PL.build_concept(m, corpus(), course, "c1"))
    run(PL.build_concept(m, corpus(), course, "c1", force=True))
    rows = TR.read(store.trace_path(course["id"], "c1"))
    assert rows[0]["forced"] is True
    assert any(p.startswith("facet:") for p in {r["phase"] for r in rows})


def test_the_trace_route_serves_the_written_events(_root):
    import json

    from lara.serve.routes import learn as LR

    m = model()
    course = run(PL.map_course(m, corpus(), run(SC.begin(m, "learn pretraining"))))
    run(PL.build_concept(m, corpus(), course, "c1"))
    resp = LR.concept_trace(course["id"], "c1")
    events = json.loads(resp.body)["events"]
    assert events and events[0]["type"] == "build_start"
    resp2 = LR.concept_trace(course["id"], "c1", since=events[-1]["seq"])
    assert json.loads(resp2.body)["events"] == []
    assert LR.concept_trace("nope", "c1").status_code == 404
    assert LR.concept_trace(course["id"], "nope").status_code == 404
