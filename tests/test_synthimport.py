"""Tests for lara.serve.synthimport -- copying autoresearch's synthesis work, copy-only."""
from __future__ import annotations

import json

from lara.serve import synthesizer as SY
from lara.serve import synthimport as SI
from lara.serve import synthruns as SR


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _dst(tmp_path):
    return dict(facts=tmp_path / "d" / "facts.jsonl", goals=tmp_path / "d" / "goals.jsonl",
                suggestions=tmp_path / "d" / "sug.jsonl", profile=tmp_path / "d" / "profile.json")


def _source(tmp_path):
    src = tmp_path / "src"
    _write_jsonl(src / "facts.jsonl", [{"id": "r-0", "fact": "f", "embedding": [1.0]}])
    _write_jsonl(src / "goal_embeddings.jsonl", [{"run_id": "r", "goal": "g", "embedding": [1.0]}])
    _write_jsonl(src / "followup_suggestions.jsonl", [{"kind": "shown", "run_id": "r"}])
    (src / "interest_profile.json").write_text('{"summary": "theirs"}')
    return src


def test_memory_is_copied_then_a_repeat_adds_nothing(tmp_path):
    src, dst = _source(tmp_path), _dst(tmp_path)
    first = SI.import_memory(src, **dst)
    assert first["facts"] == {"copied": 1, "skipped": 0} and first["profile"] == "copied"
    again = SI.import_memory(src, **dst)
    assert again["facts"] == {"copied": 0, "skipped": 1} and again["goal_embeddings"]["copied"] == 0
    assert len(dst["facts"].read_text().splitlines()) == 1


def test_rows_already_in_the_destination_are_kept_and_new_ones_appended(tmp_path):
    src, dst = _source(tmp_path), _dst(tmp_path)
    _write_jsonl(dst["facts"], [{"id": "mine", "fact": "own"}])
    SI.import_memory(src, **dst)
    ids = [json.loads(line)["id"] for line in dst["facts"].read_text().splitlines()]
    assert ids == ["mine", "r-0"]


def test_an_existing_profile_is_never_overwritten(tmp_path):
    src, dst = _source(tmp_path), _dst(tmp_path)
    dst["profile"].parent.mkdir(parents=True)
    dst["profile"].write_text('{"summary": "mine"}')
    out = SI.import_memory(src, **dst)
    assert out["profile"] == "kept the existing profile"
    assert "mine" in dst["profile"].read_text()


def test_a_missing_source_copies_nothing_and_does_not_fail(tmp_path):
    out = SI.import_memory(tmp_path / "nothing", **_dst(tmp_path))
    assert out["facts"]["copied"] == 0 and out["profile"] == "none to copy"


def _run(rid, *, status="done", deliverable="Report [1].", plan=True):
    d = {"id": rid, "goal": f"goal {rid}", "status": status, "started": 10.0, "finished": 20.0,
         "parent": "", "verdict": {"kind": "success", "because": "ok"}, "tokens_in": 5,
         "tokens_out": 6, "deliverable": deliverable, "references": {"1": {"key": "1"}},
         "deliverable_short": "short", "followups": ["a"], "orchestrated_only": "x"}
    if plan:
        d["plan"] = {"objective": "obj", "round": 3, "experiments": [],
                     "goals": {"g1": {"id": "g1", "text": "q", "status": "done", "summary": "s"}}}
    return d


def test_only_finished_synthesis_runs_are_imported_with_their_graph(tmp_path):
    runs = tmp_path / "src" / "runs"
    runs.mkdir(parents=True)
    for rid, d in {"a": _run("a"), "b": _run("b", deliverable=""), "c": _run("c", plan=False),
                   "d": _run("d", status="interrupted")}.items():
        (runs / f"{rid}.json").write_text(json.dumps(d))
    (runs / "bad.json").write_text("{not json")
    root, states = tmp_path / "lara" / "runs", tmp_path / "lara"
    out = SI.import_runs(tmp_path / "src", root=root, states=states)
    assert sorted(out["copied"]) == ["a", "d"]
    rec = SR.load_record("a", root=root)
    assert rec["imported"] == "autoresearch" and rec["status"] == SR.DONE
    assert rec["created"] == 10.0 and rec["rounds"] == 3 and rec["deliverable_short"] == "short"
    assert rec["followups"] == ["a"] and "orchestrated_only" not in rec
    assert SR.load_record("d", root=root)["status"] == SR.INTERRUPTED
    assert SY.load("a", root=states).goals["g1"].summary == "s"


def test_a_repeat_skips_runs_already_copied_and_never_overwrites(tmp_path):
    runs = tmp_path / "src" / "runs"
    runs.mkdir(parents=True)
    (runs / "a.json").write_text(json.dumps(_run("a")))
    root, states = tmp_path / "lara" / "runs", tmp_path / "lara"
    SI.import_runs(tmp_path / "src", root=root, states=states)
    mine = SR.load_record("a", root=root)
    mine["deliverable"] = "edited here"
    SR.save_record(mine, root=root)
    out = SI.import_runs(tmp_path / "src", root=root, states=states)
    assert out == {"copied": [], "skipped": ["a"]}
    assert SR.load_record("a", root=root)["deliverable"] == "edited here"
