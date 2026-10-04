"""lara.learn.profile: the cross-course knowledge ledger, built only from quiz evidence,
and the digest derived from it that personalizes a new lesson's objective."""
from __future__ import annotations

import asyncio

import pytest

from lara.learn import learner as LN
from lara.learn import profile as PR
from lara.learn import store
from learn_helpers import llm


def run(c):
    return asyncio.run(c)


@pytest.fixture(autouse=True)
def _root(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PROFILE_PATH", tmp_path / "learner" / "profile.json")


def test_record_quiz_answer_uses_the_same_mastery_formula_learner_py_uses():
    PR.record_quiz_answer("Warmup", True, 3, "mcq")
    expect = round(LN.mastery_after(0.0, True, 3, "mcq"), 4)
    data = store.load_profile()
    assert data["concepts"][store.slug("Warmup", 60)]["score"] == expect


def test_record_quiz_answer_keys_by_slugged_title_across_calls():
    PR.record_quiz_answer("Warmup Schedules", True, 3, "mcq")
    PR.record_quiz_answer("warmup schedules", False, 2, "short")   # same title, different case
    PR.record_quiz_answer("Decay", True, 3, "mcq")
    data = store.load_profile()
    assert set(data["concepts"]) == {store.slug("Warmup Schedules", 60), store.slug("Decay", 60)}
    assert len(data["concepts"][store.slug("Warmup Schedules", 60)]["evidence"]) == 2


def test_evidence_is_capped_to_the_most_recent_max_evidence_points():
    for i in range(PR.MAX_EVIDENCE + 3):
        PR.record_quiz_answer("Warmup", i % 2 == 0, 2, "mcq")
    data = store.load_profile()
    entry = data["concepts"][store.slug("Warmup", 60)]
    assert len(entry["evidence"]) == PR.MAX_EVIDENCE


def test_a_blank_title_is_ignored_not_crashed_on():
    PR.record_quiz_answer("   ", True, 3, "mcq")
    assert store.load_profile().get("concepts", {}) == {}


def test_snapshot_buckets_known_shaky_and_leaves_unassessed_concepts_out_entirely():
    for _ in range(6):   # enough correct mcq answers to cross LN.MASTERED (0.75)
        PR.record_quiz_answer("Mastered topic", True, 3, "mcq")
    PR.record_quiz_answer("Shaky topic", False, 3, "mcq")
    snap = PR.snapshot()
    by_title = {c["title"]: c for c in snap["concepts"]}
    assert by_title["Mastered topic"]["bucket"] == "known"
    assert by_title["Shaky topic"]["bucket"] == "shaky"
    assert "Never assessed" not in by_title


def test_snapshot_orders_most_recently_assessed_first():
    PR.record_quiz_answer("First", True, 2, "mcq")
    PR.record_quiz_answer("Second", True, 2, "mcq")
    titles = [c["title"] for c in PR.snapshot()["concepts"]]
    assert titles == ["Second", "First"]


def test_digest_is_empty_with_nothing_ever_assessed():
    assert run(PR.digest(llm())) == ""


def test_digest_regenerates_once_then_caches_until_new_evidence(monkeypatch):
    PR.record_quiz_answer("Warmup", True, 3, "mcq")
    m = llm(("summarizing one learner", "Solid on warmup."))

    first = run(PR.digest(m))
    assert first == "Solid on warmup."
    assert len(m.calls) == 1

    second = run(PR.digest(m))
    assert second == "Solid on warmup." and len(m.calls) == 1, "unchanged ledger -- no new call"

    PR.record_quiz_answer("Decay", False, 2, "mcq")
    m2 = llm(("summarizing one learner", "Solid on warmup, shaky on decay."))
    third = run(PR.digest(m2))
    assert third == "Solid on warmup, shaky on decay." and len(m2.calls) == 1


def test_digest_lists_every_assessed_concept_with_its_bucket(monkeypatch):
    captured = {}

    async def complete(cfg, prompt, *, system="", model=None, max_tokens=0):
        captured["prompt"] = prompt
        return "a summary"
    from lara.learn.llm import Llm
    m = Llm(complete=complete, window=200_000)

    for _ in range(6):
        PR.record_quiz_answer("Warmup", True, 3, "mcq")
    PR.record_quiz_answer("KL penalty", False, 3, "mcq")
    run(PR.digest(m))

    assert "Warmup: known" in captured["prompt"]
    assert "KL penalty: shaky" in captured["prompt"]
