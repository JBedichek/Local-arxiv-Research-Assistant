"""Follow-up recommendations: the standing profile, similarity retrieval, and the
implicit accept/skip signal from past suggestions."""

from __future__ import annotations

import asyncio

import pytest

from lara.serve import interest as I
import types


@pytest.fixture(autouse=True)
def _isolated_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "PROFILE_PATH", tmp_path / "interest_profile.json")
    monkeypatch.setattr(I, "GOAL_EMBEDDINGS_STORE", tmp_path / "goal_embeddings.jsonl")
    monkeypatch.setattr(I, "SUGGESTIONS_STORE", tmp_path / "followup_suggestions.jsonl")


def _plan(goal="does depth help?"):
    return types.SimpleNamespace(goal=goal)


def _writer(text):
    async def complete(cfg, prompt, **kw):
        return text
    return complete


class _FakeEmbedder:
    """Deterministic: a text's "embedding" is derived from its own characters, so two
    similar strings land close together without a real model."""

    def encode(self, texts, convert_to_numpy=True):
        import numpy as np
        return np.array([[float(ord(c)) for c in (t + " " * 8)[:8]] for t in texts])


class _OneHotEmbedder:
    """Distinct texts are orthogonal, identical texts identical: for tests where the
    char-code `_FakeEmbedder` would call every pair of strings near-duplicates."""

    def encode(self, texts, convert_to_numpy=True):
        import zlib

        import numpy as np
        out = np.zeros((len(texts), 256))
        for i, t in enumerate(texts):
            out[i, zlib.crc32(t.encode()) % 256] = 1.0
        return out


# ── the standing profile ─────────────────────────────────────────────────────────

def test_profile_round_trips():
    assert I.load_profile() == ""
    I.save_profile("Interested in retrieval and memory architectures.")
    assert I.load_profile() == "Interested in retrieval and memory architectures."


def test_no_profile_file_yet_is_empty_not_an_error(tmp_path):
    assert not I.PROFILE_PATH.exists()
    assert I.load_profile() == ""


def test_update_profile_persists_the_revised_summary():
    captured = {}

    async def fake_complete(cfg, prompt, **kw):
        captured["prompt"] = prompt
        return "Keeps returning to retrieval and memory systems."

    result = asyncio.run(I.update_profile({}, "improve retrieval", "found X about retrieval",
                                          complete=fake_complete))
    assert result == "Keeps returning to retrieval and memory systems."
    assert I.load_profile() == result
    assert "improve retrieval" in captured["prompt"]
    assert "(none yet" in captured["prompt"]  # first-ever goal, no prior summary


def test_update_profile_shows_the_current_summary_not_none_on_the_second_call():
    I.save_profile("Existing theme: agentic workflows.")
    captured = {}

    async def fake_complete(cfg, prompt, **kw):
        captured["prompt"] = prompt
        return "Existing theme: agentic workflows. New theme: retrieval."

    asyncio.run(I.update_profile({}, "new goal", "excerpt", complete=fake_complete))
    assert "Existing theme: agentic workflows." in captured["prompt"]
    assert "(none yet" not in captured["prompt"]


# ── embedding-similarity retrieval, not recency ─────────────────────────────────

def test_similar_past_goals_ranks_by_cosine_not_recency():
    I.record_goal_embedding("r1", "improve retrieval ranking", [1.0, 0.0, 0.0])
    I.record_goal_embedding("r2", "unrelated topic entirely", [0.0, 1.0, 0.0])
    I.record_goal_embedding("r3", "improve retrieval scoring", [0.9, 0.1, 0.0])
    # r3 is the most recent goal recorded, but a query close to r1/r3's direction
    # should rank both of them above r2 regardless of insertion order.
    got = I.similar_past_goals([1.0, 0.0, 0.0], k=2)
    assert got == ["improve retrieval ranking", "improve retrieval scoring"]


def test_similar_past_goals_excludes_its_own_run():
    I.record_goal_embedding("r1", "goal one", [1.0, 0.0])
    got = I.similar_past_goals([1.0, 0.0], exclude_run_id="r1")
    assert got == []


def test_similar_past_goals_on_no_history_is_empty():
    assert I.similar_past_goals([1.0, 0.0]) == []


def test_similar_past_goals_on_no_embedding_is_empty():
    I.record_goal_embedding("r1", "goal one", [1.0, 0.0])
    assert I.similar_past_goals([]) == []


# ── implicit preference from suggestion uptake ──────────────────────────────────

def test_acceptance_note_is_empty_with_nothing_shown_yet():
    assert I.acceptance_note() == ""


def test_acceptance_note_is_empty_when_nothing_was_ever_clicked():
    """Cold start must read as cold, not report a fabricated pattern."""
    I.record_shown("r1", "goal", ["a", "b", "c", "d", "e"])
    assert I.acceptance_note() == ""


def test_acceptance_note_reports_what_was_picked_and_skipped():
    I.record_shown("r1", "goal", ["Research retrieval more", "Research something else"])
    I.record_clicked("r1", "Research retrieval more")
    note = I.acceptance_note()
    assert "Research retrieval more" in note
    assert "Research something else" in note
    assert "picked" in note and "not picked" in note


# ── parsing the model's 5-line response ─────────────────────────────────────────

def test_parse_suggestions_reads_the_numbered_format():
    text = "1. First prompt\n2. Second prompt\n3. Third\n4. Fourth\n5. Fifth"
    assert I.parse_suggestions(text) == [
        "First prompt", "Second prompt", "Third", "Fourth", "Fifth"]


def test_parse_suggestions_tolerates_fewer_than_five():
    assert I.parse_suggestions("1. Only one") == ["Only one"]


def test_parse_suggestions_on_garbage_is_empty():
    assert I.parse_suggestions("no numbered lines here") == []


# ── the whole pipeline ───────────────────────────────────────────────────────────

def test_recommend_followups_on_empty_deliverable_is_empty():
    result = asyncio.run(I.recommend_followups({}, _plan(), "", complete=_writer("x")))
    assert result == []


def test_recommend_followups_returns_parsed_suggestions_and_records_them():
    async def fake_complete(cfg, prompt, *, system="", **kw):
        if system == I.PROFILE_UPDATE_SYSTEM:
            return "Interested in retrieval."
        return "1. Do X\n2. Do Y\n3. Do Z\n4. Do W\n5. Do V"

    result = asyncio.run(I.recommend_followups(
        {}, _plan("improve retrieval"), "The deliverable found retrieval helps.",
        run_id="run-1", embedder=_FakeEmbedder(), complete=fake_complete))
    assert result == ["Do X", "Do Y", "Do Z", "Do W", "Do V"]

    shown = [r for r in I._read_jsonl(I.SUGGESTIONS_STORE) if r["kind"] == "shown"]
    assert len(shown) == 1 and shown[0]["run_id"] == "run-1" and shown[0]["suggestions"] == result

    embeds = I._read_jsonl(I.GOAL_EMBEDDINGS_STORE)
    assert len(embeds) == 1 and embeds[0]["run_id"] == "run-1"


def test_recommend_followups_includes_profile_and_acceptance_note_in_the_prompt():
    I.save_profile("Keeps returning to retrieval.")
    I.record_shown("r0", "earlier goal", ["Earlier suggestion"])
    I.record_clicked("r0", "Earlier suggestion")

    captured = {}

    async def fake_complete(cfg, prompt, *, system="", **kw):
        if system == I.RECOMMEND_SYSTEM:
            captured["prompt"] = prompt
            return "1. A\n2. B\n3. C\n4. D\n5. E"
        return "Keeps returning to retrieval."  # unchanged profile-update response

    asyncio.run(I.recommend_followups({}, _plan(), "Full deliverable text.",
                                      complete=fake_complete))
    assert "Keeps returning to retrieval." in captured["prompt"]
    assert "Earlier suggestion" in captured["prompt"]


def test_recommend_followups_with_no_run_id_does_not_record_anything():
    """A caller that does not pass run_id (e.g. a dry-run preview) must not pollute
    the suggestion/embedding history with rows nothing can ever match a click to."""
    async def fake_complete(cfg, prompt, *, system="", **kw):
        return "1. A\n2. B\n3. C\n4. D\n5. E"

    asyncio.run(I.recommend_followups({}, _plan(), "text", complete=fake_complete))
    assert I._read_jsonl(I.SUGGESTIONS_STORE) == []
    assert I._read_jsonl(I.GOAL_EMBEDDINGS_STORE) == []


def test_recommend_followups_without_learning_reads_the_profile_and_records_nothing():
    """A backfill suggests on old deliverables: it must not revise the standing profile
    (the update prompt would run once per old run, in arbitrary order) nor log
    suggestions as shown, which would read as a long run of ignored ones."""
    I.save_profile("Keeps returning to retrieval.")
    prompts = []

    async def fake_complete(cfg, prompt, *, system="", **kw):
        prompts.append((system, prompt))
        return "1. A\n2. B\n3. C\n4. D\n5. E"

    out = asyncio.run(I.recommend_followups({}, _plan(), "text", run_id="old-run",
                                            complete=fake_complete, learn=False))
    assert out == ["A", "B", "C", "D", "E"]
    assert [s for s, _ in prompts] == [I.RECOMMEND_SYSTEM], "no profile-update call"
    assert "Keeps returning to retrieval." in prompts[0][1]
    assert I.load_profile() == "Keeps returning to retrieval."
    assert I._read_jsonl(I.SUGGESTIONS_STORE) == []


def test_refreshing_shows_the_model_the_old_suggestions_and_does_not_relearn_the_goal():
    I.save_profile("Keeps returning to retrieval.")
    prompts = []

    async def fake_complete(cfg, prompt, *, system="", **kw):
        prompts.append((system, prompt))
        return "1. N1\n2. N2\n3. N3\n4. N4\n5. N5"

    out = asyncio.run(I.recommend_followups(
        {}, _plan(), "text", run_id="r1", embedder=_OneHotEmbedder(),
        complete=fake_complete, avoid=["old one", "old two"]))
    assert out == ["N1", "N2", "N3", "N4", "N5"]
    assert [s for s, _ in prompts] == [I.RECOMMEND_SYSTEM], "no profile revision"
    assert "- old one" in prompts[0][1] and "- old two" in prompts[0][1]
    assert I.load_profile() == "Keeps returning to retrieval."
    assert I._read_jsonl(I.GOAL_EMBEDDINGS_STORE) == [], "the goal was already embedded"
    shown = [r for r in I._read_jsonl(I.SUGGESTIONS_STORE) if r["kind"] == "shown"]
    assert shown[0]["suggestions"] == ["N1", "N2", "N3", "N4", "N5"]


def test_was_shown_reports_whether_a_run_has_a_logged_set():
    assert I.was_shown("r1") is False
    I.record_shown("r1", "goal", ["a"])
    assert I.was_shown("r1") is True and I.was_shown("r2") is False


def test_previously_shown_is_distinct_recent_and_per_run():
    I.record_shown("r1", "g", ["a", "b"])
    I.record_shown("r2", "g", ["other"])
    I.record_shown("r1", "g", ["b", "c"])
    assert I.previously_shown("r1") == ["a", "b", "c"]
    assert I.previously_shown("r1", limit=2) == ["b", "c"]
    assert I.previously_shown("nobody") == []


def test_refresh_drops_a_suggestion_that_repeats_one_it_was_told_to_avoid():
    """The model repeats items from the avoid list verbatim, so asking is not enough."""
    calls = []

    async def fake_complete(cfg, prompt, *, system="", **kw):
        calls.append(prompt)
        if len(calls) == 1:
            return ("1. Old idea about caching layers\n2. Brand new angle on evaluation\n"
                    "3. Old idea about caching layers\n4. Another fresh direction on cost\n"
                    "5. Third fresh thing about data quality")
        return ("1. Fourth fresh topic on latency budgets\n2. Fifth fresh topic on drift\n"
                "3. Old idea about caching layers\n4. Sixth topic on labelling\n5. x y z")

    out = asyncio.run(I.recommend_followups(
        {}, _plan(), "text", complete=fake_complete, avoid=["Old idea about caching layers"]))
    assert "Old idea about caching layers" not in out
    assert len(out) == 5, "the dropped repeats were replaced by a second ask"
    assert out[:3] == ["Brand new angle on evaluation", "Another fresh direction on cost",
                       "Third fresh thing about data quality"]
    assert len(calls) == 2
    assert "Brand new angle on evaluation" in calls[1], \
        "the retry also tells the model what it just produced"


def test_refresh_gives_up_after_its_retries_and_returns_what_was_novel():
    async def stubborn(cfg, prompt, *, system="", **kw):
        return "1. Same old thing\n2. Same old thing\n3. Same old thing\n4. Same old thing\n5. Same old thing"

    out = asyncio.run(I.recommend_followups(
        {}, _plan(), "text", complete=stubborn, avoid=["Same old thing"]))
    assert out == []


def test_refresh_treats_a_near_identical_embedding_as_a_restatement():
    class Close(_OneHotEmbedder):
        """Anything about caching embeds identically; everything else is distinct."""

        def encode(self, texts, convert_to_numpy=True):
            out = super().encode(texts)
            for i, t in enumerate(texts):
                if "cach" in t.lower():
                    out[i] = 0.0
                    out[i, 0] = 1.0
            return out

    async def fake_complete(cfg, prompt, *, system="", **kw):
        return ("1. Put a cache in front of it\n2. Measure latency\n3. Audit labels\n"
                "4. Compare costs\n5. Check drift")

    out = asyncio.run(I.recommend_followups(
        {}, _plan(), "text", embedder=Close(), complete=fake_complete,
        avoid=["Add a caching layer"]))
    assert "Put a cache in front of it" not in out
