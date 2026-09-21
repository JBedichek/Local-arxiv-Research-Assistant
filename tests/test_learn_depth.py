import asyncio
import json
import re

import pytest

from lara.learn import depth as DP
from lara.learn import learner as LN
from lara.learn import pipeline as PL
from lara.learn import scope as SC
from lara.learn import store
from learn_helpers import FakeCorpus, corpus, llm, model, passage

LONG = "warmup avoids loss spikes early in training. " * 6
CONCEPT = {"id": "c1", "title": "warmup", "summary": "s", "goal": "pretrain an LLM", "prereqs": []}


def run(c):
    return asyncio.run(c)


def claim(key, text, chunk=1, arxiv="2401.1", certainty="single-source"):
    return {"key": key, "text": text, "certainty": certainty, "conditions": "", "kind": "finding",
            "passage": {"text": LONG, "chunk_id": chunk, "arxiv_id": arxiv, "title": "P", "date": "2024-01-01"},
            "corroborated_by": [], "conflicts": [], "superseded_by": "", "flags": []}


def content():
    return {"claims": [claim("c1", "Warmup avoids early loss spikes.", 1), claim("c2", "Warmup of 1000 steps suited the 1B model.", 2, "2401.2")],
            "lesson": {"sections": [{"heading": "H", "sentences": [{"text": "x", "claims": ["c1"]}]}], "generated": 5.0},
            "quiz": {"items": [{"id": "c1-q1", "claim": "c1"}, {"id": "c1-q2", "claim": "c2"}], "dropped": 0}, "conflicts": []}


SECTIONS = {
    "What warmup is": ("gradually raises the learning rate linear ramp shape",
                       ["Warmup gradually raises the learning rate from near zero.", "A linear ramp is the most common warmup shape."]),
    "Why it stabilises training": ("second moment estimates early large updates loss spikes",
                                   ["Adam second moment estimates are unreliable in the first steps.", "Early large updates can cause irrecoverable loss spikes."]),
    "Choosing a warmup length": ("percent of total steps larger batches shorter",
                                 ["One percent of total steps is a typical warmup length.", "Larger batches usually tolerate shorter warmups."]),
}


def outline_reply(n=3):
    heads = list(SECTIONS)[:n]
    return json.dumps({"sections": [{"heading": h, "focus": SECTIONS[h][0]} for h in heads]})


def extract_by_focus(system, prompt):
    """Distinct claims per section, chosen by the FOCUS line, so nothing is a repeat."""
    for head, (_, facts) in SECTIONS.items():
        if head in prompt:
            return json.dumps([{"passage": 1, "claim": f, "conditions": ""} for f in facts])
    return "[]"


def section_writer(system, prompt):
    keys = re.findall(r"^\[(c\d+)\]", prompt, re.M)
    return "\n".join(f"Sentence {i} of the section [{k}]." for i, k in enumerate(keys[:3], 1)) or "INSUFFICIENT"


def fake_llm(outline=None):
    return llm(("plan a self-study lesson", outline or outline_reply()),
               ("extract atomic claims", extract_by_focus), ("write ONE section", section_writer),
               ("strict fact-checker", "supports"), ("Would this claim help", "yes"),
               ("Do these two claims", "different"), ("compare two claims", '{"relation": "agree", "note": ""}'))


def sources():
    return FakeCorpus(default=[passage(50 + i, LONG, arxiv=f"2410.{i}") for i in range(4)])


def test_pages_map_to_a_bounded_section_count():
    assert [DP.section_count(p) for p in (1, 5, 10, 20)] == [3, 6, 12, 16]
    assert DP.clamp_pages(0) == 1 and DP.clamp_pages(99) == 20 and DP.clamp_pages("x") == DP.THOROUGH_PAGES


def test_the_outline_is_parsed_and_bad_entries_dropped():
    m = llm(("plan a self-study lesson", json.dumps({"sections": [{"heading": "A", "focus": "f"}, {"heading": ""}, "junk"]})))
    assert run(DP.plan_outline(m, CONCEPT, [], 5))[0] == [{"heading": "A", "focus": "f"}]
    assert run(DP.plan_outline(llm(("plan a self-study lesson", "no")), CONCEPT, [], 5)) == ([], "no")


def test_each_claim_goes_to_exactly_one_section():
    outline = [{"heading": "Warmup length", "focus": "choose length"}, {"heading": "Loss spikes", "focus": "why spikes occur"}]
    cl = [claim("c1", "Choose the warmup length by batch size."), claim("c2", "Loss spikes occur early without warmup.")]
    buckets = DP.assign(cl, outline)
    assert [[c["key"] for c in b] for b in buckets] == [["c1"], ["c2"]]


def test_deepen_researches_each_section_and_writes_all_the_supported_ones():
    lesson, changes = run(DP.deepen(fake_llm(), sources(), CONCEPT, content(), 3))
    assert [s["heading"] for s in lesson["sections"]] == list(SECTIONS) and lesson["dropped_sections"] == []
    assert lesson["stats"]["grounded_pct"] == 100 and not lesson["insufficient"]
    keys = [c["key"] for c in changes["new_claims"]]
    assert keys == ["c3", "c4", "c5", "c6", "c7", "c8"], "numbered on from the concept's own claims"
    assert len(changes["claims"]) == 2 + 6 and lesson["target_pages"] == 3


def test_a_claim_is_cited_in_one_section_only():
    lesson, _ = run(DP.deepen(fake_llm(), sources(), CONCEPT, content(), 3))
    per_section = [{k for s in sec["sentences"] for k in s["claims"]} for sec in lesson["sections"]]
    assert len(per_section) == 3 and all(per_section)
    for i, a in enumerate(per_section):
        for b in per_section[i + 1:]:
            assert not (a & b)


def test_a_section_the_corpus_cannot_support_is_dropped_and_the_shortfall_said():
    barren = FakeCorpus(default=[])
    lesson, _ = run(DP.deepen(fake_llm(), barren, CONCEPT, content(), 8))
    # only the two existing claims exist, so at most one section clears the two-claim minimum
    assert len(lesson["sections"]) <= 1 and lesson["dropped_sections"]
    assert "supported about" in lesson.get("shortfall", "") or lesson["insufficient"]


def test_no_outline_means_no_lesson_rather_than_a_padded_one():
    lesson, changes = run(DP.deepen(fake_llm(outline="{}"), sources(), CONCEPT, content(), 5))
    assert lesson["insufficient"] and changes == {}


def test_merging_back_keeps_withdrawn_flags_the_dataclass_does_not_know():
    withdrawn = dict(claim("c1", "Old."), withdrawn=True)
    from lara.learn.claims import Claim
    merged = DP._merge_back([withdrawn], [Claim.from_dict(withdrawn)])
    assert merged[0]["withdrawn"] is True


# ── pipeline ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def _root(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path / "courses")
    PL._building.clear()
    PL._editing.clear()
    PL._writing.clear()


def built():
    m = model()
    course = run(PL.map_course(m, corpus(), run(SC.begin(m, "learn pretraining"))))
    run(PL.build_concept(m, corpus(), course, "c1"))
    return course


def test_variant_keys():
    assert PL.variant_key("tldr") == ("tldr", None) and PL.variant_key("thorough") == ("thorough", 5)
    assert PL.variant_key("pages", 7) == ("pages-7", 7) and PL.variant_key("pages", 500) == ("pages-20", 20)
    with pytest.raises(ValueError):
        PL.variant_key("essay")


def test_a_tldr_is_written_from_the_existing_claims_and_kept_beside_the_standard_lesson(_root):
    course = built()
    m = llm(("write a lesson", lambda s, p: "## Key points\nWarmup avoids early loss spikes [c1]."), ("strict fact-checker", "supports"))
    out = run(PL.write_variant(m, corpus(), course, "c1", "tldr"))
    saved = store.load_concept(course["id"], "c1")
    assert out["variant"] == "tldr" and "tldr" in saved["lessons"] and saved["lesson"]["sections"]
    assert "TL;DR of at most about 70 words" in m.calls[0][1], "two claims carry about 70 words"
    assert store.load_build(course["id"], "c1")["stage"] == "done"


def test_a_thorough_lesson_adds_claims_and_tops_up_the_quiz(_root):
    course = built()
    before = store.load_concept(course["id"], "c1")
    n_claims, n_items = len(before["claims"]), len(before["quiz"]["items"])
    quiz_items = json.dumps([{"type": "short", "question": f"Question number {i}?", "answer": "a", "claim": "c1", "explanation": "e"} for i in range(3)])
    extra = llm(("plan a self-study lesson", outline_reply()), ("extract atomic claims", extract_by_focus), ("write ONE section", section_writer),
                ("strict fact-checker", "supports"), ("Would this claim help", "yes"), ("Do these two claims", "different"),
                ("compare two claims", '{"relation": "agree", "note": ""}'), ("Write quiz items", quiz_items), ("ONLY the passage", "a"),
                ("The answer is:", "supports"))
    run(PL.write_variant(extra, sources(), course, "c1", "thorough"))
    saved = store.load_concept(course["id"], "c1")
    assert len(saved["claims"]) > n_claims and "thorough" in saved["lessons"]
    ids = [i["id"] for i in saved["quiz"]["items"]]
    assert len(ids) == len(set(ids)) and len(ids) >= n_items, "no id collisions with existing items"
    assert saved["lessons"]["thorough"]["target_pages"] == 5


def test_a_variant_cannot_be_written_before_the_concept_is_built(_root):
    m = model()
    course = run(PL.map_course(m, corpus(), run(SC.begin(m, "goal"))))
    with pytest.raises(ValueError):
        run(PL.write_variant(m, corpus(), course, "c1", "tldr"))


def test_a_failed_variant_is_recorded_and_the_existing_lesson_untouched(_root):
    course = built()

    async def boom(cfg, prompt, *, system="", **kw):
        raise RuntimeError("model down")

    bad = type(model())(complete=boom, window=200_000)
    with pytest.raises(RuntimeError):
        run(PL.write_variant(bad, corpus(), course, "c1", "thorough"))
    assert store.load_build(course["id"], "c1")["stage"] == "error"
    assert "lessons" not in store.load_concept(course["id"], "c1") and store.load_concept(course["id"], "c1")["lesson"]


def test_an_expansion_belongs_to_the_version_it_was_asked_about(_root):
    course = built()
    ct = store.load_concept(course["id"], "c1")
    ct.setdefault("lessons", {})["tldr"] = {"sections": [{"heading": "K", "sentences": [{"text": "t", "claims": ["c1"]}]}], "generated": 777.0}
    store.save_concept(course["id"], "c1", ct)
    m = llm(("learner highlighted", "Warmup avoids early loss spikes [c1].\nIt is corroborated [c1, c2]."), ("strict fact-checker", "supports"))
    out = run(PL.expand_selection(m, corpus(), course, "c1", selection="Warmup", variant="tldr"))
    assert out["lesson_generated"] == 777.0 and out["variant"] == "tldr"
    with pytest.raises(ValueError):
        run(PL.expand_selection(m, corpus(), course, "c1", selection="Warmup", variant="thorough"))


def test_withdrawing_a_claim_marks_only_the_variants_that_cite_it_stale():
    ct = content()
    ct["claims"][0]["passage"]["text"] = LONG
    ct["lessons"] = {"tldr": {"sections": [{"heading": "K", "sentences": [{"text": "t", "claims": ["c1"]}]}], "generated": 1.0},
                     "other": {"sections": [{"heading": "K", "sentences": [{"text": "t", "claims": ["c2"]}]}], "generated": 2.0}}
    run(LN.recheck_claim(llm(("strict fact-checker", "unrelated")), ct, "c1", "wrong"))
    assert ct["lessons"]["tldr"].get("stale") is True and not ct["lessons"]["other"].get("stale")


def test_two_requests_for_one_variant_share_a_single_write(_root):
    async def go():
        m = model()
        course = await PL.map_course(m, corpus(), await SC.begin(m, "goal"))
        await PL.build_concept(m, corpus(), course, "c1")
        a = PL.ensure_variant(llm(("write a lesson", "## K\nWarmup avoids early loss spikes [c1]."), ("strict fact-checker", "supports")), corpus(), course, "c1", "tldr")
        b = PL.ensure_variant(m, corpus(), course, "c1", "tldr")
        assert a is b
        await a
    run(go())


def test_rebuilding_a_concepts_claims_drops_versions_and_answers_that_cite_the_old_keys(_root):
    m = model()
    course = built()
    ct = store.load_concept(course["id"], "c1")
    ct["lessons"] = {"tldr": {"sections": [], "generated": 1.0}}
    ct["expansions"] = [{"id": "x1", "claims": []}]
    store.save_concept(course["id"], "c1", ct)
    run(PL.build_concept(m, corpus(), course, "c1", force=True))
    saved = store.load_concept(course["id"], "c1")
    assert "lessons" not in saved and "expansions" not in saved


def test_a_sections_length_is_capped_by_what_its_claims_can_carry():
    m = llm(("write ONE section", "Sentence one [c1]."), ("strict fact-checker", "supports"))
    run(DP.write_section(m, CONCEPT, {"heading": "H", "focus": "f"}, [claim("c1", "One claim.")], 400))
    assert f"at most about {DP.WORDS_PER_CLAIM} words" in m.calls[0][1]


def test_the_tldr_length_scales_down_for_a_concept_with_few_claims():
    from lara.learn import lesson as LE
    assert "at most about 35 words" in LE.tldr_note(1) and "at most about 150 words" in LE.tldr_note(30)


def test_a_malformed_outline_is_retried_once():
    replies = iter(["not json {", outline_reply()])
    m = llm(("plan a self-study lesson", lambda s, p: next(replies)))
    assert [s["heading"] for s in run(DP.plan_outline(m, CONCEPT, [], 5))[0]] == list(SECTIONS)
    assert len([c for c in m.calls if "plan a self-study lesson" in c[0]]) == 2
    assert run(DP.plan_outline(llm(("plan a self-study lesson", "nope")), CONCEPT, [], 5)) == ([], "nope")


def test_a_failed_attempt_is_reported_and_never_replaces_an_existing_version(_root):
    course = built()
    ct = store.load_concept(course["id"], "c1")
    ct["lessons"] = {"thorough": {"sections": [{"heading": "K", "sentences": [{"text": "old", "claims": ["c1"]}]}], "generated": 9.0}}
    store.save_concept(course["id"], "c1", ct)
    empty_outline = llm(("plan a self-study lesson", "{}"))
    with pytest.raises(ValueError):
        run(PL.write_variant(empty_outline, corpus(), course, "c1", "thorough"))
    saved = store.load_concept(course["id"], "c1")
    assert saved["lessons"]["thorough"]["generated"] == 9.0
    b = store.load_build(course["id"], "c1")
    assert b["stage"] == "error" and "outline" in b["error"]


def test_a_section_is_cut_back_to_its_share_of_the_requested_length():
    sents = [{"text": "word " * 10, "claims": ["c1"]} for _ in range(10)]      # 100 words
    assert len(DP.trim(sents, 35)) == 3 and len(DP.trim(sents, 1)) == 1, "always keeps one"
    assert DP.trim([], 50) == [] and len(DP.trim(sents, 1000)) == 10


def test_write_section_never_runs_far_past_the_words_it_was_given():
    many = "\n".join(f"This is sentence number {i} of the section [c1]." for i in range(40))
    m = llm(("write ONE section", many), ("strict fact-checker", "supports"))
    sentences, _ = run(DP.write_section(m, CONCEPT, {"heading": "H", "focus": "f"}, [claim("c1", "One.")] * 8, 80))
    assert 0 < sum(len(s["text"].split()) for s in sentences) <= 80 * DP.OVERRUN + 10


def test_a_failed_outline_says_what_the_model_replied():
    lesson, changes = run(DP.deepen(fake_llm(outline="I cannot help with that."), sources(), CONCEPT, content(), 5))
    assert lesson["insufficient"] and "I cannot help with that." in lesson["message"] and changes == {}
