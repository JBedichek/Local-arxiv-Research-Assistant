"""Calibrated lessons: the evidence-based profile, the subject decomposition, need levels on the
map, the diagnostic, the treatment plan written into a lesson's objective, the simulated
reader, and the feedback that keeps the profile current."""

import asyncio
import json

import pytest

from lara.learn import decompose as DC
from lara.learn import diagnostic as DG
from lara.learn import graph as G
from lara.learn import learner as LN
from lara.learn import pipeline as PL
from lara.learn import profile as PR
from lara.learn import reader as RD
from lara.learn import research as RS
from lara.learn import scope as SC
from lara.learn import store
from lara.learn import treatment as TM
from learn_helpers import corpus, llm, model

DAY = 86_400


@pytest.fixture(autouse=True)
def _root(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path / "courses")
    monkeypatch.setattr(store, "PROFILE_PATH", tmp_path / "learner" / "profile.json")
    PL._building.clear()
    PL._editing.clear()


def run(c):
    return asyncio.run(c)


# ── the profile ──────────────────────────────────────────────────────────────────


def test_with_no_evidence_the_distribution_is_the_prior():
    assert PR.distribution(None) == pytest.approx(list(PR.PRIOR))


def test_a_strong_diagnostic_answer_moves_the_level_up():
    p = PR.load()
    PR.observe(p, "Attention", "probe", level=4, now=1000)
    s = PR.have(p, "Attention", now=1000)
    assert s["level"] >= 3 and s["expected"] > 2.5
    assert p["concepts"]["attention"]["evidence"][0]["level"] == 4


def test_old_evidence_counts_for_less_than_new():
    old, new = PR.load(), PR.load()
    PR.observe(old, "X", "probe", level=4, now=0)
    PR.observe(new, "X", "probe", level=4, now=0)
    assert (PR.have(old, "X", now=PR.HALF_LIFE_DAYS * 3 * DAY)["expected"]
            < PR.have(new, "X", now=0)["expected"])


def test_a_background_guess_alone_is_not_relied_on():
    p = PR.load()
    PR.observe(p, "X", "anchor", level=4, now=0)
    s = PR.have(p, "X", now=0)
    assert s["cautious"] < s["level"] or s["confidence"] < PR.CONFIDENT


def test_quiz_evidence_recorded_before_the_level_model_still_counts():
    legacy = {"title": "Warmup", "score": 0.9,
              "evidence": [{"correct": True, "confidence": 3, "ts": 0}] * 5, "updated": 0}
    assert PR.summary(legacy, now=0)["expected"] > PR.summary(None, now=0)["expected"]


def test_explain_highlights_on_a_term_lower_its_level():
    p = PR.load()
    PR.observe(p, "jargon", "explain", term=True, now=0)
    assert PR.have(p, "jargon", term=True, now=0)["expected"] < PR.summary(None, now=0)["expected"]


def _course():
    return {"id": "k-1", "goal": "g", "competencies": [],
            "concepts": [{"id": "c1", "title": "Basics", "prereqs": [], "need": 2, "summary": ""},
                         {"id": "c2", "title": "Middle", "prereqs": ["c1"], "need": 2, "summary": ""},
                         {"id": "c3", "title": "Frontier", "prereqs": ["c2"], "need": 3, "summary": ""}]}


def test_a_pass_is_evidence_for_prerequisites_and_a_fail_for_dependents():
    p = PR.load()
    assert PR.propagate(p, _course(), "c3", 3, now=0) == ["Middle", "Basics"]
    assert PR.propagate(p, _course(), "c1", 0, now=0) == ["Middle", "Frontier"]
    assert p["concepts"]["basics"]["evidence"][0]["kind"] == "prereq_of_passed"
    assert p["concepts"]["frontier"]["evidence"][0]["kind"] == "dependent_of_failed"


def test_any_new_evidence_marks_the_digest_stale():
    p = {**PR.load(), "digest": "old", "digest_stale": False}
    PR.observe(p, "X", "self", level=3)
    assert p["digest_stale"] is True


# ── treatment ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("need,have,expected", [
    (2, 2, TM.USE), (1, 0, TM.GLOSS), (2, 0, TM.INTUITION), (3, 2, TM.REFRESHER), (3, 1, TM.SECTION),
    (4, 0, TM.SECTION), (0, 0, TM.USE)])
def test_the_treatment_table(need, have, expected):
    assert TM.treatment(need, have) == expected


def _plan(*items):
    return {"concept": {"title": "C", "need": 3, "have": 1},
            "items": [{"title": t, "treatment": tr, "uncovered": False} for t, tr in items]}


def test_the_brief_says_what_to_use_explain_and_research():
    text = TM.brief(_plan(("A", TM.USE), ("B", TM.SECTION), ("C", TM.INTUITION)))
    assert "use freely, do not explain: A" in text and "explain fully, from the ground up" in text
    assert "ONCE in the whole lesson" in text and "Background:" not in text
    assert "do not explain it from general knowledge" in text


def test_an_explained_item_the_lesson_never_cites_is_marked_uncovered():
    plan = _plan(("loss spike", TM.INTUITION), ("Adam", TM.SECTION), ("SGD", TM.USE))
    lesson = {"sections": [{"heading": "", "sentences": [
        {"text": "A loss spike is a sudden jump in loss.", "claims": ["1"]},
        {"text": "Adam is also mentioned here.", "claims": []}]}]}
    assert TM.mark_uncovered(plan, lesson) == ["Adam"]
    assert TM.public(plan)["uncovered"] == ["Adam"]


def test_the_plan_goes_into_the_lessons_research_objective():
    objective = RS._lesson_objective({"title": "Warmup", "summary": "s", "goal": "g"},
                                     brief="THIS READER: needs this")
    assert objective.endswith("THIS READER: needs this")


# ── the decomposition and the map ────────────────────────────────────────────────

def researcher(text="Warmup [11] underlies Decay [12]."):
    calls = []

    async def research(question):
        calls.append(question)
        return {"text": text, "references": {"11": {"chunk_id": 11, "arxiv_id": "2401.1"}},
                "stopped_because": "" if text else "no evidence"}

    research.calls = calls
    return research


def test_the_decomposition_question_is_a_fixed_template_quoting_the_learners_words():
    r = researcher()
    run(DC.decompose(r, {"goal": "I want to learn  about\nvoice models", "competencies": []}))
    assert r.calls == ['Find the foundational concepts, from the simplest to the most complex, that '
                       'someone needs to understand for a course based on this request: '
                       '"I want to learn about voice models"']


def test_a_decomposition_is_researched_once_and_cached_per_request():
    r = researcher()
    course = {"goal": "learn schedules", "competencies": []}
    first = run(DC.decompose(r, course))
    second = run(DC.decompose(r, course))
    assert len(r.calls) == 1 and first["subject"] == "learn schedules"
    assert not first["cached"] and second["cached"] and second["text"] == first["text"]
    run(DC.decompose(r, course, force=True))
    assert len(r.calls) == 2


def test_an_empty_decomposition_is_an_error_not_a_cached_map():
    with pytest.raises(ValueError, match="no evidence"):
        run(DC.decompose(researcher(text=""), {"goal": "g", "competencies": []}))
    assert store.decomposition_get("g") is None


def test_the_mapping_objective_starts_from_the_decomposition():
    objective = G._course_objective({"goal": "g", "competencies": []},
                                    {"text": "Foundations: tokenization; then attention."})
    assert "foundations the subject builds on to its frontier" in objective
    assert "Foundations: tokenization; then attention." in objective


NEEDS = ("judge, for each concept", json.dumps({"concepts": {"c1": {"need": 4, "tier": "foundation"},
                                                             "c2": {"need": 9}}}))


def test_needs_are_assigned_and_clamped_and_a_missing_one_defaults():
    course = {"goal": "g", "concepts": [{"id": "c1", "title": "A"}, {"id": "c2", "title": "B"},
                                        {"id": "c3", "title": "C"}]}
    assert run(G.assign_needs(llm(NEEDS), course)) == 2
    assert [c["need"] for c in course["concepts"]] == [4, 4, 2]
    assert course["concepts"][0]["tier"] == "foundation" and "tier" not in course["concepts"][2]


async def _topic_graph(objective: str) -> dict:
    _topic_graph.objectives.append(objective)
    return {"subjects": [{"title": "Fundamentals", "summary": "",
                          "concepts": [{"title": "Warmup", "summary": "What warmup does [1].",
                                        "prereqs_text": "", "competencies": []},
                                       {"title": "Decay", "summary": "Decay after Warmup [1].",
                                        "prereqs_text": "Warmup", "competencies": []}]}],
            "references": {"1": {"chunk_id": 1, "arxiv_id": "1234.5678", "title": "A paper"}},
            "degraded": False, "tokens_in": 10, "tokens_out": 5, "rounds": 1}


def mapped(m, research=None):
    _topic_graph.objectives = []
    course = run(SC.begin(m, "learn pretraining"))
    return run(PL.map_course(m, corpus(), course, topic_graph_synth=_topic_graph,
                             decompose_research=research))


def test_mapping_decomposes_the_subject_first_and_assigns_needs():
    course = mapped(model(NEEDS), researcher())
    assert course["decomposition"]["subject"] == "learn pretraining"
    assert "Warmup [11] underlies Decay [12]." in _topic_graph.objectives[0]
    assert course["concepts"][0]["need"] == 4 and course["status"] == "awaiting_approval"


def test_a_failed_decomposition_does_not_fail_the_course():
    course = mapped(model(), researcher(text=""))
    assert course["status"] == "awaiting_approval" and "error" in course["decomposition"]
    assert "A MAP OF THE WHOLE SUBJECT" not in _topic_graph.objectives[0]


# ── the diagnostic ───────────────────────────────────────────────────────────────

PROBE = json.dumps({"question": "Why X?", "reference": "Because Y.", "rubric": ["0", "1", "2", "3", "4"]})


def test_the_conversation_records_use_and_anchors_then_stops():
    reply = json.dumps({"question": None, "use": "tune my runs",
                        "anchors": [{"domain": "software engineering", "depth": "expert"}]})
    diag, profile = DG.blank(), PR.load()
    run(DG.converse(llm(("getting to know a learner", reply)), _course(), diag, profile))
    assert diag["pending"] is None and profile["use"] == "tune my runs"
    assert profile["anchors"] == [{"domain": "software engineering", "depth": "expert"}]


def test_the_conversation_ends_at_the_turn_limit_even_if_the_model_keeps_asking():
    m = llm(("getting to know a learner", json.dumps({"question": "More?", "anchors": []})))
    diag = {**DG.blank(), "turns": [{"question": "q", "answer": "a"}] * DG.MAX_TURNS}
    run(DG.converse(m, _course(), diag, PR.load()))
    assert diag["pending"] is None


def test_anchors_seed_weak_prior_evidence():
    m = llm(("estimate how well a learner", json.dumps({"levels": {"c1": 3, "zz": 4}})))
    profile = {**PR.load(), "anchors": [{"domain": "ml", "depth": "solid"}]}
    assert run(DG.seed_from_anchors(m, _course(), profile, now=0)) == 1
    assert profile["concepts"]["basics"]["evidence"][0]["kind"] == "anchor"


def test_questions_are_prepared_from_short_research_runs():
    r = researcher(text="Basics are basic [11].")
    diag = DG.blank()
    run(DG.prepare(llm(("diagnostic question", PROBE)), r, _course(), diag))
    assert diag["state"] == "probing" and set(diag["probes"]) == {"c1", "c2", "c3"}
    assert len(r.calls) == 3 and diag["probes"]["c1"]["reference"] == "Because Y."


def test_a_question_whose_research_finds_nothing_is_never_asked():
    diag = DG.blank()
    run(DG.prepare(llm(("diagnostic question", PROBE)), researcher(text=""), _course(), diag))
    assert all("error" in p for p in diag["probes"].values())
    assert DG.next_probe(_course(), diag, PR.load()) is None


def _prepared():
    return {**DG.blank(), "state": "probing",
            "probes": {cid: json.loads(PROBE) for cid in ("c1", "c2", "c3")}}


def test_the_first_question_is_the_middle_of_the_outline():
    assert DG.next_probe(_course(), _prepared(), PR.load(), now=0) == "c2"


def test_i_dont_know_is_graded_without_a_model_call():
    m = llm()
    diag, profile = _prepared(), PR.load()
    assert run(DG.answer(m, _course(), diag, profile, "c2", "no idea", now=0)) == \
        {"level": 0, "feedback": "", "self": True}
    assert not m.calls and profile["concepts"]["middle"]["evidence"][0]["kind"] == "self"
    assert profile["concepts"]["frontier"]["evidence"][0]["kind"] == "dependent_of_failed"


def test_a_graded_answer_is_recorded_and_the_search_moves_on():
    m = llm(("grade a learner", json.dumps({"level": 3, "feedback": "Good."})))
    diag, profile = _prepared(), PR.load()
    assert run(DG.answer(m, _course(), diag, profile, "c2", "Because Y.", now=0))["level"] == 3
    assert diag["results"]["c2"]["feedback"] == "Good." and diag["current"] in ("c1", "c3", None)


def test_the_diagnostic_stops_at_its_budget():
    m = llm(("grade a learner", json.dumps({"level": 1, "feedback": ""})))
    diag = {**_prepared(), "budget": 1}
    run(DG.answer(m, _course(), diag, PR.load(), "c2", "something", now=0))
    assert diag["state"] == "done" and diag["current"] is None


def test_what_the_profile_is_sure_of_is_marked_known_in_the_course():
    profile = PR.load()
    for _ in range(3):
        PR.observe(profile, "Basics", "probe", level=4, now=0)
    learner = LN.blank()
    assert DG.apply_to_course(_course(), learner, profile, now=0) == ["c1"]
    assert learner["concepts"]["c1"]["inferred"] and learner["pretest"]["state"] == "done"


def test_reference_answers_are_hidden_until_answered():
    out = DG.public(_course(), {**_prepared(), "current": "c2"}, PR.load(), now=0)
    assert out["current"] == {"concept": "c2", "title": "Middle", "question": "Why X?"}
    assert "Because Y." not in json.dumps(out)


def test_uncertain_prerequisites_are_offered_as_quick_checks():
    profile = PR.load()
    assert [c["concept"] for c in DG.checks_for(_course(), _prepared(), profile, "c3", now=0)] == ["c2", "c1"]
    for _ in range(4):
        PR.observe(profile, "Middle", "probe", level=3, now=0)
    assert [c["concept"] for c in DG.checks_for(_course(), _prepared(), profile, "c3", now=0)] == ["c1"]


# ── lessons written to the plan ──────────────────────────────────────────────────

TERMS = ("list the technical terms", json.dumps({"terms": [{"term": "loss spike", "need": 2},
                                                           {"term": "Adam", "need": 3}]}))


def _lesson_synth():
    seen = []

    async def synth(objective: str) -> dict:
        seen.append(objective)
        return {"deliverable": "## Warmup\nA loss spike is a sudden jump in loss [1].\nWarmup avoids them [1, 2].",
                "references": {"1": {"chunk_id": 1, "arxiv_id": "1234.5678", "title": "A paper",
                                     "claim": "Loss spikes are sudden jumps."},
                               "2": {"chunk_id": 2, "arxiv_id": "1234.5678", "title": "A paper",
                                     "claim": "Warmup avoids them."}},
                "degraded": False, "tokens_in": 10, "tokens_out": 5, "rounds": 1}

    synth.seen = seen
    return synth


def built(m):
    course = mapped(m)
    PL.approve_plan(course)
    synth = _lesson_synth()
    content = run(PL.build_concept(m, corpus(), course, "c1", lesson_synth=synth))
    return course, content, synth


def test_a_lesson_is_researched_and_written_to_this_readers_plan():
    course, content, synth = built(model(TERMS))
    assert "THIS READER" in synth.seen[0] and "loss spike" in synth.seen[0]
    by = {i["title"]: i for i in content["plan"]["items"]}
    assert by["loss spike"]["treatment"] == TM.INTUITION and by["Adam"]["treatment"] == TM.SECTION
    assert content["plan"]["uncovered"] == ["Adam"]


def test_a_term_the_learner_knows_is_used_freely():
    profile = PR.load()
    for _ in range(4):
        PR.observe(profile, "loss spike", "probe", level=3, term=True)
    PR.save(profile)
    _, content, synth = built(model(TERMS))
    assert {i["title"]: i for i in content["plan"]["items"]}["loss spike"]["treatment"] == TM.USE
    assert "use freely, do not explain: loss spike" in synth.seen[0]


def test_the_simulated_reader_rewrites_only_what_the_judge_accepts():
    lesson = {"sections": [{"heading": "H", "sentences": [
        {"text": "Warmup avoids spikes.", "claims": ["1"]},
        {"text": "It uses a ramp.", "claims": ["1"]}]}]}
    claims = [{"key": "1", "text": "Warmup raises the rate gradually.", "certainty": "single-source",
               "passage": {"date": ""}, "conditions": ""}]
    m = llm(("specific reader", json.dumps([{"n": 2, "missing": "what a ramp is"}])),
            ("could not follow", json.dumps([{"n": 2, "text": "Warmup raises the rate gradually [1]."}])),
            ("strict fact-checker", "supports"))
    assert run(RD.review(m, lesson, claims, {"items": [], "concept": {}})) == {"flagged": 1, "rewritten": 1}
    assert lesson["sections"][0]["sentences"][1]["text"].startswith("Warmup raises")
    assert lesson["sections"][0]["sentences"][0]["text"] == "Warmup avoids spikes."


def test_the_simulated_reader_runs_only_when_turned_on(monkeypatch):
    monkeypatch.setattr(store, "shared_get", lambda *a, **k: None)   # build both from scratch
    m = model(TERMS)
    _, content, _ = built(m)
    assert "reader" not in content["lesson"]
    course = mapped(m)
    PL.approve_plan(course)
    content = run(PL.build_concept(m, corpus(), course, "c1", lesson_synth=_lesson_synth(),
                                   simulated_reader=True))
    assert content["lesson"]["reader"] == {"flagged": 0, "rewritten": 0}


# ── feedback ─────────────────────────────────────────────────────────────────────


def test_explain_on_a_term_is_evidence_and_deeper_raises_the_need():
    course = mapped(model())
    PL.record_highlight(course, "c1", "loss spikes", "explain")
    assert PR.load()["terms"]["loss-spikes"]["evidence"][0]["kind"] == "explain"
    before = course["concepts"][0].get("need", 2)
    PL.record_highlight(course, "c1", "a longer passage about warmup and its schedule here", "deeper")
    assert store.load_course(course["id"])["concepts"][0]["need"] == before + 1


def test_i_know_this_is_evidence_and_passes_the_concept():
    course = mapped(model())
    learner = LN.blank()
    PL.mark_known(course, learner, "c1")
    assert learner["concepts"]["c1"]["passed"]
    assert PR.load()["concepts"]["warmup"]["evidence"][0] == {
        **PR.load()["concepts"]["warmup"]["evidence"][0], "kind": "self", "level": 3}


def test_saying_whether_you_know_a_lesson_topic_is_term_evidence():
    course = mapped(model())
    content = {"topics": [{"id": "t1", "title": "cosine annealing"}], "stages": {}}
    store.save_concept(course["id"], "c1", content)
    run(PL.set_familiarity(model(), corpus(), course, LN.blank(), "c1", "t1", "yes", ""))
    ev = PR.load()["terms"]["cosine-annealing"]["evidence"][0]
    assert ev["kind"] == "self" and ev["level"] == 3


# ── compression ──────────────────────────────────────────────────────────────────

from lara.learn import compress as CP  # noqa: E402


def _long_lesson():
    return {"insufficient": False, "generated": 1.0, "stats": {}, "sections": [
        {"heading": "Spectrograms", "sentences": [
            {"text": "A spectrogram shows frequency content over time.", "claims": ["11"]},
            {"text": "It is computed with the STFT.", "claims": ["12"]}]},
        {"heading": "What the concept is", "sentences": [
            {"text": "A spectrogram, again, shows frequency over time.", "claims": ["11"]}]}]}


def test_each_level_is_its_own_fixed_prompt():
    m = llm(("Shorten and compress", "## Spectrograms\nA spectrogram shows frequency over time [11]."))
    run(CP.compress(m, _long_lesson(), "high"))
    run(CP.compress(m, _long_lesson(), "low"))
    systems = [s for s, _ in m.calls]
    assert systems[0].startswith("Shorten and compress this lesson text with high discarding")
    assert systems[1].startswith("Shorten and compress this lesson text with low discarding")


def test_compression_keeps_only_real_citations_and_reports_the_ratio():
    m = llm(("Shorten and compress", "## Spectrograms\nA spectrogram shows frequency over time [11, 99]."))
    out = run(CP.compress(m, _long_lesson(), "med"))
    assert out["sections"] == [{"heading": "Spectrograms", "sentences": [
        {"text": "A spectrogram shows frequency over time.", "claims": ["11"]}]}]
    assert out["compression"]["level"] == "med" and out["compression"]["ratio"] < 1


def test_a_long_lesson_is_compressed_in_chunks_each_seeing_what_was_kept(monkeypatch):
    monkeypatch.setattr(CP, "CHUNK_WORDS", 5)
    m = llm(("Shorten and compress", "## Spectrograms\nA spectrogram shows frequency over time [11]."))
    run(CP.compress(m, _long_lesson(), "high"))
    assert len(m.calls) == 2
    assert "ALREADY KEPT:\n(nothing yet)" in m.calls[0][1]
    assert "A spectrogram shows frequency over time [11]" in m.calls[1][1].split("TEXT TO COMPRESS")[0]


def test_an_unknown_level_or_missing_lesson_is_refused():
    with pytest.raises(ValueError):
        run(CP.compress(llm(), _long_lesson(), "extreme"))
    with pytest.raises(ValueError):
        run(CP.compress(llm(), None, "high"))


def test_a_compressed_version_is_stored_beside_the_standard_lesson():
    m = model(TERMS, ("Shorten and compress", "## Warmup\nWarmup avoids them [1, 2]."))
    course, content, _ = built(m)
    lesson = run(PL.write_variant(m, corpus(), course, "c1", "compress-high"))
    stored = store.load_concept(course["id"], "c1")
    assert stored["lessons"]["compress-high"]["variant"] == "compress-high"
    assert stored["lesson"]["sections"] == content["lesson"]["sections"]
    assert lesson["sections"][0]["sentences"][0]["claims"] == ["1", "2"]
    with pytest.raises(ValueError):
        PL.variant_key("compress-extreme")


def test_refreshing_a_research_driven_lesson_keeps_the_researched_lesson():
    """"Refresh from the corpus" (force=True) used to run the legacy lesson stage after the
    research-driven one, and depth.deepen overwrote the lesson research had just written."""
    m = model(TERMS)
    course, _, _ = built(m)
    synth = _lesson_synth()
    content = run(PL.build_concept(m, corpus(), course, "c1", lesson_synth=synth, force=True))
    assert len(synth.seen) == 1
    texts = [s["text"] for sec in content["lesson"]["sections"] for s in sec["sentences"]]
    assert texts == ["A loss spike is a sudden jump in loss.", "Warmup avoids them."]
    assert not any("plan a self-study lesson" in sys for sys, _ in m.calls[-40:])


# ── scope, length and reorganization ─────────────────────────────────────────────

from lara.learn import reorganize as RG  # noqa: E402


def test_a_lesson_objective_names_the_other_lessons_and_a_length():
    objective = RS._lesson_objective({"title": "Residual stream", "summary": "s", "goal": "g"},
                                     others=["Superposition", "Sparse autoencoders"],
                                     words=RS.lesson_words(4))
    assert "SCOPE:" in objective and "- Superposition" in objective and "- Sparse autoencoders" in objective
    assert "LENGTH: about 6000 words" in objective
    assert RS.lesson_words(1) < RS.lesson_words(3) < RS.lesson_words(4) and RS.lesson_words("x") == 2000


def test_a_lessons_research_is_told_its_scope_and_length():
    course, content, synth = built(model(TERMS))
    assert "SCOPE:" in synth.seen[0] and "- Decay" in synth.seen[0] and "- Warmup" not in synth.seen[0]
    assert "LENGTH: about" in synth.seen[0]


def _messy_lesson():
    return {"insufficient": False, "generated": 1.0, "stats": {}, "sections": [
        {"heading": "Where sources disagree", "sentences": [{"text": "Papers differ on X.", "claims": ["2"]}]},
        {"heading": "Superposition aside", "sentences": [{"text": "Superposition packs features.", "claims": ["3"]}]},
        {"heading": "The residual stream", "sentences": [{"text": "The stream is a running sum.", "claims": ["1"]}]}]}


OUTLINE = json.dumps({"sections": [
    {"heading": "A running sum", "establishes": "what the stream is", "from": ["s3", "s1"], "words": 200},
    {"heading": "Nothing", "establishes": "x", "from": ["s9"]}]})


def test_reorganizing_plans_an_outline_and_rewrites_from_the_lessons_own_text():
    m = llm(("reorganize an existing lesson", OUTLINE),
            ("rewrite part of an existing lesson", "The stream is a running sum [1].\nPapers differ on X [2, 99]."))
    out = run(RG.reorganize(m, _messy_lesson(), {"title": "Residual stream"}, others=["Superposition"], words=300))
    assert [s["heading"] for s in out["sections"]] == ["A running sum"]
    assert [x["claims"] for x in out["sections"][0]["sentences"]] == [["1"], ["2"]]
    assert out["reorganized"]["source_sections"] == 3 and out["reorganized"]["sections"] == 1
    outline_prompt = m.calls[0][1]
    assert "OTHER LESSONS IN THIS COURSE:\n- Superposition" in outline_prompt and "[s2] Superposition aside" in outline_prompt
    section_prompt = m.calls[1][1]
    assert "The stream is a running sum [1]." in section_prompt and "Superposition packs" not in section_prompt


def test_reorganizing_without_a_usable_outline_is_an_error():
    with pytest.raises(ValueError, match="outline"):
        run(RG.reorganize(llm(("reorganize an existing lesson", "{}")), _messy_lesson(), {"title": "t"}))
    with pytest.raises(ValueError):
        run(RG.reorganize(llm(), None, {"title": "t"}))


def test_a_reorganized_version_is_stored_beside_the_standard_lesson():
    m = model(TERMS, ("reorganize an existing lesson", json.dumps({"sections": [
        {"heading": "Warmup, in order", "establishes": "e", "from": ["s1"], "words": 200}]})),
              ("rewrite part of an existing lesson", "Warmup avoids them [1, 2]."))
    course, content, _ = built(m)
    run(PL.write_variant(m, corpus(), course, "c1", "reorganized"))
    stored = store.load_concept(course["id"], "c1")
    assert stored["lessons"]["reorganized"]["sections"][0]["heading"] == "Warmup, in order"
    assert stored["lesson"]["sections"] == content["lesson"]["sections"]
