import asyncio
import json
import time

import pytest

from lara.learn import learner as LN
from lara.learn import pipeline as PL
from lara.learn import scope as SC
from lara.learn import store
from learn_helpers import corpus, llm, model


@pytest.fixture(autouse=True)
def _root(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path / "courses")
    PL._building.clear()
    PL._building_topic.clear()


def run(c):
    return asyncio.run(c)


def ready_course(m):
    course = run(SC.begin(m, "learn pretraining"))
    return run(PL.map_course(m, corpus(), course))


def test_mapping_records_concepts_in_prerequisite_order():
    course = ready_course(model())
    assert course["status"] == "ready" and [c["title"] for c in course["concepts"]] == ["Warmup", "Decay"]
    assert course["concepts"][1]["prereqs"] == ["c1"]
    assert store.load_course(course["id"])["status"] == "ready"


def test_mapping_an_empty_corpus_fails_the_course_honestly():
    m = model()
    course = run(SC.begin(m, "goal"))
    course = run(PL.map_course(llm(("concept map", "{}")), corpus(), course))
    assert course["status"] == "failed" and "nothing" in course["error"]


def test_building_a_concept_runs_every_stage_and_persists_them():
    m = model()
    course = ready_course(m)
    content = run(PL.build_concept(m, corpus(), course, "c1"))
    assert [c["key"] for c in content["claims"]] == ["c1", "c2"]
    assert content["claims"][0]["certainty"] == "established"
    assert content["lesson"]["stats"]["grounded_pct"] == 100 and content["quiz"]["items"][0]["validated"]
    assert set(content["stages"]) == set(PL.STAGES) and store.load_build(course["id"], "c1")["stage"] == "done"
    assert store.load_concept(course["id"], "c1")["title"] == "Warmup"


def test_the_trace_is_written_incrementally_so_a_build_can_be_watched_live(monkeypatch):
    m = model()
    course = ready_course(m)
    saves = []
    real_save = store.save_concept

    def spy_save(course_id, cid, content):
        saves.append(json.loads(json.dumps(content.get("trace") or {})))
        real_save(course_id, cid, content)

    monkeypatch.setattr(store, "save_concept", spy_save)
    run(PL.build_concept(m, corpus(), course, "c1"))
    trace_saves = [s for s in saves if s]
    empties = [i for i, s in enumerate(trace_saves) if s.get("rounds") == []]
    populated = [i for i, s in enumerate(trace_saves) if s.get("rounds")]
    assert empties and populated, "both an empty-trace save (build just started) and a populated one exist"
    assert empties[0] < populated[0], "the empty trace reached disk before the populated one -- genuinely incremental"


def test_build_json_gets_a_started_timestamp_and_growing_token_counts():
    m = model()
    course = ready_course(m)
    before = time.time()
    run(PL.build_concept(m, corpus(), course, "c1"))
    after = time.time()
    b = store.load_build(course["id"], "c1")
    assert before <= b["started"] <= after
    assert b["tokens_in"] > 0 and b["tokens_out"] > 0, "every stage's own calls added to the running total"


def test_tokens_and_started_are_not_reset_by_a_no_op_rebuild():
    m = model()
    course = ready_course(m)
    run(PL.build_concept(m, corpus(), course, "c1"))
    done = store.load_build(course["id"], "c1")
    run(PL.build_concept(m, corpus(), course, "c1"))          # already fully built; nothing to do
    still = store.load_build(course["id"], "c1")
    assert still["started"] == done["started"] and still["tokens_in"] == done["tokens_in"]


def test_a_forced_rebuild_starts_the_token_count_over_rather_than_accumulating():
    m = model()
    course = ready_course(m)
    run(PL.build_concept(m, corpus(), course, "c1"))
    first = store.load_build(course["id"], "c1")
    run(PL.build_concept(m, corpus(), course, "c1", force=True))
    second = store.load_build(course["id"], "c1")
    assert second["started"] >= first["started"]
    # Same scripted calls happen again (force reruns every stage), so a genuine reset-then-
    # recount lands on the same total -- not first["tokens_in"] + more, which would mean the
    # old build's tokens were never cleared.
    assert second["tokens_in"] == first["tokens_in"] > 0


def test_built_stages_are_not_rebuilt_and_force_rebuilds():
    m = model()
    course = ready_course(m)
    run(PL.build_concept(m, corpus(), course, "c1"))
    before = len(m.calls)
    run(PL.build_concept(m, corpus(), course, "c1"))
    assert len(m.calls) == before
    run(PL.build_concept(m, corpus(), course, "c1", force=True))
    assert len(m.calls) > before


def test_a_second_course_reuses_a_shared_concept_instead_of_rebuilding_it():
    m = model()
    first = ready_course(m)
    run(PL.build_concept(m, corpus(), first, "c1"))
    second = ready_course(m)
    before = len(m.calls)
    content = run(PL.build_concept(m, corpus(), second, "c1"))
    assert content["reused"] and len(m.calls) == before
    # A reused concept skips every stage (nothing left to do), so run_stage's own save never
    # fires for it -- the reuse path has to persist it itself, or this course's own concept
    # file is silently never written even though build.json says "done".
    on_disk = store.load_concept(second["id"], "c1")
    assert on_disk is not None and on_disk["claims"] and on_disk["lesson"]


def test_a_failing_stage_is_recorded_on_the_course_and_partial_work_is_kept():
    m = model()
    course = ready_course(m)

    async def boom(cfg, prompt, *, system="", **kw):
        if "plan a self-study lesson" in system:
            raise RuntimeError("model down")
        return await m.complete(cfg, prompt, system=system, **kw)

    bad = type(m)(complete=boom, window=200_000)
    with pytest.raises(RuntimeError):
        run(PL.build_concept(bad, corpus(), course, "c1"))
    assert store.load_build(course["id"], "c1")["stage"] == "error" and "model down" in store.load_build(course["id"], "c1")["error"]
    assert store.load_concept(course["id"], "c1")["claims"], "claims survived the lesson failure"


def test_two_requests_for_one_concept_share_a_single_build():
    async def go():
        m = model()
        course = await PL.map_course(m, corpus(), await SC.begin(m, "goal"))
        a = PL.ensure_concept(m, corpus(), course, "c1")
        b = PL.ensure_concept(m, corpus(), course, "c1")
        assert a is b
        await a
    run(go())


def test_public_items_hide_the_answer_until_graded():
    item = {"id": "c1-q1", "concept": "c1", "type": "mcq", "question": "q", "choices": ["a"], "answer": "A",
            "explanation": "e", "claim": "c1", "source": {"title": "t"}}
    assert PL.public_item(item) == {"id": "c1-q1", "concept": "c1", "type": "mcq", "question": "q", "choices": ["a"]}


def test_answering_records_mastery_and_reveals_the_answer():
    m = model()
    course = ready_course(m)
    run(PL.build_concept(m, corpus(), course, "c1"))
    learner = LN.blank()
    graded = run(PL.answer_item(m, course, learner, "c1-q1", "A", 3))
    assert graded["correct"] and graded["answer"].startswith("A.") and learner["concepts"]["c1"]["mastery"] > 0
    assert store.load_learner(course["id"])["items"]["c1-q1"]["seen"] == 1
    with pytest.raises(KeyError):
        run(PL.answer_item(m, course, learner, "c1-q99", "A"))


def test_a_pretest_builds_only_claims_and_quiz_and_a_pass_skips_prerequisites():
    m = model()
    course = ready_course(m)
    learner = LN.blank()
    items = run(PL.start_pretest(m, corpus(), course, learner))
    assert {i["concept"] for i in items} == {"c1", "c2"}
    assert "lesson" not in store.load_concept(course["id"], "c1")
    for i in items:
        run(PL.answer_item(m, course, learner, i["id"], "A", 3))
    assert learner["pretest"]["state"] == "done"
    ov = PL.overview(course, learner)
    assert ov["pretest"] == "done" and all(c["passed"] or c["mastery"] >= 0.6 for c in ov["concepts"])


def test_overview_reports_progress_and_hides_the_next_items_answer():
    m = model()
    course = ready_course(m)
    run(PL.build_concept(m, corpus(), course, "c1"))
    learner = LN.blank()
    LN.concept_state(learner, "c1")["lesson_read"] = True
    ov = PL.overview(course, learner)
    assert ov["next"]["action"] == "quiz" and "answer" not in ov["next"]["item"]
    assert [c["built"] for c in ov["concepts"]][0] == list(PL.STAGES) and ov["concepts"][1]["unlocked"] is False


def test_flagging_a_claim_the_source_does_not_support_withdraws_it_on_disk():
    m = model()
    course = ready_course(m)
    run(PL.build_concept(m, corpus(), course, "c1"))
    learner = LN.blank()
    flagger = llm(("strict fact-checker", "unrelated"))
    out = run(PL.flag_claim(flagger, course, learner, "c1", "c1", "this is wrong"))
    saved = store.load_concept(course["id"], "c1")
    assert out["withdrawn"] and saved["claims"][0]["withdrawn"] and saved["lesson"]["stale"]
    assert saved["quiz"]["items"] == [] and learner["flags"][0]["withdrawn"] is True


def test_a_stale_lesson_is_regenerated_without_redoing_the_other_stages():
    m = model()
    course = ready_course(m)
    run(PL.build_concept(m, corpus(), course, "c1"))
    content = store.load_concept(course["id"], "c1")
    content["lesson"]["stale"] = True
    store.save_concept(course["id"], "c1", content)
    m.calls.clear()
    run(PL.build_concept(m, corpus(), course, "c1"))
    # The standard lesson researches its own outline section by section, so regenerating it
    # does re-plan and re-write -- but the concept's fixed passages are already fully used by
    # the claims stage, so no section's own research actually finds (or extracts) anything new.
    assert any("plan a self-study lesson" in s for s, _ in m.calls)
    assert any("write ONE section" in s for s, _ in m.calls)
    assert not any("extract atomic" in s for s, _ in m.calls)
    assert not store.load_concept(course["id"], "c1")["lesson"].get("stale")


def test_critique_updates_mastery_only_when_the_claims_speak_to_the_response():
    m = model()
    course = ready_course(m)
    run(PL.build_concept(m, corpus(), course, "c1"))
    learner = LN.blank()
    assert run(PL.submit_critique(m, course, learner, "c1", "my plan")) == []
    assert learner["concepts"] == {}


def topics_model():
    topics_reply = json.dumps([{"title": "Adam's beta_2", "note": "the momentum decay term"}])
    doc_reply = "Beta_2 controls how quickly the second-moment estimate adapts [c1]."
    return model(("about to read this lesson", topics_reply), ("background note on ONE topic", doc_reply))


def test_building_a_concept_indexes_topics_from_the_finished_lesson():
    m = topics_model()
    course = ready_course(m)
    content = run(PL.build_concept(m, corpus(), course, "c1"))
    assert content["topics"] == [{"id": "t1", "title": "Adam's beta_2", "note": "the momentum decay term"}]
    assert "topics" in content["stages"]


async def _ready(m):
    return await PL.map_course(m, corpus(), await SC.begin(m, "learn pretraining"))


def test_a_familiarity_answer_of_yes_is_recorded_without_building_a_doc():
    async def go():
        m = topics_model()
        course = await _ready(m)
        await PL.build_concept(m, corpus(), course, "c1")
        learner = LN.blank()
        state = await PL.set_familiarity(m, corpus(), course, learner, "c1", "t1", "yes", "")
        assert state["topics"]["t1"] == {"answer": "yes", "explain": ""}
        await asyncio.sleep(0)
        assert not PL._building_topic
        assert "topic_docs" not in (store.load_concept(course["id"], "c1") or {})
    run(go())


def test_a_familiarity_answer_of_no_builds_a_grounded_topic_doc():
    async def go():
        m = topics_model()
        course = await _ready(m)
        await PL.build_concept(m, corpus(), course, "c1")
        learner = LN.blank()
        await PL.set_familiarity(m, corpus(), course, learner, "c1", "t1", "no", "")
        await PL.ensure_topic_doc(m, corpus(), course, "c1", "t1")
        entry = store.load_concept(course["id"], "c1")["topic_docs"]["t1"]
        assert entry["status"] == "done" and entry["doc"]["insufficient"] is False
        assert [s["claims"] for sec in entry["doc"]["sections"] for s in sec["sentences"]] == [["c1"]]
    run(go())


def test_a_partial_answer_passes_the_learners_own_words_to_the_doc():
    async def go():
        m = topics_model()
        course = await _ready(m)
        await PL.build_concept(m, corpus(), course, "c1")
        learner = LN.blank()
        await PL.set_familiarity(m, corpus(), course, learner, "c1", "t1", "partial", "I know it decays.")
        task = PL._building_topic[(course["id"], "c1", "t1")]
        await task
        prompt = next(p for s, p in m.calls if "background note on ONE topic" in s)
        assert "READER ALREADY KNOWS: I know it decays." in prompt
        assert store.load_concept(course["id"], "c1")["topic_docs"]["t1"]["tailor"] == "I know it decays."
    run(go())


def test_two_familiarity_answers_for_one_topic_share_a_single_build():
    async def go():
        m = topics_model()
        course = await _ready(m)
        await PL.build_concept(m, corpus(), course, "c1")
        a = PL.ensure_topic_doc(m, corpus(), course, "c1", "t1")
        b = PL.ensure_topic_doc(m, corpus(), course, "c1", "t1")
        assert a is b
        await a
    run(go())


def test_rebuilding_a_concepts_claims_drops_stale_topic_docs():
    m = topics_model()
    course = ready_course(m)
    run(PL.build_concept(m, corpus(), course, "c1"))
    content = store.load_concept(course["id"], "c1")
    content.setdefault("topic_docs", {})["t1"] = {"status": "done", "doc": {"insufficient": False}}
    store.save_concept(course["id"], "c1", content)
    run(PL.build_concept(m, corpus(), course, "c1", force=True))
    assert "topic_docs" not in store.load_concept(course["id"], "c1")


def test_concurrent_builds_keep_their_own_progress():
    async def go():
        m = model()
        course = await PL.map_course(m, corpus(), await SC.begin(m, "goal"))
        # Each build gets its own copy of the course, as separate requests do.
        import copy
        a, b = copy.deepcopy(course), copy.deepcopy(course)
        await asyncio.gather(PL.build_concept(m, corpus(), a, "c1"), PL.build_concept(m, corpus(), b, "c2"))
        assert store.load_build(course["id"], "c1")["stage"] == "done"
        assert store.load_build(course["id"], "c2")["stage"] == "done"
    run(go())
