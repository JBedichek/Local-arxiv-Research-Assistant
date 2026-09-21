import asyncio

from lara.learn import learner as LN
from learn_helpers import llm

COURSE = {"competencies": [{"id": "k1", "text": "do A"}, {"id": "k2", "text": "do B"}],
          "concepts": [{"id": "c1", "title": "Basics", "prereqs": [], "competencies": ["k1"]},
                       {"id": "c2", "title": "Mid", "prereqs": ["c1"], "competencies": ["k1"]},
                       {"id": "c3", "title": "Adv", "prereqs": ["c2"], "competencies": ["k2"]}]}


def item(cid, n, kind="short"):
    return {"id": f"{cid}-q{n}", "concept": cid, "type": kind, "claim": "c1"}


def content(cid, n_items=2):
    return {"lesson": {"sections": []}, "quiz": {"items": [item(cid, i + 1) for i in range(n_items)]}, "claims": []}


def test_mastery_rises_with_correct_answers_and_confident_errors_cost_more():
    up = LN.mastery_after(0.0, True, 2, "short")
    assert 0.3 < up < 0.4 and LN.mastery_after(0.0, True, 1, "short") < up
    assert LN.mastery_after(0.8, False, 3, "short") < LN.mastery_after(0.8, False, 1, "short") < 0.8
    assert LN.mastery_after(0.5, True, 2, "mcq") < LN.mastery_after(0.5, True, 2, "short")


def test_a_lapse_comes_back_in_minutes_and_success_spaces_out():
    s = LN.schedule({}, False, 2, 1000.0)
    assert s["due"] == 1000.0 + LN.RETRY_SECONDS and s["lapses"] == 1
    s = LN.schedule({}, True, 2, 0.0)
    assert s["interval"] == 1.0 and s["due"] == LN.DAY
    s = LN.schedule(s, True, 2, LN.DAY)
    s = LN.schedule(s, True, 2, 4 * LN.DAY)
    assert s["interval"] > 3.0 and s["reps"] == 3


def test_record_answer_updates_mastery_schedule_and_log():
    lr = LN.blank()
    LN.record_answer(lr, item("c1", 1), True, 2, now=5.0)
    assert lr["concepts"]["c1"]["mastery"] > 0 and lr["items"]["c1-q1"]["seen"] == 1
    assert lr["log"] == [{"item": "c1-q1", "correct": True, "confidence": 2, "ts": 5.0}]


def test_concepts_unlock_when_their_prerequisites_are_learned():
    lr = LN.blank()
    assert LN.unlocked(COURSE, lr, "c1") and not LN.unlocked(COURSE, lr, "c2")
    LN.concept_state(lr, "c1")["mastery"] = LN.UNLOCK
    assert LN.unlocked(COURSE, lr, "c2")


def test_next_action_walks_build_lesson_quiz_then_advances():
    lr, contents = LN.blank(), {}
    assert LN.next_action(COURSE, lr, contents) == {"action": "build", "concept": "c1"}
    contents["c1"] = content("c1")
    assert LN.next_action(COURSE, lr, contents)["action"] == "lesson"
    LN.concept_state(lr, "c1")["lesson_read"] = True
    a = LN.next_action(COURSE, lr, contents)
    assert a["action"] == "quiz" and a["item"]["id"] == "c1-q1"
    for n in (1, 2):
        LN.record_answer(lr, item("c1", n), True, 3, now=0.0)
    lr["items"]["c1-q1"]["due"] = lr["items"]["c1-q2"]["due"] = 10 ** 12       # nothing due
    assert LN.next_action(COURSE, lr, contents) == {"action": "build", "concept": "c2"}
    assert LN.concept_state(lr, "c1")["passed"] and LN.concept_state(lr, "c1")["mastery"] >= LN.MASTERED


def test_due_reviews_come_before_new_material():
    lr, contents = LN.blank(), {"c1": content("c1")}
    LN.record_answer(lr, item("c1", 1), False, 2, now=0.0)
    a = LN.next_action(COURSE, lr, contents, now=LN.RETRY_SECONDS + 1)
    assert a["action"] == "review" and a["item"]["id"] == "c1-q1"


def test_a_missed_item_is_served_again_before_the_concept_is_passed():
    lr, contents = LN.blank(), {"c1": content("c1", 1)}
    LN.concept_state(lr, "c1")["lesson_read"] = True
    LN.record_answer(lr, item("c1", 1), False, 1, now=0.0)
    lr["items"]["c1-q1"]["due"] = 10 ** 12
    a = LN.next_action(COURSE, lr, contents, now=1.0)
    assert a["action"] == "quiz" and a["item"]["id"] == "c1-q1"


def test_everything_mastered_is_done():
    lr = LN.blank()
    for c in COURSE["concepts"]:
        LN.concept_state(lr, c["id"])["mastery"] = 0.9
    assert LN.next_action(COURSE, lr, {}) == {"action": "done"}


def test_pretest_samples_evenly_and_a_pass_infers_prerequisites():
    big = {"concepts": [{"id": f"c{i}", "prereqs": [], "competencies": []} for i in range(1, 13)]}
    assert LN.pretest_concepts(big, 4) == ["c1", "c5", "c8", "c12"]
    assert LN.pretest_concepts(COURSE) == ["c1", "c2", "c3"]
    lr = LN.blank()
    LN.finish_pretest(COURSE, lr, {"c3": (True, 3)})
    assert lr["concepts"]["c3"]["mastery"] == LN.PRETEST_PASS_MASTERY
    assert lr["concepts"]["c2"]["inferred"] and lr["concepts"]["c1"]["mastery"] == LN.INFERRED_MASTERY
    assert lr["pretest"]["state"] == "done"


def test_a_lucky_low_confidence_pass_does_not_skip_material():
    lr = LN.blank()
    LN.finish_pretest(COURSE, lr, {"c3": (True, 1), "c1": (False, 3)})
    assert all(s["mastery"] == 0.0 for s in lr["concepts"].values())


def test_competency_progress_averages_its_concepts_and_flags_mastery():
    lr = LN.blank()
    LN.concept_state(lr, "c1")["mastery"] = 0.9
    LN.concept_state(lr, "c2")["mastery"] = 0.8
    prog = {p["id"]: p for p in LN.competency_progress(COURSE, lr)}
    assert prog["k1"]["mastered"] and prog["k1"]["progress"] == 0.85 and not prog["k2"]["mastered"]


def _flag_content():
    return {"claims": [{"key": "c1", "text": "Warmup helps.", "passage": {"text": "p"}}],
            "lesson": {"sections": []}, "quiz": {"items": [{"id": "q1", "claim": "c1"}, {"id": "q2", "claim": "c2"}]}}


def test_a_flag_on_an_unsupported_claim_withdraws_it_and_what_rests_on_it():
    ct = _flag_content()
    out = asyncio.run(LN.recheck_claim(llm(("CLAIM", "unrelated")), ct, "c1", "seems wrong"))
    assert out["withdrawn"] and ct["claims"][0]["withdrawn"] and ct["lesson"]["stale"]
    assert [i["id"] for i in ct["quiz"]["items"]] == ["q2"]


def test_a_flag_on_a_supported_claim_is_recorded_and_the_claim_stays():
    ct = _flag_content()
    out = asyncio.run(LN.recheck_claim(llm(("CLAIM", "supports")), ct, "c1", "hm"))
    assert not out["withdrawn"] and "withdrawn" not in ct["claims"][0]
    assert ct["claims"][0]["flags"][0]["note"] == "hm" and len(ct["quiz"]["items"]) == 2


def test_a_concept_the_corpus_cannot_teach_is_skipped_without_blocking_what_follows():
    lr = LN.blank()
    thin = {"lesson": {"insufficient": True, "sections": []}, "quiz": {"items": []}, "claims": []}
    assert LN.next_action(COURSE, lr, {"c1": thin}) == {"action": "build", "concept": "c2"}
    assert LN.concept_state(lr, "c1")["unavailable"] and LN.concept_state(lr, "c1")["mastery"] == 0.0
    assert not LN.competency_progress(COURSE, lr)[0]["mastered"], "no mastery claimed without sources"
