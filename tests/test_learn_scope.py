import asyncio
import json

import pytest

from lara.learn import scope as SC
from lara.learn import store
from learn_helpers import llm


@pytest.fixture(autouse=True)
def _root(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path / "courses")


def run(c):
    return asyncio.run(c)


def reply(comps, question=None):
    return json.dumps({"competencies": [{"text": t} for t in comps], "question": question})


def test_begin_asks_a_question_when_the_model_does():
    q = {"text": "What is your compute budget?", "why": "changes the recipe", "options": ["small", "large"]}
    course = run(SC.begin(llm(("GOAL", reply(["choose a schedule", "pick a mixture"], q))), "learn pretraining"))
    assert course["status"] == "scoping" and course["pending"]["question"].startswith("What is")
    assert [c["text"] for c in course["competencies"]] == ["choose a schedule", "pick a mixture"]
    assert store.load_course(course["id"])["goal"] == "learn pretraining"


def test_no_question_means_the_scope_is_ready():
    course = run(SC.begin(llm(("GOAL", reply(["a", "b"]))), "goal"))
    assert course["status"] == "scoped" and course["pending"] is None


def test_answering_replans_and_records_what_changed():
    q = {"text": "Budget?", "why": "w", "options": ["s", "l"]}
    replies = iter([reply(["a", "b"], q), reply(["a", "c"])])
    m = llm(("GOAL", lambda s, p: next(replies)))
    course = run(SC.begin(m, "goal"))
    course = run(SC.answer(m, course, "small"))
    assert course["status"] == "scoped"
    assert course["qa"][0]["answer"] == "small" and course["qa"][0]["question"] == "Budget?"
    assert course["map_history"][-1]["diff"] == {"added": ["c"], "removed": ["b"]}
    assert "Q: Budget?\nA: small" in m.calls[-1][1]


def test_the_last_round_never_asks_again():
    q = {"text": "More?", "why": "", "options": []}
    m = llm(("GOAL", reply(["a"], q)))
    course = run(SC.begin(m, "goal"))
    for _ in range(SC.MAX_QUESTIONS):
        if course["pending"]:
            course = run(SC.answer(m, course, "x"))
    assert course["pending"] is None and course["status"] == "scoped"
    assert len(course["qa"]) == SC.MAX_QUESTIONS


def test_an_unreadable_reply_fails_the_course_instead_of_scoping_nothing():
    course = run(SC.begin(llm(("GOAL", "sorry, no")), "goal"))
    assert course["status"] == "failed"


def test_answering_with_no_question_pending_is_an_error():
    course = run(SC.begin(llm(("GOAL", reply(["a"]))), "goal"))
    with pytest.raises(ValueError):
        run(SC.answer(llm(), course, "x"))


def test_accept_takes_the_map_as_it_stands():
    q = {"text": "Q?", "why": "", "options": []}
    course = run(SC.begin(llm(("GOAL", reply(["a"], q))), "goal"))
    assert SC.accept(course)["status"] == "scoped" and course["pending"] is None


def test_store_round_trips_lists_and_deletes(tmp_path):
    course = run(SC.begin(llm(("GOAL", reply(["a"]))), "goal"))
    assert [c["id"] for c in store.list_courses()] == [course["id"]]
    assert store.delete_course(course["id"]) and store.list_courses() == []
    with pytest.raises(ValueError):
        store.course_dir("../escape")


def test_shared_concepts_expire():
    store.shared_put("Learning rate", {"claims": []})
    assert store.shared_get("Learning rate") is not None
    assert store.shared_get("Learning rate", max_age_days=-1) is None


def test_replanning_shows_the_model_the_current_competencies_to_keep():
    q = {"text": "Budget?", "why": "w", "options": ["s"]}
    replies = iter([reply(["keep me", "b"], q), reply(["keep me", "c"])])
    m = llm(("GOAL", lambda s, p: next(replies)))
    course = run(SC.begin(m, "goal"))
    assert "CURRENT COMPETENCIES" not in m.calls[0][1]
    run(SC.answer(m, course, "small"))
    assert "CURRENT COMPETENCIES:\n- keep me\n- b" in m.calls[1][1]
    assert "exact wording" in m.calls[1][0]
