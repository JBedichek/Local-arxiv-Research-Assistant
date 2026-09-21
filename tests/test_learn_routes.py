"""The /api/learn handlers, called directly in one event loop so background tasks (mapping,
building, the diagnostic) run as they do in the server."""

import asyncio
import json

import pytest

from lara.learn import pipeline as PL
from lara.learn import store
from lara.serve.routes import learn as LR
from learn_helpers import corpus, llm, model


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path / "courses")
    PL._building.clear()
    LR._tasks.clear()
    PL._editing.clear()
    PL._writing.clear()
    m = model()

    async def fake_llm():
        return m

    async def fake_corpus():
        return corpus(), None

    monkeypatch.setattr(LR, "_llm", fake_llm)
    monkeypatch.setattr(LR, "_corpus", fake_corpus)
    return m


def body(resp):
    return json.loads(resp.body)


async def settle():
    while LR._tasks or any(not t.done() for t in PL._building.values()):
        await asyncio.gather(*LR._tasks.values(), *PL._building.values(), return_exceptions=True)
        await asyncio.sleep(0)


def run(coro):
    return asyncio.run(coro)


async def ready():
    resp = await LR.start(LR.GoalRequest(goal="learn pretraining"))
    cid = body(resp)["id"]
    assert resp.status_code == 201
    await LR.map_it(cid)
    await settle()
    return cid


def test_an_empty_goal_is_refused_and_unknown_courses_404():
    async def go():
        assert (await LR.start(LR.GoalRequest(goal="  "))).status_code == 400
        assert LR.show("nope").status_code == 404
        assert LR.concept("nope", "c1").status_code == 404
        assert (await LR.map_it("nope")).status_code == 404
    run(go())


def test_scope_map_and_show_a_course():
    async def go():
        cid = await ready()
        shown = body(LR.show(cid))
        assert shown["status"] == "ready" and [c["title"] for c in shown["concepts"]] == ["Warmup", "Decay"]
        assert shown["scope"]["competencies"][0]["text"] == "choose a schedule"
        assert shown["next"] == {"action": "build", "concept": "c1"}
        assert body(LR.courses())["courses"][0]["id"] == cid
    run(go())


def test_mapping_refuses_a_course_still_being_scoped(_env):
    async def go():
        LR_llm = llm(("design the scope", json.dumps({"competencies": [{"text": "a"}], "question":
                      {"text": "Budget?", "why": "w", "options": ["s"]}})))

        async def scoping():
            return LR_llm

        LR._llm = scoping
        cid = body(await LR.start(LR.GoalRequest(goal="g")))["id"]
        assert body(LR.show(cid))["scope"]["pending"]["question"] == "Budget?"
        assert (await LR.map_it(cid)).status_code == 409
        assert (await LR.answer(cid, LR.AnswerRequest(answer="small"))).status_code == 200
    run(go())


def test_building_a_concept_in_the_background_then_reading_it():
    async def go():
        cid = await ready()
        assert (await LR.build(cid, "c1", None)).status_code == 202
        await settle()
        c = body(LR.concept(cid, "c1"))
        assert c["lesson"]["stats"]["grounded_pct"] == 100 and c["quiz"]["items"] == 1
        assert c["claims"][0]["certainty"] == "established" and c["build"]["stage"] == "done"
        assert body(LR.show(cid))["next"]["action"] == "lesson"
        after = body(LR.read(cid, "c1"))
        assert after["next"]["action"] == "quiz" and "answer" not in after["next"]["item"]
    run(go())


def test_answering_an_item_grades_it_and_returns_the_new_overview():
    async def go():
        cid = await ready()
        await LR.build(cid, "c1", None)
        await settle()
        LR.read(cid, "c1")
        out = body(await LR.answer_item(cid, "c1-q1", LR.ItemAnswer(response="A", confidence=3)))
        assert out["graded"]["correct"] and out["graded"]["answer"].startswith("A.")
        assert out["overview"]["concepts"][0]["mastery"] > 0
        assert (await LR.answer_item(cid, "c1-q9", LR.ItemAnswer(response="A"))).status_code == 404
    run(go())


def test_the_diagnostic_builds_in_the_background_and_a_pass_skips_ahead():
    async def go():
        cid = await ready()
        assert (await LR.pretest_start(cid)).status_code == 202
        await settle()
        state = body(LR.pretest(cid))
        assert state["state"] == "active" and len(state["items"]) == 2
        assert all("answer" not in i for i in state["items"])
        for item in state["items"]:
            await LR.answer_item(cid, item["id"], LR.ItemAnswer(response="A", confidence=3))
        assert body(LR.pretest(cid))["state"] == "done"
        assert body(LR.show(cid))["pretest"] == "done"
    run(go())


def test_skipping_the_diagnostic_marks_it_done():
    async def go():
        cid = await ready()
        assert body(LR.pretest_skip(cid))["state"] == "done"
    run(go())


def test_flagging_and_critique_and_delete(_env):
    async def go():
        cid = await ready()
        await LR.build(cid, "c1", None)
        await settle()
        assert body(await LR.critique(cid, "c1", LR.TextRequest(text="my plan")))["points"] == []
        assert (await LR.critique(cid, "c1", LR.TextRequest(text=" "))).status_code == 400

        async def strict():
            return llm(("strict fact-checker", "unrelated"))

        LR._llm = strict
        out = body(await LR.flag(cid, "c1", "c1", LR.FlagRequest(note="wrong")))
        assert out["withdrawn"] is True
        assert (await LR.flag(cid, "c1", "zz", LR.FlagRequest())).status_code == 404
        assert body(LR.remove(cid))["deleted"] is True and LR.show(cid).status_code == 404
    run(go())


def test_starting_the_lessons_declines_the_diagnostic():
    async def go():
        cid = await ready()
        await LR.build(cid, "c1", None)
        await settle()
        assert body(LR.show(cid))["pretest"] == "todo"
        LR.read(cid, "c1")
        assert body(LR.show(cid))["pretest"] == "done"
    run(go())


def test_a_multiple_choice_answer_is_graded_without_reaching_for_a_model(_env):
    async def go():
        cid = await ready()
        await LR.build(cid, "c1", None)
        await settle()

        async def unreachable():
            raise AssertionError("no model should be needed for a multiple-choice grade")

        LR._llm = unreachable
        out = body(await LR.answer_item(cid, "c1-q1", LR.ItemAnswer(response="A")))
        assert out["graded"]["correct"] is True
    run(go())


def test_writing_another_lesson_version_validates_then_runs_in_the_background(_env):
    async def go():
        cid = await ready()
        assert (await LR.write_lesson("nope", "c1", LR.LessonRequest(variant="tldr"))).status_code == 404
        assert (await LR.write_lesson(cid, "c1", LR.LessonRequest(variant="essay"))).status_code == 400
        assert (await LR.write_lesson(cid, "c1", LR.LessonRequest(variant="tldr"))).status_code == 409, "not built yet"
        await LR.build(cid, "c1", None)
        await settle()

        async def tldr_llm():
            return llm(("write a lesson", "## Key points\nWarmup avoids early loss spikes [c1]."), ("strict fact-checker", "supports"))

        LR._llm = tldr_llm
        resp = await LR.write_lesson(cid, "c1", LR.LessonRequest(variant="pages", pages=99))
        assert resp.status_code == 202 and body(resp) == {"status": "writing", "variant": "pages-20", "pages": 20}
        for task in list(PL._writing.values()):
            await asyncio.gather(task, return_exceptions=True)
        shown = body(LR.concept(cid, "c1"))
        assert set(shown["lessons"]) >= {"standard"}
    run(go())


def test_the_concept_lists_every_version_and_expansions_accept_a_variant(_env):
    async def go():
        cid = await ready()
        await LR.build(cid, "c1", None)
        await settle()
        content = store.load_concept(cid, "c1")
        content.setdefault("lessons", {})["tldr"] = {"sections": [{"heading": "K", "sentences": [{"text": "t", "claims": ["c1"]}]}],
                                                    "generated": 42.0, "variant": "tldr"}
        store.save_concept(cid, "c1", content)
        shown = body(LR.concept(cid, "c1"))
        assert set(shown["lessons"]) == {"standard", "tldr"} and shown["lesson"] == shown["lessons"]["standard"]

        async def expander():
            return llm(("learner highlighted", "Warmup avoids early loss spikes [c1].\nIt is corroborated [c1, c2]."), ("strict fact-checker", "supports"))

        LR._llm = expander
        out = body(await LR.expand(cid, "c1", LR.ExpandRequest(selection="Warmup avoids", variant="tldr")))
        assert out["lesson_generated"] == 42.0
        assert (await LR.expand(cid, "c1", LR.ExpandRequest(selection="Warmup avoids", variant="thorough"))).status_code == 409
    run(go())
