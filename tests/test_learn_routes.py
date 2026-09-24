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
    PL._building_topic.clear()
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
    pending = lambda: [*LR._tasks.values(), *PL._building.values(), *PL._building_topic.values()]
    while LR._tasks or any(not t.done() for t in pending()):
        await asyncio.gather(*pending(), return_exceptions=True)
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


def test_a_built_concept_exposes_its_retrieval_trace():
    async def go():
        cid = await ready()
        await LR.build(cid, "c1", None)
        await settle()
        trace = body(LR.concept(cid, "c1"))["trace"]
        assert "coverage" in trace and "budget" in trace and trace["rounds"], "trace is populated, not just present"
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


def _topics_llm():
    topics_reply = json.dumps([{"title": "Adam's beta_2", "note": "the momentum decay term"}])
    doc_reply = "Beta_2 controls how quickly the second-moment estimate adapts [c1]."
    return model(("about to read this lesson", topics_reply), ("background note on ONE topic", doc_reply))


def test_a_built_concept_lists_its_topics_and_a_no_answer_builds_a_doc():
    async def go():
        m = _topics_llm()

        async def fake_llm():
            return m

        LR._llm = fake_llm
        cid = await ready()
        await LR.build(cid, "c1", None)
        await settle()
        c = body(LR.concept(cid, "c1"))
        assert c["topics"] == [{"id": "t1", "title": "Adam's beta_2", "note": "the momentum decay term",
                                "doc_status": "todo"}]
        resp = await LR.familiarity(cid, "c1", LR.FamiliarityRequest(topic_id="t1", answer="no"))
        assert resp.status_code == 200 and body(resp)["topics"]["t1"]["answer"] == "no"
        await settle()
        c = body(LR.concept(cid, "c1"))
        assert c["topics"][0]["doc_status"] == "done"
        doc = body(LR.topic_doc(cid, "c1", "t1"))
        assert doc["status"] == "done" and doc["doc"]["insufficient"] is False and doc["title"] == "Adam's beta_2"
    run(go())


def test_a_partial_answer_requires_an_explanation_and_a_yes_needs_no_model():
    async def go():
        m = _topics_llm()

        async def fake_llm():
            return m

        LR._llm = fake_llm
        cid = await ready()
        await LR.build(cid, "c1", None)
        await settle()
        bad = await LR.familiarity(cid, "c1", LR.FamiliarityRequest(topic_id="t1", answer="partial"))
        assert bad.status_code == 400
        ok = await LR.familiarity(cid, "c1", LR.FamiliarityRequest(topic_id="t1", answer="yes"))
        assert ok.status_code == 200
        assert (await LR.familiarity(cid, "c1", LR.FamiliarityRequest(topic_id="t9", answer="yes"))).status_code == 404
        assert (await LR.familiarity(cid, "c1", LR.FamiliarityRequest(topic_id="t1", answer="maybe"))).status_code == 400
        assert LR.topic_doc(cid, "c1", "t9").status_code == 404
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


# ── the figure lookup injected into CorpusRetriever ──────────────────────────────────


def test_figure_lookup_is_none_with_no_cached_html():
    import types
    state = types.SimpleNamespace(raw_html_path=lambda arxiv_id: None)
    assert LR._figure_lookup(state)("2401.00001", 1, "S4.F2") is None


def test_figure_lookup_reads_the_cached_html_when_there_is_one(tmp_path, monkeypatch):
    import types

    from lara.serve import papers as papers_mod

    seen = []
    monkeypatch.setattr(papers_mod, "figure_image",
                        lambda path, arxiv_id, version, anchor: seen.append(
                            (path, arxiv_id, version, anchor)) or {"src": "https://x/y.png", "caption": "c"})
    path = tmp_path / "2401.00001.arxiv_html.html.zst"
    state = types.SimpleNamespace(raw_html_path=lambda arxiv_id: path)
    out = LR._figure_lookup(state)("2401.00001", 0, "S4.F2")
    assert out["src"] == "https://x/y.png"
    # version 0 (unknown) becomes 1, the same fallback resolve_full_paper-style callers use.
    assert seen == [(str(path), "2401.00001", 1, "S4.F2")]
