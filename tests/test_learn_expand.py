import asyncio
import json

import pytest

from lara.learn import expand as EX
from lara.learn import learner as LN
from lara.learn import lesson as LE
from lara.learn import pipeline as PL
from lara.learn import scope as SC
from lara.learn import store
from learn_helpers import FakeCorpus, corpus, llm, model, passage

LONG = "warmup avoids loss spikes early in training. " * 6
CONCEPT = {"id": "c1", "title": "warmup", "summary": "s", "goal": "pretrain an LLM"}


def run(c):
    return asyncio.run(c)


def claim(key, text, chunk=1, arxiv="2401.1"):
    return {"key": key, "text": text, "certainty": "single-source", "conditions": "", "kind": "finding",
            "passage": {"text": LONG, "chunk_id": chunk, "arxiv_id": arxiv, "title": "P", "date": "2024-01-01"},
            "corroborated_by": [], "conflicts": [], "superseded_by": "", "flags": []}


def content():
    return {"claims": [claim("c1", "Warmup avoids early loss spikes.", 1),
                       claim("c2", "Warmup length of 1000 steps was used for the 1B model.", 2, "2401.2"),
                       claim("c3", "Unrelated tokenizer fact.", 3, "2401.3")],
            "lesson": {"sections": [], "generated": 111.0}, "expansions": []}


ANSWER = "learner highlighted"


def test_the_concepts_own_claims_answer_first_and_no_search_happens():
    reply = ("Warmup lasted 1000 steps for the 1B model [c2].\nIt avoids early loss spikes [c1, c2]."
             "\nLoss spikes are more likely without it [c1].\nThe effect is strongest at the start of training [c1]."
             "\nIt was validated on the 1B model specifically [c2].")
    m = llm((ANSWER, reply), ("strict fact-checker", "supports"))
    c = FakeCorpus(default=[passage(9, LONG)])
    out = run(EX.expand(m, c, CONCEPT, content(), selection="Warmup avoids spikes", selection_claims=["c1"], section=2,
                        lesson_generated=111.0))
    assert out["searched"] is False and out["claims"] == [] and c.queries == []
    assert out["id"] == "x1" and out["section"] == 2 and out["lesson_generated"] == 111.0
    assert [s["claims"] for s in out["answer"]["sections"][0]["sentences"][:2]] == [["c2"], ["c1", "c2"]]


def test_a_generic_answer_that_only_restates_the_highlighted_claims_falls_back_to_search():
    only_seen = "Warmup avoids early loss spikes [c1].\nIt matters early in training [c1]."
    extracted = json.dumps([{"passage": 1, "claim": "Warmup of 2000 steps stabilised the 7B run.", "conditions": "7B"}])
    grounded = "A 2000-step warmup stabilised the 7B run [x1c1].\nThe loss spikes it prevents occur early [c1]."
    replies = iter([only_seen, grounded])
    m = llm((ANSWER, lambda s, p: next(replies)), ("extract atomic claims", extracted), ("strict fact-checker", "supports"),
            ("Would this claim help", "yes"), ("compare two claims", '{"relation": "agree", "note": ""}'))
    c = FakeCorpus(default=[passage(9, LONG, arxiv="2409.9")])
    out = run(EX.expand(m, c, CONCEPT, content(), selection="Warmup avoids spikes", selection_claims=["c1"]))
    assert out["searched"] is True and c.queries and "Warmup avoids spikes" in c.queries[0]
    assert [x["key"] for x in out["claims"]] == ["x1c1"] and out["claims"][0]["passage"]["arxiv_id"] == "2409.9"


def test_a_specific_question_may_be_answered_from_broader_concept_claims_without_a_search():
    reply = ("Warmup avoids early loss spikes [c1].\nA 1000-step warmup was used for the 1B model [c2]."
             "\nThe mechanism is a gradual learning-rate ramp-up [c1].\nWithout it, spikes occur in early batches [c1]."
             "\nThis holds regardless of batch size [c1].")
    m = llm((ANSWER, reply), ("strict fact-checker", "supports"))
    c = FakeCorpus(default=[passage(9, LONG)])
    out = run(EX.expand(m, c, CONCEPT, content(), selection="Warmup avoids spikes", question="Why does it avoid spikes?",
                        selection_claims=["c1"]))
    assert out["searched"] is False and c.queries == [] and out["question"] == "Why does it avoid spikes?"
    assert "REQUEST: Why does it avoid spikes?" in m.calls[0][1]


def test_a_specific_question_answered_only_by_restating_the_highlighted_claim_still_searches():
    """A question narrows what the model is asked -- not what it may answer *from*. An answer
    that just restates the same claim the highlighted text already cites has not actually
    addressed anything new, question or not, so it must still trigger a search."""
    only_seen = ("Warmup avoids early loss spikes [c1].\nIt was needed from the first steps [c1]."
                "\nThe mechanism is a gradual learning-rate ramp-up [c1].\nWithout it, spikes occur in early batches [c1]."
                "\nThis holds regardless of batch size [c1].")
    extracted = json.dumps([{"passage": 1, "claim": "Warmup of 2000 steps stabilised the 7B run.", "conditions": "7B"}])
    grounded = "A 2000-step warmup stabilised the 7B run [x1c1]."
    replies = iter([only_seen, grounded])
    m = llm((ANSWER, lambda s, p: next(replies)), ("extract atomic claims", extracted), ("strict fact-checker", "supports"),
            ("Would this claim help", "yes"), ("compare two claims", '{"relation": "agree", "note": ""}'))
    c = FakeCorpus(default=[passage(9, LONG, arxiv="2409.9")])
    out = run(EX.expand(m, c, CONCEPT, content(), selection="Warmup avoids spikes", question="Why does it avoid spikes?",
                        selection_claims=["c1"]))
    assert out["searched"] is True and c.queries


def test_when_nothing_can_be_supported_the_learner_is_told_and_nothing_is_stored():
    m = llm((ANSWER, "INSUFFICIENT"), ("extract atomic claims", "[]"))
    out = run(EX.expand(m, FakeCorpus(default=[passage(9, LONG)]), CONCEPT, content(), selection="x"))
    assert out["insufficient"] is True and "nothing more" in out["message"]


def test_new_searches_skip_passages_the_concept_already_used():
    m = llm((ANSWER, "INSUFFICIENT"), ("extract atomic claims", "[]"))
    c = FakeCorpus(default=[passage(1, LONG), passage(2, LONG, arxiv="2401.2"), passage(77, LONG, arxiv="2409.9")])
    run(EX.expand(m, c, CONCEPT, content(), selection="x"))
    extract_prompt = [p for s, p in m.calls if "extract atomic claims" in s][0]
    assert extract_prompt.count("[1]") == 1 and "[2]" not in extract_prompt, "only the one unseen passage"


def test_a_new_claim_that_repeats_an_existing_one_is_not_added():
    dup = json.dumps([{"passage": 1, "claim": "Warmup avoids early loss spikes."}])
    m = llm((ANSWER, "INSUFFICIENT"), ("extract atomic claims", dup), ("strict fact-checker", "supports"), ("Would this claim help", "yes"))
    out = run(EX.expand(m, FakeCorpus(default=[passage(9, LONG, arxiv="2409.9")]), CONCEPT, content(), selection="x"))
    assert out["insufficient"] is True


def test_expansion_ids_and_claim_keys_continue_from_what_is_stored():
    ct = content()
    ct["expansions"] = [{"id": "x1"}, {"id": "x3"}]
    assert EX._next_id(ct["expansions"]) == 4 and EX._next_id([]) == 1


def test_lesson_parsing_accepts_expansion_claim_keys():
    [(_, sents)] = LE.parse("Detail here [x2c1, c3].", {"x2c1", "c3"})
    assert sents[0]["claims"] == ["x2c1", "c3"] and sents[0]["text"] == "Detail here."


# ── persistence, flags, routes ───────────────────────────────────────────────────

@pytest.fixture
def _root(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path / "courses")
    PL._building.clear()
    PL._editing.clear()


def built(m):
    course = run(PL.map_course(m, corpus(), run(SC.begin(m, "learn pretraining"))))
    run(PL.build_concept(m, corpus(), course, "c1"))
    return course


def test_an_expansion_is_saved_with_the_concept(_root):
    m = model()
    course = built(m)
    m2 = llm(("learner highlighted", "Warmup avoids early loss spikes [c1].\nIt is corroborated [c1, c2]."), ("strict fact-checker", "supports"))
    out = run(PL.expand_selection(m2, corpus(), course, "c1", selection="Warmup avoids", selection_claims=[], section=0))
    saved = store.load_concept(course["id"], "c1")
    assert [e["id"] for e in saved["expansions"]] == [out["id"]] and out["lesson_generated"] == saved["lesson"]["generated"]
    assert PL.delete_expansion(course["id"], "c1", out["id"]) is True
    assert store.load_concept(course["id"], "c1")["expansions"] == [] and PL.delete_expansion(course["id"], "c1", "x9") is False


def test_expanding_a_concept_with_no_lesson_is_refused(_root):
    m = model()
    course = run(PL.map_course(m, corpus(), run(SC.begin(m, "goal"))))
    with pytest.raises(ValueError):
        run(PL.expand_selection(m, corpus(), course, "c1", selection="x"))


def test_flagging_a_claim_an_expansion_added_marks_only_that_expansion_stale():
    ct = content()
    ct["expansions"] = [{"id": "x1", "stale": False, "claims": [claim("x1c1", "An added claim.")]}]
    out = run(LN.recheck_claim(llm(("strict fact-checker", "unrelated")), ct, "x1c1", "wrong"))
    assert out["withdrawn"] and ct["expansions"][0]["stale"] is True and ct["expansions"][0]["claims"][0]["withdrawn"]
    assert not ct["lesson"].get("stale"), "the lesson does not rest on it"


def test_the_expand_route_validates_and_returns_the_expansion(_root, monkeypatch):
    from lara.serve.routes import learn as LR

    m = model()
    course = built(m)

    async def fake_llm():
        return llm(("learner highlighted", "Warmup avoids early loss spikes [c1].\nIt is corroborated [c1, c2]."),
                   ("strict fact-checker", "supports"))

    async def fake_corpus():
        return corpus(), None

    monkeypatch.setattr(LR, "_llm", fake_llm)
    monkeypatch.setattr(LR, "_corpus", fake_corpus)

    async def go():
        assert (await LR.expand("nope", "c1", LR.ExpandRequest(selection="x"))).status_code == 404
        assert (await LR.expand(course["id"], "c1", LR.ExpandRequest(selection="  "))).status_code == 400
        resp = await LR.expand(course["id"], "c1", LR.ExpandRequest(selection="Warmup avoids", claims=["c1"], section=0))
        assert resp.status_code == 200 and json.loads(resp.body)["id"] == "x1"
        shown = json.loads(LR.concept(course["id"], "c1").body)
        assert [e["id"] for e in shown["expansions"]] == ["x1"]
        assert json.loads(LR.remove_expansion(course["id"], "c1", "x1").body)["deleted"] is True
        assert (await LR.expand(course["id"], "c2", LR.ExpandRequest(selection="x"))).status_code == 409
    run(go())
