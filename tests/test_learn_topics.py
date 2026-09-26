import asyncio
import json

from lara.learn import topics as TP
from learn_helpers import FakeCorpus, llm, passage

LONG = "warmup avoids loss spikes early in training. " * 6
CONCEPT = {"id": "c1", "title": "warmup", "summary": "s", "goal": "pretrain an LLM"}


def run(c):
    return asyncio.run(c)


def lesson(sentences):
    return {"sections": [{"heading": "H", "sentences": [{"text": t, "claims": ["c1"]} for t in sentences]}]}


def test_topics_are_parsed_and_capped():
    reply = json.dumps([{"title": f"topic {i}", "note": f"note {i}"} for i in range(8)])
    m = llm(("about to read this lesson", reply))
    out = run(TP.extract_topics(m, CONCEPT, lesson(["Warmup ramps the rate up gradually."])))
    assert len(out) == TP.MAX_TOPICS
    assert out[0] == {"id": "t1", "title": "topic 0", "note": "note 0"}


def test_no_lesson_or_an_insufficient_one_extracts_nothing_without_calling_the_model():
    m = llm(("about to read this lesson", "should not be called"))
    assert run(TP.extract_topics(m, CONCEPT, None)) == []
    assert run(TP.extract_topics(m, CONCEPT, {"insufficient": True, "sections": []})) == []
    assert m.calls == []


def test_a_bad_reply_degrades_to_no_topics_rather_than_a_guess():
    m = llm(("about to read this lesson", "not json"))
    assert run(TP.extract_topics(m, CONCEPT, lesson(["x"]))) == []


def test_build_doc_writes_a_grounded_note_and_an_optional_chart():
    ps = [passage(1, "a " + LONG, arxiv="2401.1"), passage(2, "b " + LONG, arxiv="2402.2")]
    fc = FakeCorpus(default=ps)
    extract_reply = json.dumps([{"passage": 1, "claim": "Adam's beta_2 smooths the second moment."},
                                {"passage": 2, "claim": "A high beta_2 slows adaptation to new gradients."}])
    note = "A high beta_2 [c1, c2]."
    m = llm(("extract atomic claims", extract_reply), ("strict fact-checker", "supports"),
            ("background note on ONE topic", note))
    out = run(TP.build_doc(m, fc, CONCEPT, {"id": "t1", "title": "Adam's beta_2", "note": "the momentum term"}))
    assert out["insufficient"] is False
    assert [s["claims"] for sec in out["sections"] for s in sec["sentences"]] == [["c1", "c2"]]
    assert {c["key"] for c in out["claims"]} == {"c1", "c2"}


def test_build_doc_tells_the_model_what_the_reader_already_knows():
    ps = [passage(1, "a " + LONG, arxiv="2401.1"), passage(2, "b " + LONG, arxiv="2402.2")]
    extract_reply = json.dumps([{"passage": 1, "claim": "Claim one."}, {"passage": 2, "claim": "Claim two."}])
    m = llm(("extract atomic claims", extract_reply), ("strict fact-checker", "supports"),
            ("background note on ONE topic", "Some new detail [c1, c2]."))
    run(TP.build_doc(m, FakeCorpus(default=ps), CONCEPT, {"id": "t1", "title": "Adam's beta_2"},
                     tailor="I know it's an EMA decay rate but not the typical value."))
    prompt = next(p for s, p in m.calls if "background note on ONE topic" in s)
    assert "READER ALREADY KNOWS: I know it's an EMA decay rate" in prompt


def test_build_doc_is_insufficient_when_the_corpus_has_too_little():
    m = llm(("extract atomic claims", "[]"))
    out = run(TP.build_doc(m, FakeCorpus(default=[]), CONCEPT, {"id": "t1", "title": "obscure term"}))
    assert out["insufficient"] is True and out["sections"] == []
