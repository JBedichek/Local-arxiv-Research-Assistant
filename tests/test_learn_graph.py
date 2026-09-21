import asyncio
import json

from lara.learn import graph as G
from learn_helpers import FakeCorpus, llm, passage

LONG = "x " * 120


def run(c):
    return asyncio.run(c)


def course():
    return {"goal": "pretrain an LLM", "competencies": [{"id": "sched", "text": "choose a schedule"},
                                                        {"id": "mix", "text": "pick a data mixture"}]}


def corpus():
    return FakeCorpus(default=[passage(1, LONG, arxiv="2401.1", title="A Survey of LLMs"),
                               passage(2, LONG, arxiv="2401.2", title="Some paper"),
                               passage(3, LONG, arxiv="2401.3", title="Scaling laws")])


def reply(concepts):
    return ("PASSAGES", json.dumps({"concepts": concepts}))


def test_a_concept_needs_a_source_and_is_renumbered():
    m = llm(reply([{"id": "a", "title": "Optimizers", "passages": [1, 2], "competencies": ["sched"]},
                   {"id": "b", "title": "Invented topic", "passages": [], "competencies": ["mix"]},
                   {"id": "c", "title": "Bogus cites", "passages": [99]}]))
    out = run(G.build(m, corpus(), course()))
    assert [c["title"] for c in out["concepts"]] == ["Optimizers"] and out["dropped"] == 2
    assert out["concepts"][0]["id"] == "c1" and len(out["concepts"][0]["sources"]) == 2
    assert out["uncovered"] == ["pick a data mixture"]


def test_prerequisites_are_remapped_and_order_is_topological():
    m = llm(reply([{"id": "z", "title": "Advanced", "passages": [1], "prereqs": ["y"], "competencies": ["sched"]},
                   {"id": "y", "title": "Basics", "passages": [1], "competencies": ["mix"]}]))
    out = run(G.build(m, corpus(), course()))
    titles = [c["title"] for c in out["concepts"]]
    assert titles == ["Basics", "Advanced"]
    adv = out["concepts"][1]
    assert adv["prereqs"] == [out["concepts"][0]["id"]]


def test_cycles_are_broken_and_reported():
    m = llm(reply([{"id": "a", "title": "A", "passages": [1], "prereqs": ["b"]},
                   {"id": "b", "title": "B", "passages": [1], "prereqs": ["a"]},
                   {"id": "s", "title": "Self", "passages": [1], "prereqs": ["s"]}]))
    out = run(G.build(m, corpus(), course()))
    assert len(out["removed_edges"]) == 1
    deps = {c["id"]: c["prereqs"] for c in out["concepts"]}
    assert not (deps["c1"] == ["c2"] and deps["c2"] == ["c1"]) and deps["c3"] == []


def test_unreadable_reply_yields_no_concepts_rather_than_inventing_some():
    out = run(G.build(llm(("PASSAGES", "nope")), corpus(), course()))
    assert out["concepts"] == [] and len(out["uncovered"]) == 2


def test_skeleton_prefers_overview_papers_and_caps_per_paper():
    ps = [passage(i, LONG, arxiv="2401.9", title="Plain") for i in range(1, 6)] + \
         [passage(10, LONG, arxiv="2401.8", title="A Survey of Things")]
    got = run(G.skeleton(FakeCorpus(default=ps), "goal", []))
    assert got[0].arxiv_id == "2401.8" and sum(p.arxiv_id == "2401.9" for p in got) == G.MAX_PER_PAPER
