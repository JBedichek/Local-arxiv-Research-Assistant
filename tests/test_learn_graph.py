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


# ── the research-driven path: subjects grouping concepts, not a flat list ──────────

def _refs(*keys):
    return {k: {"chunk_id": int(k), "arxiv_id": "2401.1", "title": "A paper"} for k in keys}


async def _synth_result(objective: str) -> dict:
    return {"subjects": [
        {"title": "Fundamentals", "summary": "The basics.",
         "concepts": [{"title": "Warmup", "summary": "Warmup helps [1].",
                      "prereqs_text": "", "competencies": ["sched"]},
                     {"title": "Decay", "summary": "Decay too [2].",
                      "prereqs_text": "after Warmup", "competencies": ["sched"]}]},
        {"title": "No sources", "summary": "",
         "concepts": [{"title": "Unsupported", "summary": "Nothing cited here.",
                      "prereqs_text": "", "competencies": []}]},
    ], "references": _refs("1", "2"), "degraded": False, "tokens_in": 9, "tokens_out": 4,
           "rounds": 2}


def test_build_from_research_flattens_subjects_into_the_same_concepts_pipeline_expects():
    out = run(G.build_from_research(course(), synth=_synth_result))
    assert [c["title"] for c in out["concepts"]] == ["Warmup", "Decay"]
    assert out["dropped"] == 1, "the uncited 'Unsupported' concept is dropped, like G.build's own"


def test_build_from_research_keeps_subject_grouping_alongside_the_flat_list():
    out = run(G.build_from_research(course(), synth=_synth_result))
    assert len(out["subjects"]) == 1, "the empty 'No sources' subject drops out entirely"
    subj = out["subjects"][0]
    assert subj["title"] == "Fundamentals"
    warmup, decay = out["concepts"]
    assert subj["concept_ids"] == [warmup["id"], decay["id"]]
    assert warmup["subject"] == subj["id"] and decay["subject"] == subj["id"]


def test_build_from_research_still_resolves_prereqs_across_subject_boundaries():
    # "after Warmup" names another concept by title -- the same substring match G.build's
    # research-driven sibling already does, unaffected by which subject either sits in.
    out = run(G.build_from_research(course(), synth=_synth_result))
    warmup, decay = out["concepts"]
    assert decay["prereqs"] == [warmup["id"]]


async def _synth_no_subjects(objective: str) -> dict:
    return {"subjects": [], "references": {}, "degraded": False, "tokens_in": 0,
           "tokens_out": 0, "rounds": 0}


def test_build_from_research_with_no_subjects_yields_an_honestly_empty_course():
    out = run(G.build_from_research(course(), synth=_synth_no_subjects))
    assert out["concepts"] == [] and out["subjects"] == []


def test_revise_from_research_replaces_the_concept_map_from_the_resumed_graph():
    calls = {}

    async def revise(feedback: str) -> dict:
        calls["feedback"] = feedback
        return {"subjects": [{"title": "Fundamentals", "summary": "",
                              "concepts": [{"title": "Only decay", "summary": "Decay [2].",
                                           "prereqs_text": "", "competencies": []}]}],
               "references": _refs("2"), "degraded": False, "tokens_in": 1, "tokens_out": 1,
               "rounds": 1}

    out = run(G.revise_from_research(course(), "drop warmup", revise=revise))
    assert calls["feedback"] == "drop warmup"
    assert [c["title"] for c in out["concepts"]] == ["Only decay"]
    assert out["subjects"][0]["title"] == "Fundamentals"
