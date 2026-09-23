import asyncio
import json

from lara.learn import visuals as V
from learn_helpers import llm


def run(c):
    return asyncio.run(c)


def claim(key, text, certainty="established"):
    return {"key": key, "text": text, "certainty": certainty}


CLAIMS = [claim("c1", "The 1B model reached a loss of 2.31 after 20B tokens."),
          claim("c2", "The 7B model reached a loss of 1.98 after 20B tokens."),
          claim("c3", "Old result: 3B reached 2.5.", "superseded")]
CONCEPT = {"id": "c1", "title": "scaling"}


def test_a_chart_keeps_only_points_whose_number_is_in_their_claim():
    reply = json.dumps({"title": "Loss", "kind": "bar", "y_label": "loss", "points": [
        {"label": "1B", "value": 2.31, "claim": "c1"}, {"label": "7B", "value": 1.98, "claim": "c2"},
        {"label": "70B", "value": 1.2, "claim": "c2"}, {"label": "3B", "value": 2.5, "claim": "c3"}]})
    out = run(V.chart(llm(("CLAIMS:", reply)), CONCEPT, CLAIMS))
    assert [p["label"] for p in out["points"]] == ["1B", "7B"] and out["claims"] == ["c1", "c2"]


def test_a_chart_with_too_few_grounded_points_is_not_shown():
    reply = json.dumps({"points": [{"label": "1B", "value": 2.31, "claim": "c1"}, {"label": "x", "value": 99, "claim": "c2"}]})
    assert run(V.chart(llm(("CLAIMS:", reply)), CONCEPT, CLAIMS)) is None
    assert run(V.chart(llm(("CLAIMS:", "null")), CONCEPT, CLAIMS)) is None
    assert run(V.chart(llm(), CONCEPT, CLAIMS[:1])) is None


def test_numbers_are_read_with_commas_decimals_and_exponents():
    assert V.numbers_in("3e-4 and 1,000 and 2.5%") == {3e-4, 1000.0, 2.5}


def test_a_diagram_keeps_only_edges_the_judge_finds_in_their_claim():
    reply = json.dumps({"title": "Pipeline", "nodes": [{"id": "a", "label": "Warmup"}, {"id": "b", "label": "Stable phase"},
                                                      {"id": "c", "label": "Decay"}, {"id": "d", "label": "Orphan"}],
                        "edges": [{"from": "a", "to": "b", "label": "precedes", "claim": "c1"},
                                  {"from": "b", "to": "c", "label": "makes wrong", "claim": "c2"},
                                  {"from": "a", "to": "zz", "label": "x", "claim": "c1"}]})
    m = llm(("CLAIM: Warmup precedes", "supports"), ("CLAIM: Stable phase makes wrong", "unrelated"), ("CLAIMS:", reply))
    out = run(V.diagram(m, CONCEPT, CLAIMS))
    assert [n["label"] for n in out["nodes"]] == ["Warmup", "Stable phase"]
    assert out["edges"] == [{"from": "a", "to": "b", "label": "precedes", "claim": "c1"}]


def test_a_diagram_with_nothing_left_is_not_shown():
    reply = json.dumps({"nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                        "edges": [{"from": "a", "to": "b", "label": "l", "claim": "c1"}]})
    assert run(V.diagram(llm(("CLAIM: A l B", "unrelated"), ("CLAIMS:", reply)), CONCEPT, CLAIMS)) is None


def test_build_returns_whichever_visuals_survive():
    chart_reply = json.dumps({"points": [{"label": "1B", "value": 2.31, "claim": "c1"}, {"label": "7B", "value": 1.98, "claim": "c2"}]})
    m = llm(("quantity that at least two", chart_reply), ("small diagram", "null"))
    assert [v["kind"] for v in run(V.build(m, CONCEPT, CLAIMS))] == ["chart"]


def test_pseudocode_keeps_only_steps_the_judge_finds_in_their_claim():
    reply = json.dumps({"title": "Curriculum", "steps": [
        {"text": "sort examples by difficulty", "depth": 0, "claim": "c1"},
        {"text": "train on easiest first", "depth": 1, "claim": "c2"},
        {"text": "unrelated step", "depth": 0, "claim": "c1"}]})
    m = llm(("CLAIM: sort examples by difficulty", "supports"),
            ("CLAIM: train on easiest first", "supports"),
            ("CLAIM: unrelated step", "unrelated"),
            ("procedure", reply))
    out = run(V.pseudocode(m, CONCEPT, CLAIMS))
    assert [s["text"] for s in out["steps"]] == ["sort examples by difficulty", "train on easiest first"]
    assert out["claims"] == ["c1", "c2"] and out["steps"][1]["depth"] == 1


def test_pseudocode_with_too_few_grounded_steps_is_not_shown():
    reply = json.dumps({"steps": [{"text": "one step", "claim": "c1"}]})
    m = llm(("CLAIM:", "supports"), ("procedure", reply))
    assert run(V.pseudocode(m, CONCEPT, CLAIMS)) is None
    assert run(V.pseudocode(llm(("procedure", "null")), CONCEPT, CLAIMS)) is None


def test_pseudocode_drops_a_step_citing_no_real_claim():
    reply = json.dumps({"steps": [{"text": "a", "claim": "c1"}, {"text": "b", "claim": "nope"}]})
    m = llm(("CLAIM:", "supports"), ("procedure", reply))
    assert run(V.pseudocode(m, CONCEPT, CLAIMS)) is None  # only one real step left


def test_section_claims_keeps_only_what_the_section_cites_in_concept_order():
    section = {"sentences": [{"text": "x", "claims": ["c2"]}, {"text": "y", "claims": ["c1", "zz"]}]}
    assert [c["key"] for c in V._section_claims(CLAIMS, section)] == ["c1", "c2"]


def test_build_partitions_by_section_instead_of_the_whole_concept():
    """Two sections, each citing a disjoint pair of claims: each gets its own chart attempt
    over only its own claims, not the whole concept's."""
    lesson = {"sections": [{"sentences": [{"text": "x", "claims": ["c1", "c2"]}]},
                           {"sentences": [{"text": "y", "claims": ["c3"]}]}]}
    seen_prompts = []

    async def complete(cfg, prompt, *, system, **kw):
        seen_prompts.append(prompt)
        if "quantity that at least two" in system:
            return json.dumps({"points": [{"label": "1B", "value": 2.31, "claim": "c1"},
                                          {"label": "7B", "value": 1.98, "claim": "c2"}]})
        return "null"

    from lara.learn.llm import Llm
    m = Llm(complete=complete, window=200_000)
    out = run(V.build(m, CONCEPT, CLAIMS, lesson=lesson))
    assert [v["kind"] for v in out] == ["chart"]
    # The section with only c3 (one claim, superseded c3 aside) never got a chart attempt at
    # all -- section 2 has fewer than 2 live claims, so it is skipped before any call is made.
    assert not any("70B" in p or "c3" in p for p in seen_prompts if "quantity" in p)


def test_build_falls_back_to_the_whole_concept_with_no_lesson_to_section_by():
    chart_reply = json.dumps({"points": [{"label": "1B", "value": 2.31, "claim": "c1"}, {"label": "7B", "value": 1.98, "claim": "c2"}]})
    m = llm(("quantity that at least two", chart_reply), ("small diagram", "null"), ("procedure", "null"))
    assert [v["kind"] for v in run(V.build(m, CONCEPT, CLAIMS, lesson=None))] == ["chart"]
    assert [v["kind"] for v in run(V.build(m, CONCEPT, CLAIMS,
                                          lesson={"insufficient": True, "sections": []}))] == ["chart"]


def test_build_caps_total_visuals():
    lesson = {"sections": [{"sentences": [{"text": "x", "claims": ["c1", "c2"]}]}] * (V.MAX_VISUALS + 5)}
    m = llm(default=json.dumps({"points": [{"label": "1B", "value": 2.31, "claim": "c1"},
                                           {"label": "7B", "value": 1.98, "claim": "c2"}]}))
    out = run(V.build(m, CONCEPT, CLAIMS, lesson=lesson))
    assert len(out) == V.MAX_VISUALS


# ── figures: pulled from the source paper, not synthesized ──────────────────────────


def claim_with_passage(key, text, passage, certainty="established"):
    return {"key": key, "text": text, "certainty": certainty, "passage": passage}


CAPTION_PASSAGE = {"arxiv_id": "2401.00001", "version": 1, "anchor": "S4.F2",
                   "kind": "caption", "text": "Figure 2: loss over training."}


class FakeCorpus:
    """.figure records every call and answers from a table, like CorpusRetriever.figure."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def figure(self, arxiv_id, version, anchor):
        self.calls.append((arxiv_id, version, anchor))
        return self.answers.get((arxiv_id, version, anchor))


def test_figures_in_finds_only_caption_claims_and_never_calls_the_model():
    claims = [claim_with_passage("c1", "Loss falls with more data.", CAPTION_PASSAGE),
             claim("c2", "An ordinary finding with no figure passage.")]
    corpus = FakeCorpus({("2401.00001", 1, "S4.F2"):
                        {"src": "https://arxiv.org/html/2401.00001v1/x1.png", "caption": "Loss curve."}})
    out = run(V.figures_in(claims, corpus))
    assert corpus.calls == [("2401.00001", 1, "S4.F2")]
    assert out == [{"kind": "figure", "title": "Loss falls with more data.",
                    "src": "https://arxiv.org/html/2401.00001v1/x1.png",
                    "caption": "Loss curve.", "arxiv_id": "2401.00001", "claims": ["c1"]}]


def test_figures_in_falls_back_to_the_passage_text_with_no_caption():
    claims = [claim_with_passage("c1", "x", CAPTION_PASSAGE)]
    corpus = FakeCorpus({("2401.00001", 1, "S4.F2"): {"src": "https://x/y.png", "caption": ""}})
    out = run(V.figures_in(claims, corpus))
    assert out[0]["caption"] == "Figure 2: loss over training."


def test_figures_in_drops_anchors_with_no_image_and_dedupes_by_src():
    same = {"src": "https://x/y.png", "caption": "c"}
    claims = [claim_with_passage("c1", "a", CAPTION_PASSAGE),
             claim_with_passage("c2", "b", {**CAPTION_PASSAGE, "anchor": "S4.F3"}),
             claim_with_passage("c3", "c", {**CAPTION_PASSAGE, "anchor": "S4.T1"})]
    corpus = FakeCorpus({("2401.00001", 1, "S4.F2"): same, ("2401.00001", 1, "S4.F3"): same,
                        ("2401.00001", 1, "S4.T1"): None})
    out = run(V.figures_in(claims, corpus))
    assert len(out) == 1 and out[0]["claims"] == ["c1"]


def test_figures_in_ignores_non_caption_passages_and_no_corpus():
    claims = [claim_with_passage("c1", "x", {**CAPTION_PASSAGE, "kind": "body"})]
    assert run(V.figures_in(claims, FakeCorpus({}))) == []
    assert run(V.figures_in(claims, None)) == []


def test_build_includes_figures_alongside_synthesized_visuals():
    claims = CLAIMS + [claim_with_passage("c4", "Shown in the figure.", CAPTION_PASSAGE)]
    m = llm(("quantity that at least two", "null"), ("small diagram", "null"), ("procedure", "null"))
    corpus = FakeCorpus({("2401.00001", 1, "S4.F2"): {"src": "https://x/y.png", "caption": "cap"}})
    out = run(V.build(m, CONCEPT, claims, corpus=corpus))
    assert [v["kind"] for v in out] == ["figure"]
