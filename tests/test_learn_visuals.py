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
