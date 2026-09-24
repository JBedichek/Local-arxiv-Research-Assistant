"""Tests for lara.learn.passages -- CorpusRetriever's search() and its injected figure lookup."""
from __future__ import annotations

import asyncio
import types

from lara.index.search import Hit
from lara.learn.passages import CorpusRetriever, Passage


def _hit(**kw):
    base = dict(chunk_id=1, vector_row=0, score=0.9, arxiv_id="2401.00001", version=1,
               anchor_start="S4.F2", section_title="Results", kind="caption",
               text="Figure 2: loss over training.", paper_title="A Paper")
    return Hit(**{**base, **kw})


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.cross_encoder = None

    def retrieve(self, query, final_k=0, **kw):
        return types.SimpleNamespace(hits=self.hits)


class FakeConn:
    def execute(self, sql, params):
        return types.SimpleNamespace(fetchall=lambda: [])


def _state(hits, neighbours=None):
    return types.SimpleNamespace(retriever=FakeRetriever(hits), conn=lambda: FakeConn(),
                                 neighbours=neighbours or (lambda arxiv_id: {"cites": [], "cited_by": []}))


def test_search_carries_the_anchor_and_version_a_figure_lookup_needs():
    corpus = CorpusRetriever(_state([_hit()]))
    out = corpus.search("q")
    assert out == [Passage(chunk_id=1, arxiv_id="2401.00001", title="A Paper",
                           text="Figure 2: loss over training.", section="Results",
                           kind="caption", anchor="S4.F2", version=1)]


def test_figure_with_no_lookup_injected_returns_none():
    corpus = CorpusRetriever(_state([]))
    assert corpus.figure("2401.00001", 1, "S4.F2") is None


def test_figure_calls_the_injected_lookup():
    seen = []

    def lookup(arxiv_id, version, anchor):
        seen.append((arxiv_id, version, anchor))
        return {"src": "https://arxiv.org/html/2401.00001v1/x1.png", "caption": "Loss curve."}

    corpus = CorpusRetriever(_state([]), figure=lookup)
    out = corpus.figure("2401.00001", 1, "S4.F2")
    assert out["src"].endswith("x1.png") and seen == [("2401.00001", 1, "S4.F2")]


def test_a_failing_lookup_is_none_not_an_exception():
    def lookup(arxiv_id, version, anchor):
        raise RuntimeError("no such file")

    corpus = CorpusRetriever(_state([]), figure=lookup)
    assert corpus.figure("2401.00001", 1, "S4.F2") is None


def test_search_passes_a_paper_restriction_through_to_the_retriever():
    seen = {}

    class RestrictRetriever(FakeRetriever):
        def retrieve(self, query, final_k=0, **kw):
            seen.update(kw)
            return types.SimpleNamespace(hits=self.hits)

    corpus = CorpusRetriever(types.SimpleNamespace(
        retriever=RestrictRetriever([]), conn=lambda: FakeConn(),
        neighbours=lambda arxiv_id: {"cites": [], "cited_by": []}))
    corpus.search("q", papers=["2401.1", "2402.2"])
    assert seen["papers"] == ["2401.1", "2402.2"]


def test_neighbours_delegates_to_the_injected_state_method():
    corpus = CorpusRetriever(_state([], neighbours=lambda a: {"cites": [a + "-c"], "cited_by": []}))
    assert corpus.neighbours("2401.00001") == {"cites": ["2401.00001-c"], "cited_by": []}


def test_neighbours_degrades_to_empty_on_a_failing_lookup():
    def boom(arxiv_id):
        raise RuntimeError("db is gone")

    corpus = CorpusRetriever(_state([], neighbours=boom))
    assert corpus.neighbours("2401.00001") == {"cites": [], "cited_by": []}


def test_coverage_is_zero_when_the_connection_cannot_answer():
    # FakeConn.execute() always returns no rows, so count_matches's vocabulary lookup finds
    # nothing and reports no match -- the honest "nothing found" path, not a crash.
    corpus = CorpusRetriever(_state([]))
    assert corpus.coverage("anything") == {"chunks": 0, "papers": 0}


def test_coverage_degrades_to_zero_on_a_genuinely_broken_connection():
    def boom():
        raise RuntimeError("db is gone")

    corpus = CorpusRetriever(types.SimpleNamespace(
        retriever=FakeRetriever([]), conn=boom,
        neighbours=lambda a: {"cites": [], "cited_by": []}))
    assert corpus.coverage("anything") == {"chunks": 0, "papers": 0}
