"""Tests for lara.learn.passages -- CorpusRetriever's search() and its injected figure lookup."""
from __future__ import annotations

import asyncio
import types

import pytest

from lara.index.search import Hit
from lara.learn.passages import CorpusRetriever, Passage
from lara.store import db


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


@pytest.fixture
def real_conn(tmp_path):
    c = db.connect(tmp_path / "corpus.db")
    c.execute("INSERT INTO papers (arxiv_id, title) VALUES (?, ?)", ("2401.1", "A Paper"))
    c.execute("INSERT INTO sections (arxiv_id, version, anchor, title) VALUES (?, ?, ?, ?)",
             ("2401.1", 1, "S1", "Introduction"))
    rows = [
        (1, "2401.1", 1, 0, "S1", "S1", 0, "S1", 10, "body", 20, None, "First chunk of the paper."),
        (2, "2401.1", 1, 1, "S1", "S1", 10, "S1", 20, "body", 20, None, "Second chunk of the paper."),
        (3, "2401.1", 1, 2, "S1", "S1", 20, "S1", 30, "claim", 20, None, "A self-written claim chunk."),
    ]
    c.executemany(
        "INSERT INTO chunks (chunk_id, arxiv_id, version, ordinal, section_anchor, anchor_start, "
        "char_start, anchor_end, char_end, kind, n_chars, vector_row, text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows)
    c.commit()
    return c


def _corpus_over(conn):
    return CorpusRetriever(types.SimpleNamespace(
        retriever=FakeRetriever([]), conn=lambda: conn, neighbours=lambda a: {"cites": [], "cited_by": []}))


def test_full_paper_returns_every_chunk_in_reading_order_excluding_self_written_ones(real_conn):
    out = _corpus_over(real_conn).full_paper("2401.1", 1)
    assert [p.chunk_id for p in out] == [1, 2], "the kind=claim chunk is excluded, like search() excludes it"
    assert out[0].title == "A Paper" and out[0].section == "Introduction"


def test_full_paper_is_bounded_by_max_chunks(real_conn):
    out = _corpus_over(real_conn).full_paper("2401.1", 1, max_chunks=1)
    assert [p.chunk_id for p in out] == [1]


def test_full_paper_is_empty_for_a_paper_with_nothing_indexed():
    assert CorpusRetriever(_state([])).full_paper("2401.00001", 1) == []


def test_full_paper_degrades_to_empty_on_a_broken_connection():
    def boom():
        raise RuntimeError("db is gone")

    corpus = CorpusRetriever(types.SimpleNamespace(
        retriever=FakeRetriever([]), conn=boom, neighbours=lambda a: {"cites": [], "cited_by": []}))
    assert corpus.full_paper("2401.00001", 1) == []
