"""Tests for lara.serve.citations -- binding chunk-id brackets to references."""
from __future__ import annotations

import types

from lara.serve import citations as C


def test_parse_keys_flattens_grouped_brackets_in_order_without_repeats():
    assert C.parse_keys("A [3, 1]. B [1]. C [22].") == ["3", "1", "22"]


def test_parse_keys_ignores_things_that_are_not_citations():
    assert C.parse_keys("[TODO] [sic] [Fig. 2] [exp:x] see [12].") == ["12"]


def test_bind_resolves_known_and_reports_the_rest_unresolved():
    known = {"1": C.paper_ref(chunk_id=1, arxiv_id="9.9", paper_title="P")}
    cited = C.bind("x [1]. y [2].", known=known)
    assert list(cited.references) == ["1"] and cited.unresolved == ["2"]
    assert cited.papers == ["9.9"]


def test_bind_falls_back_to_the_corpus_for_a_key_outside_the_evidence_table(monkeypatch):
    hit = types.SimpleNamespace(to_dict=lambda: {"arxiv_id": "3.3", "paper_title": "Q",
                                                  "version": 2, "text": "t"})
    import lara.index.search as S
    monkeypatch.setattr(S, "hydrate", lambda conn, ids: {i: hit for i in ids})
    cited = C.bind("z [5]", conn=object())
    assert cited.unresolved == [] and cited.references["5"].arxiv_url.endswith("3.3v2")


def test_a_reference_survives_a_dict_round_trip_and_drops_unknown_fields():
    ref = C.paper_ref(chunk_id=4, arxiv_id="1.2", paper_title="T", section="s")
    d = {**ref.to_dict(), "kind": "paper", "commit": "abc", "verified": True}
    assert C.Reference.from_dict(d) == ref
    assert C.Reference.from_dict({"no": "key"}) is None


def test_from_claims_indexes_by_chunk_id_and_carries_the_claim():
    c = types.SimpleNamespace(chunk_id=9, arxiv_id="1.1", paper_title="T", section="s",
                              claim="a claim", score=0.5)
    refs = C.from_claims([c])
    assert refs["9"].claim == "a claim" and refs["9"].chunk_id == 9
