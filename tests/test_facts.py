"""Fact distillation: extraction from a deliverable, storage, and similarity retrieval."""

from __future__ import annotations

import asyncio

import pytest

from lara.serve import facts as FA


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(FA, "FACTS_STORE", tmp_path / "facts.jsonl")


class _FakeEmbedder:
    """Deterministic: a text's "embedding" is derived from its own characters, so two
    similar strings land close together without a real model."""

    def encode(self, texts, convert_to_numpy=True):
        import numpy as np
        return np.array([[float(ord(c)) for c in (t + " " * 8)[:8]] for t in texts])


def _writer(text):
    async def complete(cfg, prompt, **kw):
        return text
    return complete


# ── parsing ───────────────────────────────────────────────────────────────────────

def test_parse_facts_reads_tagged_numbered_lines():
    text = ("1. [performance ceiling] The embedder truncates inputs past 512 tokens.\n"
           "2. [design tradeoff] Retrieval uses brute-force cosine, not a vector index.")
    assert FA.parse_facts(text) == [
        ("performance ceiling", "The embedder truncates inputs past 512 tokens."),
        ("design tradeoff", "Retrieval uses brute-force cosine, not a vector index."),
    ]


def test_parse_facts_on_none_reply_is_empty():
    assert FA.parse_facts("NONE") == []


def test_parse_facts_ignores_malformed_lines():
    assert FA.parse_facts("Sure, here are some facts:\n1. no brackets here") == []


# ── distillation ─────────────────────────────────────────────────────────────────

def test_distill_facts_persists_up_to_the_cap():
    text = "\n".join(f"{i}. [tag{i}] fact number {i}" for i in range(1, 8))  # 7 lines
    facts = asyncio.run(FA.distill_facts(
        {}, "improve retrieval", "some deliverable text", run_id="run-1",
        embedder=_FakeEmbedder(), complete=_writer(text)))
    assert len(facts) == FA.MAX_FACTS_PER_DELIVERABLE
    assert FA.FACTS_STORE.exists()
    stored = FA._read_jsonl(FA.FACTS_STORE)
    assert len(stored) == FA.MAX_FACTS_PER_DELIVERABLE
    assert stored[0]["run_id"] == "run-1"
    assert stored[0]["goal"] == "improve retrieval"
    assert stored[0]["embedding"]


def test_distill_facts_on_empty_deliverable_is_a_noop():
    facts = asyncio.run(FA.distill_facts(
        {}, "goal", "   ", run_id="run-1", complete=_writer("1. [x] y")))
    assert facts == []
    assert not FA.FACTS_STORE.exists()


def test_distill_facts_on_none_reply_stores_nothing():
    facts = asyncio.run(FA.distill_facts(
        {}, "goal", "deliverable text", run_id="run-1", complete=_writer("NONE")))
    assert facts == []


def test_distill_facts_twice_for_one_run_stores_it_once():
    calls = []

    async def complete(cfg, prompt, **kw):
        calls.append(1)
        return "1. [tag] a fact"

    for _ in range(2):
        asyncio.run(FA.distill_facts({}, "goal", "deliverable text", run_id="run-1",
                                     complete=complete))
    assert len(FA._read_jsonl(FA.FACTS_STORE)) == 1
    assert len(calls) == 1, "the second call should not even reach the model"


def test_distill_facts_still_stores_a_different_run():
    for run_id in ("run-1", "run-2"):
        asyncio.run(FA.distill_facts({}, "goal", "deliverable text", run_id=run_id,
                                     complete=_writer("1. [tag] a fact")))
    assert len(FA._read_jsonl(FA.FACTS_STORE)) == 2


def test_distill_facts_without_an_embedder_still_stores_the_fact():
    facts = asyncio.run(FA.distill_facts(
        {}, "goal", "deliverable text", run_id="run-1",
        complete=_writer("1. [tag] a fact with no embedder available")))
    assert len(facts) == 1
    assert facts[0]["embedding"] == []


# ── retrieval ────────────────────────────────────────────────────────────────────

def test_search_facts_ranks_by_similarity():
    embedder = _FakeEmbedder()
    close = FA._embed_one(embedder, "retrieval architecture")
    far = FA._embed_one(embedder, "zzz totally unrelated")
    FA._append_jsonl(FA.FACTS_STORE, {"id": "a", "run_id": "r1", "goal": "g",
                                      "fact": "close fact", "tag": "t", "embedding": close})
    FA._append_jsonl(FA.FACTS_STORE, {"id": "b", "run_id": "r2", "goal": "g",
                                      "fact": "far fact", "tag": "t", "embedding": far})
    query = FA._embed_one(embedder, "retrieval architectures")
    hits = FA.search_facts(query, k=5)
    assert hits[0]["fact"] == "close fact"


def test_search_facts_drops_a_restatement_of_a_higher_ranked_hit():
    for id_, run, fact, emb in (("a", "r1", "original", [1.0, 0.0]),
                                ("b", "r2", "restated", [0.99, 0.05]),
                                ("c", "r3", "distinct", [0.6, 0.8])):
        FA._append_jsonl(FA.FACTS_STORE, {"id": id_, "run_id": run, "goal": "g",
                                          "fact": fact, "tag": "t", "embedding": emb})
    hits = FA.search_facts([1.0, 0.0], k=5)
    assert [h["fact"] for h in hits] == ["original", "distinct"]


def test_search_facts_still_fills_k_past_a_dropped_restatement():
    for i, emb in enumerate(([1.0, 0.0], [1.0, 0.01], [0.6, 0.8], [0.0, 1.0])):
        FA._append_jsonl(FA.FACTS_STORE, {"id": str(i), "run_id": f"r{i}", "goal": "g",
                                          "fact": f"f{i}", "tag": "t", "embedding": emb})
    assert len(FA.search_facts([1.0, 0.0], k=3)) == 3


def test_search_facts_on_empty_store_is_empty():
    assert FA.search_facts([1.0, 2.0]) == []


def test_search_facts_excludes_the_given_run():
    FA._append_jsonl(FA.FACTS_STORE, {"id": "a", "run_id": "self", "goal": "g",
                                      "fact": "own fact", "tag": "t",
                                      "embedding": [1.0, 0.0]})
    assert FA.search_facts([1.0, 0.0], exclude_run_id="self") == []


def test_search_facts_skips_unembedded_rows():
    FA._append_jsonl(FA.FACTS_STORE, {"id": "a", "run_id": "r1", "goal": "g",
                                      "fact": "no embedding", "tag": "t", "embedding": []})
    assert FA.search_facts([1.0, 0.0]) == []


def test_embedder_fn_wraps_encode_as_a_plain_callable():
    embed = FA.embedder_fn(_FakeEmbedder())
    assert embed("hello") == FA._embed_one(_FakeEmbedder(), "hello")


def test_format_facts_includes_tag_fact_and_provenance():
    text = FA.format_facts([{"tag": "design tradeoff", "fact": "uses brute-force cosine",
                             "goal": "improve retrieval"}])
    assert "[design tradeoff]" in text
    assert "uses brute-force cosine" in text
    assert "improve retrieval" in text


def test_format_facts_on_no_hits_is_empty():
    assert FA.format_facts([]) == ""
