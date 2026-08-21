"""Tests over the parts with no I/O.

`pyproject.toml` has configured pytest since the first commit and there were no tests.
That matters more than it sounds: every structural change proposed against this repo —
splitting the two large modules, deduplicating the four JSON-verdict parsers, unifying the
hit serialisers — is behaviour-preserving *by intention* and unverifiable in practice.

So this covers the functions that are pure, cheap and load-bearing: rank fusion, citation
fragments, config merging, follow-up detection, residency translation, and the sizing
heuristics. No model, no database, no network. It runs in about a second, which is the
property that decides whether anyone runs it.
"""

from __future__ import annotations

import numpy as np
import pytest


# ── citation fragments ────────────────────────────────────────────────────────────
# 29.8% of corpus chunks span two anchors, and a hand-written second copy of this format
# pointed every one of them at the wrong text.

def test_fragment_single_anchor():
    from lara.index.search import fragment_for
    assert fragment_for("1706.03762", 5, "S3.p1", 0, "S3.p1", 900) == \
        "/p/1706.03762v5#S3.p1:0-900"


def test_fragment_spanning_two_anchors_names_the_end_anchor():
    from lara.index.search import fragment_for
    assert fragment_for("1706.03762", 5, "S3.p1", 0, "S3.p4", 900) == \
        "/p/1706.03762v5#S3.p1:0-S3.p4:900"


def test_fragment_treats_empty_end_anchor_as_single():
    from lara.index.search import fragment_for
    assert fragment_for("x", 1, "S1.p1", 0, "", 50) == "/p/xv1#S1.p1:0-50"


def test_hit_to_dict_uses_section_not_section_title():
    # agent.format_excerpts and the answer prompt both read "section"; a near-miss here
    # drops the section from every excerpt shown to the model without raising.
    from lara.index.search import Hit
    d = Hit(chunk_id=1, vector_row=0, score=0.5, section_title="Ablations").to_dict()
    assert d["section"] == "Ablations"
    assert "section_title" not in d


def test_hit_to_dict_falls_back_to_anchor_when_untitled():
    from lara.index.search import Hit
    d = Hit(chunk_id=1, vector_row=0, score=0.0, section_anchor="S2").to_dict()
    assert d["section"] == "S2"


# ── reciprocal rank fusion ────────────────────────────────────────────────────────

def test_rrf_rewards_agreement_across_lists():
    from lara.index.search import reciprocal_rank_fusion
    # NOT exactly-reversed lists: those tie by construction, since 1/61 + 1/63 equals
    # 1/62 + 1/62 to the precision that matters. Id 2 here is near the top of both.
    order, _ = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [2, 3, 1]}, k=60)
    assert order[0] == 2, "the id both lists rank highly should win"


def test_rrf_penalises_the_consistently_middling_id():
    """A genuinely counter-intuitive property, worth pinning down.

    With exactly-reversed lists, the ids that are FIRST in one and LAST in the other beat
    the id that is second in both: 1/61 + 1/63 > 2/62, because 1/x is convex. So RRF does
    not reward "acceptable to everyone" — it rewards being strongly preferred somewhere,
    which is the behaviour you want when fusing a dense ranker with a lexical one that
    disagree about what matters.
    """
    from lara.index.search import reciprocal_rank_fusion
    order, parts = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [3, 2, 1]}, k=60)
    assert order[-1] == 2, "the id ranked 2nd by both should come last"
    assert round(parts[1]["rrf"], 12) == round(parts[3]["rrf"], 12), \
        "first-and-last in mirrored lists is symmetric"


def test_rrf_is_rank_based_not_score_based():
    from lara.index.search import reciprocal_rank_fusion
    order, parts = reciprocal_rank_fusion({"only": [7, 8]}, k=60)
    assert order == [7, 8]
    assert parts[7]["only"] > parts[8]["only"]


def test_rrf_handles_an_empty_list():
    from lara.index.search import reciprocal_rank_fusion
    order, _ = reciprocal_rank_fusion({"a": [1], "b": []}, k=60)
    assert order == [1]


# ── config layering ───────────────────────────────────────────────────────────────

def test_deep_merge_overrides_leaves_and_keeps_siblings():
    from lara.config import deep_merge
    out = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 9}})
    assert out == {"a": {"x": 1, "y": 9}}


def test_deep_merge_replaces_lists_rather_than_appending():
    # forbid_paths: [] in the local file has to be able to mean "nothing is forbidden".
    from lara.config import deep_merge
    assert deep_merge({"p": ["/data"]}, {"p": []}) == {"p": []}


def test_deep_merge_does_not_mutate_its_input():
    from lara.config import deep_merge
    base = {"a": {"x": 1}}
    deep_merge(base, {"a": {"x": 2}})
    assert base == {"a": {"x": 1}}


# ── residency translation ─────────────────────────────────────────────────────────
# Where a silent bug is unfalsifiable: returning a local index where a caller expects a
# global vector row points at the wrong passage without raising.

def _resident(rows):
    from lara.index.backends import _Resident
    r = _Resident()
    r.row_ids = np.asarray(rows, dtype=np.int64)
    return r


def test_to_local_and_back_round_trips():
    r = _resident([10, 20, 30])
    assert r.to_global(r.to_local(np.array([20]))).tolist() == [20]


def test_to_local_drops_rows_that_are_not_resident():
    r = _resident([10, 20, 30])
    assert r.to_local(np.array([15, 20, 99])).tolist() == [1]


def test_to_local_is_identity_when_whole_corpus_is_resident():
    from lara.index.backends import _Resident
    r = _Resident()
    r.row_ids = None
    assert r.to_local(np.array([5, 9])).tolist() == [5, 9]
    assert not r.resident


def test_to_local_of_nothing_resident_is_empty_not_wrong():
    r = _resident([10, 20])
    assert r.to_local(np.array([99, 100])).size == 0


# ── sizing heuristics ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,expected", [(0, 5), (15, 5), (46, 9), (78, 16), (300, 30),
                                        (5000, 30)])
def test_paper_k_is_clamped_at_both_tails(n, expected):
    from lara.serve.hierarchy import paper_k
    assert paper_k(n) == expected


def test_keep_fraction_never_exceeds_one():
    from lara.setup import OPTIONS_BY_KEY, keep_fraction_for
    opt = OPTIONS_BY_KEY["torch-fp16"]
    assert keep_fraction_for(1000.0, 4.4, opt, 1000) == 1.0


def test_keep_fraction_shrinks_when_the_budget_does():
    from lara.setup import OPTIONS_BY_KEY, keep_fraction_for
    opt = OPTIONS_BY_KEY["torch-fp16"]
    big = keep_fraction_for(64.0, 4.4, opt, 28_723_432)
    small = keep_fraction_for(10.0, 4.4, opt, 28_723_432)
    assert small < big


# ── follow-up detection ───────────────────────────────────────────────────────────
# Deliberately over-eager: a needless rewrite costs one short call, a missed follow-up
# retrieves the wrong passages and answers confidently about the wrong thing.

@pytest.mark.parametrize("q", ["why is that?", "and then?", "explain",
                               "what about the other one",
                               "Compare it to the former approach"])
def test_followups_are_detected(q):
    from lara.serve.thread import looks_like_followup
    assert looks_like_followup(q)[0]


@pytest.mark.parametrize("q", [
    "How does GQA reduce memory and latency in transformer inference?",
    "What is the learning rate schedule used in Chinchilla?",
])
def test_self_contained_questions_are_left_alone(q):
    from lara.serve.thread import looks_like_followup
    assert not looks_like_followup(q)[0]


def test_empty_question_is_not_a_followup():
    from lara.serve.thread import looks_like_followup
    assert not looks_like_followup("")[0]


def test_history_block_is_empty_without_turns_or_summary():
    from lara.serve.thread import history_block
    assert history_block([]) == ""


def test_history_block_presents_a_summary_as_context():
    from lara.serve.thread import history_block
    out = history_block([], summary="Discussed LiMuon and STORM.")
    assert "LiMuon" in out and "Summary of earlier turns" in out


def test_prior_chunk_ids_are_newest_first_and_deduplicated():
    from lara.serve.thread import prior_chunk_ids
    turns = [{"answer": "see [111] and [222]"}, {"answer": "again [222] plus [333]"}]
    assert prior_chunk_ids(turns)[:3] == [222, 333, 111]


# ── synthesis diversity ───────────────────────────────────────────────────────────

def test_cap_per_paper_limits_one_paper_from_filling_a_round():
    from lara.serve.synthesis import cap_per_paper
    hits = [{"arxiv_id": "a"}] * 5 + [{"arxiv_id": "b"}]
    out = cap_per_paper(hits, 2)
    assert sum(h["arxiv_id"] == "a" for h in out) == 2
    assert sum(h["arxiv_id"] == "b" for h in out) == 1


def test_mmr_returns_k_and_keeps_the_top_scorer():
    from lara.serve.synthesis import mmr
    hits = [{"chunk_id": i, "score": 1.0 - i / 10} for i in range(6)]
    vecs = {i: np.eye(4, dtype=np.float32)[i % 4] for i in range(6)}
    out = mmr(hits, vecs, k=3)
    assert len(out) == 3
    assert out[0]["chunk_id"] == 0


def test_mmr_is_a_noop_when_k_exceeds_the_pool():
    from lara.serve.synthesis import mmr
    hits = [{"chunk_id": 1, "score": 1.0}]
    assert mmr(hits, {}, k=5) == hits


# ── library graph ─────────────────────────────────────────────────────────────────

def test_depth_is_longest_path_so_columns_read_as_progression():
    from lara.serve.library_graph import Edge, Node, _assign_depth
    nodes = [Node(id=x, label=x, summary="") for x in "abc"]
    _assign_depth(nodes, [Edge("a", "b", "follows-up"), Edge("b", "c", "follows-up"),
                          Edge("a", "c", "same-topic")])
    assert [n.depth for n in nodes] == [0, 1, 2], "a->c must not shortcut c to depth 1"


def test_unconnected_nodes_stay_at_depth_zero():
    from lara.serve.library_graph import Node, _assign_depth
    nodes = [Node(id="solo", label="solo", summary="")]
    _assign_depth(nodes, [])
    assert nodes[0].depth == 0


def test_graph_fingerprint_is_stable_across_processes():
    # hash() is randomised per process; a fingerprint built from it can never hit.
    import hashlib
    ids = ["b", "a", "c"]
    digest = hashlib.sha1("\n".join(sorted(ids)).encode()).hexdigest()[:12]
    assert digest == hashlib.sha1("\n".join(sorted(ids)).encode()).hexdigest()[:12]


# ── device resolution ─────────────────────────────────────────────────────────────

def test_resolve_all_collapses_duplicates():
    from lara import device as dev
    assert len(dev.resolve_all([0, 0, 0])) == 1


def test_resolve_all_accepts_auto_and_bare_scalars():
    # "auto"[0] is "a" — indexing a string here silently resolved to a fallback device.
    from lara import device as dev
    assert dev.resolve_all("auto") == dev.available()
    assert dev.resolve_all(None) == dev.available()
    assert len(dev.resolve_all(0)) == 1


def test_explicit_cpu_is_always_honoured():
    from lara import device as dev
    assert dev.resolve("cpu") == "cpu"


def test_empty_string_means_auto():
    from lara import device as dev
    assert dev.resolve("") == dev.resolve(None)


# ── embedding prefixes ────────────────────────────────────────────────────────────
# A train/serve format mismatch has already cost measurable accuracy here (0.909 -> 0.741
# cosine). Four hand-written copies is how that recurs.

def test_all_modules_agree_on_the_query_prefix():
    from lara.finetune import kfold, train
    from lara.index.embed import QUERY_PROMPT
    assert kfold.QUERY_PREFIX == train.QUERY_PREFIX == QUERY_PROMPT


def test_untitled_doc_prefix_matches_document_text():
    from lara.index.embed import DOC_PREFIX_UNTITLED, document_text
    assert DOC_PREFIX_UNTITLED == document_text(None, None, "")


def test_document_text_omits_a_section_already_in_the_title():
    from lara.index.embed import document_text
    assert document_text("Attention", "attention", "x") == "title: Attention | text: x"
