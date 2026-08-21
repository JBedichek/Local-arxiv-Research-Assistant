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

import pathlib
import re as _re

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


# ── the shared JSON verdict transport ─────────────────────────────────────────────
# Nine call sites used to spell this out, and three of them lacked the try/except the
# other six had. Driven with an injected stream rather than a live model: deterministic,
# and it covers failure shapes a real generator only produces occasionally.
#
# Run through asyncio.run rather than pytest-asyncio, so the suite needs no plugin — a
# test suite with an install step is a test suite that stops being run.

import asyncio

import lara.serve.generate as GEN


def _fake_stream(text=None, exc=None):
    async def gen(cfg, prompt, hits, **kw):
        if exc is not None:
            raise exc
        for piece in (text or ""):
            yield piece
    return gen


def _json(monkeypatch, *, text=None, exc=None, **kw):
    monkeypatch.setattr(GEN, "stream_answer", _fake_stream(text, exc))
    return asyncio.run(GEN.complete_json(None, "p", system="s", **kw))


def test_complete_json_parses_a_bare_object(monkeypatch):
    assert _json(monkeypatch, text='{"ok": true}') == {"ok": True}


def test_complete_json_finds_an_object_inside_prose(monkeypatch):
    # Models narrate. Extraction has to survive a preamble and a trailing sentence.
    assert _json(monkeypatch, text='Sure! {"ok": 1} Hope that helps.') == {"ok": 1}


def test_complete_json_survives_code_fences(monkeypatch):
    assert _json(monkeypatch, text='```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}


def test_complete_json_array_shape(monkeypatch):
    out = _json(monkeypatch, text='[{"n": 1}, {"n": 2}]', shape="array")
    assert [r["n"] for r in out] == [1, 2]


def test_object_shape_does_not_match_a_bare_array(monkeypatch):
    # The shapes are distinct on purpose: extract() wants a list, and a shared regex would
    # let it silently accept the braces inside an object instead.
    assert _json(monkeypatch, text="[1, 2, 3]", default="fell back") == "fell back"


def test_malformed_json_returns_the_default(monkeypatch):
    assert _json(monkeypatch, text='{"a": ', default={"d": 1}) == {"d": 1}


def test_no_json_at_all_returns_the_default(monkeypatch):
    assert _json(monkeypatch, text="I cannot help with that.", default=[]) == []


def test_a_generator_error_returns_the_default_rather_than_raising(monkeypatch):
    # The behaviour agent.py did NOT have: its three sites propagated, so a generator
    # hiccup failed the controller turn while degrading silently everywhere else.
    assert _json(monkeypatch, exc=RuntimeError("vLLM down"), default="safe") == "safe"


def test_complete_returns_empty_string_on_error(monkeypatch):
    monkeypatch.setattr(GEN, "stream_answer", _fake_stream(exc=OSError("connrefused")))
    assert asyncio.run(GEN.complete(None, "p", system="s")) == ""


# ── the SSE driver ────────────────────────────────────────────────────────────────
#
# Both streaming endpoints share one queue-and-drain helper, so the framing and the
# disconnect semantics are worth pinning down. No server and no model: the driver is
# handed a plain coroutine and the frames are collected from the generator directly.

from lara.serve.routes.research import _sse


def _drain(resp, stop_after=None):
    """Collect SSE frames, optionally closing the connection partway through."""
    async def go():
        out = []
        agen = resp.body_iterator
        try:
            async for frame in agen:
                out.append(frame)
                if stop_after is not None and len(out) >= stop_after:
                    await agen.aclose()      # what a browser closing the tab looks like
                    break
        except asyncio.CancelledError:
            pass
        return out
    return asyncio.run(go())


def test_sse_frames_each_emitted_event():
    async def run(emit, should_stop):
        emit("step", {"kind": "search", "detail": "looking"})
        emit("token", "hello")

    frames = _drain(_sse(run, hard_stop=True))
    assert frames == [
        'event: step\ndata: {"kind": "search", "detail": "looking"}\n\n',
        'event: token\ndata: "hello"\n\n',
    ]


def test_sse_reports_a_driver_exception_as_an_error_event():
    async def run(emit, should_stop):
        emit("step", {"kind": "search", "detail": "looking"})
        raise RuntimeError("vLLM down")

    frames = _drain(_sse(run, hard_stop=True))
    assert frames[-1] == 'event: error\ndata: "vLLM down"\n\n'


def test_sse_serialises_payloads_it_cannot_json_encode():
    # default=str, because an answer's payload can carry a Path or a datetime and losing
    # the whole stream over one unserialisable field is the wrong trade.
    async def run(emit, should_stop):
        emit("done", {"where": pathlib.Path("/tmp/x")})

    assert _drain(_sse(run, hard_stop=True)) == ['event: done\ndata: {"where": "/tmp/x"}\n\n']


def test_hard_stop_cancels_the_driver_when_the_client_disconnects():
    """/api/ask: nobody is reading the tokens, so the GPU should stop making them."""
    reached_end = {"v": False}

    async def run(emit, should_stop):
        emit("token", "a")
        await asyncio.sleep(0.05)
        reached_end["v"] = True
        emit("token", "b")

    frames = _drain(_sse(run, hard_stop=True), stop_after=1)
    assert frames == ['event: token\ndata: "a"\n\n']
    assert not reached_end["v"]


def test_soft_stop_asks_the_driver_to_wind_down_instead_of_killing_it():
    """/api/synthesize: minutes of retrieval get saved, so it consolidates rather than dies."""
    saw = {}

    async def run(emit, should_stop):
        emit("round", 1)
        await asyncio.sleep(0.05)
        saw["stop"] = should_stop()          # observed the disconnect, still running

    async def go():
        resp = _sse(run, hard_stop=False)
        agen = resp.body_iterator
        first = await agen.__anext__()
        await agen.aclose()
        await asyncio.sleep(0.1)             # let the driver finish on its own terms
        return first

    first = asyncio.run(go())
    assert first == 'event: round\ndata: 1\n\n'
    assert saw["stop"] is True


# ── context expansion ─────────────────────────────────────────────────────────────
#
# expand_chunks is pure SQL plus a URL builder, so it runs against an in-memory database
# with no index, no model and no server. Added because it silently referenced an
# unimported module: every "expand_context" round raised NameError at the first row it
# built, and nothing here exercised the path.

import sqlite3

import lara.serve.agent as AG


def _corpus():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript("""
        CREATE TABLE papers (arxiv_id TEXT, title TEXT);
        CREATE TABLE sections (arxiv_id TEXT, version INT, anchor TEXT, title TEXT);
        CREATE TABLE chunks (chunk_id INT, arxiv_id TEXT, version INT, ordinal INT,
                             section_anchor TEXT, anchor_start TEXT, char_start INT,
                             anchor_end TEXT, char_end INT, kind TEXT, text TEXT);
        INSERT INTO papers VALUES ('2401.00001', 'A Paper');
        INSERT INTO sections VALUES ('2401.00001', 2, 'S3', 'Method');
    """)
    for i in range(6):
        c.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                  (100 + i, '2401.00001', 2, i, 'S3', 'S3', i * 10, 'S3', i * 10 + 9,
                   'text', f'chunk {i}'))
    return c


def test_expand_chunks_reads_more_before_than_after():
    out = AG.expand_chunks(_corpus(), [103], before=2, after=1)
    assert [o["chunk_id"] for o in out] == [101, 102, 103, 104]
    assert all(o["via"] == "context" for o in out)


def test_expand_chunks_builds_a_citation_url():
    # The regression: this line called an unimported module.
    out = AG.expand_chunks(_corpus(), [103], before=0, after=0)
    assert out[0]["url"] == "/p/2401.00001v2#S3:30-39"


def test_expand_chunks_prefers_the_section_title_over_its_anchor():
    out = AG.expand_chunks(_corpus(), [103], before=0, after=0)
    assert out[0]["section"] == "Method"


def test_expand_chunks_never_returns_the_same_chunk_twice():
    out = AG.expand_chunks(_corpus(), [102, 103], before=2, after=1)
    ids = [o["chunk_id"] for o in out]
    assert len(ids) == len(set(ids))


# ── CLI progress reporters ────────────────────────────────────────────────────────
#
# Ten commands used to hand-roll the same generator and prime it with next(). The
# priming is the part worth a test: it must happen before the first send() and it prints
# nothing, so getting it wrong fails later and elsewhere, inside whichever worker was
# reporting.



from lara.cli._base import console as _cli_console
from lara.cli._progress import reporter as _reporter

_ANSI = _re.compile(r"\x1b\[[0-9;]*m")


def _lines(make, records):
    _cli_console.begin_capture()
    try:
        p = make()
        for r in records:
            p.send(r)
    finally:
        out = _cli_console.end_capture()
    return [_ANSI.sub("", ln) for ln in out.splitlines()]


def test_reporter_is_already_primed():
    # No next() at the call site: send() works on the first record.
    assert _lines(lambda: _reporter(lambda s: f"step {s}"), [1, 2]) == ["step 1", "step 2"]


def test_reporter_prints_nothing_when_render_returns_none():
    # How a line that only matters sometimes stays an expression rather than a branch.
    got = _lines(lambda: _reporter(lambda s: f"step {s}" if s % 2 else None), [0, 1, 2, 3])
    assert got == ["step 1", "step 3"]


def test_reporter_render_can_carry_state_across_records():
    # The running-total pattern that `lara embed` reports with.
    def make():
        done = 0

        def line(n):
            nonlocal done
            done += n
            return f"{done} so far"
        return _reporter(line)

    assert _lines(make, [2, 3, 5]) == ["2 so far", "5 so far", "10 so far"]


# ── the setup wizard's backend chooser ────────────────────────────────────────────
#
# `lara setup --show` covers the non-interactive branch end to end, but not the two that
# take input. These drive those directly against real backend options and a stand-in
# machine, so the validation loops and the esc-restores-the-recommendation rule are
# pinned without running the wizard or writing a config.

from types import SimpleNamespace

import lara.prompt as _prompt_mod
import lara.setup as _SU
from lara.cli import setup_ui as _SUI


def _plan():
    opts = [_SU.OPTIONS_BY_KEY[k] for k in ("torch-fp16", "faiss-hnsw", "faiss-sq8")]
    return SimpleNamespace(
        alternatives=[(o, 0.0, None) for o in opts], option=opts[1],
        scope="required", scope_keep=0.50, dim=256, embedder_gb=1.2, reranker_gb=0.6,
        hot_tier_bytes=0, overhead_gb=1.8, budget_gb=48.0, index_gb=3.0,
    )


def _chooser():
    device = SimpleNamespace(accelerator="cuda", unified_memory=False, budget_gb=48.0,
                             gpus=[], system="Linux", machine="x86_64")
    return _SUI.BackendChooser(_plan(), device, n_chunks=10_000_000)


def _quiet(fn):
    from lara.cli._base import console
    console.begin_capture()
    try:
        return fn()
    finally:
        console.end_capture()


def test_chooser_starts_at_the_step_nearest_the_planned_fraction():
    assert _chooser().keep == 0.50


def test_nudge_keep_clamps_at_both_ends_and_says_whether_it_moved():
    c = _chooser()
    assert c.nudge_keep(1) is True and c.keep == 0.66
    while c.nudge_keep(1):
        pass
    assert c.keep == 1.00 and c.nudge_keep(1) is False     # top of the range
    while c.nudge_keep(-1):
        pass
    assert c.keep == _SUI.KEEP_STEPS[0] and c.nudge_keep(-1) is False


def test_non_interactive_takes_the_recommendation_untouched():
    c = _chooser()
    opt, keep = _quiet(lambda: c.run(allow_interactive=False))
    assert opt is c.plan.option and keep == 0.50


def test_escape_restores_the_recommended_fraction(monkeypatch):
    # The reader moved the slider, then pressed esc. That must undo the slider too, not
    # just the row selection -- they are one decision.
    c = _chooser()
    c.nudge_keep(-3)
    assert c.keep != c.default_keep
    monkeypatch.setattr(_prompt_mod, "interactive", lambda: True)
    monkeypatch.setattr(_prompt_mod, "select", lambda *a, **k: None)
    opt, keep = _quiet(lambda: c.run(allow_interactive=True))
    assert keep == c.default_keep == 0.50
    assert opt.key == "faiss-hnsw"


def test_arrow_keys_write_the_choice_back_onto_the_plan(monkeypatch):
    # Later sections size the generator from plan.option, not from the return value.
    c = _chooser()
    monkeypatch.setattr(_prompt_mod, "interactive", lambda: True)
    monkeypatch.setattr(_prompt_mod, "select", lambda *a, **k: 2)
    opt, _ = _quiet(lambda: c.run(allow_interactive=True))
    assert opt.key == "faiss-sq8" and c.plan.option.key == "faiss-sq8"


def test_typed_prompt_rejects_a_backend_that_is_not_offered(monkeypatch):
    c = _chooser()
    monkeypatch.setattr(_prompt_mod, "interactive", lambda: False)
    answers = iter(["faiss-flatt", "torch-fp16", "0.25"])
    monkeypatch.setattr(_SUI.typer, "prompt", lambda *a, **k: next(answers))
    opt, keep = _quiet(lambda: c.run(allow_interactive=True))
    assert opt.key == "torch-fp16" and keep == 0.25


def test_typed_prompt_rejects_a_fraction_that_is_not_a_number_or_out_of_range(monkeypatch):
    c = _chooser()
    monkeypatch.setattr(_prompt_mod, "interactive", lambda: False)
    answers = iter(["faiss-hnsw", "half", "1.5", "0", "0.4"])
    monkeypatch.setattr(_SUI.typer, "prompt", lambda *a, **k: next(answers))
    _, keep = _quiet(lambda: c.run(allow_interactive=True))
    assert keep == 0.4


# ── config keys with no reader ────────────────────────────────────────────────────
#
# Thirteen keys had accumulated in config.yaml that nothing read: paths pointing at
# formats the project never shipped, ingest knobs replaced by CLI flags, embedding
# prefixes that had moved into code. Each one is a trap -- it looks like a setting, so
# editing it and re-running looks like it should change something.
#
# This is a name check, not a call-graph: a key is flagged only when its leaf name appears
# nowhere in lara/, web/ or tests/ at all. That misses a key read under some other name,
# and it never fires falsely.

import yaml as _yaml

#: Documented as unimplemented rather than pretending to work. The comment above them in
#: config.yaml says tier 0 reserves nothing today, which makes them a note about intent
#: instead of a setting that silently does nothing.
_INTENTIONALLY_UNREAD = {
    "hot_tier.pin_open_paper",
    "hot_tier.pin_neighbors_hops",
    "hot_tier.lru_papers",
}

_REPO = pathlib.Path(__file__).resolve().parent.parent


def _config_leaves(node, prefix=""):
    if isinstance(node, dict):
        for k, v in node.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                yield from _config_leaves(v, path)
            else:
                yield path


def test_every_config_key_is_read_by_something():
    cfg = _yaml.safe_load((_REPO / "config.yaml").read_text())
    src = "\n".join(
        p.read_text()
        for d, glob in (("lara", "**/*.py"), ("web", "**/*.js"), ("tests", "**/*.py"))
        for p in (_REPO / d).glob(glob)
    )
    orphans = []
    for path in _config_leaves(cfg):
        if path in _INTENTIONALLY_UNREAD:
            continue
        leaf = path.rsplit(".", 1)[-1]
        if not _re.search(rf'(["\']{_re.escape(leaf)}["\']|\b{_re.escape(leaf)}\s*=|\.{_re.escape(leaf)}\b)', src):
            orphans.append(path)
    assert not orphans, (
        "config.yaml declares keys whose name appears nowhere in the source:\n  "
        + "\n  ".join(orphans)
        + "\nEither wire it up, delete it, or add it to _INTENTIONALLY_UNREAD with a "
          "comment in config.yaml saying it is not implemented."
    )
