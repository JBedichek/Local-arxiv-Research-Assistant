"""Tests for lara.serve.deliverable -- condensed versions with citations re-bound."""
from __future__ import annotations

import asyncio

import pytest

from lara.serve import citations as C
from lara.serve import deliverable as D


def _writer(text, seen=None):
    async def complete(cfg, prompt, *, system, model=None, max_tokens=0, **kw):
        if seen is not None:
            seen.append({"prompt": prompt, "system": system, "max_tokens": max_tokens})
        return text
    return complete


def _known():
    return {"7": C.paper_ref(chunk_id=7, arxiv_id="1.1", paper_title="P")}


def test_condense_rejects_an_unknown_level():
    with pytest.raises(ValueError):
        asyncio.run(D.condense({}, "text", level="tiny", complete=_writer("x")))


def test_condense_on_empty_text_returns_nothing():
    assert asyncio.run(D.condense({}, "", level="medium", complete=_writer("x"))) == ("", [], {})


def test_condense_uses_the_level_instruction_and_the_goal():
    seen = []
    asyncio.run(D.condense({}, "Full [7].", level="medium", goal="why?",
                           complete=_writer("Short [7].", seen), known=_known()))
    asyncio.run(D.condense({}, "Full [7].", level="short", goal="why?",
                           complete=_writer("Tiny [7].", seen), known=_known()))
    assert "redundant, repeated information" in seen[0]["prompt"] and "why?" in seen[0]["prompt"]
    assert "as concisely as possible" in seen[1]["prompt"]
    assert seen[0]["system"] == D.CONDENSE_SYSTEM


def test_condense_rebinds_the_citations_it_kept_and_drops_the_rest():
    known = {**_known(), "8": C.paper_ref(chunk_id=8, arxiv_id="2.2", paper_title="Q")}
    text, keys, refs = asyncio.run(D.condense(
        {}, "A [7]. B [8].", level="medium", goal="g",
        complete=_writer("A [7]."), known=known))
    assert text == "A [7]." and keys == ["7"]
    assert list(refs) == ["7"] and refs["7"]["arxiv_url"] == "https://arxiv.org/abs/1.1"


def test_condense_short_asks_for_a_smaller_budget_than_medium():
    seen = []
    asyncio.run(D.condense({}, "x", level="medium", complete=_writer("y", seen), window=0))
    asyncio.run(D.condense({}, "x", level="short", complete=_writer("y", seen), window=0))
    assert seen[1]["max_tokens"] < seen[0]["max_tokens"]


def test_condense_returns_nothing_when_the_model_does():
    assert asyncio.run(D.condense({}, "x", level="short", complete=_writer("  "))) == ("", [], {})


def test_known_from_dicts_skips_records_that_are_not_references():
    ref = C.paper_ref(chunk_id=7, arxiv_id="1.1").to_dict()
    known = D.known_from_dicts({"7": ref, "bad": {"no": "key"}, "none": None})
    assert list(known) == ["7"]
