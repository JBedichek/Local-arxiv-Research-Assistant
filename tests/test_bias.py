"""The preference for pre-compressed evidence."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from lara.index import bias as B


@dataclass
class H:
    kind: str
    score: float
    provenance: dict = field(default_factory=dict)


def test_compressed_chunks_are_lifted():
    # a claim just behind the top passage, in a candidate set with a realistic spread
    hits = [H("body", 1.0), H("claim", 0.9), H("body", 0.2)]
    B.apply(hits, {"claim": 0.5})
    assert hits[1].score > hits[0].score


def test_the_lean_is_proportional_to_the_spread_not_the_gap():
    # in a near-tied set the bonus is small, so the bias cannot manufacture a winner out
    # of a set where retrieval found nothing to separate
    tight = [H("body", 1.0), H("claim", 0.9)]
    B.apply(tight, {"claim": 0.5})
    assert tight[1].score == pytest.approx(0.9 + 0.5 * 0.1)


def test_the_bonus_is_a_fraction_of_the_spread():
    hits = [H("body", 1.0), H("claim", 0.0)]
    B.apply(hits, {"claim": 0.25})
    assert hits[1].score == pytest.approx(0.25)          # 25% of a spread of 1.0


def test_one_weight_means_the_same_thing_at_both_scales():
    # tier-2 inner products vs cross-encoder logits: same relative nudge
    small = [H("body", 1.0), H("claim", 0.5)]
    large = [H("body", 20.0), H("claim", 10.0)]
    B.apply(small, {"claim": 0.2})
    B.apply(large, {"claim": 0.2})
    assert small[1].score == pytest.approx(0.5 + 0.2 * 0.5)
    assert large[1].score == pytest.approx(10.0 + 0.2 * 10.0)


def test_a_flat_candidate_set_is_left_alone():
    # no spread to take a fraction of; inventing one would let the bias decide everything
    hits = [H("body", 0.7), H("claim", 0.7)]
    B.apply(hits, {"claim": 0.9})
    assert [h.score for h in hits] == [0.7, 0.7]


def test_the_nudge_is_recorded_so_a_ranking_can_be_explained():
    hits = [H("body", 1.0), H("claim", 0.0)]
    B.apply(hits, {"claim": 0.25})
    assert hits[1].provenance["kind_bias"] == 0.25
    assert "kind_bias" not in hits[0].provenance


def test_empty_weights_switch_the_preference_off():
    hits = [H("body", 1.0), H("claim", 0.0)]
    B.apply(hits, {})
    assert hits[1].score == 0.0


def test_a_bias_cannot_reorder_a_wide_gap():
    # a claim that is genuinely far worse stays below; the bias leans, it does not decide
    hits = [H("body", 10.0), H("body", 5.0), H("claim", 0.0)]
    B.apply(hits, B.DEFAULT)
    assert max(hits, key=lambda h: h.score).kind == "body"


def test_unknown_kinds_are_untouched():
    hits = [H("body", 1.0), H("caption", 0.5)]
    B.apply(hits, B.DEFAULT)
    assert hits[1].score == 0.5
