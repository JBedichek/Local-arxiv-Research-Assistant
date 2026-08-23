"""Preferring pre-compressed evidence.

Some chunks in the corpus are not passages from a paper — they are a model's compression
of one: a claim extracted with its metric and conditions, or a synthesis written across
dozens of papers. Retrieving one of those costs a fraction of the context a raw passage
costs, and it has already been judged relevant once, by a model that had a question in
front of it.

So the pipeline should lean toward them. Not decide by them — a compression is lossy, and
a claim that lost the caveat is worse than the paragraph it replaced — but lean.

**The bias is expressed as a fraction of the candidate set's own score spread**, not as an
absolute number. That is what makes one configured value work in two places whose scores
are not remotely comparable: tier-2 produces inner products in roughly [0, 1], while the
cross-encoder produces raw logits that can be negative and can span tens. A bonus of
"0.15" would be decisive in the first and invisible in the second. A bonus of "15% of the
spread of these results" means the same thing in both.

**It is applied where it changes what gets read closely, not only what ranks first.** A
claim is short, and short text scores lower on raw cosine than the long passage it came
from, so a compressed chunk often dies at the shortlist cut before the reranker ever sees
it. Biasing only the final ordering would leave that failure in place.
"""

from __future__ import annotations

#: The default preference. Claims outrank syntheses because a claim is attributed to one
#: paper and carries its numbers, so a wrong one is checkable; a synthesis is a broader
#: summary with correspondingly more room to have drifted.
DEFAULT = {"claim": 0.15, "synthesis": 0.10}


def spread(hits) -> float:
    """The score range across a candidate set, which the bonus is a fraction of."""
    if not hits:
        return 0.0
    scores = [float(h.score) for h in hits]
    return max(scores) - min(scores)


def apply(hits, weights: dict | None = None) -> None:
    """Nudge compressed chunks up, in place, recording the nudge in provenance.

    The bonus is recorded rather than folded silently into the score: a hit that placed
    because of the bias should say so, otherwise the ranking cannot be explained and a
    badly-set weight is invisible.
    """
    weights = DEFAULT if weights is None else weights
    if not hits or not weights:
        return
    width = spread(hits)
    if width <= 0.0:
        # Every candidate scored alike — there is no spread to take a fraction of, and
        # inventing one would let the bias decide the whole ordering by itself.
        return
    for h in hits:
        w = float(weights.get(h.kind or "", 0.0))
        if not w:
            continue
        bonus = w * width
        h.score = float(h.score) + bonus
        if getattr(h, "provenance", None) is not None:
            h.provenance["kind_bias"] = round(bonus, 6)
