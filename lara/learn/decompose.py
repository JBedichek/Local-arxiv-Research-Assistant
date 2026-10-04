"""Decomposition: one thorough deep-research call that maps what a course on the learner's
request has to cover, from its simplest foundations to its most advanced concepts. Its
thorough answer is folded into the course-mapping objective (`graph.build_from_research`),
so the outline the mapping run writes is conditioned on that map rather than on whatever its
first goals happen to find -- and that outline is what the diagnostic's questions are written
from (`diagnostic`).

The question is a fixed template around the learner's own words (`QUESTION`), not one a model
writes: a model asked to phrase it narrowed a request for a general course down to the specific
topics it happened to think of, and the decomposition searched for those instead.

The call is long by design (`learn_research.DECOMPOSE_SYNTHESIS`), and nothing in it depends on
who is asking beyond the request itself, so its result is cached by that request: another
course started from the same words reuses it."""

from __future__ import annotations

from lara.learn import store
from lara.learn import trace as TR

QUESTION = ('Find the foundational concepts, from the simplest to the most complex, that someone '
            'needs to understand for a course based on this request: "{goal}"')


def question_for(goal: str) -> str:
    return QUESTION.format(goal=" ".join(goal.split()))


async def decompose(research, course: dict, *, force: bool = False) -> dict:
    """{"subject", "question", "text", "references", "stopped_because", "built", "cached"} --
    `subject` is the learner's request, which is also the cache key.

    `research` is `lara.serve.learn_research.decompose_research`, injected (`lara.learn`
    never imports `lara.serve`). Raises ValueError when the call comes back empty -- an
    empty map is not something to cache or build a course from."""
    TR.set_phase("subject_decomposition")
    subject = " ".join(course["goal"].split())
    question = question_for(subject)
    if not force and (cached := store.decomposition_get(subject)):
        TR.emit("decomposition_cached", subject=subject)
        return {**cached, "cached": True}
    found = await research(question)
    if not (found.get("text") or "").strip():
        reason = found.get("stopped_because") or ""
        raise ValueError(f"deep research returned nothing for {subject!r}"
                         + (f" ({reason})" if reason else ""))
    stored = store.decomposition_put(subject, {"question": question, "text": found["text"],
                                               "references": found.get("references") or {},
                                               "stopped_because": found.get("stopped_because", "")})
    return {**stored, "cached": False}
