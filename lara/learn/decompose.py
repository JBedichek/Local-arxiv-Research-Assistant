"""Decomposition: one thorough deep-research call that maps a whole subject, from the concepts
it builds on to its frontier. Its thorough answer is folded into the course-mapping objective
(`graph.build_from_research`), so the outline the mapping run writes is conditioned on a map
of the whole subject rather than on whatever its first goals happen to find -- and that outline
is what the diagnostic's questions are written from (`diagnostic`).

The call is long by design (`learn_research.DECOMPOSE_SYNTHESIS`), and nothing in it depends on
the learner, so its result is cached per subject: the next course on the same subject starts
from it."""

from __future__ import annotations

from lara.learn import store
from lara.learn import trace as TR
from lara.learn.llm import Llm

SUBJECT_SYSTEM = """You prepare one deep-research question that maps a whole subject for a \
self-study course. The question is answered by iterative retrieval over a corpus of arXiv \
papers, and its text is the first search query -- so it must use the field's own vocabulary.

Reply with JSON only: {"subject": "...", "question": "..."}

- subject: the subject as a short noun phrase a paper would use ("speculative decoding for LLM \
inference"), not the learner's sentence. Two learners with the same subject should get the same \
phrase.
- question: one question, one or two sentences, asking for the foundations, core methods and \
open problems of the subject -- the concepts and techniques the field builds on, from the most \
basic to the most advanced. Name the subject's own terms. Never ask "what would someone need to \
know": no paper answers that phrasing."""


async def plan_question(llm: Llm, goal: str, competencies: list[dict]) -> tuple[str, str]:
    """(subject, question). Falls back to the goal itself rather than failing the course: a
    worse first search query still maps something."""
    listing = "\n".join(f"- {c['text']}" for c in competencies) or "(none)"
    data = await llm.ask_json(SUBJECT_SYSTEM, f"GOAL: {goal}\n\nCOMPETENCIES:\n{listing}",
                              default=300, cap=1_000, stage="learn_subject")
    data = data if isinstance(data, dict) else {}
    subject = str(data.get("subject", "")).strip() or goal.strip()
    question = str(data.get("question", "")).strip() or (
        f"Foundations, core methods and open problems of {subject}: which concepts and "
        "techniques does the field build on, from basic to advanced?")
    return subject, question


async def decompose(llm: Llm, research, course: dict, *, force: bool = False) -> dict:
    """{"subject", "question", "text", "references", "stopped_because", "built", "cached"}.

    `research` is `lara.serve.learn_research.decompose_research`, injected (`lara.learn`
    never imports `lara.serve`). Raises ValueError when the call comes back empty -- an
    empty map is not something to cache or build a course from."""
    TR.set_phase("subject_decomposition")
    subject, question = await plan_question(llm, course["goal"], course.get("competencies", []))
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
