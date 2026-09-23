"""Inline detail on highlighted lesson text. The concept's own claims are tried first; if they
cannot add anything beyond what the highlighted sentences already cite (or answer a specific
question), the corpus is searched for new claims, which go through the same checks as any other
claim. The answer is then written and verified like a lesson: every sentence cites claims and is
re-judged against them, and when nothing survives the learner is told so."""

from __future__ import annotations

import time

from lara.learn import claims as CL
from lara.learn import lesson as LE
from lara.learn.llm import Llm

DEFAULT_REQUEST = "Explain the highlighted text in more detail."

ANSWER_SYSTEM = """A learner highlighted part of a lesson and asked for more detail. Answer \
using ONLY the numbered claims. Be as thorough as the claims allow -- there is no length limit, \
so do not cut the answer short to save space.

Format: one sentence per line, each ending -- before its full stop -- with the keys of the claims \
it rests on in brackets, like [c1] or [x1c2, c3]. A sentence with no key is not allowed.

- Draw out everything the claims support beyond the highlighted text: mechanism, conditions, \
numbers, caveats, examples, how it relates to nearby ideas. Do not add any fact that is not in \
the claims, and do not pad by restating the highlighted text itself.
- Convey certainty as marked: one paper, a hypothesis, replaced by later work.
- If the claims cannot add anything to the highlighted text, reply exactly: INSUFFICIENT"""

INSUFFICIENT = ("The paper corpus has nothing more on that which I can support. "
                "Try highlighting a narrower passage or asking a more specific question.")


def rank(claims: list[dict], focus: str, embed) -> list[dict]:
    vec = embed(focus) if embed else []
    scored = []
    for c in claims:
        score = CL.overlap(focus, c["text"])
        if vec:
            other = embed(c["text"])
            score = max(score, CL.cosine(vec, other) if other else 0.0)
        scored.append((score, c))
    return [c for _, c in sorted(scored, key=lambda t: t[0], reverse=True)]


async def _answer(llm: Llm, selection: str, request: str, claims: list[dict]):
    """(sections, stats) of the verified answer, or None when the claims cannot answer."""
    if not claims:
        return None
    by_key = {c["key"]: c for c in claims}
    prompt = (f"HIGHLIGHTED LESSON TEXT: {selection}\nREQUEST: {request}\n\nCLAIMS:\n"
              + "\n".join(LE.line(c) for c in claims))
    # cap=0: no fixed ceiling -- the answer may use whatever of the context window is left
    # once the prompt is in it (see reply_room), rather than an arbitrary token budget.
    text = await llm.ask(ANSWER_SYSTEM, prompt, default=2_000, cap=0, stage="learn_expand")
    if not text or text.upper().startswith("INSUFFICIENT"):
        return None
    sections, stats = await LE.verify(llm, LE.parse(text, set(by_key)), by_key)
    return (sections, stats) if sum(len(s["sentences"]) for s in sections) else None


def _adds_something(answer, selection_claims: set[str], specific: bool) -> bool:
    """Unless the learner asked something specific, the answer must draw on at least one claim
    the highlighted text did not already cite -- otherwise it is pure restatement, however long."""
    if answer is None:
        return False
    sentences = [s for sec in answer[0] for s in sec["sentences"]]
    return specific or any(k not in selection_claims for s in sentences for k in s["claims"])


def _next_id(expansions: list[dict]) -> int:
    nums = [int(e["id"][1:]) for e in expansions if str(e.get("id", "")).startswith("x")
            and e["id"][1:].isdigit()]
    return max(nums, default=0) + 1


async def _new_claims(llm: Llm, corpus, concept: dict, existing: list[dict], focus: str, n: int,
                      embed) -> list[CL.Claim]:
    passages = await CL.gather_passages(corpus, concept, focus=focus,
                                        exclude=frozenset(c["passage"]["chunk_id"] for c in existing))
    found, *_ = await CL.extract(llm, concept, passages, embed=embed, focus=focus)
    fresh = [c for c in found if all(CL.overlap(c.text, e["text"]) < CL.DUPLICATE_OVERLAP for e in existing)]
    for i, c in enumerate(fresh, 1):
        c.key = f"x{n}c{i}"
    # Compared against copies of the concept's claims, so the new ones can record agreement or
    # conflict with them without the lesson's own claims being altered.
    await CL.relate(llm, fresh + [CL.Claim.from_dict(e) for e in existing], embed=embed)
    return fresh


async def expand(llm: Llm, corpus, concept: dict, content: dict, *, selection: str,
                 question: str = "", selection_claims=(), section: int = 0, embed=None,
                 lesson_generated=None) -> dict:
    """The expansion to store, or {"insufficient": True, "message": ...} when nothing verifiable
    could be said."""
    question = question.strip()
    focus = f"{selection}\n{question}" if question else selection
    request = question or DEFAULT_REQUEST
    live = [c for c in content.get("claims", []) if not c.get("withdrawn")]
    context = rank(live, focus, embed)
    answer = await _answer(llm, selection, request, context)
    searched, new = False, []
    if not _adds_something(answer, set(selection_claims), bool(question)):
        searched = True
        n = _next_id(content.get("expansions", []))
        new = await _new_claims(llm, corpus, concept, live, focus, n, embed)
        pool = context + [c.to_dict() for c in new]
        answer = await _answer(llm, selection, request, pool)
        if answer is None:
            return {"insufficient": True, "message": INSUFFICIENT, "searched": True}
    cited = {k for sec in answer[0] for s in sec["sentences"] for k in s["claims"]}
    kept_new = [c.to_dict() for c in new if c.key in cited]
    return {"id": f"x{_next_id(content.get('expansions', []))}", "section": section,
            "selection": selection[:600], "question": question,
            "answer": {"sections": answer[0], "stats": answer[1]}, "claims": kept_new,
            "searched": searched, "lesson_generated": lesson_generated,
            "stale": False, "ts": time.time()}
