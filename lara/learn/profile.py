"""A cross-course knowledge profile, built only from quiz evidence (see
pipeline.answer_item) -- not pretest-driven inference or topic-familiarity self-reports,
which stay exactly as they were: per-course signals that gate unlocking and topic docs
within one course, never promoted here. (A pretest answer is itself a graded quiz item,
though -- see answer_item's own comment -- so it does feed this the same way an ordinary
quiz answer does; it is the *inferred* mastery bump pretest also grants to a concept's own
prerequisites that stays course-local.)

Unlike learner.py's per-course mastery (one learner per course, scoped to that course's
own concept ids), a concept here is keyed by its own title, slugged the same way
store.shared_get already keys a reusable built lesson across courses -- a real but
inexact match (two courses naming "the same" concept differently will not merge), accepted
for the same reason shared_get already ships with it: there is no precise global concept
taxonomy to match against instead.

Read by research.build_lesson (via pipeline.build_concept) as one short paragraph folded
into a new lesson's own objective -- see `digest`'s own docstring for why that is a
freshly generated summary, not the raw per-concept table handed to the model verbatim.
Also read directly (`snapshot`) by the learner-facing "your knowledge" view -- read-only;
the only way to change it is to answer more quizzes.
"""
from __future__ import annotations

import time

from lara.learn import learner as LN
from lara.learn import store
from lara.learn.llm import Llm

#: Evidence points kept per concept, oldest first -- enough for a digest to see a trend
#: (improving, slipping, consistently solid) without an old profile's file growing without
#: bound.
MAX_EVIDENCE = 8

DIGEST_SYSTEM = """You are summarizing one learner's demonstrated knowledge, from quiz \
results across every course they have taken, into a short note a lesson-writer will read \
before writing a new lesson for them.

Write 2-4 sentences of plain prose, no heading, no list. Name what they have clearly shown \
they understand, and separately what they have struggled with or shown only a shaky grasp \
of. Say nothing about a concept not listed below -- it has simply never been assessed, not \
that they lack it. Be specific (name the concepts) rather than vague ("some topics")."""


def _bucket(score: float, evidence: list) -> str:
    if not evidence:
        return "unknown"
    return "known" if score >= LN.MASTERED else "shaky"


def snapshot() -> dict:
    """Every concept the learner has ever been quizzed on, oldest activity last -- for the
    read-only "your knowledge" view. `{"concepts": [{"title", "score", "bucket",
    "attempts", "updated"}], "digest"}`."""
    data = store.load_profile()
    concepts = sorted(
        ({"title": c["title"], "score": c["score"],
          "bucket": _bucket(c["score"], c["evidence"]),
          "attempts": len(c["evidence"]), "updated": c["updated"]}
         for c in (data.get("concepts") or {}).values()),
        key=lambda c: c["updated"], reverse=True)
    return {"concepts": concepts, "digest": data.get("digest", "")}


def record_quiz_answer(title: str, correct: bool, confidence: int, kind: str) -> None:
    """Folds one graded quiz answer into this concept's entry, keyed by title (see the
    module docstring on why a title, not a concept id). The same confidence-weighted
    update learner.py's own per-course mastery uses (`mastery_after`), applied here to a
    global, cross-course score instead of a course-local one. A blank title (should not
    happen -- every concept has one) is simply ignored rather than polluting the ledger
    with an unattributable entry."""
    title = (title or "").strip()
    if not title:
        return
    data = store.load_profile()
    concepts = data.setdefault("concepts", {})
    key = store.slug(title, 60)
    entry = concepts.setdefault(key, {"title": title, "score": 0.0, "evidence": [],
                                      "updated": 0.0})
    entry["title"] = title    # the most recently seen phrasing wins -- cosmetic only
    entry["score"] = round(LN.mastery_after(entry["score"], correct, confidence, kind), 4)
    entry["evidence"] = (entry["evidence"]
                         + [{"correct": correct, "confidence": confidence,
                            "ts": time.time()}])[-MAX_EVIDENCE:]
    entry["updated"] = time.time()
    data["digest_stale"] = True
    store.save_profile(data)


def _render_ledger(concepts: dict) -> str:
    return "\n".join(
        f"- {c['title']}: {_bucket(c['score'], c['evidence'])} "
        f"(score {c['score']:.2f} over {len(c['evidence'])} quiz answer(s))"
        for c in concepts.values() if c["evidence"])


async def digest(llm: Llm) -> str:
    """The short paragraph `research.build_lesson` folds into a new lesson's own
    objective -- regenerated from the structured ledger whenever it changed since the
    last digest, never incrementally rewritten: each call reads the same ground truth
    fresh, so a bad write can never compound into the next the way an LLM asked to
    "update its own summary" could (the same reasoning `_apply_feedback`'s objective-
    append took over letting a revision rewrite its own prior text). Cached
    (`digest_stale`) so a course building 15 lessons in a row costs one extra call, not
    fifteen. `""` when nothing has ever been assessed -- nothing to fold in, and nothing
    for `research._lesson_objective` to add a paragraph about."""
    data = store.load_profile()
    listing = _render_ledger(data.get("concepts") or {})
    if not listing:
        return ""
    if not data.get("digest_stale", True) and data.get("digest"):
        return data["digest"]
    text = await llm.ask(DIGEST_SYSTEM, listing, default=300, cap=800,
                         stage="learner_profile_digest")
    data["digest"] = text
    data["digest_stale"] = False
    store.save_profile(data)
    return text
