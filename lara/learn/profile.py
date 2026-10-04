"""A cross-course knowledge profile: one per person, shared by every course.

For each concept (and each smaller term a lesson uses) it holds a probability over five levels
of understanding, not a single score:

    0 unknown     can't recognise it
    1 heard of    recognises the name, roughly places it
    2 intuition   can say what it is for and why it exists
    3 use         can apply it correctly in their own setting
    4 derive      can explain the mechanism and critique variants

The distribution is never stored, only the evidence: each observation -- a graded quiz answer
(`record_quiz_answer`), a graded diagnostic answer, a guess from the learner's stated
background, an "explain this" highlight, an "I know this" -- is kept with its time, and the
distribution is recomputed from the prior and the evidence whenever it is read. That is what
lets old evidence fade (each observation's weight halves every `HALF_LIFE_DAYS`) and what lets
the likelihoods below be changed later without migrating anyone's profile. Quiz evidence
recorded before this model existed (`{"correct", "confidence", "ts"}`, no `kind`) still reads
as quiz evidence.

A concept is keyed by its own title, slugged the same way store.shared_get keys a reusable
built lesson across courses -- a real but inexact match (two courses naming "the same"
concept differently will not merge), accepted for the same reason shared_get already ships
with it: there is no precise global concept taxonomy to match against instead.

Read three ways: by `treatment.plan` (via `have`), which decides how each lesson explains
each concept and term; by research.build_lesson (via `digest`), as one short paragraph folded
into a new lesson's own objective; and by the learner-facing "your knowledge" view
(`snapshot`)."""
from __future__ import annotations

import math
import time

from lara.learn import learner as LN
from lara.learn import store
from lara.learn.llm import Llm

LEVELS = ("unknown", "heard of", "intuition", "use", "derive")
N = len(LEVELS)
#: Before any evidence: most concepts in a research field are unknown to most people.
PRIOR = (0.30, 0.25, 0.20, 0.15, 0.10)
HALF_LIFE_DAYS = 120
#: A level held with less probability than this is not relied on: a treatment plan reads
#: the cautious level instead (`cautious_level`), and the course offers a quick check.
CONFIDENT = 0.6
#: Evidence points kept per entry, oldest dropped first -- by then it has decayed to little.
MAX_EVIDENCE = 40
DAY = 86_400

#: P(answer right | level) for a quiz item about the concept.
_P_CORRECT = (0.15, 0.30, 0.60, 0.85, 0.95)
#: "Explain this" on a term: someone at level 2+ rarely asks what a word means.
_EXPLAIN = (1.0, 0.8, 0.25, 0.08, 0.04)
#: A dependent concept was shown at level 2 or above: its prerequisites are probably known too.
_PREREQ_OF_PASSED = (0.25, 0.45, 1.0, 1.0, 1.0)
#: A prerequisite was shown at level 1 or below: what depends on it is probably not known.
_DEPENDENT_OF_FAILED = (1.0, 1.0, 0.45, 0.2, 0.1)

DIGEST_SYSTEM = """You are summarizing one learner's demonstrated knowledge, from quiz \
results and diagnostic answers across every course they have taken, into a short note a \
lesson-writer will read before writing a new lesson for them.

Write 2-4 sentences of plain prose, no heading, no list. Name what they have clearly shown \
they understand, and separately what they have struggled with or shown only a shaky grasp \
of. Say nothing about a concept not listed below -- it has simply never been assessed, not \
that they lack it. Be specific (name the concepts) rather than vague ("some topics")."""


def key(title: str) -> str:
    return store.slug(title, 60)


def _gauss(center: float, sigma: float) -> tuple[float, ...]:
    return tuple(math.exp(-((lvl - center) ** 2) / (2 * sigma * sigma)) for lvl in range(N))


def likelihood(ev: dict) -> tuple[float, ...]:
    """P(this observation | level), for each level. Unknown kinds are uninformative."""
    kind = ev.get("kind") or ("quiz" if "correct" in ev else "")
    if kind == "probe":
        # A graded free answer: the grader's level, with grader noise.
        return _gauss(float(ev["level"]), 0.7)
    if kind == "anchor":
        # A guess from the learner's background, before anything was asked: weak.
        return _gauss(float(ev["level"]), 1.4)
    if kind == "self":
        # The learner's own word: "I know this" (3), "partly" (1), "no idea" (0).
        return _gauss(float(ev["level"]), 0.6 if ev["level"] == 0 else 1.2)
    if kind == "quiz":
        return _P_CORRECT if ev.get("correct") else tuple(1 - p for p in _P_CORRECT)
    if kind == "explain":
        return _EXPLAIN
    if kind == "prereq_of_passed":
        return _PREREQ_OF_PASSED
    if kind == "dependent_of_failed":
        return _DEPENDENT_OF_FAILED
    return (1.0,) * N


def distribution(entry: dict | None, *, now: float | None = None) -> list[float]:
    """P(level) for one entry, from the prior and its evidence, each observation weighted by
    how recent it is. Computed in log space: forty observations multiply out to underflow."""
    now = now if now is not None else time.time()
    logp = [math.log(p) for p in PRIOR]
    for ev in (entry or {}).get("evidence", []):
        age_days = max(0.0, now - ev.get("ts", now)) / DAY
        weight = ev.get("weight", 1.0) * 0.5 ** (age_days / HALF_LIFE_DAYS)
        for lvl, p in enumerate(likelihood(ev)):
            logp[lvl] += weight * math.log(max(p, 1e-6))
    top = max(logp)
    raw = [math.exp(x - top) for x in logp]
    total = sum(raw)
    return [x / total for x in raw]


def cautious_level(dist: list[float]) -> int:
    """The highest level the learner is at or above with probability >= CONFIDENT."""
    at_least = 0.0
    for lvl in range(N - 1, -1, -1):
        at_least += dist[lvl]
        if at_least >= CONFIDENT:
            return lvl
    return 0


def summary(entry: dict | None, *, now: float | None = None) -> dict:
    """{"level", "confidence", "expected", "cautious", "dist", "evidence"} -- `level` is the
    most likely level and `confidence` its probability; `cautious` is what a plan relies on."""
    dist = distribution(entry, now=now)
    level = max(range(N), key=lambda i: dist[i])
    return {"level": level, "confidence": round(dist[level], 3),
            "expected": round(sum(i * p for i, p in enumerate(dist)), 3),
            "cautious": cautious_level(dist), "dist": [round(p, 4) for p in dist],
            "evidence": len((entry or {}).get("evidence", []))}


def load() -> dict:
    return {"anchors": [], "use": "", "concepts": {}, "terms": {}, **(store.load_profile() or {})}


def save(data: dict) -> None:
    store.save_profile(data)


def _entry(data: dict, title: str, *, term: bool) -> dict:
    bucket = data.setdefault("terms" if term else "concepts", {})
    return bucket.setdefault(key(title), {"title": title, "score": 0.0, "evidence": [],
                                          "updated": 0.0})


def have(data: dict, title: str, *, term: bool = False, now: float | None = None) -> dict:
    """The summary for a concept or term, without creating an entry for it."""
    return summary(data.get("terms" if term else "concepts", {}).get(key(title)), now=now)


def observe(data: dict, title: str, kind: str, *, term: bool = False, now: float | None = None,
            weight: float = 1.0, **detail) -> dict | None:
    """Records one observation in `data` (the caller saves it). `detail` carries the kind's
    own field (`level`, `correct`) and anything worth keeping (`source`: which course). A
    blank title is ignored rather than polluting the ledger with an unattributable entry."""
    title = (title or "").strip()
    if not title:
        return None
    ts = now if now is not None else time.time()
    e = _entry(data, title, term=term)
    e["evidence"] = (e["evidence"] + [{"kind": kind, "ts": ts, "weight": weight, **detail}])[-MAX_EVIDENCE:]
    e["updated"] = ts
    data["digest_stale"] = True
    return e


def propagate(data: dict, course: dict, cid: str, level: int, *, now: float | None = None,
              source: str = "") -> list[str]:
    """After a diagnostic answer on `cid`: a pass (level 2+) is weak evidence that everything
    it depends on is known; a fail (level 1 or below) is weak evidence that everything
    depending on it is not. Half weight: an inference, not a test. Returns the titles."""
    by_id = {c["id"]: c for c in course["concepts"]}
    if level >= 2:
        targets, kind = closure(by_id, cid, lambda c: c.get("prereqs", [])), "prereq_of_passed"
    else:
        dependents = {c["id"]: [d["id"] for d in course["concepts"] if c["id"] in d.get("prereqs", [])]
                      for c in course["concepts"]}
        targets, kind = closure(by_id, cid, lambda c: dependents[c["id"]]), "dependent_of_failed"
    titles = [by_id[t]["title"] for t in targets]
    for title in titles:
        observe(data, title, kind, now=now, weight=0.5, source=source)
    return titles


def closure(by_id: dict, start: str, step) -> list[str]:
    out, stack = [], list(step(by_id[start]))
    while stack:
        nxt = stack.pop()
        if nxt in by_id and nxt not in out and nxt != start:
            out.append(nxt)
            stack.extend(step(by_id[nxt]))
    return out


def _bucket(entry: dict, *, now: float | None = None) -> str:
    if not entry.get("evidence"):
        return "unknown"
    return "known" if summary(entry, now=now)["cautious"] >= 3 else "shaky"


def snapshot() -> dict:
    """Every concept the learner has any evidence on, most recent activity first -- for the
    "your knowledge" view. `{"concepts": [{"title", "score", "bucket", "level", "confidence",
    "attempts", "updated"}], "terms": [...], "anchors", "use", "digest"}`."""
    data = load()

    def rows(bucket: str) -> list[dict]:
        out = []
        for e in (data.get(bucket) or {}).values():
            if not e.get("evidence"):
                continue
            s = summary(e)
            out.append({"title": e["title"], "score": e.get("score", 0.0), "bucket": _bucket(e),
                        "level": s["level"], "confidence": s["confidence"],
                        "attempts": len(e["evidence"]), "updated": e.get("updated", 0.0)})
        return sorted(out, key=lambda c: c["updated"], reverse=True)

    return {"concepts": rows("concepts"), "terms": rows("terms"), "anchors": data.get("anchors", []),
            "use": data.get("use", ""), "digest": data.get("digest", "")}


def record_quiz_answer(title: str, correct: bool, confidence: int, kind: str) -> None:
    """Folds one graded quiz answer into this concept's entry: as evidence for the level
    model, and into the confidence-weighted `score` learner.py's per-course mastery uses
    (`mastery_after`), kept for readers of the older single-score ledger."""
    title = (title or "").strip()
    if not title:
        return
    data = load()
    e = observe(data, title, "quiz", correct=correct, confidence=confidence, item=kind)
    e["title"] = title    # the most recently seen phrasing wins -- cosmetic only
    e["score"] = round(LN.mastery_after(e.get("score", 0.0), correct, confidence, kind), 4)
    save(data)


def _render_ledger(concepts: dict) -> str:
    rows = []
    for c in concepts.values():
        if not c.get("evidence"):
            continue
        s = summary(c)
        rows.append(f"- {c['title']}: {_bucket(c)} (most likely level: {LEVELS[s['level']]}, "
                    f"from {len(c['evidence'])} piece(s) of evidence)")
    return "\n".join(rows)


async def digest(llm: Llm) -> str:
    """The short paragraph `research.build_lesson` folds into a new lesson's own
    objective -- regenerated from the evidence whenever it changed since the last digest,
    never incrementally rewritten: each call reads the same ground truth fresh, so a bad
    write can never compound into the next. Cached (`digest_stale`) so a course building 15
    lessons in a row costs one extra call, not fifteen. `""` when nothing was ever assessed."""
    data = load()
    listing = _render_ledger(data.get("concepts") or {})
    if not listing:
        return ""
    if not data.get("digest_stale", True) and data.get("digest"):
        return data["digest"]
    text = await llm.ask(DIGEST_SYSTEM, listing, default=300, cap=800,
                         stage="learner_profile_digest")
    data["digest"] = text
    data["digest_stale"] = False
    save(data)
    return text
