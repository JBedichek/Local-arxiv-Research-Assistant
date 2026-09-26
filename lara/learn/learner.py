"""The learner model: per-concept mastery, spaced repetition per quiz item, a diagnostic
pretest, and the choice of what to do next. Pure functions over plain dicts -- one learner per
course, stored beside it."""

from __future__ import annotations

import time

from lara.learn import judge as J
from lara.learn.llm import Llm

UNLOCK = 0.6            # prerequisite mastery that opens a concept
MASTERED = 0.75
PRETEST_CONCEPTS = 6
PRETEST_PASS_MASTERY = 0.8
INFERRED_MASTERY = UNLOCK
RETRY_SECONDS = 600
DAY = 86_400
_WEIGHT = {"mcq": 0.25, "short": 0.35, "predict": 0.35, "critique": 0.35}


def blank() -> dict:
    return {"concepts": {}, "items": {}, "pretest": {"state": "todo", "concepts": [], "answered": {}},
            "flags": [], "log": []}


def concept_state(learner: dict, cid: str) -> dict:
    return learner["concepts"].setdefault(cid, {"mastery": 0.0, "attempts": 0, "lesson_read": False,
                                                "inferred": False, "last": 0, "topics": {}})


def set_topic_familiarity(learner: dict, cid: str, topic_id: str, answer: str, explain: str = "") -> None:
    """`answer` is "yes", "no" or "partial"; `explain` is the learner's own words on what they
    already know, required for "partial" by the route, stored either way for the record."""
    cs = concept_state(learner, cid)
    cs.setdefault("topics", {})[topic_id] = {"answer": answer, "explain": explain}


def mastery_after(p: float, correct: bool, confidence: int, kind: str) -> float:
    """Right answers move mastery toward 1, more when confident; a wrong answer given with
    high confidence (a likely misconception) costs more than a wrong guess."""
    w = _WEIGHT.get(kind, 0.3)
    if correct:
        return p + (1 - p) * w * (0.7 if confidence <= 1 else 1.0)
    return p * (1 - w * (1.5 if confidence >= 3 else 1.0))


def schedule(item_state: dict, correct: bool, confidence: int, now: float) -> dict:
    """SM-2 style: quality 0-5 from correctness and confidence; a lapse retries in minutes."""
    q = (3 if confidence <= 1 else 4 if confidence == 2 else 5) if correct else (0 if confidence >= 3 else 1)
    s = {"ease": 2.5, "interval": 0.0, "reps": 0, "lapses": 0, **item_state}
    if q < 3:
        s.update(reps=0, interval=0.0, lapses=s["lapses"] + 1, due=now + RETRY_SECONDS)
    else:
        s["reps"] += 1
        s["interval"] = 1.0 if s["reps"] == 1 else 3.0 if s["reps"] == 2 else round(s["interval"] * s["ease"], 1)
        s["ease"] = max(1.3, s["ease"] + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        s["due"] = now + s["interval"] * DAY
    s.update(last_correct=correct, seen=s.get("seen", 0) + 1)
    return s


def record_answer(learner: dict, item: dict, correct: bool, confidence: int = 2, *,
                  now: float | None = None) -> None:
    now = now if now is not None else time.time()
    cs = concept_state(learner, item["concept"])
    cs.update(mastery=round(mastery_after(cs["mastery"], correct, confidence, item["type"]), 4),
              attempts=cs["attempts"] + 1, last=now, inferred=False)
    learner["items"][item["id"]] = schedule(learner["items"].get(item["id"], {}), correct, confidence, now)
    learner["log"] = (learner["log"] + [{"item": item["id"], "correct": correct,
                                          "confidence": confidence, "ts": now}])[-200:]


def passed(learner: dict, cid: str) -> bool:
    cs = concept_state(learner, cid)
    return bool(cs.get("passed")) or cs["mastery"] >= UNLOCK


def unlocked(course: dict, learner: dict, cid: str) -> bool:
    concept = next(c for c in course["concepts"] if c["id"] == cid)
    return all(passed(learner, p) for p in concept["prereqs"])


def prerequisites_of(course: dict, cid: str) -> set[str]:
    by_id = {c["id"]: c for c in course["concepts"]}
    out, stack = set(), list(by_id[cid]["prereqs"])
    while stack:
        p = stack.pop()
        if p not in out:
            out.add(p)
            stack.extend(by_id[p]["prereqs"])
    return out


def pretest_concepts(course: dict, count: int = PRETEST_CONCEPTS) -> list[str]:
    """Concepts spread evenly through the prerequisite order, so a pass or fail says
    something about the whole course."""
    ids = [c["id"] for c in course["concepts"]]
    if len(ids) <= count:
        return ids
    return [ids[round(i * (len(ids) - 1) / (count - 1))] for i in range(count)]


def finish_pretest(course: dict, learner: dict, results: dict[str, tuple[bool, int]]) -> None:
    """A concept passed with confidence counts as known -- and so does everything it depends
    on, at the unlock level, so the diagnostic skips what the learner already has."""
    for cid, (correct, confidence) in results.items():
        cs = concept_state(learner, cid)
        if correct and confidence >= 2:
            cs.update(mastery=max(cs["mastery"], PRETEST_PASS_MASTERY), lesson_read=True)
            for dep in prerequisites_of(course, cid):
                ds = concept_state(learner, dep)
                if ds["mastery"] < INFERRED_MASTERY:
                    ds.update(mastery=INFERRED_MASTERY, inferred=True, lesson_read=True)
    learner["pretest"]["state"] = "done"


def next_action(course: dict, learner: dict, contents: dict[str, dict], *, now: float | None = None) -> dict:
    """What to do now: a due review, else the first open concept's lesson and practice."""
    now = now if now is not None else time.time()
    quiz = {i["id"]: i for c in contents.values() for i in (c.get("quiz") or {}).get("items", [])}
    due = sorted((s.get("due", 0), qid) for qid, s in learner["items"].items()
                 if qid in quiz and s.get("due", 0) <= now and s.get("seen"))
    if due:
        return {"action": "review", "item": quiz[due[0][1]], "due": len(due)}
    for concept in course["concepts"]:
        cid = concept["id"]
        cs = concept_state(learner, cid)
        if cs["mastery"] >= MASTERED or cs.get("passed") or not unlocked(course, learner, cid):
            continue
        content = contents.get(cid) or {}
        if not content.get("lesson") or content.get("quiz") is None:
            return {"action": "build", "concept": cid}
        if content["lesson"].get("insufficient"):
            cs.update(passed=True, unavailable=True)      # nothing to teach: do not block what follows
            continue
        if not cs["lesson_read"]:
            return {"action": "lesson", "concept": cid}
        items = content["quiz"]["items"]
        unseen = [i for i in items if not learner["items"].get(i["id"], {}).get("seen")]
        if unseen:
            return {"action": "quiz", "item": unseen[0], "concept": cid}
        missed = [i for i in items if learner["items"][i["id"]].get("last_correct") is False]
        if missed:
            return {"action": "quiz", "item": missed[0], "concept": cid}
        # Every item answered, the last attempt on each right: the concept is learned.
        cs.update(passed=True, mastery=max(cs["mastery"], MASTERED))
    return {"action": "done"}


def competency_progress(course: dict, learner: dict) -> list[dict]:
    out = []
    for comp in course["competencies"]:
        cids = [c["id"] for c in course["concepts"] if comp["id"] in c["competencies"]]
        levels = [concept_state(learner, c)["mastery"] for c in cids]
        out.append({"id": comp["id"], "text": comp["text"], "concepts": cids,
                    "progress": round(sum(levels) / len(levels), 3) if levels else 0.0,
                    "mastered": bool(levels) and all(m >= MASTERED for m in levels)})
    return out


async def recheck_claim(llm: Llm, content: dict, key: str, note: str, *, now: float | None = None) -> dict:
    """Re-verifies a flagged claim against its own passage. A claim the passage no longer
    supports is withdrawn -- and the lesson and quiz items resting on it are marked stale or
    removed -- otherwise the learner is told the source still says it."""
    owner = None
    claim = next((c for c in content["claims"] if c["key"] == key), None)
    if claim is None:
        for expansion in content.get("expansions", []):
            claim = next((c for c in expansion.get("claims", []) if c["key"] == key), None)
            if claim is not None:
                owner = expansion
                break
    if claim is None:
        raise KeyError(key)
    verdict = await J.judge(llm, claim["text"], claim["passage"]["text"])
    claim.setdefault("flags", []).append({"note": note, "ts": now if now is not None else time.time(),
                                          "verdict": verdict})
    withdrawn = verdict != J.SUPPORTS
    if withdrawn:
        claim["withdrawn"] = True
        if owner is not None:
            owner["stale"] = True          # an added answer rests on it; the lesson does not
        else:
            if content.get("lesson"):
                content["lesson"]["stale"] = True
            for other in (content.get("lessons") or {}).values():
                if any(key in s["claims"] for sec in other["sections"] for s in sec["sentences"]):
                    other["stale"] = True
            if content.get("quiz"):
                content["quiz"]["items"] = [i for i in content["quiz"]["items"] if i["claim"] != key]
    return {"key": key, "verdict": verdict, "withdrawn": withdrawn, "passage": claim["passage"]}
