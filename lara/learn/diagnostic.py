"""The diagnostic: what the learner already knows, measured against the course outline.

It runs once the outline exists (design decision D6: after the map), in three phases:

1. **Conversation.** A few turns about what the learner will do with the subject and what
   they already know well. Their background ("anchors") becomes weak prior evidence on every
   concept in the outline.
2. **Preparation.** For a spread of outline concepts, a short deep-research call
   (`lara.serve.learn_research.probe_research`, injected) grounds one
   free-answer question and a reference answer with a level rubric. These run in parallel, all
   before the first question is asked, so grading during the conversation is one fast model
   call against the reference -- never a research run while the learner waits.
3. **Probing.** Adaptive, like a binary search down the outline: ask about the middle of what
   is still uncertain; a pass is evidence for everything that concept depends on, a fail for
   everything that depends on it; ask again in the middle of what remains. It stops when the
   question budget runs out or nothing the course depends on is still uncertain.

Every answer becomes evidence in the learner's global profile (`profile`), not a mark in this
course alone."""

from __future__ import annotations

import asyncio
import re
import time

from lara.learn import learner as LN
from lara.learn import profile as PF
from lara.learn import store
from lara.learn import trace as TR
from lara.learn.llm import Llm

MAX_TURNS = 3
#: Concepts a probe is prepared for. More than the budget, so the search has room to move.
MAX_PREPARED = 12
#: Questions asked at most.
BUDGET = 10
#: Short research runs in flight at once while preparing.
PREPARE_CONCURRENCY = 3

CONVERSE_SYSTEM = """You are getting to know a learner before a self-study course. You see their \
goal and the course outline, ordered from foundations to frontier.

Learn two things, and only these: what they will do with the subject once they know it, and \
what they already know well (fields, jobs, courses, tools). Ask ONE short question at a time, \
plainly worded, and never ask what the goal or earlier answers already say.

Reply with JSON only:
{"question": "..." or null, "use": "...", "anchors": [{"domain": "...", "depth": "some"|"solid"|"expert"}]}

- question: null once you know both things well enough, or when told this is the last turn.
- use and anchors: your best reading so far, updated with every answer."""

ANCHOR_SYSTEM = """You estimate how well a learner already understands each concept of a course, \
from their background alone, before any question is asked.

Levels: 0 cannot recognise it, 1 recognises the name, 2 knows what it is for and why it exists, \
3 can use it correctly, 4 can explain its mechanism and critique variants.

Reply with JSON only: {"levels": {"<concept id>": 0-4}}

- Judge only from the background stated. A software engineer likely has level 2+ on general \
programming ideas and level 0-1 on field-specific methods. When the background says nothing \
about a concept, give 0 or 1.
- These are guesses to be tested, not conclusions: do not be generous."""

PROBE_SYSTEM = """You write one diagnostic question about a concept, from a research answer that \
explains it. The question tells how well a learner understands the concept.

Reply with JSON only:
{"question": "...", "reference": "...", "rubric": ["level 0 ...", "level 1 ...", "level 2 ...", "level 3 ...", "level 4 ..."]}

- question: answerable in two or three sentences of plain prose; asks WHY or HOW, not for a \
definition to recite or a number to remember. Understandable to someone who has met the concept \
before, without the research answer in front of them.
- reference: a strong answer, using ONLY what the research answer says.
- rubric: what an answer at each level shows (0 cannot recognise it, 1 recognises the name, \
2 knows what it is for and why it exists, 3 can use it correctly, 4 can explain its mechanism \
and critique variants), one line each, specific to this question."""

GRADE_SYSTEM = """You grade a learner's free answer to a diagnostic question, against a reference \
answer and a level rubric.

Reply with JSON only: {"level": 0-4, "feedback": "..."}

- level: the rubric level the answer shows. Grade understanding, not wording or length; a \
short answer with the right idea outranks a long one with the wrong idea. An answer that is \
wrong or only restates the question is level 0 or 1.
- feedback: one or two plain sentences for the learner: what they had right, then what the \
reference adds."""

#: An answer that is a way of saying "I don't know" -- graded without a model call.
_NO_IDEA = re.compile(r"^\s*(\?+|no idea|(i )?(do not|don'?t) know|idk|not sure|no clue|"
                      r"never heard of (it|this)|skip|pass)\W*$", re.I)


def blank() -> dict:
    return {"state": "todo", "turns": [], "pending": None, "probes": {}, "preparing": [],
            "asked": [], "results": {}, "current": None, "budget": BUDGET}


def load(course_id: str) -> dict:
    return {**blank(), **(store.load_diagnostic(course_id) or {})}


def need(concept: dict) -> int:
    return int(concept.get("need", 2))


def _outline(course: dict) -> str:
    return "\n".join(f"- {c['id']}: {c['title']} ({c.get('tier', 'core')}, needed at level {need(c)})"
                     f" -- {c.get('summary', '')}" for c in course["concepts"])


# ── phase 1: conversation ───────────────────────────────────────────────────────

async def converse(llm: Llm, course: dict, diag: dict, profile: dict) -> dict:
    """Asks the next question, or ends the conversation. Updates the profile's anchors and
    stated use with the model's latest reading."""
    history = "\n".join(f"Q: {t['question']}\nA: {t['answer']}" for t in diag["turns"]) or "(none yet)"
    last = "\nThis is the last turn: question must be null." if len(diag["turns"]) >= MAX_TURNS else ""
    known = ""
    if profile.get("anchors") or profile.get("use"):
        known = (f"\n\nFROM EARLIER COURSES (confirm or update; do not ask again what this "
                 f"already answers): use: {profile.get('use') or '(none)'}; background: "
                 + ", ".join(f"{a['domain']} ({a.get('depth', 'some')})" for a in profile.get("anchors", [])))
    data = await llm.ask_json(CONVERSE_SYSTEM,
                              f"GOAL: {course['goal']}\n\nOUTLINE:\n{_outline(course)}{known}\n\n"
                              f"CONVERSATION SO FAR:\n{history}{last}",
                              default=400, cap=1_500, stage="learn_diagnostic")
    data = data if isinstance(data, dict) else {}
    anchors = [{"domain": str(a.get("domain", "")).strip(),
                "depth": a.get("depth") if a.get("depth") in ("some", "solid", "expert") else "some"}
               for a in data.get("anchors") or [] if isinstance(a, dict) and str(a.get("domain", "")).strip()]
    if anchors:
        profile["anchors"] = anchors
    if str(data.get("use", "")).strip():
        profile["use"] = str(data["use"]).strip()
    q = data.get("question")
    question = str(q).strip() if isinstance(q, str) else str((q or {}).get("text", "")).strip() if isinstance(q, dict) else ""
    if question and len(diag["turns"]) < MAX_TURNS:
        diag.update(state="conversation", pending={"question": question})
    else:
        diag.update(pending=None)
    return diag


async def seed_from_anchors(llm: Llm, course: dict, profile: dict, *, now: float | None = None) -> int:
    """Weak prior evidence on every outline concept, from the learner's background. Returns how
    many concepts received it."""
    if not profile.get("anchors"):
        return 0
    background = "; ".join(f"{a['domain']} ({a.get('depth', 'some')})" for a in profile["anchors"])
    data = await llm.ask_json(ANCHOR_SYSTEM,
                              f"BACKGROUND: {background}\nWILL USE IT FOR: {profile.get('use') or '(not said)'}"
                              f"\n\nCONCEPTS:\n{_outline(course)}",
                              default=600, cap=2_000, stage="learn_diagnostic")
    levels = (data or {}).get("levels") if isinstance(data, dict) else None
    by_id = {c["id"]: c for c in course["concepts"]}
    n = 0
    for cid, level in (levels or {}).items():
        if cid in by_id:
            try:
                lvl = max(0, min(4, int(level)))
            except (TypeError, ValueError):
                continue
            PF.observe(profile, by_id[cid]["title"], "anchor", level=lvl, now=now, source=course["id"])
            n += 1
    return n


# ── phase 2: preparation ────────────────────────────────────────────────────────

def probe_targets(course: dict, limit: int = MAX_PREPARED) -> list[str]:
    """Concepts to prepare a probe for: the ones the goal needs (need 2+) first, spread evenly
    through the outline's order so a search can move anywhere in it."""
    order = [c["id"] for c in course["concepts"]]
    needed = [c["id"] for c in course["concepts"] if need(c) >= 2] or order
    if len(needed) <= limit:
        return needed
    picked = [needed[round(i * (len(needed) - 1) / (limit - 1))] for i in range(limit)]
    return sorted(dict.fromkeys(picked), key=order.index)


async def prepare_one(llm: Llm, research, concept: dict) -> dict:
    """{"question", "reference", "rubric", "references"} for one concept, or {"error"}."""
    question = (f"What is {concept['title']}, how does it work, and what is it used for? "
                f"{concept.get('summary', '')}").strip()
    try:
        found = await research(question)
    except Exception as e:                                      # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:300]}
    text = (found.get("text") or "").strip()
    if not text:
        return {"error": "the short research run found nothing"}
    data = await llm.ask_json(PROBE_SYSTEM,
                              f"CONCEPT: {concept['title']} -- {concept.get('summary', '')}\n\n"
                              f"RESEARCH ANSWER:\n{text[:8_000]}",
                              default=900, cap=2_500, stage="learn_probe")
    if not isinstance(data, dict) or not str(data.get("question", "")).strip() \
            or not str(data.get("reference", "")).strip():
        return {"error": "no usable question came back"}
    rubric = [str(r) for r in data.get("rubric") or []][:5]
    return {"question": str(data["question"]).strip(), "reference": str(data["reference"]).strip(),
            "rubric": rubric, "references": found.get("references") or {}}


async def prepare(llm: Llm, research, course: dict, diag: dict, *, on_change=None) -> dict:
    """Prepares every target's probe in parallel, saving progress as each lands."""
    TR.set_phase("diagnostic_preparation")
    by_id = {c["id"]: c for c in course["concepts"]}
    todo = [cid for cid in probe_targets(course) if cid not in diag["probes"]]
    diag.update(state="preparing", preparing=todo)
    if on_change:
        on_change()
    gate = asyncio.Semaphore(PREPARE_CONCURRENCY)

    async def one(cid: str) -> None:
        async with gate:
            diag["probes"][cid] = await prepare_one(llm, research, by_id[cid])
        diag["preparing"] = [x for x in diag["preparing"] if x != cid]
        if on_change:
            on_change()

    await asyncio.gather(*(one(cid) for cid in todo))
    diag["state"] = "probing"
    return diag


# ── phase 3: probing ────────────────────────────────────────────────────────────

def _ready(diag: dict) -> list[str]:
    return [cid for cid, p in diag["probes"].items() if "question" in p]


def next_probe(course: dict, diag: dict, profile: dict, *, now: float | None = None) -> str | None:
    """The concept to ask about next, or None when the diagnostic is done.

    Uncertain = the profile is not yet CONFIDENT about it. The middle of what is still
    uncertain is asked first: whichever way it goes, its answer settles one side."""
    if len(diag["asked"]) >= diag.get("budget", BUDGET):
        return None
    order = [c["id"] for c in course["concepts"]]
    titles = {c["id"]: c["title"] for c in course["concepts"]}
    open_ = [cid for cid in order if cid in _ready(diag) and cid not in diag["asked"]
             and PF.have(profile, titles[cid], now=now)["confidence"] < PF.CONFIDENT]
    return open_[len(open_) // 2] if open_ else None


async def grade(llm: Llm, probe: dict, answer: str) -> dict:
    """{"level", "feedback", "self"} -- `self` when the learner said they don't know."""
    if not answer.strip() or _NO_IDEA.match(answer):
        return {"level": 0, "feedback": "", "self": True}
    rubric = "\n".join(probe.get("rubric") or []) or "(none)"
    data = await llm.ask_json(GRADE_SYSTEM,
                              f"QUESTION: {probe['question']}\n\nREFERENCE ANSWER: {probe['reference']}\n\n"
                              f"RUBRIC:\n{rubric}\n\nLEARNER'S ANSWER: {answer.strip()[:3_000]}",
                              default=300, cap=1_000, stage="learn_grade")
    data = data if isinstance(data, dict) else {}
    try:
        level = max(0, min(4, int(data.get("level", 0))))
    except (TypeError, ValueError):
        level = 0
    return {"level": level, "feedback": str(data.get("feedback", "")).strip(), "self": False}


async def answer(llm: Llm, course: dict, diag: dict, profile: dict, cid: str, text: str, *,
                 now: float | None = None, advance_after: bool = True) -> dict:
    """Grades one answer, records it as evidence, propagates it, and (unless this is a later
    re-check, `advance_after=False`) picks the next question. Returns the result for this answer."""
    probe = diag["probes"].get(cid) or {}
    if "question" not in probe:
        raise KeyError(cid)
    title = next(c["title"] for c in course["concepts"] if c["id"] == cid)
    result = await grade(llm, probe, text)
    if result["self"]:
        PF.observe(profile, title, "self", level=0, now=now, source=course["id"])
    else:
        PF.observe(profile, title, "probe", level=result["level"], now=now, source=course["id"])
    PF.propagate(profile, course, cid, result["level"], now=now, source=course["id"])
    if cid not in diag["asked"]:
        diag["asked"].append(cid)
    diag["results"][cid] = {"answer": text.strip()[:3_000], **result, "ts": now or time.time()}
    if advance_after:
        advance(course, diag, profile, now=now)
    return result


def advance(course: dict, diag: dict, profile: dict, *, now: float | None = None) -> None:
    nxt = next_probe(course, diag, profile, now=now)
    diag["current"] = nxt
    if nxt is None:
        diag["state"] = "done"


#: Quick checks offered before one lesson, at most.
MAX_CHECKS = 2


def checks_for(course: dict, diag: dict, profile: dict, cid: str, *,
               now: float | None = None) -> list[dict]:
    """Prerequisites of `cid` the lesson will rely on (need 2+) whose level the profile is not
    confident about -- old evidence that has faded, or none at all -- and that have a prepared
    probe. Asked before the lesson is planned, so it is not written on a guess."""
    by_id = {c["id"]: c for c in course["concepts"]}
    out = []
    for pid in PF.closure(by_id, cid, lambda c: c["prereqs"]):
        c = by_id[pid]
        probe = diag["probes"].get(pid) or {}
        if (need(c) >= 2 and "question" in probe
                and PF.have(profile, c["title"], now=now)["confidence"] < PF.CONFIDENT):
            out.append({"concept": pid, "title": c["title"], "question": probe["question"]})
    return out[:MAX_CHECKS]


def apply_to_course(course: dict, learner: dict, profile: dict, *, now: float | None = None) -> list[str]:
    """Concepts the profile is confident the learner already has at the level the goal needs
    are marked known in this course, so the path skips them -- the role the old pretest's
    `finish_pretest` played. Returns their ids."""
    known = []
    for c in course["concepts"]:
        s = PF.have(profile, c["title"], now=now)
        if s["cautious"] >= need(c) and s["evidence"]:
            cs = LN.concept_state(learner, c["id"])
            cs.update(mastery=max(cs["mastery"], LN.INFERRED_MASTERY), inferred=True, lesson_read=True)
            known.append(c["id"])
    learner["pretest"]["state"] = "done"
    return known


def public(course: dict, diag: dict, profile: dict, *, now: float | None = None) -> dict:
    """What the page may see: no reference answers or rubrics before a question is answered."""
    titles = {c["id"]: c["title"] for c in course["concepts"]}
    cur = diag.get("current")
    probe = diag["probes"].get(cur) if cur else None
    results = []
    for cid in diag["asked"]:
        r = diag["results"].get(cid, {})
        p = diag["probes"].get(cid, {})
        results.append({"concept": cid, "title": titles.get(cid, cid), "question": p.get("question", ""),
                        "answer": r.get("answer", ""), "level": r.get("level"), "feedback": r.get("feedback", ""),
                        "reference": p.get("reference", "")})
    return {"state": diag["state"], "pending": diag.get("pending"), "turns": diag["turns"],
            "preparing": len(diag.get("preparing", [])), "prepared": len(_ready(diag)),
            "failed": sum(1 for p in diag["probes"].values() if "error" in p),
            "asked": len(diag["asked"]), "budget": diag.get("budget", BUDGET),
            "current": ({"concept": cur, "title": titles.get(cur, cur), "question": probe["question"]}
                        if probe and "question" in probe else None),
            "results": results, "anchors": profile.get("anchors", []), "use": profile.get("use", ""),
            "levels": [{"concept": c["id"], "title": c["title"], "need": need(c),
                        **{k: v for k, v in PF.have(profile, c["title"], now=now).items() if k != "dist"}}
                       for c in course["concepts"]]}
