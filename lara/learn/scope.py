"""Scope: turn a goal into competencies, asking a clarifying question only when the answer
would change them, and showing how each answer changed the plan."""

from __future__ import annotations

import time

from lara.learn import store
from lara.learn.llm import Llm

#: Clarifying rounds before the map is accepted as it stands.
MAX_QUESTIONS = 4

SCOPE_SYSTEM = """You design the scope of a self-study course from the learner's goal and \
their answers so far.

Reply with JSON only:
{"competencies": [{"text": "..."}], "question": null or {"text": "...", "why": "...", "options": ["..."]}}

- competencies: 4 to 8 things the learner must be able to DO or DECIDE when finished, each \
concrete and testable ("choose a learning-rate schedule for a given budget"), not topics.
- question: ask ONE clarifying question only if its answer would materially change the \
competency list -- depth, target setting, constraints, prior knowledge. Otherwise null. \
Never ask what the goal already states or what was already answered. 2 to 4 short options; \
the learner may write their own answer instead.
- why: one sentence on how the answer would change the plan.
- If CURRENT COMPETENCIES are given, keep the exact wording of every one that still applies; \
change, add or drop only what the latest answer requires."""


def _competencies(raw) -> list[dict]:
    out, seen = [], set()
    for item in raw if isinstance(raw, list) else []:
        text = str(item.get("text", "") if isinstance(item, dict) else item).strip()
        cid = store.slug(text, 48)
        if text and cid not in seen:
            seen.add(cid)
            out.append({"id": cid, "text": text})
    return out


def diff(before: list[dict], after: list[dict]) -> dict:
    old, new = {c["id"]: c["text"] for c in before}, {c["id"]: c["text"] for c in after}
    return {"added": [t for i, t in new.items() if i not in old],
            "removed": [t for i, t in old.items() if i not in new]}


async def step(llm: Llm, goal: str, qa: list[dict],
               current: list[dict] | None = None) -> tuple[list[dict], dict | None]:
    """(competencies, next question or None) for the goal and the answers so far."""
    history = "\n".join(f"Q: {x['question']}\nA: {x['answer']}" for x in qa) or "(none yet)"
    final = "\nThis is the last round: question must be null." if len(qa) >= MAX_QUESTIONS else ""
    now = "\n".join(f"- {c['text']}" for c in current or [])
    keep = f"\n\nCURRENT COMPETENCIES:\n{now}" if now else ""
    data = await llm.ask_json(SCOPE_SYSTEM, f"GOAL: {goal}\n\nANSWERS SO FAR:\n{history}{keep}{final}",
                              default=500, cap=2_000, stage="learn_scope")
    if not isinstance(data, dict):
        return [], None
    q = data.get("question")
    question = None
    if isinstance(q, dict) and str(q.get("text", "")).strip() and len(qa) < MAX_QUESTIONS:
        question = {"question": str(q["text"]).strip(), "why": str(q.get("why", "")).strip(),
                    "options": [str(o) for o in (q.get("options") or [])][:4]}
    return _competencies(data.get("competencies")), question


def _apply(course: dict, competencies: list[dict], question: dict | None) -> None:
    if not competencies:
        course["status"] = "failed"
        course["error"] = "the model returned no readable competency map"
        return
    course["map_history"].append({"after": len(course["qa"]), "competencies": competencies,
                                  "diff": diff(course["competencies"], competencies)})
    course["competencies"] = competencies
    course["pending"] = question
    course["status"] = "scoping" if question else "scoped"


async def begin(llm: Llm, goal: str) -> dict:
    course = {"id": store.new_course_id(goal), "goal": goal.strip(), "created": time.time(),
              "status": "scoping", "qa": [], "pending": None, "competencies": [],
              "map_history": [], "concepts": []}
    _apply(course, *await step(llm, course["goal"], []))
    store.save_course(course)
    return course


async def answer(llm: Llm, course: dict, text: str) -> dict:
    """Records the answer to the pending question and re-plans."""
    pending = course.get("pending")
    if not pending:
        raise ValueError("no question is waiting for an answer")
    course["qa"].append({**pending, "answer": text.strip()})
    course["pending"] = None
    _apply(course, *await step(llm, course["goal"], course["qa"], course["competencies"]))
    store.save_course(course)
    return course


def accept(course: dict) -> dict:
    """Stop asking: take the map as it stands."""
    course["pending"] = None
    if course["competencies"]:
        course["status"] = "scoped"
    store.save_course(course)
    return course
