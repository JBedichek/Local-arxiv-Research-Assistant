"""The treatment plan: how a lesson handles each concept and term it relies on, decided before
the lesson is researched.

For every item, `need` is how well this lesson requires the learner to know it and `have` is
how well the profile says they do (the cautious level -- see `profile.summary`). The pair
picks one treatment:

    have >= need               use        use freely, no explanation
    need <= 1                  gloss      name it with a one-clause gloss
    need == 2                  intuition  a short intuition before it is used
    need >= 3, gap of 1        refresher  a brief refresher before it is used
    need >= 3, gap of 2+       section    explained fully, from the ground up, before it is used

The plan is written into the lesson's own objective (`brief`, read by
`research._lesson_objective`), so the synthesis run that writes the lesson also researches
what it is told to explain -- the research is aimed at this reader's gaps, not added after.
What it cannot ground in the corpus it must not explain from general knowledge (decision D1:
corpus only): once the lesson is written, an item it was told to explain but never explains
in a cited sentence is marked `uncovered` (`mark_uncovered`), and the page tells the reader
to look it up."""

from __future__ import annotations

import re

from lara.learn import profile as PF
from lara.learn.llm import Llm

USE, GLOSS, INTUITION, REFRESHER, SECTION = "use", "gloss", "intuition", "refresher", "section"
#: Treatments that need the lesson to explain the item, from sources.
EXPLAINED = (INTUITION, REFRESHER, SECTION)
MAX_TERMS = 12
#: New terms one section may introduce: more than a reader can hold at once.
NEW_TERMS_PER_SECTION = 3

TERMS_SYSTEM = """You list the technical terms a lesson on one concept will rely on.

Reply with JSON only: {"terms": [{"term": "...", "need": 0-4}]}

- Terms, methods, acronyms and named quantities a reader must understand to follow a lesson \
on this concept. Not the concept itself, and not the PREREQUISITES listed (they are planned \
already).
- need: how well THIS lesson requires the reader to know the term -- 1 recognise the name is \
enough, 2 know what it is for, 3 must be able to use it to follow the argument, 4 its mechanism \
is the point. Most terms are 1 or 2.
- At most 12, the most important first."""


def treatment(need: int, have: int) -> str:
    if have >= need:
        return USE
    if need <= 1:
        return GLOSS
    if need == 2:
        return INTUITION
    return REFRESHER if need - have == 1 else SECTION


def _item(profile: dict, title: str, need: int, kind: str, *, now=None) -> dict:
    s = PF.have(profile, title, term=(kind == "term"), now=now)
    if kind == "term" and not s["evidence"]:
        # A term that is also a concept the profile already has evidence on.
        concept = PF.have(profile, title, now=now)
        if concept["evidence"]:
            s = concept
    return {"title": title, "kind": kind, "need": need, "have": s["cautious"],
            "confidence": s["confidence"], "treatment": treatment(need, s["cautious"]),
            "uncovered": False}


async def plan(llm: Llm, profile: dict, course: dict, concept: dict, *,
               context: str = "", now=None) -> dict:
    """{"concept": the lesson's own concept, "items": [what it relies on]}. `context` is any
    text that says what the lesson will cover -- the concept's summary is always used; a
    built lesson's claims, when there are some, sharpen it."""
    by_id = {c["id"]: c for c in course["concepts"]}
    own_need = int(concept.get("need", 2))
    prereqs = [by_id[p] for p in concept.get("prereqs", []) if p in by_id]
    items = [_item(profile, p["title"], min(int(p.get("need", 2)), max(own_need, 1)), "concept", now=now)
             for p in prereqs]
    data = await llm.ask_json(TERMS_SYSTEM,
                              f"CONCEPT: {concept['title']} -- {concept.get('summary', '')}\n"
                              f"PREREQUISITES: {', '.join(p['title'] for p in prereqs) or '(none)'}"
                              + (f"\n\nWHAT THE LESSON COVERS:\n{context[:4_000]}" if context else ""),
                              default=500, cap=1_500, stage="learn_terms")
    seen = {PF.key(p["title"]) for p in prereqs} | {PF.key(concept["title"])}
    terms = (data or {}).get("terms", []) if isinstance(data, dict) else []
    for t in terms if isinstance(terms, list) else []:
        term = str(t.get("term", "") if isinstance(t, dict) else t).strip()
        if not term or PF.key(term) in seen:
            continue
        seen.add(PF.key(term))
        try:
            n = max(0, min(4, int(t.get("need", 1)))) if isinstance(t, dict) else 1
        except (TypeError, ValueError):
            n = 1
        items.append(_item(profile, term, n, "term", now=now))
        if len(items) >= MAX_TERMS + len(prereqs):
            break
    return {"concept": _item(profile, concept["title"], own_need, "concept", now=now), "items": items}


def brief(the_plan: dict) -> str:
    """The plan as the lesson's writer reads it, inside the lesson's objective."""
    groups = {USE: [], GLOSS: [], INTUITION: [], REFRESHER: [], SECTION: []}
    for i in the_plan["items"]:
        groups[i["treatment"]].append(i["title"])
    own = the_plan["concept"]
    lines = [f"THIS READER: needs this concept at level {own['need']} of 4 "
             f"({PF.LEVELS[own['need']]}) and is at about level {own['have']} "
             f"({PF.LEVELS[own['have']]}). Pitch the lesson at them:"]
    labels = [(USE, "already known -- use freely, do not explain"),
              (GLOSS, "define in a clause the first time it is used"),
              (INTUITION, "give a short intuition (two or three sentences) before using it"),
              (REFRESHER, "give a brief refresher before using it"),
              (SECTION, "research and explain fully, from the ground up, before the lesson "
                        "relies on it")]
    for key, label in labels:
        if groups[key]:
            lines.append(f"- {label}: {', '.join(groups[key])}")
    lines.append(f"- Introduce at most {NEW_TERMS_PER_SECTION} new terms in any one section.")
    lines.append("- Explain each of these ONCE in the whole lesson, where it is first needed. "
                 "This lesson is written a section at a time: if an earlier section already "
                 "explains a term, use it without explaining it again.")
    if any(groups[k] for k in EXPLAINED):
        lines.append("- Explain these only from what the papers say, with citations. If the "
                     "corpus has nothing that explains one, do not explain it from general "
                     "knowledge -- say plainly that it is a prerequisite to look up elsewhere.")
    return "\n".join(lines)


def _sentences(lesson: dict | None) -> list[dict]:
    return [s for sec in (lesson or {}).get("sections", []) for s in sec.get("sentences", [])]


def mark_uncovered(the_plan: dict, lesson: dict | None) -> list[str]:
    """After the lesson is written: every item it was told to explain but never names in a
    cited sentence is marked `uncovered`, in place. Returns their titles."""
    cited = " ".join(s["text"].lower() for s in _sentences(lesson) if s.get("claims"))
    out = []
    for i in the_plan["items"]:
        if i["treatment"] not in EXPLAINED:
            continue
        words = [w for w in re.findall(r"[a-z0-9]+", i["title"].lower()) if len(w) > 2] or [i["title"].lower()]
        if not all(w in cited for w in words):
            i["uncovered"] = True
            out.append(i["title"])
    return out


def public(the_plan: dict) -> dict:
    return {"concept": the_plan["concept"], "items": the_plan["items"],
            "uncovered": [i["title"] for i in the_plan["items"] if i["uncovered"]]}
