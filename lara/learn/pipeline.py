"""Runs a course: scope -> map, then each concept on demand (claims -> lesson -> topics ->
quiz -> visuals), so a learner never waits on -- or pays for -- concepts they have not reached.
A concept already built by an earlier course is reused if recent enough."""

from __future__ import annotations

import asyncio
import time

from lara.learn import claims as CL
from lara.learn import depth as DP
from lara.learn import expand as EX
from lara.learn import graph as G
from lara.learn import learner as LN
from lara.learn import lesson as LE
from lara.learn import quiz as QZ
from lara.learn import store
from lara.learn import topics as TP
from lara.learn import visuals as VS
from lara.learn.llm import Llm

STAGES = ("claims", "lesson", "topics", "quiz", "visuals")
#: What a diagnostic needs from a concept -- no lesson, topics or visuals.
PRETEST_STAGES = ("claims", "quiz")
_building: dict[tuple[str, str], asyncio.Task] = {}
_building_topic: dict[tuple[str, str, str], asyncio.Task] = {}
_editing: dict[tuple[str, str], asyncio.Lock] = {}
_writing: dict[tuple[str, str, str], asyncio.Task] = {}
#: Quiz items a concept may hold once deeper lessons have added claims to it.
QUIZ_CAP = 14


async def map_course(llm: Llm, corpus, course: dict) -> dict:
    course["status"] = "mapping"
    store.save_course(course)
    result = await G.build(llm, corpus, course)
    course.update(concepts=result["concepts"], removed_edges=result["removed_edges"],
                  uncovered=result["uncovered"], dropped_concepts=result["dropped"])
    course["status"] = "ready" if result["concepts"] else "failed"
    if not result["concepts"]:
        course["error"] = "the corpus held nothing to build a concept map from"
    store.save_course(course)
    return course


def _stage_done(content: dict, stage: str) -> bool:
    if stage == "lesson":
        return bool(content.get("lesson")) and not content["lesson"].get("stale")
    return stage in content.get("stages", {}) and content.get(stage) is not None


async def build_concept(llm: Llm, corpus, course: dict, cid: str, *, stages=STAGES, embed=None,
                        force: bool = False) -> dict:
    concept = {**next(c for c in course["concepts"] if c["id"] == cid), "goal": course["goal"]}
    content = store.load_concept(course["id"], cid) or {"concept": cid, "title": concept["title"],
                                                        "stages": {}}
    if not force and not content.get("claims") and (shared := store.shared_get(concept["title"])):
        content = {**shared, "concept": cid, "reused": True}
    prereqs = {c["id"]: c["title"] for c in course["concepts"]}
    build = store.load_build(course["id"], cid)

    def note(**kw) -> None:
        build.update(kw)
        store.save_build(course["id"], cid, build)

    async def run_stage(stage: str) -> None:
        note(stage=stage, error="")
        if stage == "claims":
            content.update(await CL.build(llm, corpus, concept, embed=embed))
            # Claims are renumbered, so answers and extra lesson versions that cite the old
            # keys would now point at different claims. Topics are re-indexed from the new
            # lesson once it is written, so old ones (and their docs) would not match either.
            content.pop("lessons", None)
            content.pop("expansions", None)
            content.pop("topic_docs", None)
        elif stage == "lesson":
            content["lesson"] = await LE.compose(llm, concept, content.get("claims", []),
                                                 content.get("conflicts", []),
                                                 [prereqs[p] for p in concept["prereqs"]])
        elif stage == "topics":
            content["topics"] = await TP.extract_topics(llm, concept, content.get("lesson"))
        elif stage == "quiz":
            content["quiz"] = await QZ.build(llm, concept, content.get("claims", []))
        elif stage == "visuals":
            content["visuals"] = await VS.build(llm, concept, content.get("claims", []),
                                               lesson=content.get("lesson"), corpus=corpus)
        content.setdefault("stages", {})[stage] = time.time()
        store.save_concept(course["id"], cid, content)

    try:
        for stage in STAGES:
            if stage in stages and (force or not _stage_done(content, stage)):
                await run_stage(stage)
        note(stage="done" if all(_stage_done(content, s) for s in STAGES) else build.get("stage", ""),
             error="")
        if all(_stage_done(content, s) for s in STAGES) and not content.get("reused"):
            store.shared_put(concept["title"], {k: v for k, v in content.items() if k != "concept"})
    except Exception as e:                                     # noqa: BLE001
        note(stage="error", error=f"{type(e).__name__}: {e}")
        raise
    return content


def ensure_concept(llm: Llm, corpus, course: dict, cid: str, **kw) -> asyncio.Task:
    """The one running build of this concept, started if there is none -- two requests for
    the same lesson share it instead of paying twice."""
    key = (course["id"], cid)
    task = _building.get(key)
    if task is None or task.done():
        task = asyncio.ensure_future(build_concept(llm, corpus, course, cid, **kw))
        _building[key] = task
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
    return task


async def set_familiarity(llm: Llm, corpus, course: dict, learner: dict, cid: str, topic_id: str,
                          answer: str, explain: str, *, embed=None) -> dict:
    """Records how familiar the learner says they are with one of the lesson's indexed topics.
    "no" and "partial" start (or reuse) that topic's background document in the background --
    "yes" just records the answer, nothing is written for a topic the learner already knows."""
    LN.set_topic_familiarity(learner, cid, topic_id, answer, explain)
    store.save_learner(course["id"], learner)
    if answer in ("no", "partial"):
        ensure_topic_doc(llm, corpus, course, cid, topic_id, tailor=explain if answer == "partial" else "", embed=embed)
    return LN.concept_state(learner, cid)


def ensure_topic_doc(llm: Llm, corpus, course: dict, cid: str, topic_id: str, *, tailor: str = "",
                     embed=None) -> asyncio.Task:
    """The one running build of this topic's document, started if there is none."""
    key = (course["id"], cid, topic_id)
    task = _building_topic.get(key)
    if task is None or task.done():
        task = asyncio.ensure_future(_build_topic_doc(llm, corpus, course, cid, topic_id, tailor, embed))
        _building_topic[key] = task
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
    return task


async def _build_topic_doc(llm: Llm, corpus, course: dict, cid: str, topic_id: str, tailor: str,
                           embed) -> None:
    concept = {**next(c for c in course["concepts"] if c["id"] == cid), "goal": course["goal"]}
    lock = _editing.setdefault((course["id"], cid), asyncio.Lock())
    async with lock:
        content = store.load_concept(course["id"], cid) or {}
        topic = next((t for t in content.get("topics", []) if t["id"] == topic_id), None)
        if topic is None:
            return
        content.setdefault("topic_docs", {})[topic_id] = {"status": "building", "tailor": tailor, "doc": None}
        store.save_concept(course["id"], cid, content)
    try:
        doc = await TP.build_doc(llm, corpus, concept, topic, tailor=tailor, embed=embed)
    except Exception as e:                                        # noqa: BLE001
        doc = {"insufficient": True, "sections": [], "claims": [], "chart": None,
              "message": f"{type(e).__name__}: {e}"}
    async with lock:
        content = store.load_concept(course["id"], cid) or {}
        content.setdefault("topic_docs", {})[topic_id] = {"status": "done", "tailor": tailor, "doc": doc}
        store.save_concept(course["id"], cid, content)


async def expand_selection(llm: Llm, corpus, course: dict, cid: str, *, selection: str,
                           question: str = "", selection_claims=(), section: int = 0,
                           embed=None, variant: str = "standard") -> dict:
    """More detail on highlighted lesson text; a verified answer is kept with the concept.
    One at a time per concept: ids and claim keys are numbered from what is already stored."""
    meta = next(c for c in course["concepts"] if c["id"] == cid)
    lock = _editing.setdefault((course["id"], cid), asyncio.Lock())
    async with lock:
        content = store.load_concept(course["id"], cid)
        lesson = LE.lessons_of(content or {}).get(variant)
        if not content or not lesson or lesson.get("insufficient"):
            raise ValueError("this concept has no lesson to expand")
        result = await EX.expand(llm, corpus, {**meta, "goal": course["goal"]}, content,
                                 selection=selection, question=question,
                                 selection_claims=selection_claims, section=section, embed=embed,
                                 lesson_generated=lesson["generated"])
        if result.get("insufficient"):
            return result
        result["variant"] = variant
        content = store.load_concept(course["id"], cid) or content
        content.setdefault("expansions", []).append(result)
        store.save_concept(course["id"], cid, content)
        return result


def variant_key(variant: str, pages=None) -> tuple[str, int | None]:
    """(storage key, target pages) for a requested lesson variant."""
    if variant == "tldr":
        return "tldr", None
    if variant == "thorough":
        return "thorough", DP.THOROUGH_PAGES
    if variant == "pages":
        n = DP.clamp_pages(pages)
        return f"pages-{n}", n
    raise ValueError(f"unknown lesson variant {variant!r}")


async def _top_up_quiz(llm: Llm, meta: dict, content: dict, new_claims: list[dict]) -> None:
    """Quiz items for claims a deeper lesson added, up to `QUIZ_CAP` in all; existing items
    (and the learner's history on them) are left alone."""
    quiz = content.get("quiz")
    if quiz is None:
        quiz = content["quiz"] = {"items": [], "dropped": 0}
    room = QUIZ_CAP - len(quiz["items"])
    if not new_claims or room <= 0:
        return
    taken = [int(i["id"].rsplit("-q", 1)[1]) for i in quiz["items"] if "-q" in i["id"]]
    extra = await QZ.build(llm, meta, new_claims, start=max(taken, default=0) + 1, limit=room)
    quiz["items"] += extra["items"]
    quiz["dropped"] = quiz.get("dropped", 0) + extra["dropped"]


async def write_variant(llm: Llm, corpus, course: dict, cid: str, variant: str, pages=None, *,
                        embed=None) -> dict:
    """Writes a TL;DR, thorough or custom-length lesson and keeps it beside the others. Deeper
    ones research new claims, which join the concept's own (and its quiz)."""
    key, n = variant_key(variant, pages)
    meta = {**next(c for c in course["concepts"] if c["id"] == cid), "goal": course["goal"]}
    titles = {c["id"]: c["title"] for c in course["concepts"]}
    async with _editing.setdefault((course["id"], cid), asyncio.Lock()):
        content = store.load_concept(course["id"], cid)
        if not content or not content.get("claims"):
            raise ValueError("build this concept before writing another version of its lesson")

        def note(**kw) -> None:
            store.save_build(course["id"], cid, {"variant": key, "error": "", **kw})

        try:
            note(stage=f"writing {key}", detail="starting")
            if key == "tldr":
                lesson = await LE.compose(llm, meta, content["claims"], content.get("conflicts", []),
                                          [titles[p] for p in meta["prereqs"]], length_note=LE.tldr_note(len(LE.usable(content["claims"]))))
                changes = {}
            else:
                lesson, changes = await DP.deepen(
                    llm, corpus, meta, content, n, embed=embed,
                    progress=lambda d: note(stage=f"writing {key}", detail=d))
            lesson["variant"] = key
            if lesson.get("insufficient"):
                # A failed attempt is reported, never stored: it must not replace a version
                # the learner already has, or appear as a lesson.
                raise ValueError(lesson.get("message") or "the corpus held too little to write this")
            content = store.load_concept(course["id"], cid) or content
            if changes:
                content["claims"], content["conflicts"] = changes["claims"], changes["conflicts"]
                await _top_up_quiz(llm, meta, content, changes["new_claims"])
            content.setdefault("lessons", {})[key] = lesson
            store.save_concept(course["id"], cid, content)
            note(stage="done", detail="")
        except Exception as e:                                 # noqa: BLE001
            note(stage="error", error=f"{type(e).__name__}: {e}")
            raise
    return lesson


def ensure_variant(llm: Llm, corpus, course: dict, cid: str, variant: str, pages=None, **kw) -> asyncio.Task:
    """The one running write of this variant, started if there is none."""
    key, _ = variant_key(variant, pages)
    slot = (course["id"], cid, key)
    task = _writing.get(slot)
    if task is None or task.done():
        task = asyncio.ensure_future(write_variant(llm, corpus, course, cid, variant, pages, **kw))
        _writing[slot] = task
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
    return task


def delete_expansion(course_id: str, cid: str, xid: str) -> bool:
    content = store.load_concept(course_id, cid)
    if not content:
        return False
    kept = [e for e in content.get("expansions", []) if e["id"] != xid]
    changed = len(kept) != len(content.get("expansions", []))
    if changed:
        content["expansions"] = kept
        store.save_concept(course_id, cid, content)
    return changed


def contents_for(course: dict) -> dict[str, dict]:
    out = {}
    for c in course["concepts"]:
        loaded = store.load_concept(course["id"], c["id"])
        if loaded:
            out[c["id"]] = loaded
    return out


def public_item(item: dict) -> dict:
    """A quiz item as the learner may see it: no answer, no explanation, until graded."""
    return {k: v for k, v in item.items() if k not in ("answer", "explanation", "claim", "source")}


def find_item(course: dict, item_id: str) -> dict | None:
    cid = item_id.rsplit("-q", 1)[0]
    content = store.load_concept(course["id"], cid) or {}
    return next((i for i in (content.get("quiz") or {}).get("items", []) if i["id"] == item_id), None)


async def start_pretest(llm: Llm, corpus, course: dict, learner: dict, *, embed=None) -> list[dict]:
    """Builds just enough (claims and quiz) for a spread of concepts and picks one item each."""
    chosen = LN.pretest_concepts(course)
    await asyncio.gather(*(ensure_concept(llm, corpus, course, cid, stages=PRETEST_STAGES, embed=embed)
                           for cid in chosen))
    items = []
    for cid in chosen:
        pool = ((store.load_concept(course["id"], cid) or {}).get("quiz") or {}).get("items", [])
        pool = sorted(pool, key=lambda i: {"predict": 0, "short": 1, "mcq": 2}.get(i["type"], 3))
        if pool:
            items.append(pool[0])
    learner["pretest"] = {"state": "active" if items else "done", "concepts": chosen,
                          "items": [i["id"] for i in items], "answered": {}}
    store.save_learner(course["id"], learner)
    return items


async def answer_item(llm: Llm, course: dict, learner: dict, item_id: str, response: str,
                      confidence: int = 2) -> dict:
    item = find_item(course, item_id)
    if item is None:
        raise KeyError(item_id)
    graded = await QZ.grade(llm, item, response)
    pretest = learner["pretest"]
    if pretest.get("state") == "active" and item_id in pretest.get("items", []):
        pretest["answered"][item_id] = [graded["correct"], confidence]
        if set(pretest["answered"]) >= set(pretest["items"]):
            results = {i.rsplit("-q", 1)[0]: tuple(pretest["answered"][i]) for i in pretest["items"]}
            LN.finish_pretest(course, learner, results)
            for iid, (ok, conf) in pretest["answered"].items():
                LN.record_answer(learner, find_item(course, iid), ok, conf)
    else:
        LN.record_answer(learner, item, graded["correct"], confidence)
    store.save_learner(course["id"], learner)
    return graded


async def submit_critique(llm: Llm, course: dict, learner: dict, cid: str, text: str) -> list[dict]:
    content = store.load_concept(course["id"], cid) or {}
    points = await QZ.critique(llm, content.get("claims", []), text)
    if points:
        ok = all(p["verdict"] == "supported" for p in points)
        LN.record_answer(learner, {"id": f"{cid}-critique", "concept": cid, "type": "critique"}, ok, 2)
        store.save_learner(course["id"], learner)
    return points


async def flag_claim(llm: Llm, course: dict, learner: dict, cid: str, key: str, note: str) -> dict:
    content = store.load_concept(course["id"], cid)
    if content is None:
        raise KeyError(cid)
    result = await LN.recheck_claim(llm, content, key, note)
    store.save_concept(course["id"], cid, content)
    learner["flags"].append({"concept": cid, "claim": key, "note": note, "ts": time.time(),
                             "withdrawn": result["withdrawn"]})
    store.save_learner(course["id"], learner)
    return result


def overview(course: dict, learner: dict) -> dict:
    contents = contents_for(course)
    concepts = []
    for c in course["concepts"]:
        cs = LN.concept_state(learner, c["id"])
        content = contents.get(c["id"], {})
        concepts.append({**{k: c[k] for k in ("id", "title", "summary", "prereqs", "competencies")},
                         "mastery": cs["mastery"], "passed": bool(cs.get("passed")),
                         "inferred": bool(cs.get("inferred")), "unavailable": bool(cs.get("unavailable")),
                         "unlocked": LN.unlocked(course, learner, c["id"]),
                         "built": [s for s in STAGES if _stage_done(content, s)],
                         "build": store.load_build(course["id"], c["id"]),
                         "sources": len(c["sources"])})
    action = LN.next_action(course, learner, contents)
    if "item" in action:
        action = {**action, "item": public_item(action["item"])}
    return {"id": course["id"], "goal": course["goal"], "status": course["status"],
            "competencies": LN.competency_progress(course, learner), "concepts": concepts,
            "uncovered": course.get("uncovered", []), "next": action,
            "pretest": learner["pretest"]["state"], "due": sum(
                1 for s in learner["items"].values() if s.get("seen") and s.get("due", 0) <= time.time())}
