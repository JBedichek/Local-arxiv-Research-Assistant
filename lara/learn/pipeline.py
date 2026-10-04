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
from lara.learn import profile as PR
from lara.learn import quiz as QZ
from lara.learn import research as RS
from lara.learn import store
from lara.learn import topics as TP
from lara.learn import trace as TR
from lara.learn import visuals as VS
from lara.learn.llm import Llm, TokenMeter, metered

STAGES = ("claims", "lesson", "topics", "quiz", "visuals")
#: What a diagnostic needs from a concept -- no lesson, topics or visuals.
PRETEST_STAGES = ("claims", "quiz")
_building: dict[tuple[str, str], asyncio.Task] = {}
_building_topic: dict[tuple[str, str, str], asyncio.Task] = {}
_editing: dict[tuple[str, str], asyncio.Lock] = {}
_writing: dict[tuple[str, str, str], asyncio.Task] = {}
#: Quiz items a concept may hold once deeper lessons have added claims to it.
QUIZ_CAP = 14


async def map_course(llm: Llm, corpus, course: dict, *, topic_graph_synth=None) -> dict:
    """`topic_graph_synth`, when given (lara.serve.learn_research.topic_graph_synth),
    researches the course before mapping it (Phase 1 of the research-driven pipeline --
    see graph.build_from_research) instead of graph.build's one blind call against a
    handful of survey passages.

    Only the research-driven branch gets a trace: `G.build`'s single blind call has
    nothing worth polling for -- it returns before a client's first poll could plausibly
    land -- so starting a tracer for it would just be an always-empty file, not a
    meaningful one. Same start/emit/finally-stop shape `build_concept` below uses, so the
    frontend's trace poller/renderer needs no format awareness of which phase it is
    reading.

    Only the research-driven branch stops at "awaiting_approval" rather than "ready",
    too: it alone has a persisted synthesis graph (see learn_research._state_id) worth
    showing the learner before any lesson is built from it, and worth revising
    (`revise_plan` below) if they ask for changes -- `G.build`'s one-shot call has neither.
    `approve_plan` is what actually lets concept-building proceed from there."""
    course["status"] = "mapping"
    store.save_course(course)
    if topic_graph_synth is not None:
        TR.start(store.map_trace_path(course["id"]))
        TR.emit("map_start")
        try:
            result = await G.build_from_research(course, synth=topic_graph_synth)
        finally:
            TR.stop()
    else:
        result = await G.build(llm, corpus, course)
    # `subjects` is only ever present from the research-driven branch (see
    # graph._concepts_from_result) -- G.build's own blind path has no grouping to offer,
    # so a course it maps simply shows no subject hierarchy in the UI.
    course.update(concepts=result["concepts"], subjects=result.get("subjects", []),
                  removed_edges=result["removed_edges"],
                  uncovered=result["uncovered"], dropped_concepts=result["dropped"])
    if not result["concepts"]:
        course["status"] = "failed"
        course["error"] = "the corpus held nothing to build a concept map from"
    else:
        course["status"] = "awaiting_approval" if topic_graph_synth is not None else "ready"
    store.save_course(course)
    return course


def approve_plan(course: dict) -> dict:
    """Locks in the concept map currently awaiting the learner's approval, letting
    concept-building proceed -- see map_course's own docstring on why only the
    research-driven branch ever stops here."""
    course["status"] = "ready"
    store.save_course(course)
    return course


async def revise_plan(course: dict, feedback: str, *, revise) -> dict:
    """Re-maps the course from the learner's feedback on the plan awaiting their
    approval, keeping the superseded version in course["plan_history"]. `revise` is
    lara.serve.learn_research.topic_graph_revise, bound to this course -- resumes the
    persisted Phase 1 graph rather than researching from scratch, so already-answered
    research survives a revision.

    Raises ValueError (propagated to the caller, course left `awaiting_approval` with
    `error` set rather than stuck `mapping`) if nothing was persisted to revise from --
    a course mapped by the old blind G.build pipeline never reaches this at all (see
    routes/learn.py's own guard), but is worth handling defensively here too."""
    course.setdefault("plan_history", []).append({
        "concepts": course["concepts"], "subjects": course.get("subjects", []),
        "removed_edges": course.get("removed_edges", []),
        "uncovered": course.get("uncovered", []), "feedback": feedback,
        "superseded": time.time()})
    course["status"] = "mapping"
    store.save_course(course)
    TR.start(store.map_trace_path(course["id"]))
    TR.emit("map_revision_start", feedback=feedback)
    try:
        result = await G.revise_from_research(course, feedback, revise=revise)
    except Exception as e:                                       # noqa: BLE001
        course["status"] = "awaiting_approval"
        course["error"] = f"revision failed: {type(e).__name__}: {e}"
        store.save_course(course)
        raise
    finally:
        TR.stop()
    course.update(concepts=result["concepts"], subjects=result.get("subjects", []),
                  removed_edges=result["removed_edges"],
                  uncovered=result["uncovered"], dropped_concepts=result["dropped"])
    if not result["concepts"]:
        course["status"] = "failed"
        course["error"] = "the revision produced no concepts"
    else:
        course["status"] = "awaiting_approval"
        course["error"] = ""
    store.save_course(course)
    return course


def _stage_done(content: dict, stage: str) -> bool:
    if stage == "lesson":
        return bool(content.get("lesson")) and not content["lesson"].get("stale")
    return stage in content.get("stages", {}) and content.get(stage) is not None


async def build_concept(llm: Llm, corpus, course: dict, cid: str, *, stages=STAGES, embed=None,
                        force: bool = False, lesson_synth=None) -> dict:
    """`lesson_synth`, when given (lara.serve.learn_research.lesson_synth), researches
    and writes this concept's standard lesson from one full synthesis run scoped to it
    (Phase 2 of the research-driven pipeline -- see research.build_lesson) instead of the
    "claims" stage's claims.build() followed by the "lesson" stage's depth.deepen()."""
    concept = {**next(c for c in course["concepts"] if c["id"] == cid), "goal": course["goal"]}
    content = store.load_concept(course["id"], cid) or {"concept": cid, "title": concept["title"],
                                                        "stages": {}}
    if not force and not content.get("claims") and (shared := store.shared_get(concept["title"])):
        content = {**shared, "concept": cid, "reused": True}
        # Adopted whole and (being shared) already past every stage, so the loop below finds
        # nothing left to run and never calls run_stage's own save -- without this, this
        # course's own concept file is never written at all: build.json still ends up saying
        # "done" (nothing failed), but GET .../concepts/{cid} finds no file and returns empty
        # claims/lesson, with no way to tell from the page that anything is wrong.
        store.save_concept(course["id"], cid, content)
    prereqs = {c["id"]: c["title"] for c in course["concepts"]}
    build = store.load_build(course["id"], cid)
    meter = TokenMeter()
    llm = metered(llm, meter)

    def note(**kw) -> None:
        build.update(kw)
        store.save_build(course["id"], cid, build)

    async def run_stage(stage: str) -> None:
        note(stage=stage, error="")
        if stage == "claims" and lesson_synth is not None:
            # The learner's cross-course quiz-evidence digest (lara.learn.profile) --
            # only on this, the first build of a lesson (see build_lesson's own
            # docstring). Routed through `llm`, already `metered` above, so its tokens
            # show up in this build's own totals without a manual addition the way
            # RS.build_lesson's own (unmetered, routed through `synth` instead) does below.
            digest = await PR.digest(llm)
            content.update(await RS.build_lesson(concept, synth=lesson_synth,
                                                 profile_digest=digest))
            # Marks this concept as having a persisted synthesis graph behind its lesson
            # (see learn_research._state_id) -- what routes/learn.py's concept response
            # uses to offer "request a revision" only where one could actually work; a
            # concept reused whole from an earlier course (see build_concept's `shared`
            # branch above) or rebuilt by the legacy pipeline never sets this.
            content["research_driven"] = True
            # RS.build_lesson's own model calls run through `synth`, not this build's
            # `llm` -- so they never pass through `metered`'s wrap of `llm.complete`
            # above, and `meter` alone would report 0 regardless of what was actually
            # spent. Its trace carries the real totals; folded in here so the tokens_in/
            # tokens_out this build reports (live, and in the final note()) stay honest.
            meter.tokens_in += content.get("trace", {}).get("tokens_in", 0)
            meter.tokens_out += content.get("trace", {}).get("tokens_out", 0)
            content.pop("lessons", None)
            content.pop("expansions", None)
            content.pop("topic_docs", None)
        elif stage == "claims":

            async def on_event(name: str, payload: dict) -> None:
                # Written straight to disk as it happens, under the same lock every other
                # in-place concept edit uses -- so a build can be watched live, poll by poll,
                # instead of only showing its trace once the whole stage is done. `content`
                # (the closure's own copy) is not touched here; run_stage's own save below,
                # once CL.build returns, is what makes it authoritative.
                async with _editing.setdefault((course["id"], cid), asyncio.Lock()):
                    live = store.load_concept(course["id"], cid) or content
                    if name == "start":
                        live["trace"] = {**payload, "rounds": []}
                    elif name == "round":
                        live.setdefault("trace", {"coverage": {}, "budget": {}, "facets": [], "rounds": []})
                        live["trace"]["rounds"].append(payload)
                    store.save_concept(course["id"], cid, live)
                # tokens_in/out ride the same live cadence as the trace -- a poll during the
                # (usually longest) claims stage sees them climb round by round, not just once
                # the whole build finishes.
                note(tokens_in=meter.tokens_in, tokens_out=meter.tokens_out)

            content.update(await CL.build(llm, corpus, concept, embed=embed, on_event=on_event))
            # Claims are renumbered, so answers and extra lesson versions that cite the old
            # keys would now point at different claims. Topics are re-indexed from the new
            # lesson once it is written, so old ones (and their docs) would not match either.
            content.pop("lessons", None)
            content.pop("expansions", None)
            content.pop("topic_docs", None)
        elif stage == "lesson":
            # The default reading path researches a small outline section by section, the
            # same machinery a "thorough" or N-page request uses (`depth.deepen`), rather
            # than one flat pass over a fixed pull of passages -- see DP.STANDARD_PAGES.
            lesson, changes = await DP.deepen(llm, corpus, concept, content, DP.STANDARD_PAGES,
                                              embed=embed, quiet_shortfall=True)
            lesson["variant"] = "standard"
            if changes:
                content["claims"], content["conflicts"] = changes["claims"], changes["conflicts"]
            content["lesson"] = lesson
        elif stage == "topics":
            TR.set_phase("topics")
            content["topics"] = await TP.extract_topics(llm, concept, content.get("lesson"))
        elif stage == "quiz":
            TR.set_phase("quiz")
            content["quiz"] = await QZ.build(llm, concept, content.get("claims", []))
        elif stage == "visuals":
            content["visuals"] = await VS.build(llm, concept, content.get("claims", []),
                                               lesson=content.get("lesson"), corpus=corpus)
        content.setdefault("stages", {})[stage] = time.time()
        store.save_concept(course["id"], cid, content)
        # Every stage's own tokens count too, not just claims' -- this is what keeps the
        # total honest once lesson/topics/quiz/visuals have also spent some.
        note(tokens_in=meter.tokens_in, tokens_out=meter.tokens_out)

    pending = [s for s in STAGES if s in stages and (force or not _stage_done(content, s))]
    if pending:
        note(started=time.time(), tokens_in=0, tokens_out=0)
    # A fresh trace for this build only: `ensure_concept` runs each build in its own asyncio
    # task, which gets its own copy of this context, so concurrent builds' tracers never
    # collide even though the tracer itself is "global" state -- see trace.py's docstring.
    TR.start(store.trace_path(course["id"], cid))
    TR.emit("build_start", variant="standard", forced=force)
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
    finally:
        TR.stop()
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
                           embed=None, variant: str = "standard", synth=None) -> dict:
    """More detail on highlighted lesson text; a verified answer is kept with the concept.
    One at a time per concept: ids and claim keys are numbered from what is already stored.

    `synth`, when given (lara.serve.learn_research.expand_synth), backs any new search
    this expansion needs with one research leaf instead of claims.py's judge-verified
    extract() -- see expand.expand's own docstring. `None` (the default, and what every
    concept built before the research-driven pipeline existed still gets) keeps the old
    behavior exactly as it was."""
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
                                 lesson_generated=lesson["generated"], synth=synth)
        if result.get("insufficient"):
            return result
        result["variant"] = variant
        content = store.load_concept(course["id"], cid) or content
        content.setdefault("expansions", []).append(result)
        store.save_concept(course["id"], cid, content)
        return result


async def revise_lesson(course: dict, cid: str, feedback: str, *, revise) -> dict:
    """Rewrites a concept's research-driven standard lesson from the learner's feedback
    on it, keeping the superseded version in the concept's own "lesson_history". Optional
    and non-blocking -- unlike revise_plan, nothing in the pipeline gates on this; a
    learner can ask for a lesson revision, or not, at any point after it is built.

    `revise` is lara.serve.learn_research.lesson_revise, bound to this course and concept.
    Raises ValueError (recorded as this build's own error, then re-raised) if there is
    nothing persisted to revise from -- a lesson written by the legacy claims.py +
    depth.deepen() pipeline, which never sets content["research_driven"] and has no
    synthesis graph behind it at all."""
    concept = {**next(c for c in course["concepts"] if c["id"] == cid), "goal": course["goal"]}
    build = store.load_build(course["id"], cid)

    def note(**kw) -> None:
        build.update(kw)
        store.save_build(course["id"], cid, build)

    async with _editing.setdefault((course["id"], cid), asyncio.Lock()):
        content = store.load_concept(course["id"], cid)
        if not content or not content.get("lesson"):
            raise ValueError("build this concept's lesson first")
        note(stage="revising_lesson", error="")
        try:
            result = await RS.revise_lesson(concept, feedback, revise=revise)
        except Exception as e:                                   # noqa: BLE001
            note(stage="error", error=f"{type(e).__name__}: {e}")
            raise
        content = store.load_concept(course["id"], cid) or content
        content.setdefault("lesson_history", []).append({
            "lesson": content.get("lesson"), "claims": content.get("claims", []),
            "references": content.get("references", {}), "feedback": feedback,
            "superseded": time.time()})
        content.update(result)
        store.save_concept(course["id"], cid, content)
        note(stage="done", error="")
    return content


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

        TR.start(store.trace_path(course["id"], cid))
        TR.emit("build_start", variant=key, forced=False)
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
        finally:
            TR.stop()
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


def _concept_title(course: dict, cid: str) -> str:
    meta = next((c for c in course["concepts"] if c["id"] == cid), None)
    return meta["title"] if meta else ""


def _record_for_profile(course: dict, item: dict, correct: bool, confidence: int) -> None:
    """Feeds `lara.learn.profile`'s cross-course ledger -- quiz evidence only, per the
    user's own call: a pretest answer is itself a graded quiz item (the diagnostic just
    samples the same quiz pool up front), so it counts here the same an ordinary one does;
    it is only the *inferred* mastery bump pretest separately grants to a concept's own
    prerequisites (see LN.finish_pretest) that stays course-local and never reaches this."""
    PR.record_quiz_answer(_concept_title(course, item["concept"]), correct, confidence,
                          item["type"])


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
                pitem = find_item(course, iid)
                LN.record_answer(learner, pitem, ok, conf)
                _record_for_profile(course, pitem, ok, conf)
    else:
        LN.record_answer(learner, item, graded["correct"], confidence)
        _record_for_profile(course, item, graded["correct"], confidence)
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
        concepts.append({**{k: c.get(k) for k in ("id", "title", "summary", "prereqs", "competencies", "subject")},
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
            "subjects": course.get("subjects", []),
            "uncovered": course.get("uncovered", []), "next": action,
            "pretest": learner["pretest"]["state"], "due": sum(
                1 for s in learner["items"].values() if s.get("seen") and s.get("due", 0) <= time.time())}
