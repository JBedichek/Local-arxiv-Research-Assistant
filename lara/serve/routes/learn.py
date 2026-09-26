"""`/api/learn` -- courses: scope a goal, map it, build concepts on demand, learn and quiz.

Long work (mapping a course, building a concept, the diagnostic) runs as a background task
the page polls: the course's own `status` and each concept's `build` say where it is."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from lara.learn import learner as LN
from lara.learn import lesson as LE
from lara.learn import pipeline as PL
from lara.learn import scope as SC
from lara.learn import store
from lara.learn import trace as TR
from lara.learn.llm import Llm
from lara.learn.passages import CorpusRetriever, embed_fn
from lara.serve import generate as G
from lara.serve.deps import require_state

router = APIRouter()
logger = logging.getLogger(__name__)

_tasks: dict[str, asyncio.Task] = {}
_shared_llm: Llm | None = None


class GoalRequest(BaseModel):
    goal: str


class AnswerRequest(BaseModel):
    answer: str


class ItemAnswer(BaseModel):
    response: str = ""
    confidence: int = 2


class TextRequest(BaseModel):
    text: str


class FlagRequest(BaseModel):
    note: str = ""


class BuildRequest(BaseModel):
    force: bool = False


class ExpandRequest(BaseModel):
    selection: str
    question: str = ""
    section: int = 0
    claims: list[str] = []
    variant: str = "standard"


class LessonRequest(BaseModel):
    variant: str                # "tldr", "thorough" or "pages"
    pages: int | None = None


class FamiliarityRequest(BaseModel):
    topic_id: str
    answer: str                 # "yes", "no" or "partial"
    explain: str = ""


def _err(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


async def _llm() -> Llm | None:
    """One model wrapper shared by every course, so their calls share a concurrency limit. It
    uses lara's own generator: whatever `serving.vllm` is configured to reach."""
    global _shared_llm
    if _shared_llm is None:
        cfg = require_state().cfg
        vcfg = cfg.get_in("serving.vllm") or {}
        model = vcfg.get("default_model") or None
        window = 0
        try:
            window = await G.context_limit(vcfg.get("base_url", ""), model or "") or 0
        except Exception:                                      # noqa: BLE001
            pass
        _shared_llm = Llm(complete=G.complete, cfg=cfg, model=model, window=window)
    return _shared_llm


def _figure_lookup(state):
    """(arxiv_id, version, anchor) -> {"src", "caption"} | None, over whatever HTML the
    crawl already cached locally -- see lara.serve.papers.figure_image. A paper with no
    cached HTML (crawled for metadata only, never for fulltext) has nothing to look up in,
    which is the ordinary case for most of the corpus, not an error."""
    def lookup(arxiv_id: str, version: int, anchor: str) -> dict | None:
        path = state.raw_html_path(arxiv_id)
        if path is None:
            return None
        from lara.serve import papers as papers_mod

        return papers_mod.figure_image(str(path), arxiv_id, version or 1, anchor)
    return lookup


async def _corpus():
    state = require_state()
    return (CorpusRetriever(state, figure=_figure_lookup(state)),
            embed_fn(getattr(state.retriever, "embedder", None)))


def _course(course_id: str) -> dict | None:
    try:
        return store.load_course(course_id)
    except ValueError:
        return None


def _keep(key: str, task: asyncio.Task) -> None:
    _tasks[key] = task
    task.add_done_callback(lambda t: (_tasks.pop(key, None), t.exception() if not t.cancelled() else None))


@router.get("/api/learn/courses")
def courses() -> JSONResponse:
    return JSONResponse({"courses": store.list_courses()})


@router.post("/api/learn/courses")
async def start(req: GoalRequest) -> JSONResponse:
    goal = req.goal.strip()
    if not goal:
        return _err("say what you want to learn", 400)
    llm = await _llm()
    if llm is None:
        return _err("no generator replica was reachable", 503)
    course = await SC.begin(llm, goal)
    if course["status"] == "failed":
        return _err(course.get("error", "could not scope that goal"), 502)
    return JSONResponse(course, status_code=201)


@router.get("/api/learn/courses/{course_id}")
def show(course_id: str) -> JSONResponse:
    course = _course(course_id)
    if course is None:
        return _err(f"no course {course_id}", 404)
    learner = store.load_learner(course_id) or LN.blank()
    body = {"scope": {k: course.get(k) for k in ("qa", "pending", "competencies", "map_history")},
            "error": course.get("error", "")}
    if course["concepts"]:
        body.update(PL.overview(course, learner))
    else:
        body.update(id=course["id"], goal=course["goal"], status=course["status"], concepts=[],
                    competencies=[], next={"action": "scope"}, uncovered=[], pretest="todo", due=0)
    return JSONResponse(body)


@router.delete("/api/learn/courses/{course_id}")
def remove(course_id: str) -> JSONResponse:
    return JSONResponse({"deleted": store.delete_course(course_id)})


@router.post("/api/learn/courses/{course_id}/answer")
async def answer(course_id: str, req: AnswerRequest) -> JSONResponse:
    course = _course(course_id)
    if course is None:
        return _err(f"no course {course_id}", 404)
    if not course.get("pending"):
        return _err("no question is waiting for an answer", 409)
    llm = await _llm()
    if llm is None:
        return _err("no generator replica was reachable", 503)
    return JSONResponse(await SC.answer(llm, course, req.answer))


@router.post("/api/learn/courses/{course_id}/accept")
def accept(course_id: str) -> JSONResponse:
    course = _course(course_id)
    if course is None:
        return _err(f"no course {course_id}", 404)
    return JSONResponse(SC.accept(course))


@router.post("/api/learn/courses/{course_id}/map")
async def map_it(course_id: str) -> JSONResponse:
    course = _course(course_id)
    if course is None:
        return _err(f"no course {course_id}", 404)
    if course["status"] not in ("scoped", "failed", "ready"):
        return _err(f"the course is {course['status']}; finish scoping first", 409)
    if f"map:{course_id}" in _tasks:
        return JSONResponse({"status": "mapping"}, status_code=202)
    llm = await _llm()
    if llm is None:
        return _err("no generator replica was reachable", 503)
    corpus, _ = await _corpus()
    _keep(f"map:{course_id}", asyncio.ensure_future(PL.map_course(llm, corpus, course)))
    return JSONResponse({"status": "mapping"}, status_code=202)


@router.get("/api/learn/courses/{course_id}/concepts/{cid}")
def concept(course_id: str, cid: str) -> JSONResponse:
    course = _course(course_id)
    if course is None:
        return _err(f"no course {course_id}", 404)
    meta = next((c for c in course["concepts"] if c["id"] == cid), None)
    if meta is None:
        return _err(f"no concept {cid}", 404)
    content = store.load_concept(course_id, cid) or {}
    learner = store.load_learner(course_id) or LN.blank()
    quiz = content.get("quiz") or {}
    familiarity = LN.concept_state(learner, cid).get("topics", {})
    docs = content.get("topic_docs") or {}
    topics = [{**t, **familiarity.get(t["id"], {}),
              "doc_status": (docs.get(t["id"]) or {}).get("status", "todo")}
             for t in content.get("topics", [])]
    return JSONResponse({
        "id": cid, "title": meta["title"], "summary": meta["summary"], "prereqs": meta["prereqs"],
        "claims": content.get("claims", []), "conflicts": content.get("conflicts", []),
        "lesson": content.get("lesson"), "lessons": LE.lessons_of(content),
        "visuals": content.get("visuals", []),
        "expansions": content.get("expansions", []),
        "topics": topics,
        "stats": content.get("stats", {}), "trace": content.get("trace", {}),
        "reused": bool(content.get("reused")),
        "quiz": {"items": len(quiz.get("items", [])), "dropped": quiz.get("dropped", 0)},
        "build": store.load_build(course_id, cid), "sources": meta["sources"],
        "state": LN.concept_state(learner, cid)})


@router.get("/api/learn/courses/{course_id}/concepts/{cid}/trace")
def concept_trace(course_id: str, cid: str, since: int = 0) -> JSONResponse:
    """Every prompt, retrieval and tool call behind this concept's most recent build -- the
    Profile tab. `since` is the last `seq` the caller already has, so polling only sends what
    is new; the file itself is overwritten at the start of each build, so a poll spanning a
    rebuild sees a gap rather than a mix of two builds' events (the client restarts from 0
    when a returned `seq` is not strictly increasing from what it last saw)."""
    course = _course(course_id)
    if course is None:
        return _err(f"no course {course_id}", 404)
    if not any(c["id"] == cid for c in course["concepts"]):
        return _err(f"no concept {cid}", 404)
    rows = TR.read(store.trace_path(course_id, cid), since=since)
    return JSONResponse({"events": rows})


@router.post("/api/learn/courses/{course_id}/concepts/{cid}/build")
async def build(course_id: str, cid: str, req: BuildRequest | None = None) -> JSONResponse:
    course = _course(course_id)
    if course is None or not any(c["id"] == cid for c in course["concepts"]):
        return _err("no such course or concept", 404)
    llm = await _llm()
    if llm is None:
        return _err("no generator replica was reachable", 503)
    corpus, embed = await _corpus()
    PL.ensure_concept(llm, corpus, course, cid, embed=embed, force=bool(req and req.force))
    return JSONResponse({"status": "building"}, status_code=202)


@router.post("/api/learn/courses/{course_id}/concepts/{cid}/expand")
async def expand(course_id: str, cid: str, req: ExpandRequest) -> JSONResponse:
    course = _course(course_id)
    if course is None or not any(c["id"] == cid for c in course["concepts"]):
        return _err("no such course or concept", 404)
    selection = req.selection.strip()
    if not selection:
        return _err("highlight some of the lesson first", 400)
    llm = await _llm()
    if llm is None:
        return _err("no generator replica was reachable", 503)
    corpus, embed = await _corpus()
    try:
        result = await PL.expand_selection(llm, corpus, course, cid, selection=selection[:2000],
                                           question=req.question.strip()[:500],
                                           selection_claims=req.claims, section=req.section,
                                           embed=embed, variant=req.variant)
    except ValueError as e:
        return _err(str(e), 409)
    return JSONResponse(result)


@router.post("/api/learn/courses/{course_id}/concepts/{cid}/familiarity")
async def familiarity(course_id: str, cid: str, req: FamiliarityRequest) -> JSONResponse:
    """The learner's yes/no/partial answer on one of the lesson's indexed topics. A "no" or
    "partial" starts that topic's background document in the background; the page polls the
    concept (each topic carries its own `doc_status`) or the topic's own endpoint below."""
    course = _course(course_id)
    if course is None or not any(c["id"] == cid for c in course["concepts"]):
        return _err("no such course or concept", 404)
    if req.answer not in ("yes", "no", "partial"):
        return _err("answer must be yes, no or partial", 400)
    if req.answer == "partial" and not req.explain.strip():
        return _err("say what you already know for a partial answer", 400)
    content = store.load_concept(course_id, cid) or {}
    if not any(t["id"] == req.topic_id for t in content.get("topics", [])):
        return _err(f"no topic {req.topic_id}", 404)
    llm = await _llm() if req.answer != "yes" else None
    if llm is None and req.answer != "yes":
        return _err("no generator replica was reachable", 503)
    corpus, embed = await _corpus()
    learner = store.load_learner(course_id) or LN.blank()
    state = await PL.set_familiarity(llm, corpus, course, learner, cid, req.topic_id, req.answer,
                                     req.explain.strip(), embed=embed)
    return JSONResponse(state)


@router.get("/api/learn/courses/{course_id}/concepts/{cid}/topics/{topic_id}")
def topic_doc(course_id: str, cid: str, topic_id: str) -> JSONResponse:
    """A sub-lesson's own content -- what the standalone tab it opens in polls and renders."""
    course = _course(course_id)
    if course is None:
        return _err(f"no course {course_id}", 404)
    meta = next((c for c in course["concepts"] if c["id"] == cid), None)
    content = store.load_concept(course_id, cid) or {}
    topic = next((t for t in content.get("topics", []) if t["id"] == topic_id), None)
    if meta is None or topic is None:
        return _err(f"no topic {topic_id}", 404)
    entry = (content.get("topic_docs") or {}).get(topic_id) or {"status": "todo", "doc": None}
    return JSONResponse({"id": topic_id, "title": topic["title"], "note": topic.get("note", ""),
                         "concept_title": meta["title"], "status": entry.get("status", "todo"),
                         "doc": entry.get("doc")})


@router.post("/api/learn/courses/{course_id}/concepts/{cid}/lesson")
async def write_lesson(course_id: str, cid: str, req: LessonRequest) -> JSONResponse:
    """Starts writing another version of a built concept's lesson in the background; the page
    polls the concept, whose `build` says how far along it is."""
    course = _course(course_id)
    if course is None or not any(c["id"] == cid for c in course["concepts"]):
        return _err("no such course or concept", 404)
    try:
        key, pages = PL.variant_key(req.variant, req.pages)
    except ValueError as e:
        return _err(str(e), 400)
    content = store.load_concept(course_id, cid)
    if not content or not content.get("claims"):
        return _err("build this concept first", 409)
    llm = await _llm()
    if llm is None:
        return _err("no generator replica was reachable", 503)
    corpus, embed = await _corpus()
    PL.ensure_variant(llm, corpus, course, cid, req.variant, req.pages, embed=embed)
    return JSONResponse({"status": "writing", "variant": key, "pages": pages}, status_code=202)


@router.delete("/api/learn/courses/{course_id}/concepts/{cid}/expansions/{xid}")
def remove_expansion(course_id: str, cid: str, xid: str) -> JSONResponse:
    if _course(course_id) is None:
        return _err(f"no course {course_id}", 404)
    return JSONResponse({"deleted": PL.delete_expansion(course_id, cid, xid)})


@router.post("/api/learn/courses/{course_id}/concepts/{cid}/read")
def read(course_id: str, cid: str) -> JSONResponse:
    course = _course(course_id)
    if course is None or not any(c["id"] == cid for c in course["concepts"]):
        return _err("no such course or concept", 404)
    learner = store.load_learner(course_id) or LN.blank()
    LN.concept_state(learner, cid)["lesson_read"] = True
    if learner["pretest"]["state"] == "todo":
        learner["pretest"]["state"] = "done"      # starting the lessons is declining the diagnostic
    store.save_learner(course_id, learner)
    return JSONResponse(PL.overview(course, learner))


@router.get("/api/learn/courses/{course_id}/pretest")
def pretest(course_id: str) -> JSONResponse:
    course = _course(course_id)
    if course is None:
        return _err(f"no course {course_id}", 404)
    state = (store.load_learner(course_id) or LN.blank())["pretest"]
    items = [PL.public_item(i) for i in (PL.find_item(course, iid) for iid in state.get("items", [])) if i]
    return JSONResponse({"state": state["state"], "items": items,
                         "answered": list(state.get("answered", {}))})


@router.post("/api/learn/courses/{course_id}/pretest/start")
async def pretest_start(course_id: str) -> JSONResponse:
    course = _course(course_id)
    if course is None or course["status"] != "ready":
        return _err("the course is not ready", 409)
    key = f"pretest:{course_id}"
    if key in _tasks:
        return JSONResponse({"state": "building"}, status_code=202)
    llm = await _llm()
    if llm is None:
        return _err("no generator replica was reachable", 503)
    corpus, embed = await _corpus()
    learner = store.load_learner(course_id) or LN.blank()
    learner["pretest"] = {"state": "building", "concepts": [], "answered": {}}
    store.save_learner(course_id, learner)
    _keep(key, asyncio.ensure_future(PL.start_pretest(llm, corpus, course, learner, embed=embed)))
    return JSONResponse({"state": "building"}, status_code=202)


@router.post("/api/learn/courses/{course_id}/pretest/skip")
def pretest_skip(course_id: str) -> JSONResponse:
    if _course(course_id) is None:
        return _err(f"no course {course_id}", 404)
    learner = store.load_learner(course_id) or LN.blank()
    learner["pretest"]["state"] = "done"
    store.save_learner(course_id, learner)
    return JSONResponse({"state": "done"})


@router.post("/api/learn/courses/{course_id}/items/{item_id}/answer")
async def answer_item(course_id: str, item_id: str, req: ItemAnswer) -> JSONResponse:
    course = _course(course_id)
    if course is None:
        return _err(f"no course {course_id}", 404)
    item = PL.find_item(course, item_id)
    if item is None:
        return _err(f"no item {item_id}", 404)
    # A multiple-choice answer is graded by comparing letters; not loading a model for it
    # keeps the first answer after a restart instant.
    llm = None if item["type"] == "mcq" else await _llm()
    if llm is None and item["type"] != "mcq":
        return _err("no generator replica was reachable", 503)
    learner = store.load_learner(course_id) or LN.blank()
    try:
        graded = await PL.answer_item(llm, course, learner, item_id, req.response,
                                      max(1, min(3, req.confidence)))
    except KeyError:
        return _err(f"no item {item_id}", 404)
    return JSONResponse({"graded": graded, "overview": PL.overview(course, learner)})


@router.post("/api/learn/courses/{course_id}/concepts/{cid}/critique")
async def critique(course_id: str, cid: str, req: TextRequest) -> JSONResponse:
    course = _course(course_id)
    if course is None or not any(c["id"] == cid for c in course["concepts"]):
        return _err("no such course or concept", 404)
    if not req.text.strip():
        return _err("write something to be reviewed", 400)
    llm = await _llm()
    if llm is None:
        return _err("no generator replica was reachable", 503)
    learner = store.load_learner(course_id) or LN.blank()
    points = await PL.submit_critique(llm, course, learner, cid, req.text)
    return JSONResponse({"points": points})


@router.post("/api/learn/courses/{course_id}/concepts/{cid}/claims/{key}/flag")
async def flag(course_id: str, cid: str, key: str, req: FlagRequest) -> JSONResponse:
    course = _course(course_id)
    if course is None:
        return _err(f"no course {course_id}", 404)
    llm = await _llm()
    if llm is None:
        return _err("no generator replica was reachable", 503)
    learner = store.load_learner(course_id) or LN.blank()
    try:
        return JSONResponse(await PL.flag_claim(llm, course, learner, cid, key, req.note))
    except KeyError:
        return _err("no such concept or claim", 404)
