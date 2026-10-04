"""The prerequisite graph of concepts. Its skeleton comes from survey and overview passages,
not from the model's idea of the field, and every concept must cite the passages that
justify it -- a concept nothing in the corpus discusses is dropped."""

from __future__ import annotations

import asyncio
import re

from lara.learn import trace as TR
from lara.learn.llm import Llm
from lara.learn.passages import Passage

SKELETON_PASSAGES = 36
MAX_PER_PAPER = 3
MIN_PASSAGE_CHARS = 200

GRAPH_SYSTEM = """You design the concept map for a self-study course from numbered passages \
of survey and overview papers.

Reply with JSON only:
{"concepts": [{"id": "k1", "title": "...", "summary": "...", "prereqs": ["k2"], \
"competencies": ["<competency id>"], "passages": [1, 4]}]}

- 10 to 24 concepts, each a teachable unit: not a whole field, not a single fact.
- Only concepts the passages actually discuss. "passages" lists the numbers that justify it.
- prereqs: concepts the learner must understand first. No cycles.
- competencies: which of the listed competency ids this concept serves."""


async def skeleton(corpus, goal: str, competencies: list[dict]) -> list[Passage]:
    queries = [goal, f"survey of {goal}", f"introduction to {goal}"] + [c["text"] for c in competencies]
    seen, per_paper, found = set(), {}, []
    for q in queries:
        for p in await asyncio.to_thread(corpus.search, q, 8):
            if p.key in seen or len(p.text) < MIN_PASSAGE_CHARS:
                continue
            seen.add(p.key)
            found.append(p)
    found.sort(key=lambda p: not p.is_overview)          # stable: overview papers first
    out = []
    for p in found:
        if per_paper.get(p.arxiv_id, 0) < MAX_PER_PAPER:
            per_paper[p.arxiv_id] = per_paper.get(p.arxiv_id, 0) + 1
            out.append(p)
    return out[:SKELETON_PASSAGES]


def break_cycles(prereqs: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Removes, in place, the prerequisite edges that close a cycle. Returns (concept,
    prereq) for each removed edge."""
    removed, state = [], {}

    def visit(node: str) -> None:
        state[node] = 1
        for dep in list(prereqs.get(node, [])):
            if state.get(dep) == 1:
                prereqs[node].remove(dep)
                removed.append((node, dep))
            elif dep not in state:
                visit(dep)
        state[node] = 2

    for node in list(prereqs):
        if node not in state:
            visit(node)
    return removed


def order(prereqs: dict[str, list[str]], sequence: list[str]) -> list[str]:
    """Topological order that keeps the model's own sequence where prerequisites allow."""
    done, out = set(), []
    remaining = list(sequence)
    while remaining:
        ready = [c for c in remaining if all(d in done for d in prereqs.get(c, []))]
        pick = ready[0] if ready else remaining[0]
        out.append(pick)
        done.add(pick)
        remaining.remove(pick)
    return out


async def build(llm: Llm, corpus, course: dict) -> dict:
    """{"concepts", "removed_edges", "uncovered", "dropped"} for a scoped course."""
    comps = course["competencies"]
    passages = await skeleton(corpus, course["goal"], comps)
    body = "\n\n".join(f"[{i}] ({p.title}) {p.text}" for i, p in enumerate(passages, 1))
    listing = "\n".join(f"- {c['id']}: {c['text']}" for c in comps)
    data = await llm.ask_json(GRAPH_SYSTEM,
                              f"GOAL: {course['goal']}\n\nCOMPETENCIES:\n{listing}\n\nPASSAGES:\n{body}",
                              default=3_000, cap=10_000, stage="learn_graph")
    raw = (data.get("concepts") if isinstance(data, dict) else data) or []
    valid = {c["id"] for c in comps}
    ids, concepts, dropped = {}, [], 0
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict) or not str(item.get("title", "")).strip():
            continue
        sources = [passages[n - 1] for n in item.get("passages") or []
                   if isinstance(n, int) and 1 <= n <= len(passages)]
        if not sources:
            dropped += 1
            continue
        cid = f"c{len(concepts) + 1}"
        ids[str(item.get("id"))] = cid
        concepts.append({"id": cid, "title": str(item["title"]).strip(),
                         "summary": str(item.get("summary", "")).strip(), "_prereqs": item.get("prereqs") or [],
                         "competencies": [x for x in item.get("competencies") or [] if x in valid],
                         "sources": [{"chunk_id": p.chunk_id, "arxiv_id": p.arxiv_id, "title": p.title}
                                     for p in sources]})
    prereqs = {c["id"]: [ids[p] for p in c.pop("_prereqs") if p in ids and ids[p] != c["id"]]
               for c in concepts}
    removed = break_cycles(prereqs)
    by_id = {c["id"]: c for c in concepts}
    for cid in order(prereqs, [c["id"] for c in concepts]):
        by_id[cid]["prereqs"] = prereqs[cid]
    ordered = [by_id[cid] for cid in order(prereqs, [c["id"] for c in concepts])]
    covered = {x for c in ordered for x in c["competencies"]}
    return {"concepts": ordered, "removed_edges": [list(e) for e in removed], "dropped": dropped,
            "uncovered": [c["text"] for c in comps if c["id"] not in covered]}


#: A citation bracket, matching lara.serve.citations.CITATION -- duplicated, not imported:
#: lara.learn never imports lara.serve (see passages.CorpusRetriever's own docstring), and
#: this is a two-line regex, not real coupling to that module's Reference/bind machinery.
_CITE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_CITE_KEYS = re.compile(r"\d+")


def cited_keys(text: str) -> list[str]:
    """Every citation key in `text`, in order of first appearance, deduplicated."""
    keys: list[str] = []
    for m in _CITE.finditer(text or ""):
        keys.extend(_CITE_KEYS.findall(m.group(0)))
    return list(dict.fromkeys(keys))


def _concepts_from_result(course: dict, result: dict) -> dict:
    """{"concepts", "subjects", "removed_edges", "uncovered", "dropped"} from a `synth`/
    `revise` result -- the shared tail of `build_from_research` (a fresh run) and
    `revise_from_research` (a resumed one): both hand this the same shape (see
    lara.serve.learn_research's `_parse_topic_graph`), so there is exactly one place that
    turns a synthesis result into a concept map.

    `result["subjects"]` groups this run's own candidate concepts into a hierarchy (see
    synthesizer._finalize_topic_graph's `organize_curriculum` pass) -- flattened here into
    the same course["concepts"] list every downstream stage (build_concept, learner
    tracking, quiz caps, prereqs) already expects, so nothing past this function needs to
    know a subject exists. `course["subjects"]` is that grouping, kept alongside it purely
    for the UI: each entry names the concept ids (assigned here) that belong to it, in the
    teaching order the curriculum-organizing pass itself chose."""
    comps = course["competencies"]
    refs = result.get("references") or {}
    valid = {c["id"] for c in comps}
    concepts: list[dict] = []
    subjects: list[dict] = []
    dropped = 0
    for s_idx, s in enumerate(result.get("subjects") or [], 1):
        if not isinstance(s, dict):
            continue
        sid = f"s{s_idx}"
        concept_ids: list[str] = []
        for item in s.get("concepts") or []:
            if not isinstance(item, dict) or not str(item.get("title", "")).strip():
                continue
            summary = str(item.get("summary", "")).strip()
            sources = [{"chunk_id": refs[k].get("chunk_id"), "arxiv_id": refs[k].get("arxiv_id"),
                       "title": refs[k].get("paper_title") or refs[k].get("title", "")}
                      for k in cited_keys(summary) if k in refs]
            if not sources:
                dropped += 1
                continue
            cid = f"c{len(concepts) + 1}"
            concept_ids.append(cid)
            concepts.append({
                "id": cid, "subject": sid, "title": str(item["title"]).strip(),
                "summary": summary, "_prereqs_text": str(item.get("prereqs_text", "")).strip(),
                "competencies": [x for x in item.get("competencies") or [] if x in valid],
                "sources": sources})
        if concept_ids:
            subjects.append({"id": sid, "title": str(s.get("title") or "").strip(),
                             "summary": str(s.get("summary") or "").strip(),
                             "concept_ids": concept_ids})

    prereqs = {}
    for c in concepts:
        text = c.pop("_prereqs_text").lower()
        prereqs[c["id"]] = [o["id"] for o in concepts
                            if o["id"] != c["id"] and o["title"].lower() in text]
    removed = break_cycles(prereqs)
    by_id = {c["id"]: c for c in concepts}
    for cid in order(prereqs, [c["id"] for c in concepts]):
        by_id[cid]["prereqs"] = prereqs[cid]
    ordered = [by_id[cid] for cid in order(prereqs, [c["id"] for c in concepts])]
    covered = {x for c in ordered for x in c["competencies"]}
    return {"concepts": ordered, "subjects": subjects,
           "removed_edges": [list(e) for e in removed], "dropped": dropped,
           "uncovered": [c["text"] for c in comps if c["id"] not in covered]}


def _course_objective(course: dict) -> str:
    comps = course["competencies"]
    listing = "\n".join(f"- {c['id']}: {c['text']}" for c in comps)
    return ("Map the concepts a course on the following goal should teach, in the "
           "order a learner should meet them, from the paper corpus.\n\n"
           f"GOAL: {course['goal']}"
           + (f"\n\nCOMPETENCIES the course must cover:\n{listing}" if listing else ""))


async def build_from_research(course: dict, *, synth) -> dict:
    """{"concepts", "removed_edges", "uncovered", "dropped"} for a scoped course -- same
    shape `build` above returns, so `pipeline.map_course` does not change. Unlike `build`,
    which makes one blind call against a handful of survey passages, this asks one full
    course-level synthesis run (`synth`, injected -- see lara.serve.learn_research) to
    research the course's goal first and write the concept map directly from what it
    found: every concept still must cite the research that justifies it (`cited_keys`
    below), the same "nothing a learner is asked to trust without a source" guarantee
    `build`'s own passage-citation check keeps.

    `synth` researches the whole course in clusters (`synthesizer._cluster_goals`), each
    written up independently without seeing the others' concepts -- so unlike `build`,
    which gets clean prereq ids straight from the model, a concept only has its
    prerequisites in plain words (`prereqs_text`). Resolved here by matching another
    concept's title inside that text; a real but inexact substitute for an id the writer
    genuinely did not have available to it."""
    TR.set_phase("course_research")
    result = await synth(_course_objective(course))
    TR.emit("topic_graph_research", concepts_proposed=len(result.get("concepts") or []),
           degraded=result.get("degraded"), rounds=result.get("rounds"))
    return _concepts_from_result(course, result)


async def revise_from_research(course: dict, feedback: str, *, revise) -> dict:
    """Phase 1 revision: re-maps the course from a learner's feedback on the plan
    currently awaiting their approval, by resuming the persisted course-mapping
    synthesis graph (`revise`, injected -- see lara.serve.learn_research.
    topic_graph_revise) instead of researching from scratch. Same return shape
    `build_from_research` gives, so `pipeline.revise_plan` applies it the same way."""
    TR.set_phase("course_research_revision")
    result = await revise(feedback)
    TR.emit("topic_graph_revision", concepts_proposed=len(result.get("concepts") or []),
           degraded=result.get("degraded"), rounds=result.get("rounds"))
    return _concepts_from_result(course, result)
