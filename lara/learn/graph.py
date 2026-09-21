"""The prerequisite graph of concepts. Its skeleton comes from survey and overview passages,
not from the model's idea of the field, and every concept must cite the passages that
justify it -- a concept nothing in the corpus discusses is dropped."""

from __future__ import annotations

import asyncio

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
