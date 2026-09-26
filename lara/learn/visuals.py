"""Visuals drawn from data, never from imagination: a chart keeps only points whose number
appears in the claim it cites, a diagram keeps only edges the judge finds in a claim, and
pseudocode keeps only steps the judge finds in a claim. What fails is dropped, and a visual
with too little left is not shown.

Generated per lesson section rather than once for the whole concept -- a concept with six
sections and forty claims produced one diagram for all of them, asked to stand in for
material it could not summarise. Each section's own claim subset gets its own attempt at
each kind, so a section with nothing to show contributes nothing rather than diluting one
visual for the whole lesson. `v["claims"]` still says which claims a visual rests on, which
is what the client uses to place it under whichever section it actually belongs to -- this
module has no notion of "section" beyond partitioning claims by one."""

from __future__ import annotations

import asyncio
import re

from lara.learn import judge as J
from lara.learn import trace as TR
from lara.learn.llm import Llm

MIN_POINTS = 2
MAX_NODES = 8

CHART_SYSTEM = """From the numbered claims, pick ONE quantity that at least two claims report \
in a comparable way (the same metric across methods, settings or scales).

Reply with JSON only: {"title": "...", "kind": "bar"|"line", "x_label": "...", "y_label": "...", \
"points": [{"label": "...", "value": 1.5, "claim": "c1"}]}
or the word null if no such quantity exists. Every value must be a number stated in its claim."""

DIAGRAM_SYSTEM = """Show how the parts of this concept relate (components, steps, or cause and \
effect) as a small diagram, using only relations the numbered claims assert.

Reply with JSON only: {"title": "...", "nodes": [{"id": "a", "label": "..."}], \
"edges": [{"from": "a", "to": "b", "label": "...", "claim": "c1"}]}
or the word null. At most 8 nodes; each edge cites the claim that asserts it."""

PSEUDOCODE_SYSTEM = """If the numbered claims describe a procedure -- an algorithm, a training \
loop, a pipeline with an order to it -- write its steps as pseudocode. If nothing here is a \
procedure (a concept that is only a property, a comparison or a finding is not one), reply null.

Reply with JSON only: {"title": "...", "steps": [{"text": "...", "depth": 0, "claim": "c1"}]}
or the word null. Each step is one line of pseudocode-style prose (an action, a loop, a \
condition -- "for each candidate: score it with the reward model", not a sentence explaining \
it). `depth` is the nesting level (0 = top level, 1 = inside the nearest depth-0 step above \
it, and so on) so a loop or a branch can contain steps under it. Every step cites the one \
claim that states it; never invent a step the claims do not support."""

#: A pseudocode block with only one step is a sentence with delusions of structure.
MIN_STEPS = 2

#: Sections/concepts beyond this many visuals are not worth the calls -- MAX_NODES-sized
#: diagrams and short charts/pseudocode blocks stay well under it in practice; this is a
#: backstop against a pathological lesson with many sections, not a target.
MAX_VISUALS = 10

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numbers_in(text: str) -> set[float]:
    out = set()
    for tok in _NUM.findall(text or ""):
        try:
            out.add(float(tok.replace(",", "")))
        except ValueError:
            continue
    return out


def _listing(claims: list[dict]) -> str:
    return "\n".join(f"[{c['key']}] {c['text']}" for c in claims)


def _live(claims: list[dict]) -> list[dict]:
    return [c for c in claims if not c.get("withdrawn") and c["certainty"] != "superseded"]


async def chart(llm: Llm, concept: dict, claims: list[dict]) -> dict | None:
    TR.set_phase("visuals: chart")            # its own task, gathered alongside diagram/pseudocode
    live = _live(claims)
    if len(live) < 2:
        return None
    data = await llm.ask_json(CHART_SYSTEM, f"CONCEPT: {concept['title']}\n\nCLAIMS:\n{_listing(live)}",
                              default=600, cap=2_000, stage="learn_chart")
    if not isinstance(data, dict) or not isinstance(data.get("points"), list):
        return None
    by_key, points = {c["key"]: c for c in live}, []
    for p in data["points"]:
        try:
            value, key, label = float(p["value"]), p["claim"], str(p["label"]).strip()
        except (KeyError, ValueError, TypeError):
            continue
        if key in by_key and label and value in numbers_in(by_key[key]["text"]):
            points.append({"label": label, "value": value, "claim": key})
    if len(points) < MIN_POINTS:
        return None
    return {"kind": "chart", "chart": "line" if data.get("kind") == "line" else "bar",
            "title": str(data.get("title", concept["title"])), "x_label": str(data.get("x_label", "")),
            "y_label": str(data.get("y_label", "")), "points": points,
            "claims": sorted({p["claim"] for p in points})}


async def diagram(llm: Llm, concept: dict, claims: list[dict]) -> dict | None:
    TR.set_phase("visuals: diagram")          # its own task, gathered alongside chart/pseudocode
    live = _live(claims)
    if not live:
        return None
    data = await llm.ask_json(DIAGRAM_SYSTEM, f"CONCEPT: {concept['title']}\n\nCLAIMS:\n{_listing(live)}",
                              default=700, cap=2_500, stage="learn_diagram")
    if not isinstance(data, dict):
        return None
    labels = {str(n.get("id")): str(n.get("label", "")).strip() for n in data.get("nodes") or []
              if isinstance(n, dict) and str(n.get("label", "")).strip()}
    by_key = {c["key"]: c for c in live}
    edges = [e for e in data.get("edges") or [] if isinstance(e, dict)
             and str(e.get("from")) in labels and str(e.get("to")) in labels and e.get("claim") in by_key]
    verdicts = await asyncio.gather(*(
        J.judge(llm, f"{labels[str(e['from'])]} {e.get('label', 'relates to')} {labels[str(e['to'])]}",
                by_key[e["claim"]]["text"]) for e in edges))
    kept = [e for e, v in zip(edges, verdicts) if v == J.SUPPORTS]
    used = {str(x) for e in kept for x in (e["from"], e["to"])}
    nodes = [{"id": i, "label": labels[i]} for i in labels if i in used][:MAX_NODES]
    kept = [e for e in kept if str(e["from"]) in {n["id"] for n in nodes} and str(e["to"]) in {n["id"] for n in nodes}]
    if len(nodes) < 2 or not kept:
        return None
    return {"kind": "diagram", "title": str(data.get("title", concept["title"])), "nodes": nodes,
            "edges": [{"from": str(e["from"]), "to": str(e["to"]), "label": str(e.get("label", "")),
                       "claim": e["claim"]} for e in kept],
            "claims": sorted({e["claim"] for e in kept})}


async def pseudocode(llm: Llm, concept: dict, claims: list[dict]) -> dict | None:
    TR.set_phase("visuals: pseudocode")       # its own task, gathered alongside chart/diagram
    live = _live(claims)
    if not live:
        return None
    data = await llm.ask_json(PSEUDOCODE_SYSTEM, f"CONCEPT: {concept['title']}\n\nCLAIMS:\n{_listing(live)}",
                              default=700, cap=2_500, stage="learn_pseudocode")
    if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
        return None
    by_key = {c["key"]: c for c in live}
    raw = [s for s in data["steps"] if isinstance(s, dict) and str(s.get("text", "")).strip()
           and s.get("claim") in by_key]
    verdicts = await asyncio.gather(*(
        J.judge(llm, str(s["text"]), by_key[s["claim"]]["text"]) for s in raw))
    steps = [{"text": str(s["text"]).strip(), "depth": max(0, int(s.get("depth") or 0)),
              "claim": s["claim"]}
             for s, v in zip(raw, verdicts) if v == J.SUPPORTS]
    if len(steps) < MIN_STEPS:
        return None
    return {"kind": "pseudocode", "title": str(data.get("title", concept["title"])),
            "steps": steps, "claims": sorted({s["claim"] for s in steps})}


def _section_claims(claims: list[dict], section: dict) -> list[dict]:
    """The claims a lesson section actually cites, in the order the concept has them."""
    cited = {k for s in section.get("sentences", []) for k in s.get("claims", [])}
    return [c for c in claims if c["key"] in cited]


async def figures_in(claims: list[dict], corpus) -> list[dict]:
    """One visual per distinct figure a claim's own passage already rests on. No model call
    and no judge, unlike the other three kinds: a claim was already built and judged against
    its passage in claims.py, so its source image is exactly as grounded as the claim
    itself -- this only has to find the image, not check it. `corpus` is a CorpusRetriever
    (or anything with its `.figure` method); None (the diagnostic-only path, or a caller with
    no corpus at all) yields nothing, same as `corpus.figure` itself already does when no
    lookup was injected into it."""
    if corpus is None:
        return []
    candidates = [c for c in _live(claims) if (p := c.get("passage") or {}).get("kind") == "caption"
                 and p.get("anchor") and p.get("arxiv_id")]
    found = await asyncio.gather(*(
        asyncio.to_thread(corpus.figure, c["passage"]["arxiv_id"], c["passage"].get("version") or 1,
                          c["passage"]["anchor"])
        for c in candidates))
    seen_src: set[str] = set()
    out = []
    for c, img in zip(candidates, found):
        if not img or not img.get("src") or img["src"] in seen_src:
            continue
        seen_src.add(img["src"])
        out.append({"kind": "figure", "title": c["text"][:120], "src": img["src"],
                    "caption": img.get("caption") or c["passage"].get("text", ""),
                    "arxiv_id": c["passage"]["arxiv_id"], "claims": [c["key"]]})
    return out


async def _visuals_for(llm: Llm, concept: dict, claims: list[dict], corpus=None) -> list[dict]:
    found, figs = await asyncio.gather(
        asyncio.gather(chart(llm, concept, claims), diagram(llm, concept, claims),
                       pseudocode(llm, concept, claims)),
        figures_in(claims, corpus))
    return [v for v in found if v] + figs


async def build(llm: Llm, concept: dict, claims: list[dict], lesson: dict | None = None,
                corpus=None) -> list[dict]:
    """One attempt at each kind per lesson section, so a visual is grounded in -- and
    findable from -- the claims one section actually cites, not the concept's claims as an
    undifferentiated pile. Falls back to one whole-concept attempt when there is no lesson
    yet to section by (a diagnostic-only build, or one that failed to write a lesson at all).

    `corpus`, when given, backs the figure kind (see `figures_in`) -- optional because most
    callers in this package's own tests have no HTML cache to resolve one against.
    """
    lesson = lesson or {}
    sections = [] if lesson.get("insufficient") else lesson.get("sections", [])
    if not sections:
        return (await _visuals_for(llm, concept, claims, corpus))[:MAX_VISUALS]
    groups = [g for s in sections if len(g := _section_claims(claims, s)) >= 2]
    per_section = await asyncio.gather(*(_visuals_for(llm, concept, g, corpus) for g in groups))
    return [v for found in per_section for v in found][:MAX_VISUALS]
