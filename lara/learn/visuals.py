"""Visuals drawn from data, never from imagination: a chart keeps only points whose number
appears in the claim it cites, and a diagram keeps only edges the judge finds in a claim. What
fails is dropped, and a visual with too little left is not shown."""

from __future__ import annotations

import asyncio
import re

from lara.learn import judge as J
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


async def build(llm: Llm, concept: dict, claims: list[dict]) -> list[dict]:
    found = await asyncio.gather(chart(llm, concept, claims), diagram(llm, concept, claims))
    return [v for v in found if v]
