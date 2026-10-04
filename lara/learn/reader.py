"""The simulated reader: a model told it knows exactly what the learner's profile says, and
nothing else of the field, reads a written lesson and flags every sentence it cannot follow.
Flagged sentences are rewritten for that reader from the same claims and re-judged like any
lesson sentence; a rewrite the judge rejects leaves the original in place.

Optional (config `learn.simulated_reader`, off by default): it adds a model call per section,
and a model asked to un-know things tends to under-flag, so its flags are worth checking against
real "explain this" highlights before relying on it."""

from __future__ import annotations

import asyncio

from lara.learn import lesson as LE
from lara.learn import profile as PF
from lara.learn import research as RS
from lara.learn.llm import Llm

READER_SYSTEM = """You are a specific reader. You know general knowledge any educated adult has, \
plus exactly the items listed under YOU KNOW, at the level given -- nothing else about this \
field. You read a lesson section one numbered sentence at a time, and you also know whatever \
earlier sentences explained.

Flag every sentence you could not follow: a term you were never given, a step that assumes \
something you do not know, a leap you cannot reconstruct.

Reply with JSON only: [{"n": 3, "missing": "what you would need to know to follow it"}] -- [] when \
you could follow every sentence. Do not flag a sentence only because it is dense."""

REWRITE_SYSTEM = """A reader could not follow some sentences of a lesson. Rewrite each so that \
reader can follow it, using ONLY the numbered claims (background claims included). You may make \
one sentence into two. End every sentence -- before its full stop -- with the keys of the claims \
it rests on, in brackets exactly as they are written below.

Reply with JSON only: [{"n": 1, "text": "... [c1]. ... [c3]."}] -- reply "KEEP" as the text for \
a sentence the claims cannot make clearer."""


def _parse(text: str, known: set[str]) -> list[dict]:
    """A research-driven lesson cites chunk ids ([3352954]); a legacy one cites claim keys
    ([c1]) -- whichever this lesson's claims use."""
    if known and all(k.isdigit() for k in known):
        return [s for _, sents in RS._parse_sections(text, known) for s in sents]
    return [s for _, sents in LE.parse(text, known) for s in sents]


def _knows(the_plan: dict) -> str:
    rows = [f"- {i['title']}: level {i['have']} ({PF.LEVELS[i['have']]})"
            for i in the_plan["items"] if i["have"] >= 1]
    return "\n".join(rows) or "(nothing in this field)"


async def _section(llm: Llm, sec: dict, knows: str, by_key: dict[str, dict]) -> tuple[int, int]:
    numbered = "\n".join(f"{n}. {s['text']}" for n, s in enumerate(sec["sentences"], 1))
    data = await llm.ask_json(READER_SYSTEM, f"YOU KNOW:\n{knows}\n\nSECTION: {sec['heading']}\n{numbered}",
                              default=500, cap=1_500, stage="learn_reader")
    flags = {}
    for item in data if isinstance(data, list) else []:
        try:
            n = int(item["n"])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= n <= len(sec["sentences"]):
            flags[n] = str(item.get("missing", "")).strip()
    if not flags:
        return 0, 0
    cited = {k for n in flags for k in sec["sentences"][n - 1]["claims"]}
    # The flagged sentences' own claims, plus every background claim: the explanation the
    # reader is missing is most likely in one of those.
    pool = {k: c for k, c in by_key.items() if k in cited or c.get("role") == "background"}
    listing = "\n".join(LE.line(c) for c in pool.values())
    asks = "\n".join(f"{n}. {sec['sentences'][n - 1]['text']} [{', '.join(sec['sentences'][n - 1]['claims'])}]"
                     f" -- the reader is missing: {why or 'unclear'}" for n, why in flags.items())
    data = await llm.ask_json(REWRITE_SYSTEM, f"CLAIMS:\n{listing}\n\nSENTENCES:\n{asks}",
                              default=900, cap=3_000, stage="learn_reader_rewrite")
    rewrites = {}
    for item in data if isinstance(data, list) else []:
        try:
            n, text = int(item["n"]), str(item["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if n in flags and text.upper().rstrip(".") != "KEEP":
            rewrites[n] = text
    rewritten = 0
    for n in sorted(rewrites, reverse=True):       # replace from the end: indices stay valid
        parsed = _parse(rewrites[n], set(pool))
        if not parsed or not all(s["claims"] for s in parsed):
            continue
        out, stats = await LE.verify(llm, [("", parsed)], pool)
        if out and stats["dropped"] == 0:
            sec["sentences"][n - 1:n] = [{**s, "reader_rewrite": True} for s in out[0]["sentences"]]
            rewritten += 1
    return len(flags), rewritten


async def review(llm: Llm, lesson: dict, claims: list[dict], the_plan: dict) -> dict:
    """Reviews `lesson` in place; returns {"flagged", "rewritten"}."""
    if lesson.get("insufficient") or not lesson.get("sections"):
        return {"flagged": 0, "rewritten": 0}
    by_key = {c["key"]: c for c in claims if not c.get("withdrawn")}
    knows = _knows(the_plan)
    results = await asyncio.gather(*(_section(llm, sec, knows, by_key) for sec in lesson["sections"]))
    return {"flagged": sum(f for f, _ in results), "rewritten": sum(r for _, r in results)}
