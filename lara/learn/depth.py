"""Longer lessons by researching deeper, not by writing more.

A lesson is limited by what the corpus can support, so a request for N pages plans an outline
of about N sections, runs a focused corpus search for each, and writes each section only from
the claims found for it -- every sentence cited and re-judged, as in any lesson. Each claim is
used in exactly one section, so sections do not repeat each other. A section the corpus cannot
support is dropped, and the lesson says how much of the requested length it could stand behind
rather than padding to it."""

from __future__ import annotations

import asyncio
import time

from lara.learn import claims as CL
from lara.learn import lesson as LE
from lara.learn.llm import Llm, parse_json

WORDS_PER_PAGE = 450
#: What one claim can carry once explained. A section asked for more than this is being asked to
#: pad, and padding is where unsupported sentences come from.
WORDS_PER_CLAIM = 55
#: A section may run this far past its share of the requested length before trailing sentences
#: are cut: "about N pages" is approximate, but 3 pages should not come back as 4.5.
OVERRUN = 1.15
#: A malformed outline reply is retried once before giving up on the lesson.
OUTLINE_ATTEMPTS = 2
SEARCH_BREADTH = 12
MIN_PAGES, MAX_PAGES = 1, 20
THOROUGH_PAGES = 5
MIN_SECTIONS, MAX_SECTIONS = 3, 16
MIN_SECTION_CLAIMS = 2
MAX_SECTION_CLAIMS = 24
#: A page is enough of a shortfall to mention only when the lesson lands below this share.
SHORTFALL_BELOW = 0.75

OUTLINE_SYSTEM = """You plan a self-study lesson from the learner's goal and what is already known \
about one concept.

Reply with JSON only: {"sections": [{"heading": "...", "focus": "..."}]}

- Exactly the number of sections asked for, in the order a learner should read them, from what \
the concept is through how it works, its variants and edge cases, to how to use it for their goal.
- focus: one sentence saying what that section must establish -- specific enough to search a \
paper corpus for.
- Cover the concept end to end; do not repeat a topic across sections."""

SECTION_SYSTEM = """You write ONE section of a lesson, using ONLY the numbered claims provided.

Format: one sentence per line, each ending -- before its full stop -- with the keys of the claims \
it rests on in brackets, like [c1] or [c2, c5]. A sentence with no key is not allowed. Do not \
write a heading.

- Do not add any fact that is not in the claims. Connecting or explaining claims is fine only if \
every part is supported by a claim you cite.
- State each idea once, even when several claims support it -- cite them together.
- Convey certainty as marked: one paper, a hypothesis, replaced by later work.
- Aim for about the length asked for, but never go beyond what the claims support: fewer \
sentences is right when the claims are few."""


def clamp_pages(pages) -> int:
    try:
        return max(MIN_PAGES, min(MAX_PAGES, int(pages)))
    except (TypeError, ValueError):
        return THOROUGH_PAGES


def section_count(pages: int) -> int:
    return max(MIN_SECTIONS, min(MAX_SECTIONS, round(pages * 1.2)))


def count_words(sections: list[dict]) -> int:
    return sum(len(s["text"].split()) for sec in sections for s in sec["sentences"])


async def plan_outline(llm: Llm, concept: dict, claims: list[dict], pages: int) -> tuple[list[dict], str]:
    """(sections, the last raw reply) -- the reply is kept so a failure can say what came back."""
    n = section_count(pages)
    known = "\n".join(f"- {c['text']}" for c in claims[:25]) or "(nothing yet)"
    prompt = (f"LEARNER'S GOAL: {concept.get('goal') or '(not given)'}\nCONCEPT: {concept['title']} -- "
              f"{concept.get('summary', '')}\nSECTIONS: {n} (about {pages} page(s) in all)\n\nALREADY KNOWN:\n{known}")
    reply = ""
    for _ in range(OUTLINE_ATTEMPTS):
        reply = await llm.ask(OUTLINE_SYSTEM, prompt, default=1_800, cap=5_000, stage="learn_outline")
        data = parse_json(reply)
        raw = data.get("sections") if isinstance(data, dict) else None
        out = [{"heading": str(s["heading"]).strip(), "focus": str(s.get("focus", "")).strip()}
               for s in raw or [] if isinstance(s, dict) and str(s.get("heading", "")).strip()]
        if out:
            return out[:MAX_SECTIONS], reply
    return [], reply


def _key_number(key: str) -> int:
    digits = key[1:]
    return int(digits) if key.startswith("c") and digits.isdigit() else 0


async def research(llm: Llm, corpus, concept: dict, existing: list[dict], outline: list[dict],
                   *, embed=None) -> list[CL.Claim]:
    """New claims for every section, searched in parallel and merged once: repeats of what
    the concept already had (or of each other) are dropped, and the rest are numbered on from
    the concept's own keys."""
    used = frozenset(c["passage"]["chunk_id"] for c in existing)

    async def one(sec: dict) -> list[CL.Claim]:
        focus = f"{sec['heading']}. {sec['focus']}".strip()
        passages = await CL.gather_passages(corpus, concept, focus=focus, exclude=used,
                                            per_query=SEARCH_BREADTH)
        found, *_ = await CL.extract(llm, concept, passages, embed=embed, focus=focus)
        return found

    per_section = await asyncio.gather(*(one(s) for s in outline))
    seen = [c["text"] for c in existing]
    fresh: list[CL.Claim] = []
    next_key = max((_key_number(c["key"]) for c in existing), default=0) + 1
    for found in per_section:
        for c in found:
            if all(CL.overlap(c.text, t) < CL.DUPLICATE_OVERLAP for t in seen):
                c.key = f"c{next_key}"
                next_key += 1
                seen.append(c.text)
                fresh.append(c)
    return fresh


def assign(claims: list[dict], outline: list[dict], embed=None) -> list[list[dict]]:
    """Each claim goes to the one section it fits best, so no idea is written twice."""
    vecs = [embed(f"{s['heading']}. {s['focus']}") if embed else [] for s in outline]
    buckets: list[list[dict]] = [[] for _ in outline]
    for c in claims:
        cv = embed(c["text"]) if embed else []
        best, best_score = 0, -1.0
        for i, s in enumerate(outline):
            score = CL.overlap(f"{s['heading']} {s['focus']}", c["text"])
            if cv and vecs[i]:
                score = max(score, CL.cosine(cv, vecs[i]))
            if score > best_score:
                best, best_score = i, score
        buckets[best].append(c)
    return buckets


async def write_section(llm: Llm, concept: dict, sec: dict, claims: list[dict],
                        words: int) -> tuple[list[dict], dict]:
    """(verified sentences, stats) for one section."""
    by_key = {c["key"]: c for c in claims}
    words = min(words, WORDS_PER_CLAIM * len(claims))
    prompt = (f"CONCEPT: {concept['title']}\nSECTION: {sec['heading']} -- {sec['focus']}\n"
              f"LENGTH: at most about {words} words; fewer is right if the claims say less\n\nCLAIMS:\n"
              + "\n".join(LE.line(c) for c in claims))
    text = await llm.ask(SECTION_SYSTEM, prompt, default=1_200, cap=4_000, stage="learn_section")
    parsed = LE.parse(text, set(by_key))
    flat = [(sec["heading"], [s for _, sents in parsed for s in sents])]
    out, stats = await LE.verify(llm, flat, by_key)
    return trim(out[0]["sentences"] if out else [], words * OVERRUN), stats


def trim(sentences: list[dict], budget: float) -> list[dict]:
    """The leading sentences that fit in `budget` words -- always at least one."""
    kept, used = [], 0
    for s in sentences:
        n = len(s["text"].split())
        if kept and used + n > budget:
            break
        kept.append(s)
        used += n
    return kept


def _merge_back(existing: list[dict], updated: list[CL.Claim]) -> list[dict]:
    """Stored claim dicts carry extras the dataclass does not (`withdrawn`), so what relating
    changed is folded into the originals rather than replacing them."""
    by_key = {c.key: c for c in updated}
    merged = [{**c, **by_key[c["key"]].to_dict()} if c["key"] in by_key else c for c in existing]
    return merged + [c.to_dict() for c in updated if c.key not in {m["key"] for m in existing}]


async def deepen(llm: Llm, corpus, concept: dict, content: dict, pages: int, *, embed=None,
                 progress=None) -> tuple[dict, dict]:
    """(lesson, changes): the lesson, and what it added to the concept -- `claims` (the full
    merged list), `conflicts`, and `new_claims` (just the additions)."""
    note = progress or (lambda detail: None)
    existing = [c for c in content.get("claims", []) if not c.get("withdrawn")]
    outline, reply = await plan_outline(llm, concept, LE.usable(existing), pages)
    if not outline:
        return _empty(f"the outline could not be planned (the model replied: {reply[:160]!r})", pages), {}
    note(f"planned {len(outline)} sections; searching the papers for each")
    fresh = await research(llm, corpus, concept, existing, outline, embed=embed)
    everything = [CL.Claim.from_dict(c) for c in content.get("claims", [])] + fresh
    await CL.relate(llm, everything, embed=embed, involving={c.key for c in fresh})
    merged = _merge_back(content.get("claims", []), everything)
    usable = LE.usable(merged)
    note(f"found {len(fresh)} new claims; writing")

    words = max(60, round(pages * WORDS_PER_PAGE / len(outline)))
    buckets = [b[:MAX_SECTION_CLAIMS] for b in assign(usable, outline, embed)]
    live = [(s, b) for s, b in zip(outline, buckets) if len(b) >= MIN_SECTION_CLAIMS]
    written = await asyncio.gather(*(write_section(llm, concept, s, b, words) for s, b in live))
    sections = [{"heading": s["heading"], "sentences": sents}
                for (s, _), (sents, _) in zip(live, written) if sents]
    stats = {"written": 0, "kept_first_pass": 0, "repaired": 0, "dropped": 0}
    for _, st in written:
        for k in stats:
            stats[k] += st[k]
    stats["grounded_pct"] = round(100 * stats["kept_first_pass"] / stats["written"]) if stats["written"] else 0
    conflicts = CL.conflicts([CL.Claim.from_dict(c) for c in merged if not c.get("withdrawn")])
    achieved = round(count_words(sections) / WORDS_PER_PAGE, 1)
    lesson = {"insufficient": not sections, "sections": sections, "generated": time.time(),
              "target_pages": pages, "achieved_pages": achieved, "stats": stats,
              "outline": [s["heading"] for s in outline],
              "dropped_sections": [s["heading"] for s, b in zip(outline, buckets) if len(b) < MIN_SECTION_CLAIMS]}
    if not sections:
        lesson["message"] = "The paper corpus held too little to write this lesson responsibly."
    elif achieved < pages * SHORTFALL_BELOW:
        lesson["shortfall"] = (f"The corpus supported about {achieved:g} of the {pages} page(s) asked for; "
                               "the rest would have been padding.")
    return lesson, {"claims": merged, "conflicts": conflicts, "new_claims": [c.to_dict() for c in fresh]}


def _empty(reason: str, pages: int) -> dict:
    return {"insufficient": True, "sections": [], "generated": time.time(), "target_pages": pages,
            "achieved_pages": 0, "outline": [], "dropped_sections": [],
            "message": f"Could not write this lesson: {reason}.",
            "stats": {"written": 0, "kept_first_pass": 0, "repaired": 0, "dropped": 0, "grounded_pct": 0}}

