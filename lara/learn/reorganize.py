"""Reorganization: an already-written lesson rewritten into a teaching order, on request.

Lessons written one research theme at a time read as a pile of separate reports, in research
order, each with its own "Where sources disagree"; and before lessons were scoped
(`research._lesson_objective`'s SCOPE), one could re-teach most of its course. This plans a
teaching outline over the lesson's own sections, then writes each new section from the old
sections it draws on -- the same outline-then-write shape a newly researched lesson is now
written in (`synthesizer._write_lesson_outlined`), applied to finished text. Nothing is
researched again, the lesson's citations are kept, and nothing the lesson did not say is
added; material that belongs to the course's other lessons is cut down to what this one
needs from it."""

from __future__ import annotations

import time

from lara.learn import compress as CP
from lara.learn import trace as TR
from lara.learn.llm import Llm

#: Words of a section shown to the outline call -- enough to know what the section is about.
PREVIEW_WORDS = 45
#: A section's length bounds. Without an upper bound an outline sized one 36,000-word lesson
#: into six 2,000-word walls of text.
MIN_SECTION_WORDS, MAX_SECTION_WORDS = 150, 900
#: A rewritten section citing a smaller share of its sentences than this fraction of its
#: source's share is written again once -- the first live run kept citations on 55% of
#: sentences, from a lesson that had them on 81%.
CITATION_KEEP = 0.8

OUTLINE_SYSTEM = """You reorganize an existing lesson into the order a learner should meet \
its material. Below are the lesson's concept, its scope, and every current section: an id, \
its heading, its length and how it begins.

Reply with JSON only:
{"sections": [{"heading": "...", "establishes": "...", "from": ["<section id>"], "words": N}]}

- Order the new sections for a learner: what the concept is and why it matters, then how it \
works, then the evidence and how it is used, then its limits and open questions. Each builds \
on the ones before and never needs a later one.
- heading: what the section is about, specifically -- never a generic label such as "What \
the concept is" or "Where sources disagree".
- establishes: one sentence -- what the learner understands after reading it.
- from: the current sections whose material it is written from. Merge sections that cover \
the same ground; a current section may feed several new ones. Material about the point a \
disagreement concerns goes with that point -- never a section of disagreements on their own.
- Leave out current sections whose material belongs to the OTHER LESSONS listed, beyond what \
this lesson needs from them.
- words: this section's share of the LENGTH given; together they add up to it. A section is \
300 to 800 words: a long topic becomes several sections, each with its own specific heading."""

SECTION_SYSTEM = """You rewrite part of an existing lesson into ONE section of its new \
outline. The outline is given for orientation; write only the section named as yours, under \
no heading of your own (the caller adds it).

Format: one sentence per line.

- Establish what the outline says this section establishes, at about the length given, using \
only the source text provided. Prose that teaches: connect each point to the one before it.
- End every sentence that states a fact with the citation brackets of the source sentences \
it comes from, exactly as written there, e.g. [12345] or [c3]. When you merge source \
sentences, carry all of their brackets. Never invent a citation and never add a fact the \
source does not state.
- When the source says the same thing more than once, say it once. Never explain again what \
ALREADY COVERED lists.
- Where the source reports that findings disagree, keep that, with what each side found."""


def _sections_listing(sections: list[dict]) -> str:
    rows = []
    for i, sec in enumerate(sections, 1):
        words = " ".join(s["text"] for s in sec.get("sentences", [])).split()
        preview = " ".join(words[:PREVIEW_WORDS]) + ("..." if len(words) > PREVIEW_WORDS else "")
        rows.append(f"[s{i}] {sec.get('heading') or '(untitled)'} ({len(words)} words): {preview}")
    return "\n".join(rows)


def _cited_share(sections: list[dict]) -> float:
    flat = [s for sec in sections for s in sec.get("sentences", [])]
    return sum(1 for s in flat if s.get("claims")) / len(flat) if flat else 0.0


def _parse_outline(data, known: set[str]) -> list[dict]:
    out = []
    for sec in (data.get("sections") if isinstance(data, dict) else None) or []:
        if not isinstance(sec, dict) or not str(sec.get("heading", "")).strip():
            continue
        sources = [f for f in sec.get("from") or [] if isinstance(f, str) and f in known]
        if not sources:
            continue
        try:
            words = max(MIN_SECTION_WORDS, min(MAX_SECTION_WORDS, int(sec.get("words") or 400)))
        except (TypeError, ValueError):
            words = 400
        out.append({"heading": str(sec["heading"]).strip(),
                    "establishes": str(sec.get("establishes", "")).strip(),
                    "from": sources, "words": words})
    return out


async def reorganize(llm: Llm, lesson: dict, concept: dict, *, others=(), words: int = 0) -> dict:
    """A reorganized copy of `lesson`, in the same shape. Raises ValueError when there is no
    lesson, or no usable outline came back."""
    if not lesson or lesson.get("insufficient") or not lesson.get("sections"):
        raise ValueError("there is no written lesson to reorganize")
    sections = lesson["sections"]
    by_id = {f"s{i}": sec for i, sec in enumerate(sections, 1)}
    known = {k for sec in sections for s in sec["sentences"] for k in s.get("claims", [])}
    source_words = CP._words(sections)
    target = words or source_words
    scope = ("\n".join(f"- {t}" for t in others)) or "(none)"
    TR.set_phase("reorganize: outline")
    data = await llm.ask_json(OUTLINE_SYSTEM,
                              f"CONCEPT: {concept['title']} -- {concept.get('summary', '')}\n"
                              f"OTHER LESSONS IN THIS COURSE:\n{scope}\n"
                              f"LENGTH: about {target} words\n\nCURRENT SECTIONS:\n{_sections_listing(sections)}",
                              default=3_000, cap=0, stage="learn_reorganize_outline")
    outline = _parse_outline(data, set(by_id))
    if not outline:
        raise ValueError("no usable outline came back; try again")

    plan = "\n".join(f"{i}. {sec['heading']} -- {sec['establishes']}" for i, sec in enumerate(outline, 1))
    written: list[dict] = []
    for i, sec in enumerate(outline, 1):
        TR.set_phase(f"reorganize: {sec['heading']}")
        already = CP._render([{"heading": w["heading"], "sentences": w["sentences"][:1]}
                              for w in written]) or "(nothing yet)"
        sources = [by_id[f] for f in sec["from"]]
        source = CP._render(sources)
        prompt = (f"CONCEPT: {concept['title']}\nNEW OUTLINE:\n{plan}\n\n"
                  f"YOUR SECTION: {i}. {sec['heading']}\nIt establishes: {sec['establishes']}\n"
                  f"LENGTH: about {sec['words']} words\n\nALREADY COVERED:\n{already}\n\n"
                  f"SOURCE TEXT:\n{source}")
        want = _cited_share(sources) * CITATION_KEEP
        sentences: list[dict] = []
        for attempt in range(2):
            text = await llm.ask(SECTION_SYSTEM, prompt if attempt == 0 else prompt + (
                "\n\nYour last version dropped citations: every factual sentence must end with "
                "the brackets of the source sentences it comes from."),
                default=max(500, int(sec["words"] * 2.2)), cap=0, stage="learn_reorganize_section")
            got = [s for _, sents in CP._parse(text, known) for s in sents]
            if not sentences or _cited_share([{"sentences": got}]) > _cited_share([{"sentences": sentences}]):
                sentences = got
            if _cited_share([{"sentences": sentences}]) >= want:
                break
        if sentences:
            written.append({"heading": sec["heading"], "sentences": sentences})
    if not written:
        raise ValueError("reorganizing produced no text; try again")
    out_words = CP._words(written)
    flat = [s for sec in written for s in sec["sentences"]]
    cited = sum(1 for s in flat if s["claims"])
    return {"insufficient": False, "sections": written, "generated": time.time(),
            "reorganized": {"source_sections": len(sections), "sections": len(written),
                            "source_words": source_words, "words": out_words},
            "stats": {"written": len(flat), "kept_first_pass": cited, "repaired": 0, "dropped": 0,
                      "grounded_pct": round(100 * cited / len(flat)) if flat else 0}}
