"""Compression: a shorter version of a finished lesson, written on request, deliberately lossy.

The learner picks how much to throw away, and each level is one fixed prompt (`PROMPTS`) -- not
a page target the model negotiates with. The lesson is compressed from its own text, not
researched again, so it keeps the lesson's citations and adds nothing the lesson did not say.

A long lesson is compressed in consecutive chunks of whole sections (`CHUNK_WORDS`), each call
shown what the earlier chunks already kept, so a point the lesson made several times is kept
once rather than shortened several times over."""

from __future__ import annotations

import time

from lara.learn import lesson as LE
from lara.learn import research as RS
from lara.learn import trace as TR
from lara.learn.llm import Llm

#: level -> (the word the prompt uses, roughly what share of the words to keep)
LEVELS = {"high": ("high", 0.15), "med": ("medium", 0.35), "low": ("low", 0.6)}
#: Words of lesson text sent in one call.
CHUNK_WORDS = 3_000
#: How much already-kept text a later chunk is shown, at most.
KEPT_CONTEXT_CHARS = 6_000

COMPRESS_SYSTEM = """Shorten and compress this lesson text with {degree} discarding of \
conceptually unimportant detail.

Format: "## Heading" lines, and under each heading one sentence per line.

- Keep the ideas a learner needs to understand the concept; drop {drop}.
- When the text explains the same idea more than once, keep it once. Drop anything the \
ALREADY KEPT text below says already.
- Keep each kept sentence's citation brackets exactly as written, e.g. [12345] or [c3]. Never \
invent a citation and never add a fact the text does not state.
- Merge or rename headings so they say what each part is about; drop generic headings that \
only repeat ("What the concept is", "Where sources disagree") when nothing new sits under them.
- Aim for about {pct}% of the original length."""

_DROP = {"high": "examples, numbers, caveats and side points unless the main idea depends on them",
         "med": "secondary examples, repeated numbers and side points",
         "low": "only repetition and clearly minor asides"}


def variant(level: str) -> str:
    return f"compress-{level}"


def _render(sections: list[dict]) -> str:
    out = []
    for sec in sections:
        if sec.get("heading"):
            out.append(f"## {sec['heading']}")
        for s in sec.get("sentences", []):
            text = s["text"]
            if s.get("claims"):
                # Before the full stop, the way the lesson writer cites.
                end = text[-1] if text[-1:] in ".!?" else ""
                text = f"{text[:len(text) - len(end)]} [{', '.join(s['claims'])}]{end}"
            out.append(text)
    return "\n".join(out)


def _chunks(sections: list[dict]) -> list[list[dict]]:
    chunks, cur, words = [], [], 0
    for sec in sections:
        n = sum(len(s["text"].split()) for s in sec.get("sentences", []))
        if cur and words + n > CHUNK_WORDS:
            chunks.append(cur)
            cur, words = [], 0
        cur.append(sec)
        words += n
    if cur:
        chunks.append(cur)
    return chunks


def _parse(text: str, known: set[str]) -> list[tuple[str, list[dict]]]:
    """Research-driven lessons cite chunk ids ([3352954]); legacy ones claim keys ([c1])."""
    if known and all(k.isdigit() for k in known):
        return RS._parse_sections(text, known)
    return LE.parse(text, known)


def _words(sections: list[dict]) -> int:
    return sum(len(s["text"].split()) for sec in sections for s in sec.get("sentences", []))


async def compress(llm: Llm, lesson: dict, level: str) -> dict:
    """A compressed copy of `lesson` at `level` ("high", "med" or "low"), in the same shape."""
    if level not in LEVELS:
        raise ValueError(f"unknown compression level {level!r}")
    if not lesson or lesson.get("insufficient") or not lesson.get("sections"):
        raise ValueError("there is no written lesson to compress")
    degree, share = LEVELS[level]
    system = COMPRESS_SYSTEM.format(degree=degree, drop=_DROP[level], pct=round(share * 100))
    known = {k for sec in lesson["sections"] for s in sec["sentences"] for k in s.get("claims", [])}
    TR.set_phase(f"compress: {degree}")
    kept: list[dict] = []
    for n, chunk in enumerate(_chunks(lesson["sections"]), 1):
        already = _render(kept)[-KEPT_CONTEXT_CHARS:] or "(nothing yet)"
        words = _words(chunk)
        text = await llm.ask(system, f"ALREADY KEPT:\n{already}\n\nTEXT TO COMPRESS (part {n}):\n{_render(chunk)}",
                             default=max(400, int(words * share * 2.2)), cap=0, stage="learn_compress")
        kept += [{"heading": h, "sentences": sents} for h, sents in _parse(text, known) if sents]
    source_words, out_words = _words(lesson["sections"]), _words(kept)
    sentences = [s for sec in kept for s in sec["sentences"]]
    cited = sum(1 for s in sentences if s["claims"])
    return {"insufficient": not kept, "sections": kept, "generated": time.time(),
            "message": "" if kept else "Compression returned nothing usable; try another level.",
            "compression": {"level": level, "source_words": source_words, "words": out_words,
                            "ratio": round(out_words / source_words, 2) if source_words else 0.0},
            "stats": {"written": len(sentences), "kept_first_pass": cited, "repaired": 0,
                      "dropped": 0, "grounded_pct": round(100 * cited / len(sentences)) if sentences else 0}}
