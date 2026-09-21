"""Shorter versions of a finished deliverable, with every citation intact.

A synthesis deliverable is written once at full length; `condense` makes the medium
(redundancy removed) and short (a summary that answers the original question) versions
from it. Each version's citations are re-bound rather than trusted to still resolve just
because the bracket text survived.
"""
from __future__ import annotations

from lara.serve import citations as C
from lara.serve import context as CX

#: What to ask for when nobody could say how wide the model's window is; above the longest
#: deliverable measured, so the fallback is not itself a ceiling.
DEFAULT_DELIVERABLE_TOKENS = 6_000

#: The ceiling when the window is known. A reply that reaches it is a model that failed to
#: stop, not a thorough answer.
MAX_DELIVERABLE_TOKENS = 16_384

CONDENSE_SYSTEM = """You are condensing a research deliverable that has already been \
written, into a shorter version for a reader who wants less to read.

The deliverable you are given carries citation keys in square brackets, exactly as \
written — like [3352954]. These were bound once already; you are not assigning new ones. \
Rules:

- Every claim you keep must keep its own citation bracket(s), copied exactly as they \
appear in the source — character for character. Never retype, paraphrase, merge, or \
invent one.
- A claim you cut takes its citation bracket with it. Never attach a citation to a \
sentence it did not originally support, and never leave a bracket dangling with no claim \
around it.
- Never introduce a citation key that is not already present in the source text.
- Do not add claims, numbers, or reasoning that are not already in the source. You are \
condensing, not researching further.
- No preamble, no sign-off, no meta-commentary about what was removed or why.

Write prose. Markdown is fine."""

#: "medium" keeps the full deliverable's ceiling: removing redundancy is not a length
#: target, and a deliverable that earned its length should not be cut for having it.
DEFAULT_MEDIUM_TOKENS = DEFAULT_DELIVERABLE_TOKENS
MAX_MEDIUM_TOKENS = MAX_DELIVERABLE_TOKENS

#: "short" is a genuine summary, so a much lower ceiling.
DEFAULT_SHORT_TOKENS = 800
MAX_SHORT_TOKENS = 3_000

_CONDENSE_INSTRUCTION = {
    "medium": "Get rid of all redundant, repeated information in this deliverable, "
              "based on the original question: {goal}",
    "short": "Summarize the information in this document as concisely as possible, "
             "and use it to answer the original question: {goal}",
}


def known_from_dicts(references: dict | None) -> dict[str, C.Reference]:
    """A citation table built from already-bound reference dicts (a run record's
    `references`)."""
    known: dict[str, C.Reference] = {}
    for raw in (references or {}).values():
        ref = C.Reference.from_dict(raw)
        if ref is not None:
            known[ref.key] = ref
    return known


async def condense(cfg, full_text: str, *, level: str, goal: str = "", model=None,
                   window: int = 0, conn=None, complete=None,
                   known: dict[str, C.Reference] | None = None,
                   ) -> tuple[str, list[str], dict[str, dict]]:
    """A shorter version of `full_text` at `level` ("medium" or "short").

    Returns (text, citation keys kept, their references as dicts); text is "" when the
    model returned nothing. Citations are re-bound against `known` and then the corpus.
    """
    if level not in _CONDENSE_INSTRUCTION:
        raise ValueError(f"level must be one of {sorted(_CONDENSE_INSTRUCTION)}, got {level!r}")
    if not full_text.strip():
        return ("", [], {})
    if complete is None:
        from lara.serve.generate import complete

    prompt = f"{_CONDENSE_INSTRUCTION[level].format(goal=goal)}\n\nDeliverable:\n\n{full_text}"
    default = DEFAULT_MEDIUM_TOKENS if level == "medium" else DEFAULT_SHORT_TOKENS
    cap = MAX_MEDIUM_TOKENS if level == "medium" else MAX_SHORT_TOKENS
    room = CX.reply_room(window, prompt, CONDENSE_SYSTEM, stage=f"deliverable_{level}",
                         default=default, cap=cap)
    text = ((await complete(cfg, prompt, system=CONDENSE_SYSTEM, model=model,
                            max_tokens=room)) or "").strip()
    if not text:
        return ("", [], {})
    cited = C.bind(text, known=known or {}, conn=conn)
    # Chunk ids are dense integers, so a key the model mistyped or invented usually exists in the
    # corpus and would resolve to an unrelated paper. Only keys the source itself carried count.
    source = set(C.parse_keys(full_text))
    kept = {k: r for k, r in cited.references.items() if k in source}
    return cited.text, list(kept), {k: r.to_dict() for k, r in kept.items()}
