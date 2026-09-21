"""Distills a finished deliverable into a handful of reusable facts, so a later run can
reuse settled information instead of re-deriving it (see `synthesizer.py`'s
`retrieve_facts` tool, the reader).

The extraction test is the fact's *implication*, not its truth: only facts that would
change how a future analysis reasons -- a performance ceiling, a safety boundary, a
measured baseline -- are worth keeping. What gets stored is the fact alone, never the
implication itself, so a later run draws its own conclusion for its own question rather
than inheriting one drawn for a different question. This also keeps extraction from
drifting too fine-grained: a fact with no real implication fails the test and is dropped,
and a fact too broad to be one checkable statement was never a candidate.

Storage follows `interest.py`'s precedent: append-only JSONL under `~/.lara`, no
database. Retrieval is brute-force cosine over the whole file, matching `methods.py`'s
`MethodIndex` and `interest.py`'s `similar_past_goals` -- the store is expected to stay in
the low thousands of rows (a handful of facts per deliverable), where a vector index would
be pure overhead.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from lara.serve import context as CX

#: Append-only, one row per fact: {"id", "run_id", "goal", "fact", "tag", "embedding": [floats],
#: "ts"}. Never pruned here -- see module docstring on expected scale.
FACTS_STORE = Path.home() / ".lara" / "facts.jsonl"

#: Most deliverables license 1-3 real facts, not 5 -- this is a ceiling, not a target.
MAX_FACTS_PER_DELIVERABLE = 5

#: How many past facts a single retrieval call returns.
DEFAULT_RETRIEVE_K = 5

#: A retrieved fact this similar to one already ranked higher is dropped from the result.
#: Measured on the backfilled store: cross-run restatements of one fact stay true
#: restatements down to ~0.85, and ~1 in 4 top-5 result sets contained one without this.
DUPLICATE_SIMILARITY = 0.85

DISTILL_SYSTEM = """You distill a finished research deliverable into a small number of \
standalone facts worth remembering for future research.

Extract a fact only when it has an important implication -- something a future analysis \
would reason differently because of it (a performance ceiling, a safety boundary, a \
design tradeoff, a hard constraint, a measured number that changes a decision). Do not \
extract facts that are merely true but inert: they must license something.

Store only the fact itself, as one self-contained sentence a reader could use without \
having read the deliverable. Do NOT write the implication -- state what is true, not what \
follows from it; a future reader draws their own conclusion for their own question.

Tag each fact with a short (2-4 word) label naming the KIND of implication it supports \
(e.g. "performance ceiling", "safety boundary", "design tradeoff", "known limitation", \
"measured baseline") -- not a summary of the fact itself.

Extract at most 5 facts, fewer if fewer clear this bar -- a deliverable with nothing that \
qualifies extracts none.

Respond with one line per fact, formatted exactly as: "1. [tag] fact text" through \
"5. [tag] fact text", nothing before or after, no other commentary. If nothing qualifies, \
respond with the single word "NONE"."""


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _embed_one(embedder, text: str) -> list[float]:
    if embedder is None or not text.strip():
        return []
    try:
        vec = embedder.encode([text], convert_to_numpy=True)[0]
        return [float(x) for x in vec]
    except Exception:                                          # noqa: BLE001
        return []


def embedder_fn(embedder):
    """Wraps a raw embedder object as a plain `str -> list[float]` callable, for a
    caller (`synthesizer.py`) that deliberately never imports an embedder object
    directly -- see that module's own docstring on why."""
    def _fn(text: str) -> list[float]:
        return _embed_one(embedder, text)
    return _fn


_FACT_LINE = re.compile(r"^\s*[1-5]\.\s*\[([^\]]+)\]\s*(.+)$", re.MULTILINE)


def parse_facts(text: str) -> list[tuple[str, str]]:
    """(tag, fact) pairs from `DISTILL_SYSTEM`'s "1. [tag] fact" format. A "NONE" reply,
    or anything else that matches no line, parses to an empty list -- not an error."""
    return [(tag.strip(), fact.strip()) for tag, fact in _FACT_LINE.findall(text or "")
            if tag.strip() and fact.strip()]


async def distill_facts(cfg, goal: str, deliverable_text: str, *, run_id: str,
                        embedder=None, model=None, window: int = 0,
                        complete=None) -> list[dict]:
    """Extracts and persists up to `MAX_FACTS_PER_DELIVERABLE` facts from a just-finished
    deliverable. Returns the stored rows (each already written to `FACTS_STORE`) so a
    caller can report what happened; `[]` for an empty deliverable or a reply with
    nothing that cleared the implication bar."""
    if not deliverable_text.strip():
        return []
    if run_id and any(r.get("run_id") == run_id for r in _read_jsonl(FACTS_STORE)):
        return []
    if complete is None:
        from lara.serve.generate import complete

    prompt = f"Goal: {goal}\n\nDeliverable:\n\n{deliverable_text}\n\nExtract the facts."
    room = CX.reply_room(window, prompt, DISTILL_SYSTEM, stage="fact_distill",
                         default=400, cap=1_000)
    text = await complete(cfg, prompt, system=DISTILL_SYSTEM, model=model, max_tokens=room)
    pairs = parse_facts(text or "")[:MAX_FACTS_PER_DELIVERABLE]

    stored = []
    for i, (tag, fact) in enumerate(pairs):
        row = {"id": f"{run_id}-{i}", "run_id": run_id, "goal": goal, "fact": fact,
              "tag": tag, "embedding": _embed_one(embedder, fact), "ts": time.time()}
        _append_jsonl(FACTS_STORE, row)
        stored.append(row)
    return stored


def search_facts(embedding: list[float], *, k: int = DEFAULT_RETRIEVE_K,
                 exclude_run_id: str = "") -> list[dict]:
    """The k stored facts whose own embedding is most similar to `embedding`, skipping any
    that restates a higher-ranked hit (`DUPLICATE_SIMILARITY`). Empty if the store is
    empty or `embedding` is empty (no embedder available to the caller)."""
    rows = [r for r in _read_jsonl(FACTS_STORE)
            if r.get("run_id") != exclude_run_id and r.get("embedding")]
    if not rows or not embedding:
        return []
    scored = sorted(rows, key=lambda r: _cosine(embedding, r["embedding"]), reverse=True)
    picked: list[dict] = []
    for row in scored:
        if any(_cosine(row["embedding"], p["embedding"]) >= DUPLICATE_SIMILARITY
               for p in picked):
            continue
        picked.append(row)
        if len(picked) == k:
            break
    return picked


def format_facts(rows: list[dict]) -> str:
    """Renders retrieved facts as the plain-text tool result a model reads -- provenance
    (the originating goal) included so the model can judge relevance, no citation
    binding attempted (these are informational context, not a bound reference)."""
    if not rows:
        return ""
    return "\n".join(
        f"- [{r.get('tag', '')}] {r.get('fact', '')} "
        f"(from a past goal: {r.get('goal', '')[:80]})"
        for r in rows)
