"""Multi-round retrieval agent with three tools.

The model gets three capabilities beyond answering:

``clarify``          the question is too vague to search usefully — ask the reader first
``search``           the excerpts do not answer it — search again with a better query
``expand_context``   a chunk is unintelligible alone — pull its neighbours or whole section

**Decisions are JSON, not native tool calls.** vLLM's tool-calling needs a per-model parser
enabled at launch (``--tool-call-parser``), which would tie the agent to one generator and
break the model picker the moment someone selects a different checkpoint. A small JSON
verdict works on any instruction-tuned model and degrades to "answer directly" if parsing
fails, so a model that cannot follow the schema still produces an answer rather than an
error.

**Every round emits progress events.** A search that renders nothing for twenty seconds is
indistinguishable from a hang, and the honest fix is to show the work rather than to hide
the latency.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from lara.serve.generate import SYSTEM, stream_answer

# ── the speed <-> accuracy spectrum ────────────────────────────────────────────────
#
# One dial, because every knob here spends the same currency: latency for recall. The
# estimates are shown in the UI, since a slider that does not say what it costs invites
# people to max it out and then think the app has hung.

@dataclass
class Breadth:
    name: str
    label: str
    max_rounds: int
    k: int
    candidates: int
    expand_context: bool
    allow_clarify: bool
    budget_sec: float
    estimate: str


# Estimates are MEASURED end-to-end medians, not search overhead. They matter: an earlier
# version advertised "~0.5s" for Instant, which measured 5.3s, because it counted only the
# retrieval leg. A dial that understates its cost is worse than one with no numbers at all.
#
# Note the floor. Generation alone is ~3s to first token plus ~2s of streaming, so the
# bottom three settings differ far less than their names suggest — the dial buys search
# depth, and search was never the expensive part.
SPECTRUM: list[Breadth] = [
    Breadth("instant", "Instant", 1, 5, 100, False, False, 20, "~5s"),
    Breadth("fast", "Fast", 1, 8, 200, False, False, 30, "~5s"),
    Breadth("balanced", "Balanced", 2, 8, 200, True, False, 60, "~6s"),
    Breadth("thorough", "Thorough", 3, 12, 300, True, True, 120, "~6-9s"),
    Breadth("exhaustive", "Exhaustive", 5, 20, 400, True, True, 300, "~10-17s"),
]
BY_NAME = {b.name: b for b in SPECTRUM}
DEFAULT = "balanced"


def resolve(name: str | None) -> Breadth:
    return BY_NAME.get((name or DEFAULT).lower(), BY_NAME[DEFAULT])


# ── the decision step ──────────────────────────────────────────────────────────────

DECIDE_SYSTEM = """You are the retrieval controller for a question-answering system over \
arXiv machine-learning papers. You do not answer the question yourself. You decide what \
the system should do next, and you reply with a single JSON object and nothing else.

Schema:
{"action": "answer" | "search" | "clarify" | "expand",
 "reason": "<one short sentence>",
 "queries": ["<search query>", ...],      // only for "search", 1-3 queries
 "question": "<what to ask the reader>",  // only for "clarify"
 "options": ["<suggestion>", ...],        // only for "clarify", 2-4 concrete refinements
 "chunk_ids": [<int>, ...]}               // only for "expand"

Choose "answer" when the excerpts contain enough to answer, even partially. Prefer it.
Choose "search" when the excerpts are on-topic but miss the specific fact asked for; the \
queries should use the paper's own terminology rather than repeating the question.
Choose "clarify" ONLY when the question is so ambiguous that any search would be a guess \
-- for example it names no method, dataset, or property, or could plausibly mean several \
unrelated things.
Choose "expand" when an excerpt is clearly a fragment whose meaning depends on \
surrounding text -- a bare equation, a sentence beginning "This shows that...", a table \
row without its caption."""


@dataclass
class Step:
    kind: str                      # search | decide | expand | clarify | answer
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)


def format_excerpts(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        where = h.get("section") or ""
        title = h.get("paper_title") or h.get("arxiv_id", "")
        lines.append(f"[{i}] (id={h.get('chunk_id')}) ({title} > {where}) {h.get('text','').strip()}")
    return "\n\n".join(lines) if lines else "(nothing retrieved yet)"


_JSON = re.compile(r"\{.*\}", re.S)


def parse_decision(text: str) -> dict[str, Any]:
    """Extract the JSON verdict, tolerating prose or fences around it.

    A model that ignores the schema should not break the request — an unparseable verdict
    means "just answer", which is the safe default.
    """
    m = _JSON.search(text or "")
    if not m:
        return {"action": "answer", "reason": "no decision returned"}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"action": "answer", "reason": "unparseable decision"}
    if not isinstance(d, dict) or d.get("action") not in {"answer", "search", "clarify", "expand"}:
        return {"action": "answer", "reason": "unknown action"}
    return d


async def decide(cfg, question: str, hits: list[dict], breadth: Breadth,
                 rounds_done: int, model: str | None) -> dict[str, Any]:
    """One controller turn. Returns the parsed verdict."""
    remaining = breadth.max_rounds - rounds_done
    prompt = (
        f"Question: {question}\n\n"
        f"Excerpts retrieved so far:\n{format_excerpts(hits)}\n\n"
        f"Search rounds remaining: {remaining}. "
        f"{'Clarification is not available; do not choose it.' if not breadth.allow_clarify else ''} "
        f"{'Context expansion is not available; do not choose it.' if not breadth.expand_context else ''}\n"
        "Reply with the JSON object only."
    )
    buf = ""
    async for tok in stream_answer(
        cfg, prompt, [], system=DECIDE_SYSTEM, model=model,
        temperature=0.0, max_tokens=400, raw_user=True,
    ):
        buf += tok
    return parse_decision(buf)


# ── context expansion ──────────────────────────────────────────────────────────────

def expand_chunks(conn, chunk_ids: list[int], before: int = 2, after: int = 1,
                  limit: int = 12) -> list[dict]:
    """Fetch neighbouring chunks so a fragment can be read in context.

    Cheap by construction: chunks carry a per-paper ``ordinal``, so this is an index
    lookup with no embedding and no search. Reads more before than after, because the
    usual failure is a chunk that refers backwards ("this shows that...", "the above
    bound").
    """
    if not chunk_ids:
        return []
    out: list[dict] = []
    seen: set[int] = set()
    for cid in chunk_ids[:6]:
        row = conn.execute(
            "SELECT arxiv_id, version, ordinal FROM chunks WHERE chunk_id=?", (cid,)
        ).fetchone()
        if row is None:
            continue
        rows = conn.execute(
            """SELECT c.chunk_id, c.arxiv_id, c.version, c.ordinal, c.section_anchor,
                      c.anchor_start, c.char_start, c.anchor_end, c.char_end, c.kind, c.text,
                      p.title AS paper_title, s.title AS section_title
               FROM chunks c
               JOIN papers p ON p.arxiv_id = c.arxiv_id
               LEFT JOIN sections s ON s.arxiv_id=c.arxiv_id AND s.version=c.version
                                   AND s.anchor=c.section_anchor
               WHERE c.arxiv_id=? AND c.version=? AND c.ordinal BETWEEN ? AND ?
               ORDER BY c.ordinal""",
            (row["arxiv_id"], row["version"], row["ordinal"] - before, row["ordinal"] + after),
        ).fetchall()
        for r in rows:
            if r["chunk_id"] in seen or len(out) >= limit:
                continue
            seen.add(r["chunk_id"])
            out.append({
                "chunk_id": r["chunk_id"], "arxiv_id": r["arxiv_id"], "version": r["version"],
                "url": f"/p/{r['arxiv_id']}v{r['version']}#{r['anchor_start']}:"
                       f"{r['char_start']}-{r['char_end']}",
                "anchor": r["anchor_start"], "char_start": r["char_start"],
                "char_end": r["char_end"], "anchor_end": r["anchor_end"],
                "section": r["section_title"] or r["section_anchor"], "kind": r["kind"],
                "score": 0.0, "paper_title": r["paper_title"], "text": r["text"],
                "via": "context",
            })
    return out


def merge_hits(existing: list[dict], new: list[dict], cap: int) -> tuple[list[dict], int]:
    """Add only chunks not already present. Returns (merged, n_added).

    Deduplicating across rounds matters: without it round 2 re-retrieves most of round 1,
    the model sees no new evidence, and the loop burns its whole budget standing still.
    """
    seen = {h.get("chunk_id") for h in existing}
    added = 0
    for h in new:
        if h.get("chunk_id") in seen:
            continue
        seen.add(h.get("chunk_id"))
        existing.append(h)
        added += 1
        if len(existing) >= cap:
            break
    return existing, added
