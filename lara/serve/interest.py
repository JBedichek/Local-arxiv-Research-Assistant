"""Follow-up recommendations for a finished deliverable, tailored by what this
installation's goals have actually been about.

**Three mechanisms, one pipeline.** A goal's own deliverable is always the primary input
to `recommend_followups` — the five suggestions are follow-ups on *this* answer, not
generic. What makes them tailored rather than generic-per-goal is the other two inputs:

1. **A standing interest profile** (`PROFILE_PATH`), one short paragraph, updated after
   every deliverable rather than recomputed from scratch each time. Bounded cost
   regardless of how many goals this installation has run — the alternative, re-deriving
   a digest from the full goal history on every recommendation, gets more expensive
   exactly as the history that makes it useful grows.
2. **Embedding-similarity retrieval**, not recency, decides which past goals the profile
   update is shown. The last five goals are not necessarily the *related* five — a person
   who just asked about memory architectures for the third time this month, after two
   unrelated goals in between, should have that pattern surface, and recency alone would
   miss it.

**Implicit preference from suggestion uptake**, the third mechanism, lives in
`SUGGESTIONS_STORE`: every set of five shown is logged, and `record_clicked` is called
when a follow-up actually gets submitted (see `serve/routes/runs.py`'s `follow_up`).
`acceptance_note` turns that log into a short note — which kinds of suggestions this
installation's user tends to act on versus skip — folded into the recommendation prompt
itself, not the profile update. Cold-start is real and not hidden: with nothing recorded
yet, `acceptance_note` returns "".

**Pooled across everyone**, deliberately, not per-user. There is no attribution today of
which auth token submitted which goal (`serve/runs.py`'s `owner_pid` tracks the owning
*process*, not a person) — pooling was the explicit choice made instead of building that
first. If this installation gets meaningfully multi-person, that gap needs closing before
per-user profiles/suggestions mean anything.

Storage follows `prompt_memory.py`'s and `gaps.py`'s own precedent: append-only JSONL (or
a single small JSON file for the profile) under `~/.lara`, read across runs, no
database. Same "live, auto-adopt, no held-out validation" trade `prompt_memory.py`'s
module docstring names for itself — if the profile or the recommendations start drifting
badly, the fix is a human review step before either is trusted further, deliberately
deferred here the same way it was there.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from lara.serve import context as CX

#: The standing interest profile: {"summary": str, "updated": float}. One paragraph,
#: not a list of goals — see module docstring's mechanism 1.
PROFILE_PATH = Path.home() / ".lara" / "interest_profile.json"

#: Append-only, one row per goal: {"run_id", "goal", "embedding": [floats]}. Read back
#: for embedding-similarity retrieval (mechanism 2) -- a rolling history, never pruned
#: here (an installation would need many tens of thousands of goals before this file's
#: size is a real concern relative to everything else this system stores per-run).
GOAL_EMBEDDINGS_STORE = Path.home() / ".lara" / "goal_embeddings.jsonl"

#: Append-only log of shown/clicked follow-up suggestions (mechanism 3). Two row kinds:
#: {"kind": "shown", "run_id", "goal", "suggestions": [...5 strings...], "ts"} and
#: {"kind": "clicked", "run_id", "text", "ts"} -- "run_id" on a clicked row is the
#: *parent* run (the one the suggestions were shown on), matching `follow_up`'s own
#: parent-run framing, not the new run the click created.
SUGGESTIONS_STORE = Path.home() / ".lara" / "followup_suggestions.jsonl"

#: How many similarity-retrieved past goals the profile update sees. Small on purpose --
#: this is "which past goals resemble this one", not a history dump.
RELATED_GOALS_K = 5

#: How many past shown/clicked rows `acceptance_note` looks at. Recent enough that a
#: changed pattern of interest shows up in a few goals, not stuck averaging over a
#: whole installation's lifetime.
ACCEPTANCE_WINDOW = 30

RECOMMEND_SYSTEM = """You suggest follow-up research prompts for someone who just read a \
finished research deliverable.

You are given the deliverable, a short note on this installation's recurring interests, \
and — when available — a note on which kinds of past suggestions this person tends to act \
on versus skip. Rules:

- Suggest exactly 5 follow-up prompts, each one a complete, concrete prompt someone could \
submit as-is — not a topic, not a question about the topic, the actual prompt.
- Ground every suggestion in something the deliverable actually said: a finding, a \
limitation it named, a question it left open, a "not settled" it flagged. Do not suggest \
something the deliverable gives no reason to ask.
- Use the interest note to prefer directions this installation keeps returning to, when \
the deliverable gives more than one reasonable direction to follow up in — but never at \
the cost of relevance to *this* deliverable. A good match to interest that ignores what \
was just found is worse than an on-topic suggestion that ignores interest.
- Respond with exactly 5 lines, each starting "1. " through "5. ", nothing before or \
after, no other commentary."""

PROFILE_UPDATE_SYSTEM = """You maintain a short standing summary of what someone keeps \
asking a research system about, updated after each new goal.

You are given the current summary (empty if this is the first goal ever), the new goal, a \
short excerpt of what its deliverable found, and any past goals that were found to \
resemble this one. Rules:

- Write one paragraph, at most 6 sentences. Name the 2-4 recurring themes you can \
actually support from what you were given — do not pad to length.
- A genuinely new interest, unrelated to anything in the current summary, is folded in \
alongside the existing ones, not used to replace them — this is a standing profile, not \
a most-recent-goal summary. But a theme the summary has carried for a long time with \
nothing reinforcing it recently may be dropped to make room.
- No preamble, no "Updated summary:", no meta-commentary. Just the paragraph."""

#: Deliverable text this long or shorter is passed to the profile updater whole; longer
#: is truncated. The updater needs "roughly what this was about", not the full argument
#: the way the recommender itself does.
PROFILE_DELIVERABLE_EXCERPT_CHARS = 1_500


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


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


def load_profile() -> str:
    """The current standing summary, or "" if nothing has been recorded yet."""
    return _load_json(PROFILE_PATH, {}).get("summary", "") or ""


def save_profile(summary: str) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps({"summary": summary, "updated": time.time()}))


def record_goal_embedding(run_id: str, goal: str, embedding: list[float]) -> None:
    _append_jsonl(GOAL_EMBEDDINGS_STORE, {"run_id": run_id, "goal": goal,
                                          "embedding": list(embedding)})


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def similar_past_goals(embedding: list[float], *, exclude_run_id: str = "",
                       k: int = RELATED_GOALS_K) -> list[str]:
    """The k past goals whose own embedding is most similar to `embedding` — similarity,
    not recency, per the module docstring's mechanism 2. Empty on a fresh installation
    (nothing recorded yet) or if no embedder was available to compute `embedding` in the
    first place (an empty list embeds as maximally dissimilar to everything, which is
    the honest answer, not a crash)."""
    rows = [r for r in _read_jsonl(GOAL_EMBEDDINGS_STORE) if r.get("run_id") != exclude_run_id]
    if not rows or not embedding:
        return []
    scored = sorted(rows, key=lambda r: _cosine(embedding, r.get("embedding") or []),
                    reverse=True)
    return [r["goal"] for r in scored[:k]]


def record_shown(run_id: str, goal: str, suggestions: list[str]) -> None:
    _append_jsonl(SUGGESTIONS_STORE, {"kind": "shown", "run_id": run_id, "goal": goal,
                                      "suggestions": list(suggestions), "ts": time.time()})


def was_shown(run_id: str) -> bool:
    return any(r.get("kind") == "shown" and r.get("run_id") == run_id
               for r in _read_jsonl(SUGGESTIONS_STORE))


def previously_shown(run_id: str, *, limit: int = 15) -> list[str]:
    """The most recent `limit` distinct suggestions ever shown for a run, oldest first --
    what a refresh must not circle back to, beyond the set it is replacing."""
    seen: list[str] = []
    for row in _read_jsonl(SUGGESTIONS_STORE):
        if row.get("kind") == "shown" and row.get("run_id") == run_id:
            for text in row.get("suggestions", []):
                if text in seen:
                    seen.remove(text)
                seen.append(text)
    return seen[-limit:]


def record_clicked(run_id: str, text: str) -> None:
    """Called when a person actually submits one of the shown suggestions as a follow-up
    goal (see `serve/routes/runs.py`'s `follow_up`) — the implicit signal `acceptance_note`
    reads back. `run_id` is the PARENT run the suggestions were shown on, matching
    `follow_up`'s own framing, not the new run the click creates."""
    _append_jsonl(SUGGESTIONS_STORE, {"kind": "clicked", "run_id": run_id, "text": text,
                                      "ts": time.time()})


def acceptance_note(*, limit: int = ACCEPTANCE_WINDOW) -> str:
    """A short note on which recent suggestions were acted on versus not, or "" with
    nothing recorded yet — the honest cold-start answer, not a fabricated pattern."""
    rows = _read_jsonl(SUGGESTIONS_STORE)
    shown = [r for r in rows if r.get("kind") == "shown"][-limit:]
    if not shown:
        return ""
    clicked_texts = {r.get("text", "") for r in rows if r.get("kind") == "clicked"}
    accepted = [s for row in shown for s in row.get("suggestions", []) if s in clicked_texts]
    skipped = [s for row in shown for s in row.get("suggestions", []) if s not in clicked_texts]
    if not accepted:
        return ""
    parts = [f"Of the last {len(shown)} suggestion sets, these were picked: "
            + "; ".join(accepted[-10:])]
    if skipped:
        parts.append("These were shown but not picked: " + "; ".join(skipped[-10:]))
    return " ".join(parts)


def _embed_one(embedder, text: str) -> list[float]:
    if embedder is None or not text.strip():
        return []
    try:
        vec = embedder.encode([text], convert_to_numpy=True)[0]
        return [float(x) for x in vec]
    except Exception:                                          # noqa: BLE001
        return []


async def update_profile(cfg, goal: str, deliverable_text: str, *, embedder=None,
                         model=None, window: int = 0, complete=None) -> str:
    """Revises and persists the standing profile for this new goal, returns the revised
    text (so `recommend_followups` uses the just-updated profile, not a stale read)."""
    if complete is None:
        from lara.serve.generate import complete

    embedding = _embed_one(embedder, goal)
    related = similar_past_goals(embedding, k=RELATED_GOALS_K)
    current = load_profile()
    excerpt = deliverable_text[:PROFILE_DELIVERABLE_EXCERPT_CHARS]

    prompt = (f"Current summary: {current or '(none yet — this is the first goal)'}\n\n"
             f"New goal: {goal}\n\n"
             f"What its deliverable found (excerpt): {excerpt}\n\n"
             + (f"Past goals that resemble this one: "
                + "; ".join(related) + "\n\n" if related else "")
             + "Write the updated summary.")
    room = CX.reply_room(window, prompt, PROFILE_UPDATE_SYSTEM, stage="interest_profile",
                      default=300, cap=600)
    revised = await complete(cfg, prompt, system=PROFILE_UPDATE_SYSTEM, model=model,
                             max_tokens=room)
    revised = (revised or "").strip()
    if revised:
        save_profile(revised)
    return revised or current


_SUGGESTION_LINE = re.compile(r"^\s*[1-5]\.\s+(.+)$", re.MULTILINE)


def parse_suggestions(text: str) -> list[str]:
    """Up to 5 follow-up prompts from RECOMMEND_SYSTEM's "1. "..."5. " format. Fewer
    than 5 is a model that did not fill every slot, not an error — the caller shows
    whatever came back rather than discarding a partial, useful answer."""
    return [m.strip() for m in _SUGGESTION_LINE.findall(text or "") if m.strip()]


#: A refresh drops a suggestion that restates one already shown: word overlap this high, or
#: (when an embedder is available) embeddings this close. Verbatim repeats are the common
#: case and score 1.0 on both; rewordings land in between.
RESTATEMENT_WORD_OVERLAP = 0.6
RESTATEMENT_COSINE = 0.9
#: Extra model calls a refresh may make to replace suggestions dropped as repeats.
REFRESH_RETRIES = 2
#: How many already-shown suggestions the refresh prompt lists (the most recent).
REFRESH_AVOID_IN_PROMPT = 25


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _novel(candidates: list[str], against: list[str], embedder=None) -> list[str]:
    """`candidates` that restate neither anything in `against` nor an earlier candidate."""
    seen = [(a, _words(a), _embed_one(embedder, a)) for a in against]
    out: list[str] = []
    for cand in candidates:
        words, emb = _words(cand), _embed_one(embedder, cand)
        repeat = any(
            (words and w and len(words & w) / len(words | w) >= RESTATEMENT_WORD_OVERLAP)
            or (emb and e and _cosine(emb, e) >= RESTATEMENT_COSINE)
            for _, w, e in seen)
        if not repeat:
            out.append(cand)
            seen.append((cand, words, emb))
    return out


async def recommend_followups(cfg, plan, full_deliverable_text: str, *, run_id: str = "",
                              embedder=None, model=None, window: int = 0, complete=None,
                              learn: bool = True, avoid: list[str] | None = None,
                              ) -> list[str]:
    """The five recommended follow-ups for a just-finished deliverable — the whole
    pipeline the module docstring describes: profile update, similarity-aware, folded
    together with the acceptance pattern from past suggestions, then one call asking
    for exactly 5. Returns [] if the deliverable is empty (nothing to follow up on) or
    the model returned nothing parseable.

    `learn=False` reads the standing profile without revising it and records nothing
    shown -- for suggesting on a deliverable that is not new (a backfill), where updating
    the profile in arbitrary order and logging suggestions nobody saw would both distort
    the signals later runs read.

    `avoid` makes this a refresh of a deliverable that already has suggestions: the model
    is shown them and asked for different ones. The profile is not revised again and the
    goal is not embedded again -- both already counted this deliverable once -- but the
    new set is still logged as shown.
    """
    if not full_deliverable_text.strip():
        return []
    if complete is None:
        from lara.serve.generate import complete

    goal = getattr(plan, "goal", "") or ""
    if learn and not avoid:
        profile = await update_profile(cfg, goal, full_deliverable_text, embedder=embedder,
                                       model=model, window=window, complete=complete)
    else:
        profile = load_profile()
    note = acceptance_note()

    async def ask(avoiding: list[str]) -> list[str]:
        prompt = (f"Deliverable:\n\n{full_deliverable_text}\n\n"
                 + (f"This installation's recurring interests: {profile}\n\n" if profile else "")
                 + (f"Suggestion acceptance pattern: {note}\n\n" if note else "")
                 + ("Already suggested. Propose 5 follow-ups on different aspects of the "
                    "deliverable: none may restate, rephrase or narrow any of these:\n"
                    + "\n".join(f"- {a}" for a in avoiding[-REFRESH_AVOID_IN_PROMPT:])
                    + "\n\n" if avoiding else "")
                 + "Suggest the 5 follow-up prompts.")
        room = CX.reply_room(window, prompt, RECOMMEND_SYSTEM, stage="followup_recommend",
                          default=400, cap=1_000)
        text = await complete(cfg, prompt, system=RECOMMEND_SYSTEM, model=model,
                              max_tokens=room)
        return parse_suggestions(text or "")

    suggestions = await ask(avoid or [])
    if avoid:
        # Asking is not enough: the model repeats items from the list it was told to
        # avoid, verbatim. So repeats are dropped here and replacements asked for.
        against, kept = list(avoid), []
        for attempt in range(REFRESH_RETRIES + 1):
            kept += _novel(suggestions, against + kept, embedder)
            if len(kept) >= 5 or attempt == REFRESH_RETRIES:
                break
            against += suggestions
            suggestions = await ask(against)
        suggestions = kept[:5]
    if suggestions and run_id and learn:
        record_shown(run_id, goal, suggestions)
        embedding = [] if avoid else _embed_one(embedder, goal)
        if embedding:
            record_goal_embedding(run_id, goal, embedding)
    return suggestions
