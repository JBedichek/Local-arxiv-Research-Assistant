"""Inline detail on highlighted lesson text, tried cheapest-first: (1) the concept's own already-
fetched claims; if they cannot add anything beyond what the highlighted sentences already cite,
(2) the fuller text of the most relevant paper(s) the lesson already cites, which the excerpted
claims may have only partly captured; only if that still cannot answer does (3) the general
corpus get searched for new material. Tier 1's answer is verified like a lesson: every sentence
cites claims and is re-judged against them. Tiers 2 and 3 trust their own citation writing
directly -- see `_paper_answer`'s and `_new_claims_from_research`'s own docstrings for why. When
nothing survives any tier the learner is told so."""

from __future__ import annotations

import asyncio
import re
import time

from lara.learn import claims as CL
from lara.learn import lesson as LE
from lara.learn.llm import Llm

DEFAULT_REQUEST = "Explain the highlighted text in more detail."

ANSWER_SYSTEM = """A learner highlighted part of a lesson and asked for more detail. Answer \
using ONLY the numbered claims. Be as thorough as the claims allow -- there is no length limit, \
so do not cut the answer short to save space.

Format: one sentence per line, each ending -- before its full stop -- with the keys of the claims \
it rests on in brackets, like [c1] or [x1c2, c3]. A sentence with no key is not allowed.

- Draw out everything the claims support beyond the highlighted text: mechanism, conditions, \
numbers, caveats, examples, how it relates to nearby ideas. Do not add any fact that is not in \
the claims, and do not pad by restating the highlighted text itself.
- Convey certainty as marked: one paper, a hypothesis, replaced by later work.
- If the claims cannot add anything to the highlighted text, reply exactly: INSUFFICIENT"""

INSUFFICIENT = ("The paper corpus has nothing more on that which I can support. "
                "Try highlighting a narrower passage or asking a more specific question.")

#: How many of the lesson's own cited papers tier 2 will read in full, most relevant first --
#: bounded the same way `lara.serve.synthesis.MAX_FULL_PAPERS_PER_RUN` bounds a synthesis run's
#: own full-paper reads, for the identical reason: a paper is tens of thousands of tokens, and
#: reading every paper a lesson happens to cite for one question would cost far more than the
#: general-corpus search this tier exists to avoid falling back to.
FULL_PAPER_CAP = 3

PAPER_ANSWER_SYSTEM = """A learner highlighted part of a lesson and asked for more detail. The \
lesson already cites the paper below among its sources -- its excerpted claims may not have \
captured everything relevant, so you are reading more of the paper itself. Answer using ONLY \
the paper text given, labelled by chunk id.

Format: one sentence per line, each ending -- before its full stop -- with the chunk id(s) it \
rests on in brackets, like [12345] or [12345, 67890].

- Draw out everything the paper supports beyond the highlighted text: mechanism, conditions, \
numbers, caveats, examples, how it relates to nearby ideas. Do not add any fact the paper text \
does not state, and do not pad by restating the highlighted text itself.
- If the paper does not address the request, reply exactly: INSUFFICIENT"""


def rank(claims: list[dict], focus: str, embed) -> list[dict]:
    vec = embed(focus) if embed else []
    scored = []
    for c in claims:
        score = CL.overlap(focus, c["text"])
        if vec:
            other = embed(c["text"])
            score = max(score, CL.cosine(vec, other) if other else 0.0)
        scored.append((score, c))
    return [c for _, c in sorted(scored, key=lambda t: t[0], reverse=True)]


async def _answer(llm: Llm, selection: str, request: str, claims: list[dict]):
    """(sections, stats) of the verified answer, or None when the claims cannot answer."""
    if not claims:
        return None
    by_key = {c["key"]: c for c in claims}
    prompt = (f"HIGHLIGHTED LESSON TEXT: {selection}\nREQUEST: {request}\n\nCLAIMS:\n"
              + "\n".join(LE.line(c) for c in claims))
    # cap=0: no fixed ceiling -- the answer may use whatever of the context window is left
    # once the prompt is in it (see reply_room), rather than an arbitrary token budget.
    text = await llm.ask(ANSWER_SYSTEM, prompt, default=2_000, cap=0, stage="learn_expand")
    if not text or text.upper().startswith("INSUFFICIENT"):
        return None
    sections, stats = await LE.verify(llm, LE.parse(text, set(by_key)), by_key)
    return (sections, stats) if sum(len(s["sentences"]) for s in sections) else None


def _adds_something(answer, selection_claims: set[str]) -> bool:
    """The answer must draw on at least one claim the highlighted text did not already cite --
    otherwise it is pure restatement, however long, whether or not the learner phrased their
    ask as a specific question. A specific question used to skip this check entirely, which is
    what let "why does it avoid spikes?" come back as the same sentence the highlighted text
    already made, with no search and nothing new: the question narrowed what the model was
    asked to answer, not what it was allowed to answer *from*."""
    if answer is None:
        return False
    sentences = [s for sec in answer[0] for s in sec["sentences"]]
    return any(k not in selection_claims for s in sentences for k in s["claims"])


def _next_id(expansions: list[dict]) -> int:
    nums = [int(e["id"][1:]) for e in expansions if str(e.get("id", "")).startswith("x")
            and e["id"][1:].isdigit()]
    return max(nums, default=0) + 1


async def _new_claims(llm: Llm, corpus, concept: dict, existing: list[dict], focus: str, n: int,
                      embed) -> list[CL.Claim]:
    passages = await CL.gather_passages(corpus, concept, focus=focus,
                                        exclude=frozenset(c["passage"]["chunk_id"] for c in existing))
    found, *_ = await CL.extract(llm, concept, passages, embed=embed, focus=focus)
    fresh = [c for c in found if all(CL.overlap(c.text, e["text"]) < CL.DUPLICATE_OVERLAP for e in existing)]
    for i, c in enumerate(fresh, 1):
        c.key = f"x{n}c{i}"
    # Compared against copies of the concept's claims, so the new ones can record agreement or
    # conflict with them without the lesson's own claims being altered.
    await CL.relate(llm, fresh + [CL.Claim.from_dict(e) for e in existing], embed=embed)
    return fresh


#: A citation bracket, matching lara.serve.citations.CITATION -- duplicated, not imported,
#: the same small-shape-adapter duplication graph.py and research.py already use for the
#: identical pair (see either's own comment): lara.learn never imports lara.serve.
_CITE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_CITE_KEYS = re.compile(r"\d+")


def _cited_keys(text: str) -> list[str]:
    keys: list[str] = []
    for m in _CITE.finditer(text or ""):
        keys.extend(_CITE_KEYS.findall(m.group(0)))
    return list(dict.fromkeys(keys))


def _pseudo_claim(key: str, ref: dict) -> dict:
    """Duplicated from lara.learn.research._pseudo_claim, not imported -- same reasoning
    as this module's own _CITE/_cited_keys duplication above."""
    return {"key": key, "text": ref.get("claim") or ref.get("text", ""),
           "passage": {"chunk_id": ref.get("chunk_id"), "arxiv_id": ref.get("arxiv_id"),
                       "title": ref.get("paper_title") or ref.get("title", ""),
                       "section": ref.get("section", ""), "text": ref.get("text", "")},
           "conditions": "", "kind": "finding", "corroborated_by": [], "conflicts": [],
           "superseded_by": "", "flags": [], "certainty": "single-source"}


async def _new_claims_from_research(synth, existing: list[dict], focus: str, n: int) -> list[dict]:
    """New pseudo-claims from one targeted research leaf (`synth`, injected -- see
    lara.serve.learn_research.expand_synth), in place of CL.gather_passages() ->
    CL.extract() -> CL.relate()'s judge-verified search: consistent with the confirmed,
    course-wide decision that a synthesis-driven answer uses its own citation writing
    directly with no independent re-check (see lara.learn.research.build_lesson) -- this
    on-demand path should not be the one place still paying for, and trusting, a
    different verification story than the lesson it is expanding.

    `n` numbers keys `x{n}c{i}`, the same convention `_new_claims`'s freshly extracted
    claims already used, so a search's origin stays visible in the claim key either way."""
    result = await synth(focus)
    text, refs = (result.get("text") or "").strip(), result.get("references") or {}
    if not text or not refs:
        return []
    fresh, i = [], 0
    for k in _cited_keys(text):
        ref = refs.get(k)
        if ref is None:
            continue
        i += 1
        claim = _pseudo_claim(f"x{n}c{i}", ref)
        if any(CL.overlap(claim["text"], e["text"]) >= CL.DUPLICATE_OVERLAP for e in existing):
            continue
        fresh.append(claim)
    return fresh


def _ranked_papers(context: list[dict], cap: int = FULL_PAPER_CAP) -> list[tuple[str, int]]:
    """Up to `cap` distinct (arxiv_id, version) the lesson cites, most relevant first --
    `context` is already `rank()`'s own best-first order, so a paper's rank is just wherever
    its best-scoring claim first appears; only the first occurrence of each paper matters.
    `version` falls back to 1 when a pseudo-claim's passage does not carry one (synthesis's
    `Reference` has no version field) -- the same fallback `lara/learn/visuals.py`'s own
    `full_paper()` fallback already uses for the identical gap."""
    seen: dict[str, int] = {}
    for c in context:
        aid = c["passage"].get("arxiv_id")
        if not aid or aid in seen:
            continue
        seen[aid] = int(c["passage"].get("version") or 1)
        if len(seen) >= cap:
            break
    return list(seen.items())


def _parse_chunk_cited(text: str, known: set[str]) -> list[tuple[str, list[dict]]]:
    """[(heading, [{"text", "claims"}])] against chunk-id citation brackets -- the same shape
    and stripping behaviour `lesson.parse()`/`research._parse_sections` already produce, and the
    same "keep every line regardless of whether it cited anything" choice `research.py` makes
    for the identical reason: there is no `LE.verify()` pass here to decide an uncited line does
    not belong (see `_paper_answer`'s own docstring). Duplicated, not imported, from
    `research._parse_sections` -- this module already duplicates `_CITE`/`_cited_keys`/
    `_pseudo_claim` from that module for the same small-shape-adapter reasoning stated above."""
    sections: list[tuple[str, list[dict]]] = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*• ").strip()
        if not line:
            continue
        if line.startswith("#"):
            sections.append((line.lstrip("#").strip(), []))
            continue
        if not sections:
            sections.append(("", []))
        keys = [k for k in _cited_keys(line) if k in known]
        clean = re.sub(r"\s+", " ", _CITE.sub("", line)).replace(" .", ".").strip()
        sections[-1][1].append({"text": clean, "claims": keys})
    return sections


async def _paper_answer(llm: Llm, passages: list, selection: str, request: str):
    """One paper's fuller text, answered directly and trusted -- no `LE.verify()`, the same
    "use its own citation writing directly, no independent re-check" decision
    `_new_claims_from_research` already made for its own synthesis-driven citations: this tier
    sits in the same no-verify escalation path as that one, not tier 1's `LE.verify()`'d
    `ANSWER_SYSTEM` path above it. Grounded either way -- the answer is written from the paper's
    actual chunk text, labelled by id, not free generation.

    Returns (sections, stats, claims) in the shape `_answer()`'s callers already expect, or None
    when this paper does not answer (explicitly, or by citing nothing)."""
    if not passages:
        return None
    by_chunk = {str(p.chunk_id): p for p in passages}
    prompt = (f"HIGHLIGHTED LESSON TEXT: {selection}\nREQUEST: {request}\n\n"
              "PAPER TEXT, by chunk id:\n"
              + "\n\n".join(f"[{p.chunk_id}] {p.text}" for p in passages))
    text = await llm.ask(PAPER_ANSWER_SYSTEM, prompt, default=1_500, cap=0,
                         stage="learn_expand_paper")
    if not text or text.upper().startswith("INSUFFICIENT"):
        return None
    raw_sections = _parse_chunk_cited(text, set(by_chunk))
    cited = {k for _, sents in raw_sections for s in sents for k in s["claims"]}
    if not cited:
        return None
    written = sum(len(sents) for _, sents in raw_sections)
    stats = {"written": written, "kept_first_pass": len(cited), "repaired": 0,
             "dropped": written - len(cited),
             "grounded_pct": round(100 * len(cited) / written) if written else 0}
    sections = [{"heading": h, "sentences": s} for h, s in raw_sections if s]
    claims = [_pseudo_claim(k, {"chunk_id": by_chunk[k].chunk_id, "arxiv_id": by_chunk[k].arxiv_id,
                                "title": by_chunk[k].title, "section": by_chunk[k].section,
                                "text": by_chunk[k].text, "claim": by_chunk[k].text})
             for k in cited]
    return sections, stats, claims


async def _paper_tier(llm: Llm, corpus, context: list[dict], selection: str, request: str):
    """Tier 2: read up to `FULL_PAPER_CAP` of the lesson's own most relevant cited papers in
    full, one at a time, stopping at the first that actually answers -- cheaper than reading all
    of them when the first candidate already does, which is this whole module's point (cheapest
    pass first). Entirely within `lara.learn`: `corpus.full_paper()` already assembles a paper's
    chunks straight from the corpus (`lara/learn/passages.py`), so unlike tier 3 below, this
    tier needs no `lara.serve` capability injected in.

    Returns (answer, claims) -- `answer` in `_answer()`'s own (sections, stats) shape -- or
    (None, []) when none of the candidate papers can answer."""
    if corpus is None:
        return None, []
    for arxiv_id, version in _ranked_papers(context):
        passages = await asyncio.to_thread(corpus.full_paper, arxiv_id, version)
        result = await _paper_answer(llm, passages, selection, request)
        if result is not None:
            sections, stats, claims = result
            return (sections, stats), claims
    return None, []


async def expand(llm: Llm, corpus, concept: dict, content: dict, *, selection: str,
                 question: str = "", selection_claims=(), section: int = 0, embed=None,
                 lesson_generated=None, synth=None) -> dict:
    """The expansion to store, or {"insufficient": True, "message": ...} when nothing verifiable
    could be said.

    Escalates cheapest-first when the concept's own claims cannot add anything: tier 2
    (`_paper_tier`) reads the fuller text of the most relevant paper(s) already cited before
    tier 3 searches the general corpus. `synth`, when given (lara.serve.learn_research.
    expand_synth), drives tier 3 with one research leaf (`_new_claims_from_research`) instead
    of the old judge-verified `_new_claims`/claims.py `extract()` path. `None` (the default) is
    the old tier-3 behavior exactly, which is what a concept built before the research-driven
    pipeline existed still gets -- tier 2 is unaffected either way, since it needs neither."""
    question = question.strip()
    focus = f"{selection}\n{question}" if question else selection
    request = question or DEFAULT_REQUEST
    live = [c for c in content.get("claims", []) if not c.get("withdrawn")]
    context = rank(live, focus, embed)
    answer = await _answer(llm, selection, request, context)
    searched, new_dicts = False, []
    if not _adds_something(answer, set(selection_claims)):
        searched = True
        paper_answer, paper_claims = await _paper_tier(llm, corpus, context, selection, request)
        if paper_answer is not None and paper_claims:
            answer, new_dicts = paper_answer, paper_claims
        else:
            n = _next_id(content.get("expansions", []))
            if synth is not None:
                new_dicts = await _new_claims_from_research(synth, live, focus, n)
            else:
                new_dicts = [c.to_dict() for c in
                            await _new_claims(llm, corpus, concept, live, focus, n, embed)]
            pool = context + new_dicts
            answer = await _answer(llm, selection, request, pool)
            if answer is None:
                return {"insufficient": True, "message": INSUFFICIENT, "searched": True}
    cited = {k for sec in answer[0] for s in sec["sentences"] for k in s["claims"]}
    kept_new = [c for c in new_dicts if c["key"] in cited]
    return {"id": f"x{_next_id(content.get('expansions', []))}", "section": section,
            "selection": selection[:600], "question": question,
            "answer": {"sections": answer[0], "stats": answer[1]}, "claims": kept_new,
            "searched": searched, "lesson_generated": lesson_generated,
            "stale": False, "ts": time.time()}
