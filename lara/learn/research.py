"""Phase 2 of the research-driven Learn pipeline: one concept's lesson, written directly
from its own full synthesis run -- see lara.serve.learn_research and lara.learn.graph's
own module docstring (build_from_research) on why the run itself is driven from
lara.serve, not here.

Confirmed design: synthesis's own citation writing is used directly, with no independent
judge re-check -- unlike claims.py + lesson.py's extract() -> judge() -> compose() ->
verify() pipeline this replaces. A `Reference` (lara.serve.citations, already resolved to
a plain dict by the time it reaches here) is repackaged as the existing Claim shape
(`_pseudo_claim`) purely for compatibility: the on-disk contract (store.py has none of its
own -- concept files are schema-free JSON) and everything downstream that already reads
that shape (the lesson renderer, pipeline.write_variant's `content.get("claims")` gate,
and -- until a follow-up rewrites them against Reference directly -- quiz.py/visuals.py)
keep working. What changed is only how a "claim" is produced, not the shape one is in:
certainty is always what Claim's own certainty property computes from an empty
corroborated_by/conflicts (`"single-source"`), since no relate()-style comparison pass
runs here."""
from __future__ import annotations

import re
import time

from lara.learn import trace as TR

#: A lesson this thin is not one a learner should be handed -- the same floor
#: lesson.compose's MIN_CLAIMS uses, applied here to citations actually used in the text
#: rather than to a claims list, since there is no separate claims list at this stage.
MIN_CITED = 2

#: A citation bracket, matching lara.serve.citations.CITATION -- duplicated, not
#: imported: lara.learn never imports lara.serve (see passages.CorpusRetriever's own
#: docstring), and this is a two-line regex, not real coupling to that module's
#: Reference/bind machinery. Every Reference this module ever sees was already resolved
#: there before it arrived, as a plain dict.
_CITE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
_CITE_KEYS = re.compile(r"\d+")


def cited_keys(text: str) -> list[str]:
    """Every citation key in `text`, in order of first appearance, deduplicated."""
    keys: list[str] = []
    for m in _CITE.finditer(text or ""):
        keys.extend(_CITE_KEYS.findall(m.group(0)))
    return list(dict.fromkeys(keys))


def _parse_sections(text: str, known: set[str]) -> list[tuple[str, list[dict]]]:
    """[(heading, [{"text", "claims"}])] -- the same shape and stripping behaviour
    lesson.parse() has always produced (so the existing lesson renderer needs no change),
    built against chunk-id citation brackets instead of lesson.py's `c1`/`c2` keys. Unlike
    parse(), every line is kept regardless of whether it cites anything: there is no
    verify() pass here to decide an uncited line does not belong (see the module
    docstring) -- a transition sentence with nothing to cite is still something the model
    legitimately wrote, not a dropped one."""
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
        keys = [k for k in cited_keys(line) if k in known]
        clean = re.sub(r"\s+", " ", _CITE.sub("", line)).replace(" .", ".").strip()
        sections[-1][1].append({"text": clean, "claims": keys})
    return sections


def _pseudo_claim(key: str, ref: dict) -> dict:
    return {"key": key, "text": ref.get("claim") or ref.get("text", ""),
           "passage": {"chunk_id": ref.get("chunk_id"), "arxiv_id": ref.get("arxiv_id"),
                       "title": ref.get("paper_title") or ref.get("title", ""),
                       "section": ref.get("section", ""), "text": ref.get("text", "")},
           "conditions": "", "kind": "finding", "corroborated_by": [], "conflicts": [],
           "superseded_by": "", "flags": [], "certainty": "single-source"}


def _insufficient(result: dict) -> dict:
    lesson = {"insufficient": True, "sections": [], "generated": time.time(),
             "message": "The paper corpus holds too little verifiable material on this "
                        "concept to teach it responsibly.",
             "stats": {"written": 0, "kept_first_pass": 0, "repaired": 0, "dropped": 0,
                      "grounded_pct": 0}}
    return {"claims": [], "conflicts": [], "facets": [], "lesson": lesson, "references": {},
           "stats": {"claims": 0, "papers": 0, "comparisons": 0,
                    "degraded": result.get("degraded", False)},
           "trace": {"tokens_in": result.get("tokens_in", 0),
                    "tokens_out": result.get("tokens_out", 0),
                    "rounds": result.get("rounds", 0)}}


def _lesson_objective(concept: dict, *, profile_digest: str = "") -> str:
    objective = (f"{concept['title']} -- {concept.get('summary', '')}\n\n"
                f"Write this as a lesson for a learner whose goal is: "
                f"{concept.get('goal') or '(not given)'}")
    if profile_digest:
        # See pipeline.build_concept's own call site: `profile_digest` is this learner's
        # whole cross-course knowledge digest (lara.learn.profile.digest), unfiltered --
        # left to the model to judge what is actually relevant to this one concept rather
        # than pre-matched by title, which is the fragile part (see profile.py's own
        # docstring on why a concept there is only ever matched by title, inexactly).
        objective += (
            "\n\nWhat this learner has already demonstrated on quizzes in other lessons: "
            f"{profile_digest}\n\nWhere that is directly relevant to this concept, state "
            "it briefly without re-deriving it from scratch, and spend the room you save "
            "going deeper on what they have not yet shown understanding of. Where it is "
            "not relevant, ignore it.")
    return objective


def _lesson_from_result(result: dict) -> dict:
    """What the "claims" stage's `content.update()`s with, from a `synth`/`revise` result
    -- the shared tail of `build_lesson` (a fresh run) and `revise_lesson` (a resumed
    one). Populating `content["lesson"]` here (not left to the "lesson" stage) is what
    lets pipeline's own `_stage_done` see the lesson already done and skip
    depth.deepen() entirely -- no change needed to pipeline's stage loop itself."""
    text = (result.get("deliverable") or "").strip()
    refs = result.get("references") or {}
    if not text or not refs:
        return _insufficient(result)

    sections = _parse_sections(text, set(refs))
    cited = {k for _, sents in sections for s in sents for k in s["claims"]}
    if len(cited) < MIN_CITED:
        return _insufficient(result)

    claims = [_pseudo_claim(k, refs[k]) for k in cited if k in refs]
    written = sum(len(sents) for _, sents in sections)
    grounded = sum(1 for _, sents in sections for s in sents if s["claims"])
    lesson = {"insufficient": False,
             "sections": [{"heading": h, "sentences": s} for h, s in sections if s],
             "generated": time.time(), "variant": "standard",
             "stats": {"written": written, "kept_first_pass": grounded, "repaired": 0,
                      "dropped": written - grounded,
                      "grounded_pct": round(100 * grounded / written) if written else 0}}
    return {"claims": claims, "conflicts": [], "facets": [], "lesson": lesson,
           "references": refs,
           "stats": {"claims": len(claims),
                    "papers": len({c["passage"]["arxiv_id"] for c in claims}),
                    "comparisons": 0, "degraded": result.get("degraded", False)},
           "trace": {"tokens_in": result.get("tokens_in", 0),
                    "tokens_out": result.get("tokens_out", 0),
                    "rounds": result.get("rounds", 0)}}


async def build_lesson(concept: dict, *, synth, profile_digest: str = "") -> dict:
    """Phase 2: one full synthesis run scoped to this concept, its own citation writing
    becoming the lesson directly -- replaces claims.build() + depth.deepen()'s standard-
    lesson path (see pipeline.build_concept's "claims" stage).

    `synth` is the injected capability (lara.serve.learn_research.lesson_synth) --
    `lara.learn` never imports `lara.serve` directly, see graph.build_from_research's own
    docstring for the same boundary. `profile_digest` -- see `_lesson_objective` -- is
    only ever given on this, the first build of a lesson; `revise_lesson` below never
    takes one, since a revision is already a specific, human-directed instruction rather
    than this general personalization."""
    TR.set_phase(f"lesson_research: {concept['title']}")
    result = await synth(_lesson_objective(concept, profile_digest=profile_digest))
    TR.emit("lesson_research", tokens_in=result.get("tokens_in", 0),
           tokens_out=result.get("tokens_out", 0), degraded=result.get("degraded"))
    return _lesson_from_result(result)


async def revise_lesson(concept: dict, feedback: str, *, revise) -> dict:
    """Phase 2 revision: rewrites this concept's lesson from a learner's feedback on it,
    by resuming the persisted concept-scoped synthesis graph (`revise`, injected -- see
    lara.serve.learn_research.lesson_revise) instead of researching from scratch. Same
    return shape `build_lesson` gives, so `pipeline.revise_lesson` applies it the same
    way."""
    TR.set_phase(f"lesson_revision: {concept['title']}")
    result = await revise(feedback)
    TR.emit("lesson_revision", tokens_in=result.get("tokens_in", 0),
           tokens_out=result.get("tokens_out", 0), degraded=result.get("degraded"))
    return _lesson_from_result(result)
