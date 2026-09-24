"""A lesson is prose written only from verified claims. Every sentence must cite the claims it
rests on, and is then re-checked against exactly those claims by the judge: a sentence that
adds anything the claims do not say is rewritten once, then dropped. When the corpus holds too
little on a concept, the lesson says so instead of padding."""

from __future__ import annotations

import asyncio
import re
import time

from lara.learn import judge as J
from lara.learn.llm import Llm

MIN_CLAIMS = 2
#: A standard lesson is one flat prompt and one generation, read in one sitting -- not a
#: retrieval-depth cap (facets/rounds/citation-walk/full-paper stay as thorough as the corpus
#: earns, see claims.py), just how much one lesson tries to teach at once. usable() sorts
#: strongest-first, so this keeps the best claims, not an arbitrary slice. A deep/"thorough"
#: lesson (depth.py) fans claims out across many sections instead and does not go through
#: this cap -- it calls usable() itself, not compose()'s capped copy.
MAX_LESSON_CLAIMS = 40

LESSON_SYSTEM = """You write a lesson on one concept for a learner, using ONLY the numbered \
claims provided.

Format: "## Heading" lines, and under each heading ONE SENTENCE PER LINE. Every sentence ends, \
before its full stop, with the keys of the claims it rests on in brackets, like [c1] or [c2, c5]. \
A sentence with no key is not allowed.

- Do not add any fact that is not in the claims. Connecting or explaining claims is fine only \
if every part is supported by a claim you cite.
- State each idea once, even when several claims support it -- cite them together.
- Order for a learner: what the concept is, then how it works, then what is known about using it.
- Convey certainty as marked: say when a point rests on one paper, is only a hypothesis, or was \
replaced by later work.
- Where CONFLICTS are listed, give a section "## Where sources disagree" stating both sides with \
their conditions and dates. Do not pick a winner.
- The learner already knows the prerequisites listed; do not re-teach them."""

REPAIR_SYSTEM = """Each numbered sentence below was NOT fully supported by the claims it cites. \
Rewrite each so it states only what those claims support, keeping the citation keys, or reply \
DROP for it. Reply with JSON only: [{"n": 1, "text": "... [c1]."}, {"n": 2, "text": "DROP"}]"""

#: A lesson's claims are c1, c2, ...; the ones an expansion adds are x1c1, x1c2, ...
_KEY = r"(?:c\d+|x\d+c\d+)"
_KEYS = re.compile(rf"\[\s*({_KEY}(?:\s*,\s*{_KEY})*)\s*\]")


def line(claim: dict) -> str:
    bits = [claim["certainty"], claim["passage"].get("date", "")]
    if claim.get("conditions"):
        bits.append(f"conditions: {claim['conditions']}")
    return f"[{claim['key']}] ({'; '.join(b for b in bits if b)}) {claim['text']}"


def usable(claims: list[dict]) -> list[dict]:
    """Claims a lesson may teach from: not withdrawn, strongest first. No count cap -- a lesson
    uses everything the corpus supported, not a fixed number picked in advance."""
    rank = {"established": 0, "single-source": 1, "contested": 2, "speculative": 3, "superseded": 4}
    live = [c for c in claims if not c.get("withdrawn")]
    return sorted(live, key=lambda c: rank.get(c["certainty"], 5))


def parse(text: str, known: set[str]) -> list[tuple[str, list[dict]]]:
    """[(heading, [{"text", "claims"}])] -- sentences with no valid key are kept with an empty
    `claims` list so the caller can count them as ungrounded."""
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
        keys = [k.strip() for m in _KEYS.findall(line) for k in m.split(",")]
        clean = re.sub(r"\s+", " ", _KEYS.sub("", line)).replace(" .", ".").strip()
        sections[-1][1].append({"text": clean, "claims": [k for k in dict.fromkeys(keys) if k in known]})
    return sections


def evidence(sentence: dict, by_key: dict[str, dict]) -> str:
    return "\n".join(by_key[k]["text"] + (f" (conditions: {by_key[k]['conditions']})"
                                          if by_key[k].get("conditions") else "")
                     for k in sentence["claims"])


TLDR_WORDS = 150
TLDR_WORDS_PER_CLAIM = 35


def tldr_note(n_claims: int) -> str:
    """Length guidance for a TL;DR: about 150 words, but never more than the claims can carry,
    so a concept with three claims gets a short one rather than a padded one."""
    words = min(TLDR_WORDS, TLDR_WORDS_PER_CLAIM * max(n_claims, 1))
    return (f"LENGTH: this is a TL;DR of at most about {words} words -- only the most important "
            "points, in one section, no background. Shorter is right if the claims say less.")


def lessons_of(content: dict) -> dict[str, dict]:
    """Every written version of a concept's lesson, by variant: the standard one plus any
    TL;DR, thorough or custom-length ones."""
    out = {"standard": content["lesson"]} if content.get("lesson") else {}
    return {**out, **(content.get("lessons") or {})}


async def compose(llm: Llm, concept: dict, claims: list[dict], conflicts: list[dict],
                  prereq_titles: list[str], *, length_note: str = "") -> dict:
    claims = usable(claims)[:MAX_LESSON_CLAIMS]
    if len(claims) < MIN_CLAIMS:
        return {"insufficient": True, "sections": [], "generated": time.time(),
                "message": "The paper corpus holds too little verifiable material on this "
                           "concept to teach it responsibly.",
                "stats": {"written": 0, "kept_first_pass": 0, "repaired": 0, "dropped": 0,
                          "grounded_pct": 0}}
    by_key = {c["key"]: c for c in claims}
    listing = "\n".join(line(c) for c in claims)
    disputes = "\n".join(f"- [{x['a']}] vs [{x['b']}] ({x['relation']}): {x['note']}"
                         for x in conflicts if x["a"] in by_key and x["b"] in by_key) or "(none)"
    prompt = (f"CONCEPT: {concept['title']} -- {concept.get('summary', '')}\n"
              f"PREREQUISITES ALREADY KNOWN: {', '.join(prereq_titles) or '(none)'}\n\n"
              f"CLAIMS:\n{listing}\n\nCONFLICTS:\n{disputes}"
              + (f"\n\n{length_note}" if length_note else ""))
    text = await llm.ask(LESSON_SYSTEM, prompt, default=2_500, cap=8_000, stage="learn_lesson")
    sections = parse(text, set(by_key))

    out, stats = await verify(llm, sections, by_key)
    return {"insufficient": False, "sections": out, "generated": time.time(), "stats": stats}


async def verify(llm: Llm, sections: list[tuple[str, list[dict]]],
                 by_key: dict[str, dict]) -> tuple[list[dict], dict]:
    """Judges every parsed sentence against the claims it cites; rewrites the failures once;
    returns the sections that survive and the counts. Shared by lessons and expansions."""
    flat = [(i, s) for i, (_, sents) in enumerate(sections) for s in sents]
    verdicts = await asyncio.gather(*(
        J.judge(llm, s["text"], evidence(s, by_key)) if s["claims"] else _ungrounded()
        for _, s in flat))
    failed = [(idx, s) for (idx, s), v in zip(flat, verdicts) if v == J.UNRELATED and s["claims"]]
    keep = {id(s) for (_, s), v in zip(flat, verdicts) if v == J.SUPPORTS}
    written, first_pass = len(flat), len(keep)

    repaired = await _repair(llm, [s for _, s in failed], by_key)
    keep |= {id(s) for s in repaired}
    out = []
    for heading, sents in sections:
        kept = [{**s, "repaired": id(s) in {id(r) for r in repaired}} for s in sents if id(s) in keep]
        if kept:
            out.append({"heading": heading, "sentences": kept})
    final = sum(len(s["sentences"]) for s in out)
    return out, {"written": written, "kept_first_pass": first_pass,
                 "repaired": final - first_pass, "dropped": written - final,
                 "grounded_pct": round(100 * first_pass / written) if written else 0}


async def _ungrounded() -> str:
    return J.UNRELATED


async def _repair(llm: Llm, failed: list[dict], by_key: dict[str, dict]) -> list[dict]:
    """Rewrites failed sentences in place; returns those whose rewrite the judge accepts."""
    if not failed:
        return []
    listing = "\n".join(f"{n}. {s['text']} [{', '.join(s['claims'])}]" for n, s in enumerate(failed, 1))
    claims = "\n".join(line(c) for c in by_key.values())
    data = await llm.ask_json(REPAIR_SYSTEM, f"CLAIMS:\n{claims}\n\nSENTENCES:\n{listing}",
                              default=800, cap=3_000, stage="learn_repair")
    rewrites = {}
    for item in data if isinstance(data, list) else []:
        try:
            n, text = int(item["n"]), str(item["text"]).strip()
        except (KeyError, ValueError, TypeError):
            continue
        if 1 <= n <= len(failed) and text.upper().rstrip(".") != "DROP":
            rewrites[n] = text
    candidates = []
    for n, text in rewrites.items():
        parsed = parse(text, set(by_key))
        sentence = parsed[0][1][0] if parsed and parsed[0][1] else {"text": "", "claims": []}
        candidates.append((failed[n - 1], sentence))
    ok = []
    verdicts = await asyncio.gather(*(J.judge(llm, p["text"], evidence(p, by_key))
                                      if p["claims"] and p["text"] else _ungrounded()
                                      for _, p in candidates))
    for (s, p), v in zip(candidates, verdicts):
        if v == J.SUPPORTS:
            s["text"], s["claims"] = p["text"], p["claims"]
            ok.append(s)
    return ok

