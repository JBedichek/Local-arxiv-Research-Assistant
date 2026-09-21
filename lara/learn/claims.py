"""Claims: what the sources actually say about a concept, and how they relate.

Each claim is extracted from one passage and re-checked against it by the judge, so a claim
the model paraphrased beyond its source never enters. Claims from different papers are then
compared pairwise: agreement corroborates, opposite conclusions under the same conditions are
a conflict (the newer one supersedes), and different conclusions explained by different
conditions are a scope difference -- taught as such, not as a contradiction."""

from __future__ import annotations

import asyncio
import re
from dataclasses import asdict, dataclass, field
from datetime import date

from lara.learn import judge as J
from lara.learn.llm import Llm
from lara.learn.passages import Passage

PASSAGES_PER_CONCEPT = 12
MAX_PER_PAPER = 2
MIN_PASSAGE_CHARS = 200
MAX_PAIR_CHECKS = 40
#: A pair of claims is worth a comparison call above either similarity.
PAIR_WORD_OVERLAP = 0.25
PAIR_COSINE = 0.65
#: Opposite conclusions this many days apart: the newer supersedes the older.
SUPERSEDE_DAYS = 180
DUPLICATE_OVERLAP = 0.8
#: Two claims from one paper this close (embedding cosine) are put to the judge as possible
#: repeats -- measured on real output, repeats scored 0.73-0.79 and the nearest non-repeat 0.63.
SAME_PAPER_COSINE = 0.62

EXTRACT_SYSTEM = """You extract atomic claims about a concept from numbered passages of \
research papers.

Reply with JSON only: [{"passage": N, "claim": "...", "conditions": "...", "kind": "finding"|"hypothesis"}]

- Each claim must be stated by the passage it cites; keep every number and qualifier.
- Self-contained: name the method or quantity, never "this", "they" or "the proposed method".
- conditions: the setting it holds in (model size, data, method), or "" if none is stated.
- kind: "finding" if the paper reports or proves it, "hypothesis" if it only proposes or speculates.
- Skip background the passage merely cites from elsewhere, and anything not about the concept.
- LEARNER'S GOAL is given: skip claims that only matter in a setting unrelated to it (a different \
field or application, or the internals of one software package). A claim must help someone \
pursuing that goal understand the concept.
- At most 3 claims per passage."""

_STOP = frozenset("the and for that with this from are was were has have not but can may their "
                  "its into than more which when also such these those over under between".split())


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOP}


def overlap(a: str, b: str) -> float:
    wa, wb = words(a), words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = sum(x * x for x in a) ** 0.5, sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class Claim:
    key: str
    text: str
    passage: dict                      # Passage.to_dict() of the source
    conditions: str = ""
    kind: str = "finding"
    corroborated_by: list[str] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)   # {"with", "relation", "note"}
    superseded_by: str = ""
    flags: list[dict] = field(default_factory=list)

    @property
    def paper(self) -> str:
        return self.passage.get("arxiv_id", "")

    @property
    def date(self) -> str:
        return self.passage.get("date", "")

    @property
    def certainty(self) -> str:
        if self.superseded_by:
            return "superseded"
        if any(c["relation"] == J.CONTRADICT for c in self.conflicts):
            return "contested"
        if self.kind == "hypothesis":
            return "speculative"
        return "established" if self.corroborated_by else "single-source"

    def to_dict(self) -> dict:
        return {**asdict(self), "certainty": self.certainty}

    @classmethod
    def from_dict(cls, d: dict) -> "Claim":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _days_apart(a: str, b: str) -> int:
    try:
        return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)
    except ValueError:
        return 0


async def gather_passages(corpus, concept: dict, *, per_query: int = 8, focus: str = "",
                          exclude: frozenset = frozenset(),
                          limit: int = PASSAGES_PER_CONCEPT) -> list[Passage]:
    """Distinct, substantial passages for a concept, at most `MAX_PER_PAPER` from one paper so
    corroboration means independent papers. `focus` (a highlighted passage and question)
    steers the search toward it; `exclude` holds chunk ids already used, so a follow-up
    search finds something new."""
    seen, per_paper, out = set(), {}, []
    goal = concept.get("goal", "")[:160]
    if focus:
        queries = [focus[:240], f"{focus[:140]} {concept['title']}"]
    else:
        queries = [concept["title"], f"{concept['title']} {concept.get('summary', '')}".strip()]
    if goal:
        queries.append(f"{concept['title']} for {goal}")
    for query in queries:
        for p in await asyncio.to_thread(corpus.search, query, per_query):
            if p.key in seen or len(p.text) < MIN_PASSAGE_CHARS or p.chunk_id in exclude:
                continue
            if per_paper.get(p.arxiv_id, 0) >= MAX_PER_PAPER:
                continue
            seen.add(p.key)
            per_paper[p.arxiv_id] = per_paper.get(p.arxiv_id, 0) + 1
            out.append(p)
    return out[:limit]


async def _merge_repeats(llm: Llm, items: list[tuple], embed) -> tuple[list[tuple], int]:
    """Drops later claims a paper makes twice in different words (abstract and conclusion
    usually). Similarity only picks the pairs; the judge decides whether they are redundant."""
    vecs = [embed(t[1]) if embed else [] for t in items]
    pairs = [(i, j) for i in range(len(items)) for j in range(i + 1, len(items))
             if items[i][0].arxiv_id == items[j][0].arxiv_id
             and (overlap(items[i][1], items[j][1]) >= PAIR_WORD_OVERLAP
                  or (vecs[i] and vecs[j] and cosine(vecs[i], vecs[j]) >= SAME_PAPER_COSINE))]
    answers = await asyncio.gather(*(J.same(llm, items[i][1], items[j][1]) for i, j in pairs))
    drop: set[int] = set()
    for (i, j), redundant in zip(pairs, answers):
        if redundant and i not in drop:
            drop.add(j)
    return [t for k, t in enumerate(items) if k not in drop], len(drop)


async def extract(llm: Llm, concept: dict, passages: list[Passage], *, embed=None,
                  focus: str = "", first_key: int = 1) -> tuple[list[Claim], int, int, int]:
    """`focus` asks for claims that speak to a specific request; `first_key` numbers them from
    there (so extra claims added later do not collide with the concept's own).

    (verified claims, count dropped as unfaithful to their passage, count merged as
    repeats of another claim from the same paper, count dropped as not serving the
    learner's goal)."""
    if not passages:
        return [], 0, 0, 0
    body = "\n\n".join(f"[{i}] {p.text}" for i, p in enumerate(passages, 1))
    data = await llm.ask_json(EXTRACT_SYSTEM,
                              f"LEARNER'S GOAL: {concept.get('goal') or '(not given)'}\n"
                              f"CONCEPT: {concept['title']}\n"
                              + (f"FOCUS (extract only claims that speak to this): {focus}\n" if focus else "")
                              + f"\nPASSAGES:\n{body}",
                              default=1_500, cap=6_000, stage="learn_claims")
    candidates = []
    for item in data if isinstance(data, list) else []:
        try:
            p = passages[int(item["passage"]) - 1]
            text = str(item["claim"]).strip()
        except (KeyError, ValueError, TypeError, IndexError):
            continue
        # Near-identical claims from different papers are corroboration, kept on purpose.
        if text and all(c[0].arxiv_id != p.arxiv_id or overlap(text, c[1]) < DUPLICATE_OVERLAP
                        for c in candidates):
            kind = "hypothesis" if str(item.get("kind")) == "hypothesis" else "finding"
            candidates.append((p, text, str(item.get("conditions") or "").strip(), kind))
    verdicts = await asyncio.gather(*(J.judge(llm, text, p.text) for p, text, _, _ in candidates))
    faithful = [c for c, v in zip(candidates, verdicts) if v == J.SUPPORTS]
    dropped = len(candidates) - len(faithful)
    goal = concept.get("goal", "")
    useful = await asyncio.gather(*(J.relevant(llm, goal, concept["title"], c[1]) for c in faithful))
    kept = [c for c, ok in zip(faithful, useful) if ok]
    off_topic = len(faithful) - len(kept)
    unique, merged = await _merge_repeats(llm, kept, embed)
    claims = [Claim(key=f"c{n}", text=text, passage=p.to_dict(), conditions=cond, kind=kind)
              for n, (p, text, cond, kind) in enumerate(unique, first_key)]
    return claims, dropped, merged, off_topic


async def relate(llm: Llm, claims: list[Claim], *, embed=None, involving: set[str] | None = None) -> int:
    """Compares claims from different papers and records agreement, conflict, scope
    differences and supersession on them in place. `involving` limits it to pairs that
    include one of those keys (claims added later, against everything already known).
    Returns the comparisons made."""
    vecs = {c.key: embed(c.text) for c in claims} if embed else {}
    pairs = []
    for i, a in enumerate(claims):
        for b in claims[i + 1:]:
            if a.paper == b.paper or (involving is not None and a.key not in involving and b.key not in involving):
                continue
            ov = overlap(a.text, b.text)
            cos = cosine(vecs[a.key], vecs[b.key]) if vecs.get(a.key) and vecs.get(b.key) else 0.0
            if ov >= PAIR_WORD_OVERLAP or cos >= PAIR_COSINE:
                pairs.append((max(ov, cos), a, b))
    pairs = sorted(pairs, key=lambda t: t[0], reverse=True)[:MAX_PAIR_CHECKS]
    results = await asyncio.gather(*(
        J.compare(llm, a.text, a.passage["text"], b.text, b.passage["text"]) for _, a, b in pairs))
    for (_, a, b), (relation, note) in zip(pairs, results):
        if relation == J.AGREE:
            a.corroborated_by.append(b.key)
            b.corroborated_by.append(a.key)
        elif relation in (J.CONTRADICT, J.SCOPE):
            a.conflicts.append({"with": b.key, "relation": relation, "note": note})
            b.conflicts.append({"with": a.key, "relation": relation, "note": note})
            if (relation == J.CONTRADICT and a.kind == b.kind == "finding"
                    and _days_apart(a.date, b.date) >= SUPERSEDE_DAYS):
                older, newer = (a, b) if a.date < b.date else (b, a)
                older.superseded_by = newer.key
    return len(pairs)


def conflicts(claims: list[Claim]) -> list[dict]:
    """Each conflicting pair once, with both sides' text, conditions and dates -- what a
    lesson's "where sources disagree" section is written from."""
    by_key, seen, out = {c.key: c for c in claims}, set(), []
    for c in claims:
        for x in c.conflicts:
            pair = tuple(sorted((c.key, x["with"])))
            if pair in seen or x["with"] not in by_key:
                continue
            seen.add(pair)
            o = by_key[x["with"]]
            out.append({"a": c.key, "b": o.key, "relation": x["relation"], "note": x["note"],
                        "sides": [{"key": s.key, "text": s.text, "conditions": s.conditions,
                                   "date": s.date, "paper": s.passage.get("title", "")}
                                  for s in (c, o)]})
    return out


async def build(llm: Llm, corpus, concept: dict, *, embed=None) -> dict:
    passages = await gather_passages(corpus, concept)
    claims, dropped, merged, off_topic = await extract(llm, concept, passages, embed=embed)
    compared = await relate(llm, claims, embed=embed)
    return {"claims": [c.to_dict() for c in claims], "conflicts": conflicts(claims),
            "stats": {"passages": len(passages), "unfaithful_dropped": dropped,
                      "repeats_merged": merged, "off_topic_dropped": off_topic,
                      "comparisons": compared}}
