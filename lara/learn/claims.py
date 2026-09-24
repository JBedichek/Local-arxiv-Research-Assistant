"""Claims: what the sources actually say about a concept, and how they relate.

A concept is researched one facet at a time (what it is, how it works, its evidence, its limits,
...) rather than by one generic search on its title, so a lesson has more than whatever passages
happen to be nearest the title to draw on. How hard each facet is researched is not fixed: a
cheap coverage probe against the corpus first (see `budget`) sizes it to how much is actually
there, and a facet's own top result can pull in its citation neighbours -- what it cites and
what cites it -- not just whatever embeds nearest the query. Each claim is extracted from one
passage and re-checked against it by the judge, so a claim the model paraphrased beyond its
source never enters. Claims from different papers are then compared pairwise: agreement
corroborates, opposite conclusions under the same conditions are a conflict (the newer one
supersedes), and different conclusions explained by different conditions are a scope difference
-- taught as such, not as a contradiction."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date

from lara.learn import judge as J
from lara.learn.llm import Llm
from lara.learn.passages import Passage

MAX_PER_PAPER = 2
MIN_PASSAGE_CHARS = 200
#: Bounds the O(claims^2) candidate pairs relate() will pay an LLM call to compare -- a compute
#: safeguard, not a content limit: pairs are already filtered by similarity before this cuts in.
MAX_PAIR_CHECKS = 80
MAX_FACETS = 6
#: A facet whose own passages support fewer claims than this is under-covered; retried once
#: against a wider slice of the corpus (excluding what every facet has already used) before
#: being accepted as genuinely thin.
MIN_FACET_CLAIMS = 2
#: A pair of claims is worth a comparison call above either similarity.
PAIR_WORD_OVERLAP = 0.25
PAIR_COSINE = 0.65
#: Opposite conclusions this many days apart: the newer supersedes the older.
SUPERSEDE_DAYS = 180
DUPLICATE_OVERLAP = 0.8
#: Two claims from one paper this close (embedding cosine) are put to the judge as possible
#: repeats -- measured on real output, repeats scored 0.73-0.79 and the nearest non-repeat 0.63.
SAME_PAPER_COSINE = 0.62

#: Coverage tiers (distinct papers a cheap FTS probe finds for the concept+goal) that pick how
#: hard retrieval tries -- see `budget`. A thin corpus researched as if it were rich just spends
#: more calls finding the same handful of passages again; a rich one capped as if it were thin
#: leaves real material unread.
COVERAGE_THIN = 3
COVERAGE_RICH = 15
#: (facets, passages per query, citation walk on?) by coverage tier -- thin/typical/rich.
BUDGETS = {"thin": {"facets": 3, "per_query": 6, "citation_walk": False},
          "typical": {"facets": MAX_FACETS, "per_query": 8, "citation_walk": True},
          "rich": {"facets": 8, "per_query": 12, "citation_walk": True}}
#: Citation neighbours (of a facet's own top result) tried per facet, at most.
CITATION_NEIGHBOURS = 6

FACETS_SYSTEM = """You are choosing what a learner needs evidence for, to fully understand one \
concept from a research-paper corpus, before a lesson on it is written.

Reply with JSON only: ["facet query 1", "facet query 2", ...]

- Up to the number of facets asked for, short and specific search queries (not sentences), each \
aimed at a DIFFERENT angle of the concept: what it is, how or why it works, concrete numbers or \
empirical results, the conditions or limits it holds under, how it compares to alternatives -- \
adapted to what actually matters for THIS concept and the learner's goal. Fewer is right when \
the concept genuinely has fewer distinct angles; skip one that does not apply to it.
- Each facet should surface different passages than the others -- do not just reword the \
concept's title several times."""

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


def coverage_tier(coverage: dict) -> str:
    papers = coverage.get("papers", 0)
    if papers < COVERAGE_THIN:
        return "thin"
    if papers >= COVERAGE_RICH:
        return "rich"
    return "typical"


def budget(coverage: dict) -> dict:
    """How hard to research this concept, from a cheap coverage probe (see
    `CorpusRetriever.coverage`) rather than one size for every concept: a corpus that barely
    touches the topic gets fewer, narrower facets -- more would just be more calls spent
    re-finding the same handful of passages, not more material. A corpus rich in it gets more
    facets, a wider net per facet, and a citation walk on top of similarity search."""
    return {"tier": (tier := coverage_tier(coverage)), **BUDGETS[tier]}


async def facets(llm: Llm, concept: dict, *, max_facets: int = MAX_FACETS) -> list[str]:
    """Search queries covering the distinct angles a lesson on this concept needs evidence for.
    Falls back to the concept's own title (the old, single-query behaviour) if the model's reply
    cannot be used, so a bad facet call degrades retrieval rather than failing the build."""
    data = await llm.ask_json(FACETS_SYSTEM,
                              f"LEARNER'S GOAL: {concept.get('goal') or '(not given)'}\n"
                              f"CONCEPT: {concept['title']} -- {concept.get('summary', '')}\n"
                              f"FACETS: up to {max_facets}",
                              default=400, cap=1_200, stage="learn_facets")
    out = [str(f).strip() for f in data][:max_facets] if isinstance(data, list) else []
    return [f for f in out if f] or [concept["title"]]


async def gather_passages(corpus, concept: dict, *, per_query: int = 8, focus: str = "",
                          exclude: frozenset = frozenset()) -> list[Passage]:
    """Distinct, substantial passages for a concept, at most `MAX_PER_PAPER` from one paper so
    corroboration means independent papers. `focus` (a facet query, or a highlighted passage and
    question) steers the search toward it; `exclude` holds chunk ids already used, so a follow-up
    search finds something new. No cap on how many come back -- that is `MAX_PER_PAPER` and
    however many facets are searched, not a fixed number picked in advance."""
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
    return out


async def gather_citation_passages(corpus, focus: str, papers: list[str], *, per_query: int = 8,
                                   exclude: frozenset = frozenset()) -> list[Passage]:
    """One search restricted to specific papers -- a facet's citation neighbours (what its own
    top result cites, and what cites it), not just whatever embeds nearest the query. Same
    passage-quality filters as `gather_passages` (MAX_PER_PAPER, MIN_PASSAGE_CHARS, exclude);
    [] straight away if there is nowhere to restrict the search to."""
    if not papers:
        return []
    seen, per_paper, out = set(), {}, []
    for p in await asyncio.to_thread(corpus.search, focus[:240], per_query, papers=papers):
        if p.key in seen or len(p.text) < MIN_PASSAGE_CHARS or p.chunk_id in exclude:
            continue
        if per_paper.get(p.arxiv_id, 0) >= MAX_PER_PAPER:
            continue
        seen.add(p.key)
        per_paper[p.arxiv_id] = per_paper.get(p.arxiv_id, 0) + 1
        out.append(p)
    return out


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


def _merge_facets(groups: list[list[Claim]]) -> list[Claim]:
    """Every facet's claims, renumbered from c1 and deduplicated against everything kept so far --
    facets often surface overlapping passages, and a passage two facets both found should not
    become two claims. Only a same-paper repeat is dropped here, matching `extract`'s own rule:
    a near-identical claim from a DIFFERENT paper is corroboration, not a repeat, and stays for
    `relate` to record as agreement."""
    seen: list[tuple[str, str]] = []       # (arxiv_id, text) of every claim kept so far
    out: list[Claim] = []
    for group in groups:
        for c in group:
            if all(c.paper != paper or overlap(c.text, text) < DUPLICATE_OVERLAP for paper, text in seen):
                c.key = f"c{len(out) + 1}"
                seen.append((c.paper, c.text))
                out.append(c)
    return out


async def build(llm: Llm, corpus, concept: dict, *, embed=None) -> dict:
    """Researches one facet at a time (see `facets`), sized by a cheap coverage probe (see
    `budget`) instead of one fixed effort for every concept, then widens the search once for
    any facet that came back thin, so a lesson has more than whatever a handful of passages
    nearest the title happened to say. Also returns a `trace`: the coverage probe, the budget
    it picked, and a per-facet record of what was searched and found -- for a build's own
    profiling view, not just its output."""
    t0 = time.time()
    coverage = await asyncio.to_thread(
        corpus.coverage, f"{concept['title']} {concept.get('summary', '')}".strip())
    bud = budget(coverage)
    queries = await facets(llm, concept, max_facets=bud["facets"])

    async def citation_round(facet: str, dense: list[Passage], exclude: frozenset) -> tuple[list[Passage], dict]:
        """The facet's own top result's citation neighbours, searched once for this facet --
        {} straight away when the budget has the citation walk off, or there is nothing to
        walk from yet."""
        if not bud["citation_walk"] or not dense:
            return [], {"tried": 0, "kept": 0}
        nb = await asyncio.to_thread(corpus.neighbours, dense[0].arxiv_id)
        have = {p.arxiv_id for p in dense}
        neighbours = list(dict.fromkeys(
            a for a in (nb.get("cites", []) + nb.get("cited_by", [])) if a not in have))[:CITATION_NEIGHBOURS]
        if not neighbours:
            return [], {"tried": 0, "kept": 0}
        found = await gather_citation_passages(corpus, facet, neighbours, per_query=bud["per_query"], exclude=exclude)
        return found, {"tried": len(neighbours), "kept": len(found)}

    async def one(facet: str, exclude: frozenset, *, widened: bool = False):
        dense = await gather_passages(corpus, concept, focus=facet, exclude=exclude, per_query=bud["per_query"])
        cited, cite_stats = await citation_round(facet, dense, exclude | frozenset(p.chunk_id for p in dense))
        passages = dense + cited
        claims, dropped, merged, off_topic = await extract(llm, concept, passages, embed=embed, focus=facet)
        round_ = {"facet": facet, "widened": widened, "dense_retrieved": len(dense),
                 "citation_papers_tried": cite_stats["tried"], "citation_passages_kept": cite_stats["kept"],
                 "claims": len(claims)}
        return claims, frozenset(p.chunk_id for p in passages), (len(passages), dropped, merged, off_topic), round_

    first = await asyncio.gather(*(one(f, frozenset()) for f in queries))
    used = frozenset(cid for _, exhausted, _, _ in first for cid in exhausted)
    thin = [f for f, (claims, _, _, _) in zip(queries, first) if len(claims) < MIN_FACET_CLAIMS]
    widened = await asyncio.gather(*(one(f, used, widened=True) for f in thin)) if thin else []

    rounds = first + widened
    claims = _merge_facets([c for c, _, _, _ in rounds])
    compared = await relate(llm, claims, embed=embed)
    stats = [s for _, _, s, _ in rounds]
    return {"claims": [c.to_dict() for c in claims], "conflicts": conflicts(claims), "facets": queries,
            "stats": {"passages": sum(s[0] for s in stats), "unfaithful_dropped": sum(s[1] for s in stats),
                      "repeats_merged": sum(s[2] for s in stats), "off_topic_dropped": sum(s[3] for s in stats),
                      "cross_facet_merged": sum(len(g) for g, _, _, _ in rounds) - len(claims),
                      "comparisons": compared, "facets_widened": len(thin)},
            "trace": {"coverage": coverage, "budget": bud, "rounds": [r for _, _, _, r in rounds],
                      "claims": len(claims), "papers": len({c.paper for c in claims}),
                      "comparisons": compared, "ms": round((time.time() - t0) * 1000)}}
