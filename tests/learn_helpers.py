"""Scripted model and corpus stand-ins for the learn tests."""

from __future__ import annotations

import json

from lara.learn.llm import Llm
from lara.learn.passages import Passage


def scripted(*rules, default=""):
    """A `complete` that answers by the first rule whose substring is in the system or user
    prompt. A rule's answer may be a string, or a callable(system, prompt) -> string. Every
    call is appended to `.calls` as (system, prompt)."""
    calls = []

    async def complete(cfg, prompt, *, system="", model=None, max_tokens=0):
        calls.append((system, prompt))
        for needle, answer in rules:
            if needle in system or needle in prompt:
                return answer(system, prompt) if callable(answer) else answer
        return default

    complete.calls = calls
    return complete


def llm(*rules, default="") -> Llm:
    c = scripted(*rules, default=default)
    out = Llm(complete=c, window=200_000)
    out.calls = c.calls
    return out


def passage(i=1, text="text", *, arxiv="2401.00001", title="A Paper", date="2024-01-01",
            cited_by=10, journal_ref="", kind="body") -> Passage:
    return Passage(chunk_id=i, arxiv_id=arxiv, title=title, text=text, section="Intro",
                   kind=kind, date=date, cited_by=cited_by, journal_ref=journal_ref)


class FakeCorpus:
    """search() returns the passages registered for the first matching substring. `coverage`
    defaults generous (rich) so existing tests that never call it keep today's unthrottled
    facet/budget behaviour; `neighbours` defaults to no citation edges, so the citation walk
    is a no-op unless a test wires some in. A `papers=` search draws from `by_paper` instead
    of the query-matched set, regardless of query text -- restricting to specific papers is
    what the real retriever's own `papers=` does too (Retriever.retrieve): it narrows the
    candidate pool, it does not change which query text matched it."""

    def __init__(self, by_query=None, default=(), coverage=None, neighbours=None, by_paper=()):
        self.by_query = by_query or {}
        self.default = list(default)
        self.queries = []
        self._coverage = {"chunks": 100, "papers": 20} if coverage is None else coverage
        self._neighbours = neighbours or {}
        self._by_paper = {p.arxiv_id: p for p in by_paper}

    def search(self, query, k=8, papers=None):
        self.queries.append(query)
        if papers is not None:
            return [self._by_paper[a] for a in papers if a in self._by_paper][:k]
        out = self.default
        for needle, found in self.by_query.items():
            if needle in query:
                out = list(found)
                break
        return out[:k]

    def relevance(self, pairs):
        return None

    def coverage(self, query):
        return dict(self._coverage)

    def neighbours(self, arxiv_id):
        found = self._neighbours.get(arxiv_id, {})
        return {"cites": list(found.get("cites", [])), "cited_by": list(found.get("cited_by", []))}


LONG = "warmup avoids loss spikes early in training. " * 6


def corpus():
    return FakeCorpus(default=[passage(1, LONG, arxiv="2401.1", title="A Survey of Training"),
                               passage(2, LONG + " More.", arxiv="2402.2", title="Second paper", date="2024-03-01")])


def model(*extra):
    """`extra` rules are appended after the built-in ones (still checked in order, so a built-in
    needle always wins unless a test's own is more specific) -- lets a test add a topics or
    topic-doc reply without restating this whole fixture."""
    concepts = [{"id": "a", "title": "Warmup", "summary": "s", "passages": [1, 2], "competencies": ["choose-a-schedule"]},
                {"id": "b", "title": "Decay", "summary": "d", "passages": [1], "prereqs": ["a"], "competencies": ["choose-a-schedule"]}]
    claims = [{"passage": 1, "claim": "Warmup avoids early loss spikes.", "conditions": "1B", "kind": "finding"},
              {"passage": 2, "claim": "Warmup avoids loss spikes early in training.", "conditions": "", "kind": "finding"}]
    quiz = [{"type": "mcq", "question": "What does warmup avoid?", "choices": ["spikes", "x", "y", "z"], "answer": "A",
             "claim": "c1", "explanation": "Early spikes."}]
    return llm(("design the scope", json.dumps({"competencies": [{"text": "choose a schedule"}], "question": None})),
               ("concept map", json.dumps({"concepts": concepts})),
               ("extract atomic claims", json.dumps(claims)),
               ("strict fact-checker", "supports"),
               ("compare two claims", '{"relation": "agree", "note": ""}'),
               ("write a lesson", "## Warmup\nWarmup avoids early loss spikes [c1].\nIt is corroborated [c1, c2]."),
               ("Write quiz items", json.dumps(quiz)),
               ("ONLY the passage", "A"),
               ("pick ONE quantity", "null"), ("small diagram", "null"),
               ("review a learner", "[]"), *extra)
