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
    """search() returns the passages registered for the first matching substring."""

    def __init__(self, by_query=None, default=()):
        self.by_query = by_query or {}
        self.default = list(default)
        self.queries = []

    def search(self, query, k=8):
        self.queries.append(query)
        for needle, found in self.by_query.items():
            if needle in query:
                return list(found)[:k]
        return self.default[:k]

    def relevance(self, pairs):
        return None


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
