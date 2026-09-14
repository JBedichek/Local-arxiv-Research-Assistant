"""One cited answer from a question and a set of already-gathered source excerpts.

Deliberately thinner than lara.serve.synthesis.run_synthesis: no retrieval, no
saturation loop, no Run/Claim persistence -- this takes sources the caller
already has in hand (lara's own retriever, a web search, anywhere else) and
asks one model call to write an answer that cites them. A caller wanting the
full iterative retrieve/extract/consolidate loop over a paper corpus wants
run_synthesis instead; a caller that already has its evidence and just wants
it written up wants this.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lara.serve.generate import complete_json

SYSTEM = """You are answering a question using only the numbered sources given to you.

Cite every claim you make with the bracket number(s) of the source(s) it came from,
like [1] or [2][3]. Do not cite a source for a claim it does not support. If the
sources do not answer the question, say so plainly rather than guessing.

Reply as JSON: {"answer": "...", "cited_ids": [1, 2, ...]}
`answer` is the written answer with inline [N] citations. `cited_ids` lists every
source number actually cited in `answer`, in the order first used."""


@dataclass
class Source:
    """One piece of already-gathered evidence -- a retrieved chunk, a fetched web
    page, a claim from elsewhere. `id` is the caller's own identifier for it
    (survives round-trip onto SynthesisAnswer.cited), not the [N] bracket number
    -- that's assigned positionally when sources are numbered for the prompt."""
    id: str
    text: str
    title: str = ""


@dataclass
class SynthesisAnswer:
    text: str
    cited: list[Source]
    raw: dict = field(default_factory=dict)


def numbered_sources(sources: list[Source], char_cap: int = 1500) -> str:
    """[N] title\\ntext, one per source, 1-indexed -- the same bracket-citation
    convention lara.serve.synthesis._numbered uses, so a reader used to one
    output recognizes the other."""
    lines = []
    for i, s in enumerate(sources, 1):
        head = f"[{i}] {s.title}".rstrip() if s.title else f"[{i}]"
        lines.append(f"{head}\n{s.text[:char_cap]}")
    return "\n\n".join(lines)


def resolve_citations(cited_ids: list, sources: list[Source]) -> list[Source]:
    """Maps the model's 1-indexed [N] citations back to the Source objects they
    name, dropping anything out of range or non-integer -- a model reply is
    untrusted input, not a guarantee it only cited numbers that exist."""
    resolved = []
    for n in cited_ids:
        idx = n - 1 if isinstance(n, int) else None
        if idx is not None and 0 <= idx < len(sources):
            resolved.append(sources[idx])
    return resolved


async def synthesize(cfg, question: str, sources: list[Source], *, model: str | None = None,
                     temperature: float = 0.0, max_tokens: int = 900) -> SynthesisAnswer:
    """One model call: question + numbered sources in, a cited answer out.

    Returns an empty answer with no citations (not an exception) if the source
    list is empty or the model call fails/returns unparseable JSON -- the same
    "failure is a value" convention generate.complete_json's callers already
    use, since a synthesis failure shouldn't crash whatever asked for it."""
    if not sources:
        return SynthesisAnswer(text="", cited=[])

    prompt = f"Question: {question}\n\nSources:\n{numbered_sources(sources)}"
    result = await complete_json(cfg, prompt, system=SYSTEM, model=model,
                                 temperature=temperature, max_tokens=max_tokens,
                                 default={})
    result = result if isinstance(result, dict) else {}
    answer = result.get("answer", "")
    cited = resolve_citations(result.get("cited_ids", []), sources)

    return SynthesisAnswer(text=answer, cited=cited, raw=result)
