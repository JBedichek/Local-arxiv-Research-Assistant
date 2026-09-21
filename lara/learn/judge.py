"""The independent checks everything else rests on: does this passage support this claim,
do two claims agree, what does a source answer.

An LLM judge, not the reranker: measured on this corpus the reranker scores an unrelated
sentence ("the capital of France is Paris") within 0.07 of a supported one and above a
contradicted one, so it cannot tell support from relevance. Each check sees only what it
needs -- a claim and a passage -- and never the intended answer.
"""

from __future__ import annotations

import re

from lara.learn.llm import Llm

SUPPORTS, CONTRADICTS, UNRELATED = "supports", "contradicts", "unrelated"
AGREE, CONTRADICT, SCOPE = "agree", "contradict", "scope"

JUDGE_SYSTEM = """You are a strict fact-checker. You are given a CLAIM and a PASSAGE.

- supports: the passage states the claim or directly entails it, including every number, \
qualifier and condition the claim includes.
- contradicts: the passage states something incompatible with the claim.
- unrelated: anything else -- including when the claim adds a detail, number or generality \
the passage does not state.

Judge only from the passage; ignore what you know. Reply with exactly one word: \
supports, contradicts or unrelated."""

COMPARE_SYSTEM = """You compare two claims from different papers on the same topic. Each is \
given with its source passage.

Reply with JSON only: {"relation": "agree"|"contradict"|"scope"|"unrelated", "note": "..."}

- agree: the same conclusion under compatible conditions.
- contradict: opposite conclusions under the same conditions.
- scope: conclusions that look different but are explained by different conditions (model \
size, data, method, setting). The note must name what differs.
- unrelated: not about the same question.
The note is one short sentence."""

SAME_SYSTEM = """Do these two claims say essentially the same thing, so that one adds no information \
the other lacks? Ignore wording. A claim that is more general or more specific than the other, \
or that reports a different result, is different. Reply with exactly one word: same or different."""

RELEVANT_SYSTEM = """A learner has a goal and is studying one concept of it. Would this claim help \
them understand the concept AS IT APPLIES TO THEIR GOAL? A true claim about a different field, \
model family or application than their goal's, or about one software package's internals, does \
not. Reply with exactly one word: yes or no."""

SOLVE_SYSTEM = """Answer the question using ONLY the passage. If the passage does not settle it, \
reply exactly: CANNOT DETERMINE. Otherwise reply with just the answer -- for a multiple-choice \
question, the single letter of the correct choice."""


async def judge(llm: Llm, claim: str, passage: str) -> str:
    """supports / contradicts / unrelated. Anything unparseable is `unrelated`: a check that
    could not be read never counts as support."""
    if not claim.strip() or not passage.strip():
        return UNRELATED
    reply = await llm.ask(JUDGE_SYSTEM, f"CLAIM: {claim}\n\nPASSAGE: {passage}",
                          default=8, cap=64, stage="learn_judge")
    word = re.sub(r"[^a-z]", "", reply.lower().split()[0]) if reply.split() else ""
    return word if word in (SUPPORTS, CONTRADICTS) else UNRELATED


async def compare(llm: Llm, claim_a: str, passage_a: str, claim_b: str,
                  passage_b: str) -> tuple[str, str]:
    """(relation, note) between two claims. `unrelated` when the reply is unreadable."""
    data = await llm.ask_json(
        COMPARE_SYSTEM,
        f"CLAIM A: {claim_a}\nPASSAGE A: {passage_a}\n\nCLAIM B: {claim_b}\nPASSAGE B: {passage_b}",
        default=120, cap=400, stage="learn_compare")
    if not isinstance(data, dict):
        return "unrelated", ""
    relation = str(data.get("relation", "")).lower()
    if relation not in (AGREE, CONTRADICT, SCOPE):
        return "unrelated", ""
    return relation, str(data.get("note", "")).strip()


async def same(llm: Llm, claim_a: str, claim_b: str) -> bool:
    """True when the judge finds the two claims redundant with each other."""
    reply = await llm.ask(SAME_SYSTEM, f"CLAIM A: {claim_a}\nCLAIM B: {claim_b}", default=8, cap=64,
                          stage="learn_same")
    return reply.lower().startswith("same")


async def relevant(llm: Llm, goal: str, concept: str, claim: str) -> bool:
    """Whether a claim serves the learner's goal. With no goal there is nothing to check
    against, so everything is relevant."""
    if not goal.strip():
        return True
    reply = await llm.ask(RELEVANT_SYSTEM, f"GOAL: {goal}\nCONCEPT: {concept}\nCLAIM: {claim}",
                          default=8, cap=64, stage="learn_relevant")
    return not reply.lower().startswith("no")


async def solve(llm: Llm, question: str, passage: str, choices: list[str] | None = None) -> str | None:
    """What a solver who sees only the passage answers; None when it says it cannot tell."""
    body = f"PASSAGE: {passage}\n\nQUESTION: {question}"
    if choices:
        body += "\n" + "\n".join(f"{chr(65 + i)}. {c}" for i, c in enumerate(choices))
    reply = await llm.ask(SOLVE_SYSTEM, body, default=60, cap=300, stage="learn_solve")
    if not reply or "cannot determine" in reply.lower():
        return None
    return reply
