"""Quiz items are generated from claims, then validated by a solver that sees only the
question and the source passage -- never the intended answer. An item the solver cannot
answer from the source, or answers differently, is ambiguous or wrong and is dropped, so every
surviving answer points at a passage."""

from __future__ import annotations

import asyncio
import re

from lara.learn import claims as CL
from lara.learn import judge as J
from lara.learn.llm import Llm

MAX_ITEMS = 10
NUMERIC_TOLERANCE = 0.15

QUIZ_SYSTEM = """Write quiz items that test understanding of the claims below.

Reply with JSON only: [{"type": "mcq"|"short"|"predict", "question": "...", \
"choices": ["...", "...", "...", "..."], "answer": "...", "claim": "c1", "explanation": "..."}]

- mcq: exactly one correct choice; "answer" is its letter A-D. Distractors plausible but \
clearly wrong according to the claim. Omit "choices" for other types.
- short: "answer" is one short phrase stated in the claim.
- predict: only for a claim reporting a number, direction or comparison. The question sets up \
the situation WITHOUT the result and asks the learner to predict it; "answer" is the result.
- Each item tests one claim and must be answerable from that claim's source passage alone, \
with no outside knowledge.
- Write every question and explanation so it stands alone: never mention "the claims", claim \
keys like c1, "the passage", "the text" or "the provided information".
- explanation: one sentence on why the answer is right."""

CRITIQUE_SYSTEM = """You review a learner's plan or answer against numbered claims from the \
literature. Comment ONLY where the claims speak to what the learner wrote.

Reply with JSON only: [{"statement": "<the learner's words>", "verdict": "supported"|"contradicted", \
"claims": ["c1"], "advice": "<one sentence>"}]

Skip anything the claims neither back nor contradict. Never use knowledge outside the claims."""

_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_LETTER_PREFIX = re.compile(r"^\(?[A-Da-d][\).:]\s+")


def _letter(text: str) -> str:
    m = re.match(r"\s*\(?([A-Da-d])\b", text or "")
    return m.group(1).upper() if m else ""


async def generate(llm: Llm, concept: dict, claims: list[dict], *, start: int = 1,
                   limit: int = MAX_ITEMS) -> list[dict]:
    live = [c for c in claims if not c.get("withdrawn") and c["certainty"] != "superseded"]
    if not live:
        return []
    listing = "\n".join(f"[{c['key']}] {c['text']}" for c in live)
    data = await llm.ask_json(QUIZ_SYSTEM, f"CONCEPT: {concept['title']}\n\nCLAIMS:\n{listing}",
                              default=2_000, cap=6_000, stage="learn_quiz")
    by_key, items, seen = {c["key"]: c for c in live}, [], []
    for raw in data if isinstance(data, list) else []:
        if not isinstance(raw, dict) or raw.get("claim") not in by_key:
            continue
        kind, question = str(raw.get("type")), str(raw.get("question", "")).strip()
        answer = str(raw.get("answer", "")).strip()
        choices = [_LETTER_PREFIX.sub("", str(c).strip()) for c in raw.get("choices") or []]
        if kind not in ("mcq", "short", "predict") or not question or not answer:
            continue
        if kind == "mcq" and (len(choices) != 4 or not _letter(answer)):
            continue
        if any(CL.overlap(question, q) > 0.8 for q in seen):
            continue
        seen.append(question)
        src = by_key[raw["claim"]]["passage"]
        items.append({"id": f"{concept['id']}-q{start + len(items)}", "concept": concept["id"],
                      "type": kind, "question": question,
                      "choices": choices if kind == "mcq" else [],
                      "answer": _letter(answer) if kind == "mcq" else answer,
                      "claim": raw["claim"], "explanation": str(raw.get("explanation", "")).strip(),
                      "source": {"arxiv_id": src.get("arxiv_id"), "title": src.get("title"),
                                 "chunk_id": src.get("chunk_id")}})
    return items[:limit]


async def _agrees(llm: Llm, item: dict, solved: str | None, passage: str) -> bool:
    if solved is None:
        return False
    if item["type"] == "mcq":
        return _letter(solved) == item["answer"]
    verdict = await J.judge(llm, f"The answer is: {solved}",
                            f"Question: {item['question']}\nCorrect answer: {item['answer']}")
    return verdict == J.SUPPORTS


async def validate(llm: Llm, items: list[dict], claims: list[dict]) -> tuple[list[dict], int]:
    """(items whose independent solve matches the intended answer, count dropped)."""
    by_key = {c["key"]: c for c in claims}
    solved = await asyncio.gather(*(
        J.solve(llm, i["question"], by_key[i["claim"]]["passage"]["text"], i["choices"] or None)
        for i in items))
    ok = await asyncio.gather(*(_agrees(llm, i, s, by_key[i["claim"]]["passage"]["text"])
                                for i, s in zip(items, solved)))
    kept = [{**i, "validated": True} for i, good in zip(items, ok) if good]
    return kept, len(items) - len(kept)


async def build(llm: Llm, concept: dict, claims: list[dict], *, start: int = 1,
                limit: int = MAX_ITEMS) -> dict:
    items, dropped = await validate(llm, await generate(llm, concept, claims, start=start, limit=limit),
                                    claims)
    return {"items": items, "dropped": dropped}


def _numbers(text: str) -> list[float]:
    return [float(x) for x in _NUM.findall(text or "")]


async def grade(llm: Llm, item: dict, response: str) -> dict:
    """{"correct", "answer", "feedback"} for the learner's response to one item."""
    response = (response or "").strip()
    correct = False
    if item["type"] == "mcq":
        correct = (_letter(response) == item["answer"]
                   or (bool(response) and response.lower() in
                       [c.lower() for c in item["choices"][ord(item["answer"]) - 65:][:1]]))
    else:
        want = _numbers(item["answer"])
        if want and item["type"] == "predict":
            got = _numbers(response)
            correct = bool(got) and any(abs(g - want[0]) <= NUMERIC_TOLERANCE * max(abs(want[0]), 1e-9)
                                        for g in got)
        if not correct and response:
            verdict = await J.judge(
                llm, f"The answer is: {response}",
                f"Question: {item['question']}\nCorrect answer: {item['answer']}")
            correct = verdict == J.SUPPORTS
    shown = (_LETTER_PREFIX.sub("", item["choices"][ord(item["answer"]) - 65])
             if item["type"] == "mcq" else item["answer"])
    return {"correct": correct, "answer": f"{item['answer']}. {shown}" if item["type"] == "mcq" else shown,
            "feedback": item.get("explanation", ""), "source": item.get("source", {})}


async def critique(llm: Llm, claims: list[dict], response: str) -> list[dict]:
    """Points about the learner's text that the claims support or contradict, each one
    re-checked by the judge. What the claims do not speak to is left unsaid."""
    live = [c for c in claims if not c.get("withdrawn")]
    by_key = {c["key"]: c for c in live}
    if not live or not response.strip():
        return []
    listing = "\n".join(f"[{c['key']}] ({c['certainty']}) {c['text']}" for c in live)
    data = await llm.ask_json(CRITIQUE_SYSTEM, f"CLAIMS:\n{listing}\n\nLEARNER:\n{response}",
                              default=1_200, cap=4_000, stage="learn_critique")
    points = [p for p in data if isinstance(p, dict) and p.get("verdict") in ("supported", "contradicted")
              and str(p.get("statement", "")).strip() and any(k in by_key for k in p.get("claims") or [])] \
        if isinstance(data, list) else []
    evidence = ["\n".join(by_key[k]["text"] for k in p["claims"] if k in by_key) for p in points]
    verdicts = await asyncio.gather(*(J.judge(llm, p["statement"], e) for p, e in zip(points, evidence)))
    want = {"supported": J.SUPPORTS, "contradicted": J.CONTRADICTS}
    return [{"statement": p["statement"], "verdict": p["verdict"],
             "claims": [k for k in p["claims"] if k in by_key], "advice": str(p.get("advice", "")).strip()}
            for p, v in zip(points, verdicts) if v == want[p["verdict"]]]
