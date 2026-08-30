"""Decide whether a fetched document belongs in the corpus the reader asked for.

Two judges, used where each is worth its cost — the same split that
``lara/finetune/judgements.py`` applies to passage relevance, but NOT the same first judge.

``embedder``   cheap and already loaded. Cosine between the goal and samples of the
               document. This is a topical-similarity question and that is what an
               embedding model is for.
``llm``        expensive, reserved for the band where the embedder is unsure. That is
               where a reading model changes the answer rather than confirming it.

**The cross-encoder was tried first and is unusable for this.** It is trained for short
query -> passage retrieval, and given a goal statement against long document text it does
not merely degrade, it inverts. Measured against a "learn single-variable calculus" goal
over Wikipedia articles:

    document                mean cross-encoder    mean embedder
    Calculus     (relevant)         0.442             0.287
    Derivative   (relevant)         0.298             0.269
    Integral     (relevant)         0.522             0.230
    Espresso     (irrelevant)       0.369             0.058
    Sourdough    (irrelevant)       0.569            -0.012
    FIFA WC      (irrelevant)       0.563            -0.042

The reranker ranks sourdough and the World Cup *above* the article on derivatives, and no
pooling — max, mean, p75, trimmed — separates the two groups. The embedder separates them
by 0.172 with room to spare. Reusing the reranker here was an assumption that looked
obvious and was wrong.

**Documents are sampled, not read whole.** A 700-page textbook has one topic and the
cross-encoder's window is 512 tokens; feeding it the first 512 tokens judges the title
page. Several passages spread through the document, scored and pooled by their best, asks
the question that actually matters: does this book contain what the reader needs, anywhere
in it?
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

#: Below LOW the document is clearly off topic; above HIGH, clearly on it. Between them
#: the generator decides. Calibrated on the table above: the worst relevant document
#: scored 0.230 and the best irrelevant one 0.058, so the band sits between them with
#: margin on both sides. Six documents is a small calibration set, which is exactly why the
#: uncertain band exists rather than a single cut point.
UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.10, 0.22

#: Samples taken through the document. Enough to cross chapter boundaries in a textbook,
#: few enough that judging a candidate costs milliseconds.
SAMPLES = 6
SAMPLE_CHARS = 1400

#: A document shorter than this is a stub, a cookie notice or an error page that returned
#: HTTP 200. Judging it wastes a model call to reach the obvious answer.
MIN_USEFUL_CHARS = 800

# M19: consecutive-failure watchdog for judge_uncertain. One failure is a
# degraded decision (documented, by design); a wall of failures is an outage.
_LLM_CONSECUTIVE_FAILURES = 0
_LLM_FAILURE_THRESHOLD = 10


@dataclass
class Verdict:
    relevant: bool
    score: float                 # pooled cross-encoder score, 0-1
    judge: str                   # heuristic | embedder | llm
    reason: str = ""

    def as_reason(self) -> str:
        return self.reason or f"{self.judge} score {self.score:.2f}"


def samples(text: str, n: int = SAMPLES, width: int = SAMPLE_CHARS) -> list[str]:
    """Evenly spaced excerpts, skipping the front matter.

    The first page of a textbook is a cover, a copyright notice and a table of contents —
    the least characteristic text in the whole document. Starting a little way in gets
    prose instead.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= width:
        return [text] if text else []
    start = min(len(text) // 20, 5000)          # skip ~5% or 5k chars, whichever is less
    body = text[start:]
    if len(body) <= width:
        return [body]
    step = max(1, (len(body) - width) // max(1, n - 1))
    return [body[i * step: i * step + width] for i in range(n)][:n]


def score_document(embedder, goal: str, text: str, title: str = "") -> float:
    """Pooled cosine between the goal and the document, roughly -1 to 1.

    Pooled by MEAN rather than maximum. Maximum was the first instinct — a reference work
    is mostly not about any one question, so take its best part — and measurement killed
    it: four of six samples from the espresso article scored below 0.15 against a calculus
    goal, but two outliers reached 0.86, and the maximum reported those. Averaging asks
    "is this document about the goal" instead of "does any paragraph of it happen to
    match", which is the question a corpus builder needs answered.
    """
    import numpy as np

    from lara.index.embed import document_text, embed_queries

    parts = samples(text)
    if not parts:
        return 0.0
    q = embed_queries(embedder, [goal])[0]
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    E = embedder.encode([document_text(title or None, None, p) for p in parts],
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=False)
    return float((E @ q).mean())


async def judge_uncertain(cfg, goal: str, doc_title: str, text: str,
                          model: str | None = None) -> Verdict | None:
    """Ask the generator about a document the reranker could not call.

    Returns None if the model is unreachable or answers unusably, so an unavailable
    generator degrades the decision to the reranker's score rather than failing the build.
    """
    from lara.serve.generate import stream_answer

    excerpt = "\n\n---\n\n".join(samples(text, n=3, width=1200))

    prompt = (
        f"A reader wants to build a study corpus for this goal:\n\n  {goal}\n\n"
        f"Candidate document: {doc_title}\n\nExcerpts:\n{excerpt}\n\n"
        'Reply with JSON only: {"include": true/false, "why": "one short sentence"}. '
        "Include it if a reader pursuing that goal would want this document searchable. "
        "Exclude marketing pages, link directories, and documents on a different subject."
    )
    buf = ""
    try:
        async for tok in stream_answer(
            cfg, prompt, [], system="You judge whether a document belongs in a study "
                                   "corpus. Reply with JSON and nothing else.",
            model=model, temperature=0.0, max_tokens=120, raw_user=True,
        ):
            buf += tok
    except Exception:                                  # noqa: BLE001
        # M19: a single failure degrades to the reranker's score (documented
        # above), but a generator that fails every call is an outage, not a
        # judgement signal — count consecutive failures and raise rather than
        # let a whole build silently run on reranker-only verdicts.
        global _LLM_CONSECUTIVE_FAILURES
        _LLM_CONSECUTIVE_FAILURES += 1
        if _LLM_CONSECUTIVE_FAILURES >= _LLM_FAILURE_THRESHOLD:
            raise RuntimeError(
                f"generator unreachable? {_LLM_CONSECUTIVE_FAILURES} consecutive "
                f"judge_uncertain LLM calls failed") from None
        return None
    _LLM_CONSECUTIVE_FAILURES = 0

    m = re.search(r"\{.*\}", buf, re.S)
    if not m:
        return None
    try:
        blob = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "include" not in blob:
        return None
    return Verdict(relevant=bool(blob["include"]), score=float("nan"), judge="llm",
                   reason=str(blob.get("why", ""))[:160])


async def validate(cfg, embedder, goal: str, doc, model: str | None = None) -> Verdict:
    """The full decision for one fetched document."""
    if doc.error:
        return Verdict(False, 0.0, "heuristic", f"could not fetch: {doc.error}")
    if doc.chars < MIN_USEFUL_CHARS:
        return Verdict(False, 0.0, "heuristic",
                       f"too little text ({doc.chars} chars) to be a document")

    score = score_document(embedder, goal, doc.text, doc.title)
    if score >= UNCERTAIN_HIGH:
        return Verdict(True, score, "embedder")
    if score <= UNCERTAIN_LOW:
        return Verdict(False, score, "embedder", f"off topic (similarity {score:.2f})")

    llm = await judge_uncertain(cfg, goal, doc.title, doc.text, model=model)
    if llm is None:
        # No generator: fall back to the reranker's own leaning rather than blocking. The
        # reader confirms every source anyway, so a borderline call errs toward showing
        # them the candidate rather than hiding it.
        return Verdict(score >= (UNCERTAIN_LOW + UNCERTAIN_HIGH) / 2, score, "embedder",
                       f"uncertain ({score:.2f}), no generator to adjudicate")
    llm.score = score
    return llm
