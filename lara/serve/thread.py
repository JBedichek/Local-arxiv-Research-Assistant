"""Conversation threads — what the previous questions were, and what "it" refers to.

Every question used to be asked in isolation: no prior turn reached the request, the
prompt, or the retriever. Asking *"why is that?"* embedded those three words and searched
29.5 M chunks for them, which matches nothing in particular. The answer prompt then saw a
fresh question with unrelated excerpts and no sign an earlier exchange had happened.

The exchanges were already being stored — `library.json` holds every question with its
answer, paper and selection — and simply never read back.

Two distinct problems, fixed in different places, because fixing only one leaves the
feature broken in a way that is hard to see:

**Reference resolution happens before retrieval.** A follow-up is rewritten into a
standalone query *first*, so the index is searched for what the reader meant rather than
for a pronoun. History in the answer prompt cannot repair this — by then the wrong
excerpts have already been fetched.

**History goes in the cached region.** The prompt is laid out stable-prefix-first so vLLM's
prefix cache survives follow-ups about one paper. History is append-only, so it belongs
immediately after the system block and *before* the excerpts, which change every turn.
Putting it after would invalidate the excerpt cache on every question.

Threads are keyed by paper, not global. Opening a different paper starts a new thread,
because most reading is several short conversations rather than one long one, and a global
history would drag a question about optimisers into an answer about tokenisation.
"""

from __future__ import annotations

import re

CORPUS_THREAD = "corpus"

#: Enough turns to resolve a reference; few enough that the prefix stays small. A
#: follow-up almost always refers to the immediately preceding exchange.
DEFAULT_TURNS = 4
MAX_ANSWER_CHARS = 700


def thread_id(arxiv_id: str | None) -> str:
    return arxiv_id or CORPUS_THREAD


def turns_for(root, arxiv_id: str | None, limit: int = DEFAULT_TURNS,
              include_compressed: bool = False) -> list[dict]:
    """The most recent question/answer pairs in this thread, oldest first.

    Turns already folded into a stored summary are skipped unless asked for: they are
    represented by the summary now, and sending both would cost the tokens compression
    was invoked to reclaim.
    """
    from lara.serve import memory as MEM

    want = thread_id(arxiv_id)
    covered = set() if include_compressed else set(
        MEM.get_thread_summary_covered(root, want))
    try:
        data = MEM.load(root)
    except Exception:
        return []
    turns = [
        e for e in data.get("entries", [])
        if e.get("kind") == "question" and thread_id(e.get("arxiv_id")) == want
        and (e.get("question") or "").strip() and e.get("id") not in covered
    ]
    turns.sort(key=lambda e: e.get("created_utc") or "")
    return turns[-limit:]


# ── follow-up detection ───────────────────────────────────────────────────────────

#: Words that point at something outside the sentence. A question containing one of these
#: and little else cannot be retrieved on its own terms.
_DEICTIC = re.compile(
    r"\b(it|its|it's|that|those|these|this|they|them|their|he|she|his|her|"
    r"the (former|latter|first|second|other|same|above|previous)|"
    r"there|then|such)\b", re.I)

_ELLIPTIC = re.compile(
    r"^\s*(and|but|so|also|what about|how about|why|why not|how so|and then|"
    r"go on|continue|more|explain|elaborate|expand|say more|really|"
    r"compared to|versus|vs\.?)\b", re.I)

#: A question this short is almost never self-contained.
_SHORT_WORDS = 6


def looks_like_followup(question: str) -> tuple[bool, str]:
    """Cheap pre-filter. Returns (is_followup, why).

    Deliberately generous about what counts: a rewrite that was not needed costs one
    short model call and returns the question roughly unchanged, while a missed follow-up
    silently retrieves the wrong passages and produces a confident answer about the wrong
    thing. The asymmetry is not close.
    """
    q = (question or "").strip()
    if not q:
        return False, ""
    words = q.split()
    if len(words) <= _SHORT_WORDS:
        return True, f"only {len(words)} words"
    if _ELLIPTIC.match(q):
        return True, "opens with a continuation"
    if _DEICTIC.search(q):
        return True, "contains a reference to something unstated"
    return False, ""


REWRITE_SYSTEM = """You rewrite a follow-up question into one that stands on its own.

The rewritten question is used to SEARCH a corpus of papers. It is not shown to the reader \
as their question, and it is not answered directly — so it should read like a search \
query stated as a full question, carrying every noun the search needs.

What your rewrite causes:
- The rewritten text is embedded and used to retrieve passages. Pronouns and references \
retrieve nothing, so every "it", "that" or "the former" must be replaced with the thing \
it names, taken from the conversation.
- If you add a term the reader did not mean, the search goes somewhere they did not ask \
about, and the answer will be confidently about the wrong subject.

Rules:
- Keep the reader's intent exactly. Do not broaden, narrow, or answer it.
- Resolve references using the conversation. Prefer the most recent mention.
- If the question already stands alone, return it unchanged.
- Return the question only, with no preamble, quotes or explanation."""


async def rewrite(cfg, question: str, turns: list[dict], model, stream_answer) -> dict:
    """Resolve a follow-up against the thread. Never raises; falls back to the original."""
    if not turns:
        return {"query": question, "rewritten": False, "why": "no prior turns"}
    is_fu, why = looks_like_followup(question)
    if not is_fu:
        return {"query": question, "rewritten": False, "why": "self-contained"}

    convo = "\n\n".join(
        f"Q: {t.get('question','').strip()}\n"
        f"A: {(t.get('answer') or '').strip()[:MAX_ANSWER_CHARS]}"
        for t in turns[-2:]
    )
    prompt = (f"Conversation so far:\n{convo}\n\n"
              f"Follow-up question: {question.strip()}\n\n"
              "Rewritten standalone question:")
    buf = ""
    try:
        async for tok in stream_answer(cfg, prompt, [], system=REWRITE_SYSTEM, model=model,
                                       temperature=0.0, max_tokens=120, raw_user=True):
            buf += tok
    except Exception:
        buf = ""

    out = (buf or "").strip().strip('"').split("\n")[0].strip()
    # A rewrite that collapses or explodes is a malfunction, not a better query. Length
    # is a blunt check, but it reliably catches the two ways this fails: an empty answer,
    # and a model that starts explaining itself instead of rewriting.
    if not out or len(out) < 8 or len(out) > 400:
        return {"query": question, "rewritten": False, "why": "rewrite unusable"}
    if out.lower() == question.strip().lower():
        return {"query": question, "rewritten": False, "why": "already standalone"}
    return {"query": out, "rewritten": True, "why": why, "original": question}


# ── prompt ────────────────────────────────────────────────────────────────────────


def history_block(turns: list[dict], max_answer: int = MAX_ANSWER_CHARS,
                  summary: str = "") -> str:
    """The conversation as the answering model sees it.

    Answers are truncated because their job here is to establish what was discussed, not
    to be re-read in full: an untruncated thread would crowd out the excerpts that carry
    the evidence for the current question.
    """
    if not turns and not summary:
        return ""
    lines = []
    if summary:
        # The compressed turns are presented as established context rather than as
        # transcript, because that is what they now are: the exchanges themselves are
        # gone and only these notes can resolve a reference back to them.
        lines.append("Summary of earlier turns in this conversation:\n" + summary.strip())
    if turns:
        lines.append("\nMost recent exchanges:" if summary
                     else "Earlier in this conversation:")
    for t in turns:
        q = (t.get("question") or "").strip()
        a = (t.get("answer") or "").strip()
        if len(a) > max_answer:
            a = a[:max_answer].rsplit(" ", 1)[0] + " …"
        lines.append(f"\nQ: {q}")
        if a:
            lines.append(f"A: {a}")
    lines.append(
        "\nThe reader can see the above. Do not repeat it; answer the new question, and "
        "use the earlier exchange only to understand what they are referring to."
    )
    return "\n".join(lines) + "\n"


def prior_chunk_ids(turns: list[dict], cap: int = 8) -> list[int]:
    """Chunk ids cited by recent answers, newest first.

    A follow-up usually concerns a passage already on screen, so the passages the last
    answers actually cited are strong candidates for this one — and they cost nothing to
    reuse, having already been retrieved and reranked.
    """
    out: list[int] = []
    for t in reversed(turns):
        for m in re.finditer(r"\[(\d{3,})\]", t.get("answer") or ""):
            cid = int(m.group(1))
            if cid not in out:
                out.append(cid)
        if len(out) >= cap:
            break
    return out[:cap]


# ── compression ───────────────────────────────────────────────────────────────────

#: Turns kept verbatim after compressing. References almost always point at the most
#: recent exchange, so summarising it away is what would actually break follow-ups.
KEEP_VERBATIM = 2

COMPRESS_SYSTEM = """You are compressing a conversation so it can keep going.

The summary REPLACES the exchanges it covers. Nothing else from them survives — so
anything the reader might refer back to has to be in it, stated plainly enough that a
pronoun can be resolved against it.

What your summary causes:
- It is prepended to future prompts in place of the full exchanges, and is what the model
  will use to work out what "it", "that" or "the second one" refers to.
- Anything you leave out is gone. If a method, dataset, number or comparison was discussed
  and you omit it, a later question about it will be answered as though it were never
  mentioned.
- Anything you add that was not discussed will be treated as established fact.

Write compact notes, not prose. Prioritise, in order:
1. Named entities — methods, papers, datasets, metrics, authors.
2. Conclusions actually reached, with their numbers and conditions.
3. What the reader seemed to be pursuing, so a vague follow-up still has a subject.
4. Anything explicitly ruled out or found unsupported.

Do not restate the questions verbatim. Do not add caveats about being a summary.
Aim for under 200 words."""


async def compress(cfg, root, arxiv_id, model, stream_answer,
                   keep: int = KEEP_VERBATIM) -> dict:
    """Summarise a thread's older turns so the conversation can continue cheaply.

    The most recent turns are left verbatim: they are what follow-ups point at, and
    compressing them is what would break reference resolution — the opposite of the point.
    """
    from lara.serve import memory as MEM

    turns = turns_for(root, arxiv_id, limit=200)
    if len(turns) <= keep:
        return {"ok": False, "reason": f"only {len(turns)} turns; nothing to compress"}

    older, recent = turns[:-keep], turns[-keep:]
    prior = MEM.get_thread_summary(root, thread_id(arxiv_id))
    convo = "\n\n".join(
        f"Q: {t.get('question','').strip()}\nA: {(t.get('answer') or '').strip()[:1500]}"
        for t in older)
    prompt = ((f"Existing summary of even earlier turns:\n{prior}\n\n" if prior else "")
              + f"Exchanges to compress:\n{convo}\n\nWrite the compact notes.")

    buf = ""
    try:
        async for tok in stream_answer(cfg, prompt, [], system=COMPRESS_SYSTEM, model=model,
                                       temperature=0.1, max_tokens=500, raw_user=True):
            buf += tok
    except Exception:
        buf = ""
    summary = (buf or "").strip()
    if len(summary) < 40:
        return {"ok": False, "reason": "the model returned no usable summary"}

    # Compare like with like. `turns` here is the WHOLE thread, but only the last
    # DEFAULT_TURNS were ever being sent, so measuring against all of them would report a
    # saving that never existed. The honest before is what the model actually saw.
    before = len(history_block(turns[-DEFAULT_TURNS:]))
    MEM.set_thread_summary(root, thread_id(arxiv_id), summary,
                           covered=[t["id"] for t in older])
    after = len(history_block(recent, summary=summary))
    return {
        "ok": True, "summary": summary,
        "compressed_turns": len(older), "kept_verbatim": len(recent),
        "chars_before": before, "chars_after": after,
        # What compression actually buys is COVERAGE: every turn is represented, instead
        # of a sliding window of the most recent DEFAULT_TURNS. On a short thread that
        # costs more characters than it saves, and saying so is better than implying a
        # win that is not there.
        "turns_before": min(len(turns), DEFAULT_TURNS),
        "turns_after": len(turns),
        "saves_chars": after < before,
    }
