"""Prompt assembly and streaming generation against vLLM.

**Prompt layout is chosen for prefix caching, not readability.** vLLM reuses the KV cache
of any shared prefix, so everything stable goes first and everything per-question goes
last::

    [ system + citation instructions + few-shot ]   stable for the whole session
    [ retrieved chunks                          ]   stable across follow-ups on a paper
    [ selected passage                          ]   changes per selection
    [ question                                  ]   changes per question

Ask three questions about the same passage and only the last block is recomputed, which is
where the "TTFT collapses on follow-ups" behaviour in PLAN.md §7 comes from. Putting the
question first would discard the cache on every keystroke of a new question.

The few-shot exemplars are load-bearing rather than decorative: under D3 the only complete
generators in cache are *base* models, which do not follow instructions. They sit inside
the cached prefix, so after the first request they cost nothing.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

OPEN, CLOSE = "<think>", "</think>"

SYSTEM = """You answer questions about arXiv machine-learning papers using only the \
excerpts provided.

Rules:
1. Ground every claim in an excerpt and cite it as [n]. A sentence carrying a factual \
claim with no citation is a bug.
2. Never let a citation stand for something the excerpt does not actually say. If an \
excerpt only partly supports a claim, say which part.
3. Distinguish what the sources state from what you are inferring. Mark inference \
explicitly ("this implies", "the excerpts do not say, but").
4. If two excerpts disagree — different numbers, opposite conclusions, incompatible \
setups — say so and cite both. Do not silently pick one.
5. Prefer the paper's own terminology and notation.
6. Be concise. Two or three sentences unless the question genuinely needs more."""


# Coverage-specific instructions appended to SYSTEM. The point is to make an honest
# non-answer a first-class outcome rather than something the model has to improvise its
# way out of — an LLM handed weak excerpts and a bare "answer the question" instruction
# will reliably paper over the gap instead of naming it.
COVERAGE_INSTRUCTIONS = {
    "full": "",
    "partial": """

IMPORTANT — the retrieved excerpts only PARTIALLY answer this question. You must:
- Open with one sentence stating plainly what the sources do not establish.
- Then give what they DO support, cited as usual.
- Do not fill the gap with background knowledge.""",
    "none": """

IMPORTANT — the retrieved excerpts do NOT answer this question. You must:
- Open by saying plainly that the sources do not contain the answer.
- Then summarise what the most relevant excerpts DO cover, cited as usual, so the reader \
can judge whether to rephrase or search elsewhere.
- Do not attempt an answer from background knowledge. An honest non-answer is correct \
here; a plausible invented one is not.""",
}


FEWSHOT = """Example.

Excerpts:
[1] (Attention Is All You Need > Model Architecture) The Transformer follows this overall \
architecture using stacked self-attention and point-wise, fully connected layers.
[2] (Attention Is All You Need > Why Self-Attention) A self-attention layer connects all \
positions with a constant number of sequentially executed operations, whereas a recurrent \
layer requires O(n) sequential operations.

Question: How does the Transformer avoid recurrence?

Answer: It replaces recurrent layers with stacked self-attention and position-wise feed-\
forward layers [1]. Because self-attention connects all positions in a constant number of \
sequential operations rather than O(n) [2], sequence order is handled without any \
recurrent step."""


def prefix_body(prefix: str) -> str:
    """The excerpt block, with the system text stripped (it moves to the system role)."""
    return prefix.split('---\n\n', 1)[-1]


def order_for_attention(hits: list[dict]) -> list[dict]:
    """Reorder excerpts so the strongest sit at the two ends of the context.

    Transformers attend most reliably to the beginning and end of a long context and
    weakest to its middle — the "lost in the middle" effect. Retrieval hands us excerpts
    in descending relevance, which puts the second- and third-best evidence exactly where
    the model is least likely to use it. Alternating outward from the ends costs nothing
    and keeps the best material where it is actually read.
    """
    out: list[dict] = []
    tail: list[dict] = []
    for i, h in enumerate(hits):
        (out if i % 2 == 0 else tail).append(h)
    return out + tail[::-1]


def build_prompt(
    query: str, hits: list[dict], selection: str | None = None
) -> tuple[str, str]:
    """Return (stable_prefix, tail). Split at the prefix-cache boundary."""
    lines = []
    # Bind each citation marker to the excerpt's ORIGINAL rank first, so reordering for
    # attention never renumbers what the reader ends up clicking.
    for i, h in enumerate(hits, 1):
        h.setdefault("marker", i)
    ordered = order_for_attention(hits)
    for h in ordered:
        where = h.get("section") or ""
        title = h.get("paper_title") or h.get("arxiv_id", "")
        date = (h.get("submitted") or "")[:7]
        stamp = f", {date}" if date else ""
        # Carry the sub-question an excerpt was retrieved for. Without it the model sees a
        # flat pile of evidence for a two-part question and typically answers whichever
        # part the strongest excerpts happen to cover, leaving the other unanswered.
        part = f" [for: {h['part']}]" if h.get("part") else ""
        lines.append(
            f"[{h['marker']}] ({title}{stamp} > {where}){part} {h.get('text', '').strip()}"
        )
    excerpts = "\n\n".join(lines) if lines else "(no excerpts retrieved)"

    prefix = f"{SYSTEM}\n\n{FEWSHOT}\n\n---\n\nExcerpts:\n{excerpts}\n"
    tail = ""
    if selection:
        tail += f"\nThe reader has highlighted this passage:\n\"{selection.strip()}\"\n"
    parts = [p for p in dict.fromkeys(h.get("part") for h in hits) if p]
    if len(parts) > 1:
        tail += ("\nThis question has several parts. Address each one, and say so "
                 "explicitly if the excerpts answer some parts but not others:\n"
                 + "\n".join(f"  - {p}" for p in parts) + "\n")
    tail += f"\nQuestion: {query.strip()}\n\nAnswer:"
    return prefix, tail


async def stream_answer(
    cfg,
    query: str,
    hits: list[dict],
    *,
    selection: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    system: str | None = None,
    raw_user: bool = False,
    coverage: str = "full",
) -> AsyncIterator[str]:
    """Stream a completion.

    ``system`` overrides the answering persona and ``raw_user`` sends ``query`` verbatim
    instead of wrapping it in the excerpt template — together they let the retrieval
    controller in :mod:`lara.serve.agent` reuse this transport for its JSON verdicts.
    """
    # Reasoning-block stripping runs over the accumulated stream, not per chunk.
    acc, sent, state = "", 0, "unknown"
    vcfg = cfg.get_in("serving.vllm")
    base_url = vcfg["base_url"].rstrip("/")
    model_name = model or vcfg.get("default_model")
    prefix, tail = build_prompt(query, hits, selection)

    # Chat completions, not raw completions: the generator is instruction tuned, so its
    # chat template is what it was trained to answer in. Raw prompting works but produces
    # noticeably worse grounding and ignores the system role.
    #
    # `enable_thinking: false` matters — Qwen3.x emits a <think> reasoning block by
    # default, which for a two-sentence grounded answer is most of the latency and none
    # of the output. The tags are also stripped below, since the flag is model-specific
    # and silently ignored by generators that do not know it.
    user_content = query if raw_user else prefix_body(prefix) + tail
    payload = {
        "model": model_name,
        "messages": [
            # Coverage instructions append to whatever persona is in force, including a
            # reader's custom prompt. They are not part of the persona — they are what
            # makes an honest non-answer possible — so overriding the prompt must not
            # silently switch them off.
            {"role": "system",
             "content": (system or SYSTEM) + COVERAGE_INSTRUCTIONS.get(coverage, "")},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0)) as client:
        try:
            async with client.stream("POST", f"{base_url}/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode()[:200]
                    raise RuntimeError(f"vLLM returned {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        return
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta", {})
                    # Some builds stream reasoning in its own field; that is easy, we
                    # simply never read it.
                    text = delta.get("content") or choice.get("text") or ""
                    if not text:
                        continue

                    acc += text
                    if state == "unknown":
                        stripped = acc.lstrip()
                        if stripped.startswith(OPEN):
                            state = "thinking"
                        elif OPEN.startswith(stripped[: len(OPEN)]):
                            # Could still turn into "<think>" — the tag arrives split
                            # across token boundaries ("<", "think", ">"), which is why
                            # a per-chunk substring test never fires. Wait for more.
                            continue
                        else:
                            state = "plain"

                    if state == "thinking":
                        if CLOSE not in acc:
                            continue
                        acc = acc.split(CLOSE, 1)[1].lstrip()
                        state, sent = "plain", 0

                    out = acc[sent:]
                    if out:
                        sent = len(acc)
                        yield out
        except httpx.ConnectError as exc:
            # Naming vLLM here was wrong on every machine that does not run vLLM, which
            # is every Mac. Report the backend that would actually serve, and the reason
            # nothing is listening — which is usually that no model is configured for it,
            # not that the user forgot to launch something.
            try:
                from lara.serve import devices as DV
                from lara.serve import generator as GEN

                backend = GEN.resolve_backend(cfg, DV.detect().accelerator,
                                              cfg.get_path("huggingface.home"))
                configured = GEN.model_for(backend, {
                    **((cfg.get_in("serving.generator") or {})),
                    "vllm": cfg.get_in("serving.vllm") or {},
                })
            except Exception:
                backend, configured = "the generation backend", None
            hint = ("run `lara setup` to choose one, then restart `lara serve`"
                    if not configured else
                    f"`lara serve` should have started it for {configured}; "
                    f"check its log, or run `lara serve-llm`")
            raise RuntimeError(
                f"nothing is answering at {base_url}. Backend is {backend} and "
                f"{'no model is configured for it' if not configured else 'it is not running'}"
                f" — {hint}."
            ) from exc
