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
import os
import re
from collections.abc import AsyncIterator

import httpx

OPEN, CLOSE = "<think>", "</think>"


def _auth_headers() -> dict:
    """The bearer token vLLM servers in this fleet require, when they require one.

    Some replicas are launched outside lara (by hand, or a systemd unit) with
    `VLLM_API_KEY` exported, which makes vLLM reject any request lacking it. Matches the
    default `autoresearch/pool.py` and `autoresearch/agent.py` already use for the same
    variable, so a replica started either way is reachable from every caller.
    """
    return {"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'vllm-local')}"}

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
    query: str, hits: list[dict], selection: str | None = None, history: str = ""
) -> tuple[str, str]:
    """Return (stable_prefix, tail). Split at the prefix-cache boundary.

    ``history`` sits between the system block and the excerpts, which is the only place it
    can go without cost. It is append-only across a thread, so everything before the
    excerpts stays byte-identical from turn to turn and the prefix cache still hits;
    putting it after the excerpts would invalidate them on every question instead.
    """
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

    past = f"{history}\n---\n\n" if history else ""
    prefix = f"{SYSTEM}\n\n{FEWSHOT}\n\n---\n\n{past}Excerpts:\n{excerpts}\n"
    tail = ""
    if selection:
        tail += f"\nThe reader has highlighted this passage:\n\"{selection.strip()}\"\n"
    parts = [p for p in dict.fromkeys(h.get("part") for h in hits) if p]
    if len(parts) > 1:
        tail += ("\nThis question has several parts. Address each one, and say so "
                 "explicitly if the excerpts answer some parts but not others:\n"
                 + "\n".join(f"  - {p}" for p in parts) + "\n")
    tail += f"\nQuestion: {query.strip()}\n\nAnswer:"
    # Returned, not stashed on the function. Module-level mutable state made two
    # concurrent /api/ask streams read each other's prompt breakdown — and every agent
    # sub-call (decide, assess, rewrite, tag) goes through here too, so even one request
    # overwrote its own parts several times before the viewer read them.
    parts = [
        ("system", SYSTEM), ("few-shot", FEWSHOT), ("history", history),
        ("excerpts", excerpts), ("question", tail),
    ]
    return prefix, tail, parts


_JSON_OBJECT = re.compile(r"\{.*\}", re.S)
_JSON_ARRAY = re.compile(r"\[.*\]", re.S)


def _balanced(text: str, opener: str, closer: str) -> str:
    """The first *complete* bracketed value in `text`, or "".

    The greedy patterns above run from the first opener to the last closer, which is right
    when the reply is one value with prose around it and wrong the moment there are two.
    A model answering in its native tool-call format emits

        <tool_call>
        {"tool": "outline", "path": "a.py"}
        <tool_call>
        {"tool": "outline", "path": "b.py"}

    and the greedy match spans both objects plus the tag between them, so it does not
    parse and the caller is told nothing came back. Measured on this machine: 34 of 39
    explorations that "read nothing" had a perfectly good object in the reply, and 47% of
    one day's explorations failed this way. Taking the first balanced value keeps the
    first answer instead of discarding both.

    Quote-aware, because a brace inside a string is not a bracket — `{"re": "\\}"}` would
    otherwise close early and produce a truncated object that happens to parse.
    """
    start = text.find(opener)
    if start < 0:
        return ""
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


async def complete(cfg, prompt: str, *, system: str, model: str | None = None,
                   temperature: float = 0.0, max_tokens: int = 400,
                   sampling: dict | None = None, error_out: dict | None = None) -> str:
    """One non-streamed controller turn, accumulated to a string.

    The twelve lines this replaces were written out at nine call sites, and had already
    drifted: three of them did not wrap the stream at all, so a generator error failed
    those turns while degrading silently everywhere else — for no stated reason. Failure
    is a value here, not an exception, because every caller already has a considered
    fallback and none of them wants a controller hiccup to fail a user's question.

    `error_out`, when given, is filled with what actually went wrong. Failure stays a
    value; what changes is that the reason survives it. Without this every cause collapsed
    into the same empty string, and the best a caller could say was "either it was cut off
    or the request was refused — this side cannot tell which". It could not, and the
    answer was sitting in the exception: `404 the model does not exist`, because the pool
    had substituted another card's endpoint while the caller went on naming the model it
    wanted. That took a vLLM log to find and should have taken reading the error.
    """
    buf = ""
    try:
        async for tok in stream_answer(cfg, prompt, [], system=system, model=model,
                                       temperature=temperature, max_tokens=max_tokens,
                                       # Forwarded, which it was not from the day the
                                       # parameter was added. `complete_json` passed it
                                       # on and this did not, so whether a caller's
                                       # `top_k` reached the server depended on which of
                                       # two adjacent functions it happened to call —
                                       # and the nineteen sites that name one all go
                                       # through here. Only `temperature` ever arrived.
                                       sampling=sampling,
                                       raw_user=True):
            buf += tok
    except Exception as exc:
        if error_out is not None:
            error_out["type"] = type(exc).__name__
            # `str(exc)` on an httpx status error is the URL and the code; the body is
            # where the server says *why*, and that is the half worth having.
            body = getattr(getattr(exc, "response", None), "text", "")
            error_out["error"] = f"{exc}{' — ' + body[:300] if body else ''}"
        return ""
    return buf


async def complete_json(cfg, prompt: str, *, system: str, shape: str = "object",
                        model: str | None = None, temperature: float = 0.0,
                        max_tokens: int = 400, default=None,
                        sampling: dict | None = None, raw_out: dict | None = None,
                        error_out: dict | None = None):
    """A controller turn parsed as JSON, or ``default`` if it cannot be.

    Only the transport and the extraction are shared. Schema validation and what to do
    when the verdict is unusable stay at the call site, because that is where the real
    variation lives and it is deliberate: one caller keeps the reranker's top pick, another
    widens a tier rather than stopping, a third treats the same failure as "answer now".
    Collapsing those into one policy would be the opposite of a simplification.
    """
    raw = await complete(cfg, prompt, system=system, model=model,
                         temperature=temperature, max_tokens=max_tokens,
                         sampling=sampling, error_out=error_out)
    # So a caller can tell *why* it got the default. `complete` turns every failure into
    # an empty string — a refused request, a dropped connection, a replica that was
    # evicted mid-call — and an empty reply is a different fact from a reply that could
    # not be parsed. Callers that report one as the other send people looking for a
    # prompt bug when the generator was simply not there.
    if raw_out is not None:
        raw_out["raw"] = raw or ""
    raw = raw or ""
    opener, closer = ("[", "]") if shape == "array" else ("{", "}")
    # Greedy first: it is what every existing caller has been parsed with, and for a reply
    # that is one value with prose around it the two agree. The balanced fallback only
    # matters when the greedy span covers more than one value.
    m = (_JSON_ARRAY if shape == "array" else _JSON_OBJECT).search(raw)
    for candidate in (m.group(0) if m else "", _balanced(raw, opener, closer)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return default


async def count_tokens(base_url: str, model: str, texts: list[str]) -> list[int] | None:
    """Exact token counts from the generator's tokenizer, or None if it cannot say.

    Asking the server beats tokenizing locally: it is the tokenizer that will actually be
    used, so the numbers match what the model sees rather than what a lookalike would.
    Returns None rather than guessing — a context viewer that silently shows estimates as
    facts is worse than one that admits it does not know.
    """
    root = base_url.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    out: list[int] = []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=3.0),
                                     headers=_auth_headers()) as client:
            for t in texts:
                if not t:
                    out.append(0)
                    continue
                r = await client.post(f"{root}/tokenize", json={"model": model, "prompt": t})
                if r.status_code != 200:
                    return None
                out.append(int(r.json().get("count", 0)))
    except Exception:
        return None
    return out


async def context_limit(base_url: str, model: str) -> int | None:
    """The model's real context window, as the server reports it."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0),
                                     headers=_auth_headers()) as client:
            r = await client.get(f"{base_url.rstrip('/')}/models")
            if r.status_code != 200:
                return None
            for m in r.json().get("data", []):
                if m.get("id") == model and m.get("max_model_len"):
                    return int(m["max_model_len"])
            for m in r.json().get("data", []):
                if m.get("max_model_len"):
                    return int(m["max_model_len"])
    except Exception:
        return None
    return None


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
    history: str = "",
    prompt_parts: list | None = None,
    usage_out: dict | None = None,
    # Sampling beyond temperature, forwarded verbatim when given. Absent by default so
    # every existing caller sends exactly what it sent before, and the server's own
    # defaults — which vLLM takes from the checkpoint's generation_config.json — apply.
    sampling: dict | None = None,
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
    prefix, tail, parts = build_prompt(query, hits, selection, history)
    if prompt_parts is not None:
        prompt_parts[:] = parts

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
        # The server counts its own tokens; anything computed here would be a lookalike.
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    # Merged rather than assigned: a caller naming `top_p` must not silently drop the
    # temperature it also asked for, and an unknown key is the caller's business — vLLM
    # rejects what it does not accept, which is a better failure than this guessing.
    payload.update({k: v for k, v in (sampling or {}).items() if v is not None})

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0),
                                 headers=_auth_headers()) as client:
        try:
            async with client.stream("POST", f"{base_url}/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode()[:200]
                    # Named by URL rather than "vLLM": this speaks to whatever is serving
                    # -- llama.cpp, MLX, Ollama, LM Studio -- and reporting the wrong one
                    # sends anyone reading the error to the wrong logs.
                    raise RuntimeError(
                        f"generator at {base_url} returned {resp.status_code}: {body}")
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
                    if chunk.get("usage") and usage_out is not None:
                        # The final frame carries exact prompt and completion counts. Into
                        # a dict the caller owns, so concurrent streams cannot collide.
                        usage_out.update(chunk["usage"])
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
                configured = GEN.model_for(backend, GEN.generator_cfg(cfg))
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
