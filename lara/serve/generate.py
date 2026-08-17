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

SYSTEM = """You answer questions about arXiv machine-learning papers using only the \
excerpts provided. Rules:

1. Ground every claim in an excerpt. If the excerpts do not contain the answer, say so \
plainly rather than guessing.
2. Cite with the bracketed marker of the excerpt you used, e.g. [1] or [3].
3. Prefer quoting the paper's own terminology and notation.
4. Be concise. Two or three sentences unless the question needs more."""

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


def build_prompt(
    query: str, hits: list[dict], selection: str | None = None
) -> tuple[str, str]:
    """Return (stable_prefix, tail). Split at the prefix-cache boundary."""
    lines = []
    for i, h in enumerate(hits, 1):
        where = h.get("section") or ""
        title = h.get("paper_title") or h.get("arxiv_id", "")
        lines.append(f"[{i}] ({title} > {where}) {h.get('text', '').strip()}")
    excerpts = "\n\n".join(lines) if lines else "(no excerpts retrieved)"

    prefix = f"{SYSTEM}\n\n{FEWSHOT}\n\n---\n\nExcerpts:\n{excerpts}\n"
    tail = ""
    if selection:
        tail += f"\nThe reader has highlighted this passage:\n\"{selection.strip()}\"\n"
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
) -> AsyncIterator[str]:
    vcfg = cfg.get_in("serving.vllm")
    base_url = vcfg["base_url"].rstrip("/")
    model_name = model or vcfg.get("default_model")
    prefix, tail = build_prompt(query, hits, selection)

    payload = {
        "model": model_name,
        "prompt": prefix + tail,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stop": ["\n\nQuestion:", "\n\nExcerpts:"],
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0)) as client:
        try:
            async with client.stream("POST", f"{base_url}/completions", json=payload) as resp:
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
                    text = (chunk.get("choices") or [{}])[0].get("text", "")
                    if text:
                        yield text
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"no vLLM server at {base_url} — start one with `lara serve-llm`"
            ) from exc
