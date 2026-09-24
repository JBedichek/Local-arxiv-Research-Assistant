"""One wrapper around a model call, so every stage takes the same injected object."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any



CHARS_PER_TOKEN = 3.8
WINDOW_MARGIN_TOKENS = 1_500
WINDOW_MARGIN_FRACTION = 0.15
REPLY_FLOOR = 2_000


def reply_room(window: int, *parts: str, default: int, cap: int = 0) -> int:
    """How long a reply may be: what the window has left once the prompt is in it, capped by
    what a plausible reply needs. With no known window, `default` stands in -- a constant is
    wrong in both directions (truncates real output on a small window, leaves a large one
    unused), and an uncapped remainder lets a model that never stops run to the window."""
    if window <= 0:
        return min(default, cap) if cap else default
    used = int(sum(len(p) for p in parts) / CHARS_PER_TOKEN)
    margin = max(WINDOW_MARGIN_TOKENS, int(used * WINDOW_MARGIN_FRACTION))
    room = max(REPLY_FLOOR, window - used - margin)
    return max(REPLY_FLOOR, min(room, cap)) if cap else room


@dataclass
class Llm:
    """`complete` is `lara.serve.generate.complete`'s shape: (cfg, prompt, *, system, model,
    max_tokens) -> str. `limit` bounds concurrent calls across every stage sharing this."""

    complete: Callable[..., Awaitable[str]]
    cfg: Any = None
    model: str | None = None
    window: int = 0
    limit: int = 8
    _sem: asyncio.Semaphore | None = field(default=None, repr=False)

    async def ask(self, system: str, prompt: str, *, default: int = 800, cap: int = 6_000,
                  stage: str = "learn") -> str:
        room = reply_room(self.window, prompt, system, default=default, cap=cap)
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.limit)
        async with self._sem:
            text = await self.complete(self.cfg, prompt, system=system, model=self.model,
                                       max_tokens=room)
        return (text or "").strip()

    async def ask_json(self, system: str, prompt: str, **kw) -> Any:
        """The parsed JSON in the reply, or None -- a malformed reply is a failed step the
        caller decides about, not an exception."""
        return parse_json(await self.ask(system, prompt, **kw))


@dataclass
class TokenMeter:
    """A running (tokens_in, tokens_out) tally for one build, estimated the same way
    `reply_room` already estimates prompt size -- chars / CHARS_PER_TOKEN. Not exact (that
    would need vLLM's own usage counts threaded through every completion call, a much larger
    change to the shared generation path every other feature also calls through); close enough
    to show real, moving numbers rather than none at all."""
    tokens_in: int = 0
    tokens_out: int = 0


def metered(llm: Llm, meter: TokenMeter) -> Llm:
    """A copy of `llm` whose calls also add to `meter` as they complete. Transparent to every
    caller -- still just an `Llm`, `.ask()`/`.ask_json()` unchanged -- so nothing downstream
    (claims, lesson, quiz, visuals, topics, judge, ...) needs to know metering is happening;
    a build only has to wrap `llm` once, before its first stage, to meter all of them.

    Creates `llm`'s concurrency semaphore up front if it does not exist yet, rather than
    leaving `Llm.ask` to lazily create one on `self` (the wrapped copy) later -- that would
    give the wrapped copy its own semaphore, separate from the shared one every other caller
    of the original `llm` is limited by, quietly doubling the real concurrency cap."""
    if llm._sem is None:
        llm._sem = asyncio.Semaphore(llm.limit)

    async def wrapped(cfg, prompt, *, system="", model=None, max_tokens=0):
        text = await llm.complete(cfg, prompt, system=system, model=model, max_tokens=max_tokens)
        meter.tokens_in += round((len(system) + len(prompt)) / CHARS_PER_TOKEN)
        meter.tokens_out += round(len(text or "") / CHARS_PER_TOKEN)
        return text
    return replace(llm, complete=wrapped)


def _literal_backslashes(text: str) -> str:
    """Doubles every backslash that is not already half of `\\` or an escaped quote, so LaTeX
    like `$\\hat{m}_t$` written with single backslashes survives as text. Only used on a
    candidate that failed strict parsing: a valid `\\n` or `\\u00e9` is never rewritten."""
    return re.sub(r'\\(?![\\"])', r'\\\\', text)


def parse_json(text: str) -> Any:
    """First JSON object or array in `text`, tolerating code fences, prose around it, and
    math with unescaped backslashes (which the model writes constantly and strict JSON rejects)."""
    text = re.sub(r"```(?:json)?", "", text or "")
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            for candidate in (text[i:], _literal_backslashes(text[i:])):
                try:
                    found = decoder.raw_decode(candidate)[0]
                except ValueError:
                    continue
                if candidate is text[i:] and _mangled(found):
                    continue
                return found
    return None


#: What LaTeX like \beta, \frac, \tau and \rho turns into when a strict parse reads its first
#: letters as escapes. None of these belongs in text we ask for. (\n is left alone: real
#: newlines are legitimate, so \nu and \nabla cannot be told apart from them.)
_MANGLED = ("\x08", "\x0c", "\t", "\r")


def _mangled(value: Any) -> bool:
    if isinstance(value, str):
        return any(c in value for c in _MANGLED)
    if isinstance(value, dict):
        return any(_mangled(v) for v in value.values())
    if isinstance(value, list):
        return any(_mangled(v) for v in value)
    return False
