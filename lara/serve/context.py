"""Sizing a model call against the window it will run in.

`reply_room` gives a reply whatever the window has left once the prompt is in it, `clipped`
and `fitted` cut text to a budget and say so when they do, and `budget_for` turns a token
window into a character budget. Nothing here enforces the window at assembly time: these
take numbers, not a window, so a caller whose prompt can outgrow it should size it first.
"""
from __future__ import annotations

#: Source code runs ~3.8 characters per token, prose a little higher.
CHARS_PER_TOKEN = 3.8

#: Share of the model's window one context block may take; the rest carries the
#: instructions and the reply.
WINDOW_FRACTION = 0.5

#: Head-room kept back beyond the prompt and the reply: the chat template, the system prompt
#: as the server counts it, and the gap between `CHARS_PER_TOKEN` and the real tokenizer.
WINDOW_MARGIN_TOKENS = 1_500

#: The estimate error scales with the prompt, so the margin is a fraction of it, not a
#: constant. A flat 1,500 was measured 10% short on a real prompt and the request was refused.
WINDOW_MARGIN_FRACTION = 0.15


def budget_for(max_model_len: int, *, fraction: float = WINDOW_FRACTION,
               chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """Characters of context a given model window can afford."""
    return max(8_000, int(max_model_len * fraction * chars_per_token))


def clipped(text: str, limit: int, *, what: str = "text") -> str:
    """`text`, cut to `limit` -- and saying so, in the text, when it cuts.

    A cut that announces itself is recoverable: the model can say it needed the rest. One
    that does not leaves it reasoning from half a text while the UI shows the whole.
    """
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    return (text[:limit]
            + f"\n\n[… cut here: {limit:,} of {len(text):,} characters of {what} shown, "
              f"{len(text) - limit:,} not. Say so if you needed the rest.]")


def fitted(full: str, short: str, limit: int, *, what: str = "text") -> str:
    """The longest whole version that fits, rather than a cut-down of the longest one.

    A complete short answer beats the opening fifth of a long one -- the cut end is where
    the conclusion lives. Truncation is the last resort, and it still announces itself.
    """
    full, short = full or "", short or ""
    if limit <= 0 or len(full) <= limit:
        return full
    if short and len(short) <= limit:
        return (short + f"\n\n[this is the short version of {what}: the full one is "
                        f"{len(full):,} characters and {limit:,} were available. Say so "
                        f"if you needed the detail.]")
    return clipped(full or short, limit, what=what)


def reply_room(window: int, *parts: str, default: int, floor: int = 2_000,
               cap: int = 0, stage: str = "") -> int:
    """How long a reply may be: whatever this window has left once the prompt is in it.

    `window` of 0 means nobody could say how wide it is, and `default` stands in. `floor`
    is what to ask for when the prompt has all but filled the window -- a refused request
    is better than a reply cut off at 200 tokens. `cap` is the other end: room left is what
    a reply *may* use, and a model that never emits a stop token decodes until it reaches
    the number it was given. `stage` names the caller and is unused here.
    """
    if window <= 0:
        return min(default, cap) if cap else default
    used = int(sum(len(p) for p in parts) / CHARS_PER_TOKEN)
    margin = max(WINDOW_MARGIN_TOKENS, int(used * WINDOW_MARGIN_FRACTION))
    room = max(floor, window - used - margin)
    return max(floor, min(room, cap)) if cap else room
