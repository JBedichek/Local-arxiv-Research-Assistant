"""Tokens per second, read from vLLM's own `/metrics` -- not estimated, not tracked at
each call site. The same mechanism `autoresearch/context_view.py` uses (scrape the
Prometheus counter, keep the last sample, report the delta as a rate), scaled down for one
generator replica instead of a pool of cards.

**Observed, not instrumented.** Nothing here is recorded by the code that sends requests,
so no call site has to remember to update it. **Deliberately not smoothed** -- a rate
refreshed every couple of seconds should show generation stopping the moment it does, and
smoothing would hide exactly that.
"""
from __future__ import annotations

import re
import time
from urllib.parse import urlsplit

import httpx

#: Short: this is a display, not a control path, and a replica busy enough to be slow here
#: is a replica whose number is about to change anyway.
METRICS_TIMEOUT = 2.0

#: vLLM has shipped this counter under more than one name across versions. An unexposed
#: metric parses as absent, which reads as "not generating" -- the same picture an
#: idle-but-reachable replica gives, so getting the name wrong costs a stale display, not
#: a crash.
_COUNTER = re.compile(r"^vllm:generation_tokens_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", re.M)

#: (timestamp, tokens_total) from the last read, keyed by base_url so this still works if
#: lara is ever pointed at more than one replica in the same process.
_last: dict[str, tuple[float, float]] = {}


def _metrics_url(base_url: str) -> str:
    """vLLM serves `/metrics` at the root, not under `/v1` -- `base_url` is `.../v1`."""
    parts = urlsplit(base_url)
    return f"{parts.scheme}://{parts.netloc}/metrics"


async def rate(base_url: str, *, timeout: float = METRICS_TIMEOUT,
               client: httpx.AsyncClient | None = None, now: float | None = None) -> dict:
    """{"tokens_per_sec", "reachable"}. 0.0 on the very first read of a replica (nothing to
    take a delta against yet) or whenever `/metrics` cannot be read or does not expose the
    counter -- never raises, since a card that will not answer is a fact about the card,
    not a reason to break the display asking about it.

    `client`, when given, is used as-is instead of opening a fresh connection each poll --
    tests pass one built on `httpx.MockTransport`; every real caller leaves this to open
    and close its own, since a poll every couple of seconds does not warrant holding a
    connection open between them. `now`, when given, stands in for the current time --
    tests control it directly rather than monkeypatching `time.time` process-wide, which
    would also shift whatever httpx and asyncio time themselves by while a test runs."""
    now = time.time() if now is None else now
    try:
        if client is not None:
            r = await client.get(_metrics_url(base_url))
        else:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.get(_metrics_url(base_url))
        r.raise_for_status()
        body = r.text
    except Exception:                                      # noqa: BLE001
        return {"tokens_per_sec": 0.0, "reachable": False}

    m = _COUNTER.search(body)
    if not m:
        return {"tokens_per_sec": 0.0, "reachable": False}
    try:
        total = float(m.group(1))
    except ValueError:
        return {"tokens_per_sec": 0.0, "reachable": False}

    before = _last.get(base_url)
    _last[base_url] = (now, total)
    if before is None:
        return {"tokens_per_sec": 0.0, "reachable": True}
    then, old_total = before
    dt = now - then
    if dt <= 0:
        return {"tokens_per_sec": 0.0, "reachable": True}
    return {"tokens_per_sec": round(max(0.0, total - old_total) / dt, 1), "reachable": True}
