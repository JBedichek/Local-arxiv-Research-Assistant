"""Every prompt, retrieval and tool call behind one concept's build, for the Profile tab.

A plain, append-only JSONL log, one line per event, overwritten at the start of each build --
the same shape `autoresearch/serve/runs.py`'s uncapped, disk-only "verbose" log uses for the
same reason: high-volume, rich payloads, never worth holding in memory or folding into the
small, capped state a course/concept file carries.

Reachable from anywhere in the call stack through two contextvars rather than a parameter
threaded through every function between `pipeline.build_concept` and `Llm.ask` three modules
away: `asyncio.gather` copies the current context into each child task, so a facet's own round
(`claims.build`'s `one`) or a section's own research/write (`depth.py`'s `one`/`write_section`)
can label only its own events by calling `set_phase` on its own task's context, without
touching its siblings or its parent -- exactly the isolation concurrent facets and sections
need, for free."""

from __future__ import annotations

import contextvars
import json
import time
from pathlib import Path

_TRACER: contextvars.ContextVar["Tracer | None"] = contextvars.ContextVar(
    "learn_tracer", default=None)
_PHASE: contextvars.ContextVar[str] = contextvars.ContextVar("learn_trace_phase", default="setup")

#: A page of trace kept in memory at once when serving a poll -- the file itself is unbounded.
READ_LIMIT = 500


class Tracer:
    """Appends one JSON object per line to `path`, truncated at construction. Writes are
    synchronous and the file is reopened each call rather than held open: a build emits at
    most a few hundred events, so the extra opens cost nothing, and there is then no flush or
    close lifecycle to manage across a task that may be cancelled mid-build."""

    def __init__(self, path: Path):
        self.path = path
        self.seq = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def write(self, phase: str, type_: str, **payload) -> None:
        self.seq += 1
        row = {"seq": self.seq, "ts": time.time(), "phase": phase, "type": type_, **payload}
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")


def start(path: Path) -> Tracer:
    """A fresh tracer, made current for this task (and everything it awaits or gathers)."""
    tracer = Tracer(path)
    _TRACER.set(tracer)
    _PHASE.set("setup")
    return tracer


def stop() -> None:
    _TRACER.set(None)


def set_phase(label: str) -> None:
    """Labels every event `emit`ted from here on, in this task's context only -- a facet's own
    round, or a section's own research/write, calls this once at its own start."""
    _PHASE.set(label)


def emit(type_: str, **payload) -> None:
    """No-op when nothing is tracing, so call sites never need to check first."""
    tracer = _TRACER.get()
    if tracer is not None:
        tracer.write(_PHASE.get(), type_, **payload)


def read(path: Path, *, since: int = 0, limit: int = READ_LIMIT) -> list[dict]:
    """Rows with seq > `since`, oldest first, capped at `limit` for one poll response."""
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("seq", 0) > since:
                out.append(row)
                if len(out) >= limit:
                    break
    return out
