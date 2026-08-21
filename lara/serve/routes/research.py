"""The long-running, model-driven endpoints: conversation threads, deep research, ask.

Two of these stream Server-Sent Events, and both use the same shape: a driver coroutine
runs the real work and pushes results through ``emit``, while the endpoint drains a queue
and frames whatever arrives. The work itself lives in :mod:`lara.serve.agent` and
:mod:`lara.serve.synthesis` — an SSE generator is transport, not a place to keep control
flow.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from lara.serve.agent import AskRequest, run_ask
from lara.serve.deps import memory_root, require_state

router = APIRouter()

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


Driver = Callable[[Callable[[str, object], None], Callable[[], bool]], Awaitable[None]]


def _sse(run: Driver, *, hard_stop: bool) -> StreamingResponse:
    """Run ``run(emit, should_stop)`` in the background and stream everything it emits.

    ``hard_stop`` decides what a closed browser tab means. For ``/api/ask`` it means
    stop now: nobody is reading the tokens and the GPU has better things to do. For
    synthesis it does not, because a run is minutes of retrieval that gets saved and can
    be reopened later — it is asked to wind down and consolidate what it has instead of
    being abandoned mid-flight.
    """
    queue: asyncio.Queue = asyncio.Queue()
    cancelled = {"v": False}

    def emit(name: str, payload: object) -> None:
        queue.put_nowait((name, payload))

    async def driver() -> None:
        try:
            await run(emit, lambda: cancelled["v"])
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            emit("error", str(exc)[:400])
        finally:
            queue.put_nowait((None, None))

    async def events() -> AsyncIterator[str]:
        task = asyncio.create_task(driver())
        try:
            while True:
                name, payload = await queue.get()
                if name is None:
                    break
                yield f"event: {name}\ndata: {json.dumps(payload, default=str)}\n\n"
        finally:
            # Reached on a normal finish too, where the task is already done and this is
            # a no-op; on client disconnect the generator is closed here instead.
            cancelled["v"] = True
            if hard_stop and not task.done():
                task.cancel()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


# ── the library as a graph ────────────────────────────────────────────────────────────

@router.get("/api/library/graph")
async def library_graph(refresh: bool = False, model: str | None = None) -> JSONResponse:
    """The library as a directed graph of conversations.

    Cached against a fingerprint of the question entries, so it is rebuilt when the
    library changes and not on every render — it costs two model calls over the whole
    library.
    """
    from lara.serve import library_graph as LG
    from lara.serve.generate import stream_answer

    s = require_state()
    root = memory_root()
    if not refresh:
        hit = LG.cached(root)
        if hit is not None:
            return JSONResponse({**hit, "cached": True})

    graph = await LG.build(s.cfg, root, model, stream_answer)
    if graph.get("nodes"):
        LG.store(root, graph)
    return JSONResponse({**graph, "cached": False})


# ── conversation threads ──────────────────────────────────────────────────────────────

class CompressRequest(BaseModel):
    paper: str | None = None
    model: str | None = None
    keep: int = 2


@router.post("/api/thread/compress")
async def thread_compress(req: CompressRequest) -> JSONResponse:
    """Summarise a thread's older turns so the conversation can keep going.

    The most recent turns stay verbatim — they are what follow-ups point at, so
    compressing them would break the reference resolution this exists to protect.
    """
    from lara.serve import thread as TH
    from lara.serve.generate import stream_answer

    s = require_state()
    out = await TH.compress(s.cfg, memory_root(), req.paper, req.model, stream_answer,
                            keep=max(0, int(req.keep)))
    return JSONResponse(out)


@router.post("/api/thread/uncompress")
def thread_uncompress(req: CompressRequest) -> JSONResponse:
    """Drop a thread's summary and send the full history again."""
    from lara.serve import memory as MEM
    from lara.serve import thread as TH

    ok = MEM.clear_thread_summary(memory_root(), TH.thread_id(req.paper))
    return JSONResponse({"ok": ok})


@router.get("/api/thread/state")
def thread_state(paper: str | None = None) -> JSONResponse:
    """What the model will be sent for this thread, and what has been compressed away."""
    from lara.serve import memory as MEM
    from lara.serve import thread as TH

    root = memory_root()
    tid = TH.thread_id(paper)
    summary = MEM.get_thread_summary(root, tid)
    live = TH.turns_for(root, paper)
    return JSONResponse({
        "thread": tid,
        "summary": summary,
        "compressed_turns": len(MEM.get_thread_summary_covered(root, tid)),
        "live_turns": len(live),
        "total_turns": len(TH.turns_for(root, paper, limit=999, include_compressed=True)),
        "history_chars": len(TH.history_block(live, summary=summary)),
    })


# ── deep research ─────────────────────────────────────────────────────────────────────

class SynthRequest(BaseModel):
    question: str
    model: str | None = None


@router.post("/api/synthesize")
async def synthesize(req: SynthRequest) -> StreamingResponse:
    """Deep automated research, streamed as it happens.

    Runs for minutes rather than seconds, so every intermediate result is emitted the
    moment it exists: a run whose progress is invisible is indistinguishable from one that
    has hung, and the user is being asked to wait on faith otherwise.
    """
    s = require_state()
    assert s.retriever is not None

    from lara.serve import synthesis as SY
    from lara.serve.generate import stream_answer

    async def run(emit, should_stop) -> None:
        await SY.run_synthesis(
            s, s.cfg, req.question, model=req.model, stream_answer=stream_answer,
            emit=emit, should_stop=should_stop,
        )

    return _sse(run, hard_stop=False)


@router.get("/api/synthesis/runs")
def synthesis_runs(limit: int = 50) -> JSONResponse:
    """Past runs, newest first."""
    from lara.serve import synthesis as SY

    return JSONResponse({"runs": SY.list_runs(require_state().conn(), limit)})


@router.get("/api/synthesis/run/{run_id}")
def synthesis_run(run_id: str) -> JSONResponse:
    """One saved run in full: rounds, claims and both answers."""
    from lara.serve import synthesis as SY

    d = SY.load_run(require_state().conn(), run_id)
    if d is None:
        raise HTTPException(404, f"no synthesis run {run_id}")
    return JSONResponse(d)


# ── ask ───────────────────────────────────────────────────────────────────────────────

@router.post("/api/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    """Stream a grounded answer over SSE.

    Streaming is the difference between ~300 ms to first token and 5–10 s to a complete
    answer. If ``hits`` are supplied the client already retrieved them speculatively, so
    generation starts immediately. The loop itself is ``agent.run_ask``.
    """
    s = require_state()
    assert s.retriever is not None

    async def run(emit, _should_stop) -> None:
        await run_ask(s, s.cfg, req, emit=emit)

    return _sse(run, hard_stop=True)
