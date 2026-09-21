"""`/api/synthesizer` -- goal-graph synthesis runs: start, watch live, read, compress, follow up.

A run executes in the background (see `lara.serve.synthruns`); the events endpoint streams its
graph as it grows, and a finished run is read from disk."""

from __future__ import annotations

import contextlib

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from lara.serve import interest as IN
from lara.serve import synthesizer as SY
from lara.serve import synthruns as SR
from lara.serve.deps import require_state

router = APIRouter()

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class StartRequest(BaseModel):
    goal: str
    parent: str = ""
    model: str | None = None
    allow_subsynthesis: bool = True
    deliverable_tokens: int = 0
    final_compression_prompt: str = ""


class PromptRequest(BaseModel):
    prompt: str


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _finished(run_id: str):
    """(record, None) for a finished run with a deliverable, else (None, error response)."""
    rec = SR.load_record(run_id)
    if rec is None:
        return None, _error(f"no synthesis run {run_id}", 404)
    if rec["status"] in SR.LIVE:
        return None, _error("that run is still going — wait for it, or cancel it first", 409)
    if not (rec.get("deliverable") or "").strip():
        return None, _error("that run produced no deliverable", 400)
    return rec, None


@router.get("/api/synthesizer/runs")
def runs() -> JSONResponse:
    return JSONResponse({"runs": SR.list_records()})


@router.post("/api/synthesizer/runs")
async def start(req: StartRequest) -> JSONResponse:
    goal = req.goal.strip()
    if not goal:
        return _error("a goal is required", 400)
    state = require_state()
    if state.retriever is None:
        return _error("the retriever is not loaded", 503)
    parent = ""
    if req.parent:
        prec = SR.load_record(req.parent)
        if prec is None:
            return _error(f"no synthesis run {req.parent}", 404)
        parent = prec["id"]
        # A follow-up submitted exactly as suggested is the accept signal the interest
        # profile reads back; edited or free-typed text is not recorded.
        if goal in (prec.get("followups") or []):
            with contextlib.suppress(Exception):
                IN.record_clicked(parent, goal)
    opts = {"model": req.model, "allow_subsynthesis": req.allow_subsynthesis,
            "deliverable_tokens": max(0, req.deliverable_tokens),
            "final_compression_prompt": req.final_compression_prompt.strip()}
    return JSONResponse(SR.start(state, goal, options=opts, parent=parent), status_code=201)


@router.get("/api/synthesizer/runs/{run_id}")
def one(run_id: str) -> JSONResponse:
    rec = SR.load_record(run_id)
    if rec is None:
        return _error(f"no synthesis run {run_id}", 404)
    graph = SR._GRAPHS.get(run_id) or SY.load(run_id)
    return JSONResponse({"run": rec, "graph": graph.to_dict() if graph else None})


@router.get("/api/synthesizer/runs/{run_id}/events")
async def events(run_id: str) -> StreamingResponse:
    return StreamingResponse(SR.stream(run_id), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


@router.post("/api/synthesizer/runs/{run_id}/cancel")
def cancel(run_id: str) -> JSONResponse:
    if not SR.cancel(run_id):
        return _error("that run is not running", 409)
    return JSONResponse({"id": run_id, "cancelling": True})


@router.delete("/api/synthesizer/runs/{run_id}")
def delete(run_id: str) -> JSONResponse:
    if run_id in SR._ACTIVE:
        return _error("that run is still going — cancel it first", 409)
    if not SR.delete_record(run_id):
        return _error(f"no synthesis run {run_id}", 404)
    return JSONResponse({"deleted": True})


@router.post("/api/synthesizer/runs/{run_id}/compress")
async def compress(run_id: str, req: PromptRequest) -> JSONResponse:
    """Answer one question from a finished run's deliverable, without re-running anything."""
    rec, err = _finished(run_id)
    if err:
        return err
    prompt = req.prompt.strip()
    if not prompt:
        return _error("a prompt is required", 400)
    g = await SR.generator(require_state(), (rec.get("options") or {}).get("model"))
    out = await SR.compress(rec, prompt, base_url=g.base_url, model=g.model,
                            api_key=g.api_key, max_model_len=g.window)
    return JSONResponse({"id": run_id, "prompt": prompt, **out})


@router.post("/api/synthesizer/runs/{run_id}/followups/refresh")
async def refresh_followups(run_id: str) -> JSONResponse:
    """Replace a finished run's follow-ups with five different ones."""
    rec, err = _finished(run_id)
    if err:
        return err
    state = require_state()
    g = await SR.generator(state, (rec.get("options") or {}).get("model"))
    try:
        fresh = await SR.refresh_followups(
            rec, cfg=g.cfg, model=g.model, window=g.window,
            embedder=getattr(state.retriever, "embedder", None))
    except Exception as exc:                                   # noqa: BLE001
        return _error(f"the model call failed: {type(exc).__name__}: {exc}", 502)
    if not fresh:
        return _error("couldn't come up with follow-ups that differ from the ones already "
                      "shown for this run; the current set is unchanged", 422)
    return JSONResponse({"id": run_id, "followups": fresh})
