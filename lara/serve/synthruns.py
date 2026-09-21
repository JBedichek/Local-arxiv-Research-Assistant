"""Goal-graph synthesis runs: start one, watch it live, keep what it produced.

A run is a `synthesizer.SynthesizerState` grown against the paper corpus, then written up as
a deliverable in three lengths (full, medium, short), with recommended follow-ups and a few
distilled facts kept for later runs. It executes as a background task of the server so a
page reload or a closed tab stops nothing.

Two things are persisted under `~/.lara/synthesizer`: the graph itself (`<id>.json`, written
by `synthesizer.save` after every change) and the run record (`runs/<id>.json`: goal,
status, the three deliverables, follow-ups). While a run is live an in-memory `Feed` carries
the change events the UI streams; a finished run is read from disk.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import types
import uuid
from pathlib import Path

import httpx

from lara.serve import citations as C
from lara.serve import deliverable as DL
from lara.serve import facts as FA
from lara.serve import interest as IN
from lara.serve import synthesizer as SY

log = logging.getLogger(__name__)

RUNS = SY.STATES / "runs"

RUNNING, DONE, FAILED = "running", "done", "failed"
CANCELLED, INTERRUPTED = "cancelled", "interrupted"
LIVE = (RUNNING,)

#: Window assumed when the server does not report one.
DEFAULT_WINDOW = 32_768

#: The run record's own keys a listing shows; the deliverables and references are large.
_LIST_KEYS = ("id", "goal", "status", "created", "ended", "verdict", "parent", "imported",
              "tokens_in", "tokens_out", "rounds")

_ACTIVE: dict[str, "Feed"] = {}
_TASKS: dict[str, asyncio.Task] = {}
#: The in-memory graph of each live run, so a watcher's snapshot is never behind the feed.
_GRAPHS: dict[str, SY.SynthesizerState] = {}


# ── records ───────────────────────────────────────────────────────────────────────


def _path(run_id: str, root: Path | None = None) -> Path:
    return (root or RUNS) / f"{run_id}.json"


def save_record(rec: dict, *, root: Path | None = None) -> bool:
    """Atomic write; never raises, since losing the record must not stop the run. Returns
    whether it was written, for a caller (an import) that must not report a write that failed."""
    where = root or RUNS
    try:
        where.mkdir(parents=True, exist_ok=True)
        tmp = where / f"{rec['id']}.writing"
        tmp.write_text(json.dumps(rec, indent=1, default=str))
        tmp.replace(_path(rec["id"], where))
        return True
    except Exception as exc:                                   # noqa: BLE001
        log.warning("synthruns: could not persist %s: %s", rec.get("id"), exc)
        return False


def load_record(run_id: str, *, root: Path | None = None) -> dict | None:
    """The stored record, with a run that was live when the server died marked interrupted."""
    try:
        rec = json.loads(_path(run_id, root).read_text())
    except FileNotFoundError:
        return None
    except Exception as exc:                                   # noqa: BLE001
        log.warning("synthruns: could not read %s: %s", run_id, exc)
        return None
    if rec.get("status") in LIVE and run_id not in _ACTIVE:
        rec["status"] = INTERRUPTED
        rec["ended"] = rec.get("ended") or time.time()
        save_record(rec, root=root)
    return rec


def list_records(*, root: Path | None = None) -> list[dict]:
    """Newest first, without the deliverables."""
    out = []
    for p in (root or RUNS).glob("*.json"):
        rec = load_record(p.stem, root=root)
        if rec is not None:
            out.append({k: rec.get(k) for k in _LIST_KEYS}
                       | {"has_deliverable": bool((rec.get("deliverable") or "").strip())})
    return sorted(out, key=lambda r: r.get("created") or 0, reverse=True)


def delete_record(run_id: str, *, root: Path | None = None) -> bool:
    """Remove a finished run's record and graph. A live run must be cancelled first."""
    if run_id in _ACTIVE:
        return False
    found = False
    for p in (_path(run_id, root), (root or RUNS).parent / f"{run_id}.json"):
        with contextlib.suppress(FileNotFoundError):
            p.unlink()
            found = True
    return found


# ── the live feed ─────────────────────────────────────────────────────────────────


class Feed:
    """The change events of one live run, replayable by any number of watchers."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []
        self.closed = False
        self._waiters: list[asyncio.Future] = []

    def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))
        self._wake()

    def close(self) -> None:
        self.closed = True
        self._wake()

    def _wake(self) -> None:
        waiters, self._waiters = self._waiters, []
        for w in waiters:
            if not w.done():
                w.set_result(None)

    async def wait(self) -> None:
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        await fut


def _frame(name: str, payload: object) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, default=str)}\n\n"


async def stream(run_id: str):
    """SSE frames for one run: a `snapshot`, then each change while it is live, then `end`.

    The snapshot and the position in the feed are taken with no await between them, so no
    event falls in the gap.
    """
    rec = load_record(run_id)
    if rec is None:
        yield _frame("end", {"because": "no such run"})
        return
    feed = _ACTIVE.get(run_id)
    graph = _GRAPHS.get(run_id) or SY.load(run_id)
    yield _frame("snapshot", {"run": rec, "graph": graph.to_dict() if graph else None})
    if feed is None:
        yield _frame("end", {"because": rec.get("status", "")})
        return
    seen = len(feed.events)
    while True:
        while seen < len(feed.events):
            name, payload = feed.events[seen]
            seen += 1
            yield _frame(name, payload)
        if feed.closed:
            yield _frame("end", {"because": (load_record(run_id) or {}).get("status", "")})
            return
        await feed.wait()


# ── the leaf and the run ──────────────────────────────────────────────────────────


def _resolve_api_key(vcfg: dict) -> str:
    return vcfg.get("api_key") or os.environ.get("VLLM_API_KEY", "vllm-local")


async def _served_model(base_url: str, api_key: str) -> str:
    """The first model the server reports, for a config that names no default."""
    async with httpx.AsyncClient(timeout=10, headers={"Authorization": f"Bearer {api_key}"}) as c:
        r = await c.get(f"{base_url.rstrip('/')}/models")
        r.raise_for_status()
        data = r.json().get("data") or []
    if not data:
        raise RuntimeError("the model server reports no loaded model")
    return str(data[0]["id"])


class NoEvidence(RuntimeError):
    """A research leaf that gathered no claims at all."""


def leaf(app_state, cfg, *, run_synthesis=None, stream_answer=None):
    """The deep-research leaf `synthesizer.run` injects: one question in, thorough and tldr
    answers out, both with their citations bound to references."""
    if run_synthesis is None:
        from lara.serve.synthesis import run_synthesis
    if stream_answer is None:
        from lara.serve.generate import stream_answer

    async def aresearch(question: str, *, model=None, base_url=None, api_key=""):
        run = await run_synthesis(app_state, cfg, question, model=model,
                                  stream_answer=stream_answer)
        if not run.claims:
            # "Nothing found" is indistinguishable here from a model that was unreachable, and
            # either way the goal did not produce an answer: land it failed, where the digest
            # and the feed show it, rather than as a finding that the literature is silent.
            raise NoEvidence(f"no relevant evidence was gathered for this question "
                             f"({run.stopped_because or 'no reason recorded'})")
        known = C.from_claims(run.claims)
        conn = app_state.conn()
        return types.SimpleNamespace(
            thorough=C.bind(run.thorough, known=known, conn=conn),
            tldr=C.bind(run.tldr, known=known, conn=conn),
            stopped_because=run.stopped_because)

    return aresearch


def new_record(goal: str, *, options: dict | None = None, parent: str = "") -> dict:
    return {"id": uuid.uuid4().hex[:12], "goal": goal, "status": RUNNING,
            "created": time.time(), "ended": None, "parent": parent,
            "options": dict(options or {}), "verdict": None, "error": "",
            "rounds": 0, "tokens_in": 0, "tokens_out": 0,
            "deliverable": "", "references": {},
            "deliverable_medium": "", "deliverable_medium_references": {},
            "deliverable_short": "", "deliverable_short_references": {},
            "followups": []}


def note(rec: dict, feed: Feed, event: str, text: str) -> None:
    """A problem worth keeping: on the record, so it survives a reload, and on the feed."""
    rec.setdefault("notes", []).append(text)
    save_record(rec)
    feed.emit(event, {"error": text})


def _diff_emitter(state: SY.SynthesizerState, feed: Feed, rec: dict):
    """A `persist()` callback: saves the graph and the record, and emits what changed."""
    last_goals: dict[str, dict] = {}
    seen = {"compressed": 0, "round": None}

    def persist() -> None:
        SY.save(rec["id"], state)
        d = state.to_dict()
        for gid, g in d["goals"].items():
            prev = last_goals.get(gid)
            if prev is None:
                feed.emit("goal.new", g)
            elif prev.get("status") != g.get("status") or prev.get("summary") != g.get("summary"):
                feed.emit("goal.update", g)
        for i in range(seen["compressed"], len(d["compressed"])):
            feed.emit("compression", {"index": i, "text": d["compressed"][i]})
        if d["round"] != seen["round"]:
            feed.emit("round", {"round": d["round"], "tokens_in": d["tokens_in"],
                                "tokens_out": d["tokens_out"]})
        last_goals.clear()
        last_goals.update({gid: dict(g) for gid, g in d["goals"].items()})
        seen["compressed"], seen["round"] = len(d["compressed"]), d["round"]
        rec.update(rounds=state.round, tokens_in=state.tokens_in, tokens_out=state.tokens_out)
        save_record(rec)

    return persist


async def _finish_up(rec: dict, feed: Feed, *, cfg, model, window, conn, embedder,
                     references: dict) -> None:
    """Facts, the medium and short versions and the follow-ups. Each step is independent
    and a failure of one is an event, not a failed run: the deliverable is already written."""
    goal, deliverable = rec["goal"], rec["deliverable"]
    with contextlib.suppress(Exception):
        await FA.distill_facts(cfg, goal, deliverable, run_id=rec["id"], embedder=embedder,
                               model=model, window=window)
    known = DL.known_from_dicts(references)
    for level in ("medium", "short"):
        feed.emit("phase", {"name": f"writing the {level} version"})
        try:
            text, _keys, refs = await DL.condense(
                cfg, deliverable, level=level, goal=goal, model=model, window=window,
                conn=conn, known=known)
        except Exception as exc:                               # noqa: BLE001
            note(rec, feed, f"deliverable_{level}.failed",
                 f"The {level} version failed: {type(exc).__name__}: {exc}")
            continue
        if not text.strip():
            note(rec, feed, f"deliverable_{level}.empty",
                 f"No {level} version was written: the model returned nothing (or the call failed).")
            continue
        rec[f"deliverable_{level}"], rec[f"deliverable_{level}_references"] = text, refs
        save_record(rec)
        feed.emit(f"deliverable_{level}", {"text": text, "references": refs})
    feed.emit("phase", {"name": "choosing follow-ups"})
    try:
        suggestions = await IN.recommend_followups(
            cfg, types.SimpleNamespace(goal=goal), deliverable, run_id=rec["id"],
            embedder=embedder, model=model, window=window)
    except Exception as exc:                                   # noqa: BLE001
        note(rec, feed, "followups.failed", f"Follow-ups failed: {type(exc).__name__}: {exc}")
        return
    if suggestions:
        rec["followups"] = suggestions
        save_record(rec)
        feed.emit("followups", {"suggestions": suggestions})
    else:
        note(rec, feed, "followups.empty",
             "No follow-ups were written: the model returned nothing (or the call failed).")


async def generator(app_state, model: str | None = None) -> types.SimpleNamespace:
    """Where and how to call the model: `cfg`, `base_url`, `api_key`, `model`, `window`."""
    from lara.serve import generate as G

    cfg = app_state.cfg
    vcfg = cfg.get_in("serving.vllm") or {}
    base_url, api_key = vcfg.get("base_url", ""), _resolve_api_key(vcfg)
    model = model or vcfg.get("default_model") or await _served_model(base_url, api_key)
    window = (await G.context_limit(base_url, model, api_key=api_key)
              or int(vcfg.get("max_model_len") or DEFAULT_WINDOW))
    return types.SimpleNamespace(cfg=cfg, base_url=base_url, api_key=api_key, model=model,
                                 window=window)


async def _drive(rec: dict, feed: Feed, app_state, *, run=SY.run) -> None:
    opts = rec["options"]
    g = await generator(app_state, opts.get("model"))
    embedder = getattr(app_state.retriever, "embedder", None)
    embed = FA.embedder_fn(embedder) if embedder is not None else None

    state = _GRAPHS[rec["id"]] = SY.SynthesizerState(objective=rec["goal"])
    persist = _diff_emitter(state, feed, rec)
    persist()
    result = await run(
        state, base_url=g.base_url, model=g.model, api_key=g.api_key, max_model_len=g.window,
        aresearch=leaf(app_state, g.cfg), on_change=persist, conn=app_state.conn(),
        deliverable_tokens=opts.get("deliverable_tokens") or None,
        allow_subsynthesis=bool(opts.get("allow_subsynthesis", True)),
        final_compression_prompt=str(opts.get("final_compression_prompt") or ""), embed=embed)

    rec.update(deliverable=result.deliverable, references=result.references,
               verdict=SY.verdict_for(result), rounds=result.rounds,
               tokens_in=result.tokens_in, tokens_out=result.tokens_out)
    if result.total_done == 0:
        # The deliverable is a placeholder. Condensing it, distilling "facts" from it and
        # suggesting follow-ups would dress an outage up as a finished run.
        rec["status"], rec["ended"] = FAILED, time.time()
        rec["error"] = (
            "No sub-question was researched successfully"
            + (f": {result.silent_reason_rounds} of {result.rounds} reasoning round(s) returned "
               "nothing. Check that the model server is up and was started with tool calling "
               "(--enable-auto-tool-choice and a --tool-call-parser matching the model; see "
               "serving.vllm.tool_call_parser)." if result.silent_reason_rounds
               else f" ({result.total_failed} failed)."))
        save_record(rec)
        feed.emit("verdict", rec["verdict"])
        return
    save_record(rec)
    feed.emit("deliverable", {"text": result.deliverable, "references": result.references})
    await _finish_up(rec, feed, cfg=g.cfg, model=g.model, window=g.window,
                     conn=app_state.conn(), embedder=embedder, references=result.references)
    rec["status"], rec["ended"] = DONE, time.time()
    save_record(rec)
    feed.emit("verdict", rec["verdict"])


async def _supervise(rec: dict, feed: Feed, app_state, driver) -> None:
    try:
        await driver(rec, feed, app_state)
    except asyncio.CancelledError:
        rec["status"] = CANCELLED
        raise
    except Exception as exc:                                   # noqa: BLE001
        log.exception("synthruns: run %s failed", rec["id"])
        rec["status"], rec["error"] = FAILED, f"{type(exc).__name__}: {exc}"[:400]
    finally:
        rec["ended"] = rec.get("ended") or time.time()
        save_record(rec)
        feed.close()
        _ACTIVE.pop(rec["id"], None)
        _TASKS.pop(rec["id"], None)
        _GRAPHS.pop(rec["id"], None)


def start(app_state, goal: str, *, options: dict | None = None, parent: str = "",
          driver=_drive) -> dict:
    """Begin a run in the background and return its record."""
    rec = new_record(goal, options=options, parent=parent)
    feed = _ACTIVE[rec["id"]] = Feed()
    save_record(rec)
    _TASKS[rec["id"]] = asyncio.get_running_loop().create_task(
        _supervise(rec, feed, app_state, driver))
    return rec


def cancel(run_id: str) -> bool:
    task = _TASKS.get(run_id)
    if task is None:
        return False
    task.cancel()
    return True


# ── after the fact ────────────────────────────────────────────────────────────────


async def compress(rec: dict, prompt: str, *, base_url: str, model: str, api_key: str,
                   max_model_len: int, answer=SY.answer_from_deliverable) -> dict:
    """Answer one question from a finished run's own deliverable, in place: the answer goes
    above the full report, and a second call answers from the original report, never from
    a previous answer plus the report."""
    full = SY.strip_prior_answer(rec["deliverable"])
    text, degraded, because, tin, tout = await answer(
        full, prompt, base_url=base_url, model=model, max_model_len=max_model_len,
        api_key=api_key)
    wrapped = SY.wrap_with_answer(prompt, text, full)
    # A degraded call saves a slice of the report as the "answer"; say so in the text, where a
    # reload will still show it, the way a fresh run's deliverable does.
    rec["deliverable"] = (SY._pressure_note(because) + "\n\n" + wrapped) if degraded else wrapped
    rec["tokens_in"] = int(rec.get("tokens_in") or 0) + tin
    rec["tokens_out"] = int(rec.get("tokens_out") or 0) + tout
    save_record(rec)
    return {"deliverable": rec["deliverable"], "degraded": degraded, "degraded_because": because}


async def refresh_followups(rec: dict, *, cfg, model: str, window: int, embedder=None) -> list[str]:
    """Five different follow-ups for a finished run. The current set is logged as shown
    first if it never was, and the model is shown everything already shown."""
    old = list(rec.get("followups") or [])
    if old and not IN.was_shown(rec["id"]):
        IN.record_shown(rec["id"], rec["goal"], old)
    avoid = IN.previously_shown(rec["id"]) or old
    fresh = await IN.recommend_followups(
        cfg, types.SimpleNamespace(goal=rec["goal"]), rec["deliverable"], run_id=rec["id"],
        embedder=embedder, model=model, window=window, avoid=avoid, learn=False)
    if fresh:
        IN.record_shown(rec["id"], rec["goal"], fresh)
        rec["followups"] = fresh
        save_record(rec)
    return fresh
