"""FastAPI app for the reader (R3, R4, R6, R8, R9).

**Endpoints are deliberately `def`, not `async def`.** Retrieval touches the GPU and
SQLite, both synchronous and both holding the GIL for milliseconds at a time. Declared
`async def`, that work runs *on the event loop* and stalls every other in-flight request —
which is what makes an otherwise fast server feel unresponsive the moment two people use
it. Starlette runs sync endpoints in a threadpool instead, so a slow retrieval blocks only
its own request. The one genuinely async endpoint is the SSE stream, which is I/O-bound
and yields between tokens.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from lara.serve import papers as papers_mod
from lara.serve.state import AppState

WEB_ROOT = Path(__file__).resolve().parent.parent.parent / "web"

app = FastAPI(title="Local arXiv Research Assistant", docs_url="/api/docs")
state: AppState | None = None


@app.on_event("startup")
def _startup() -> None:
    global state
    state = AppState(os.environ.get("LARA_CONFIG"), load_models=os.environ.get("LARA_NO_MODELS") != "1")


def _state() -> AppState:
    if state is None or not state.ready:
        raise HTTPException(503, "still warming up")
    return state


# ── UI ────────────────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/app.js")
def appjs() -> FileResponse:
    return FileResponse(WEB_ROOT / "app.js", media_type="application/javascript")


@app.get("/style.css")
def appcss() -> FileResponse:
    return FileResponse(WEB_ROOT / "style.css", media_type="text/css")


@app.get("/p/{arxiv_id:path}")
def reader(arxiv_id: str) -> FileResponse:
    """Deep links land here; the client reads the id and fragment from the URL."""
    return FileResponse(WEB_ROOT / "index.html")


# ── data ──────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    s = state
    return {
        "ready": bool(s and s.ready),
        "warmup_ms": (s.warmup_ms if s else {}),
        "vectors": (s.store.rows() if s else 0),
    }


@app.get("/api/paper/{arxiv_id:path}")
def get_paper(arxiv_id: str) -> JSONResponse:
    s = _state()
    row = s.paper(arxiv_id)
    if row is None:
        raise HTTPException(404, f"{arxiv_id} not in corpus")

    version = row["fulltext_version"] or row["latest_version"] or 1
    path = s.raw_html_path(arxiv_id)
    body = papers_mod.render(str(path), arxiv_id) if path else ""

    return JSONResponse({
        "arxiv_id": arxiv_id,
        "title": row["title"],
        "abstract": row["abstract"],
        "authors": row["authors"],
        "categories": row["categories"],
        "submitted": row["submitted_utc"],
        "version": version,
        "fulltext_status": row["fulltext_status"],
        "fulltext_source": row["fulltext_source"],
        "n_chunks": row["n_chunks"],
        "html": body,
        "sections": [dict(r) for r in s.sections(arxiv_id, version)],
    })


class RetrieveRequest(BaseModel):
    query: str
    selection: str | None = None
    paper: str | None = None
    scope: str = "corpus"          # corpus | paper | neighbourhood
    k: int = 8


@app.post("/api/retrieve")
def retrieve(req: RetrieveRequest) -> JSONResponse:
    """Retrieval only — no generation.

    The client fires this speculatively the moment a selection is made, before the user
    has finished typing. By submit time the chunks are already in hand, which removes the
    retrieval leg from perceived latency entirely.
    """
    s = _state()
    assert s.retriever is not None

    restrict: list[str] | None = None
    if req.scope == "paper" and req.paper:
        restrict = [req.paper]
    elif req.scope == "neighbourhood" and req.paper:
        n = s.neighbours(req.paper)
        restrict = [req.paper, *n["cites"], *n["cited_by"]]

    s.retriever.final_k = max(1, min(req.k, 32))
    result = s.retriever.retrieve(req.query, papers=restrict, selection=req.selection)
    return JSONResponse({
        "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
        "candidates": result.n_candidates,
        "hits": [
            {
                "chunk_id": h.chunk_id, "arxiv_id": h.arxiv_id, "version": h.version,
                "url": h.fragment(), "anchor": h.anchor_start,
                "char_start": h.char_start, "char_end": h.char_end,
                "anchor_end": h.anchor_end, "section": h.section_title or h.section_anchor,
                "kind": h.kind, "score": round(h.score, 4),
                "paper_title": h.paper_title, "text": h.text,
            }
            for h in result.hits
        ],
    })


@app.get("/api/graph/{arxiv_id:path}")
def graph(arxiv_id: str, query: str = Query(default=""), limit: int = 60) -> JSONResponse:
    """Ego network with similarity shading (R8, R9)."""
    s = _state()
    assert s.retriever is not None
    n = s.neighbours(arxiv_id, limit=limit)
    ids = list(dict.fromkeys([arxiv_id, *n["cites"], *n["cited_by"]]))[: limit * 2]

    titles, present = {}, set()
    if ids:
        ph = ",".join("?" * len(ids))
        for r in s.conn().execute(
            f"SELECT arxiv_id, title, cited_by_count FROM papers WHERE arxiv_id IN ({ph})", ids
        ):
            titles[r["arxiv_id"]] = {"title": r["title"], "cited_by": r["cited_by_count"] or 0}
            present.add(r["arxiv_id"])

    heat: dict[str, float] = {}
    if query.strip():
        from lara.index.embed import embed_queries
        q = embed_queries(s.retriever.embedder, [query])[0][: s.retriever.dim_trunc]
        q = q / max(float((q ** 2).sum() ** 0.5), 1e-12)
        heat = s.retriever.dense.paper_scores(q, s.paper_rows(list(present)), top_n=3)

    return JSONResponse({
        "root": arxiv_id,
        "nodes": [
            {
                "id": i,
                "title": titles.get(i, {}).get("title", ""),
                "cited_by": titles.get(i, {}).get("cited_by", 0),
                "in_corpus": i in present,
                "heat": round(heat.get(i, 0.0), 4),
            }
            for i in ids
        ],
        "edges": (
            [{"src": arxiv_id, "dst": d} for d in n["cites"] if d in ids]
            + [{"src": srcid, "dst": arxiv_id} for srcid in n["cited_by"] if srcid in ids]
        ),
    })


@app.get("/api/models")
def models() -> JSONResponse:
    """Generators available from the HF cache (R6, R7)."""
    s = _state()
    from lara.models import scan

    found = scan(s.cfg.get_path("huggingface.home"))
    return JSONResponse({
        "models": [
            {
                "repo": m.repo, "arch": m.arch, "size_gb": round(m.size_gb, 1),
                "quantization": m.quantization,
                "quant_options": m.runtime_quant_options(),
            }
            for m in found if m.servable
        ],
        "rejected": len([m for m in found if not m.servable]),
    })


class AskRequest(BaseModel):
    query: str
    selection: str | None = None
    paper: str | None = None
    scope: str = "corpus"
    hits: list[dict] | None = None       # reuse speculative retrieval; skip re-retrieving
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1024


@app.post("/api/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    """Stream a grounded answer over SSE.

    Streaming is the difference between ~300 ms to first token and 5–10 s to a complete
    answer. If ``hits`` are supplied the client already retrieved them speculatively, so
    generation starts immediately.
    """
    s = _state()
    assert s.retriever is not None

    hits = req.hits
    if hits is None:
        restrict = [req.paper] if (req.scope == "paper" and req.paper) else None
        result = s.retriever.retrieve(req.query, papers=restrict, selection=req.selection)
        hits = [
            {"arxiv_id": h.arxiv_id, "version": h.version, "url": h.fragment(),
             "section": h.section_title or h.section_anchor, "text": h.text,
             "paper_title": h.paper_title}
            for h in result.hits
        ]

    from lara.serve.generate import stream_answer

    async def events():
        yield f"event: hits\ndata: {json.dumps(hits)}\n\n"
        try:
            async for token in stream_answer(
                s.cfg, req.query, hits, selection=req.selection,
                model=req.model, temperature=req.temperature, max_tokens=req.max_tokens,
            ):
                yield f"event: token\ndata: {json.dumps(token)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps(str(exc)[:300])}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
