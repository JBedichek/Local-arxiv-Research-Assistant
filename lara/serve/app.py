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

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
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

    # k is passed per call rather than assigned to the shared retriever: endpoints run
    # concurrently in a threadpool, so mutating retriever state lets one request change
    # another's result count mid-flight.
    result = s.retriever.retrieve(
        req.query, papers=restrict, selection=req.selection,
        final_k=max(1, min(req.k, 32)),
    )
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


class SearchRequest(BaseModel):
    query: str
    limit: int = 20
    category: str | None = None
    year_from: int | None = None
    require_fulltext: bool = False


@app.post("/api/search")
def search_papers(req: SearchRequest) -> JSONResponse:
    """Semantic paper search — "show me papers on Muon learning rate adaptation".

    Fuses breadth (title+abstract vectors, which exist for every in-scope paper) with
    depth (full-text chunks aggregated per paper). Neither alone is adequate: abstracts
    miss methods introduced in section 4, and full text covers only the crawled minority.
    """
    import time

    from lara.index import paper_search as PS
    from lara.index.embed import embed_queries

    s = _state()
    assert s.retriever is not None
    t: dict[str, float] = {}
    clock = time.perf_counter

    t0 = clock()
    q_full = embed_queries(s.retriever.embedder, [req.query])[0]
    dim = s.retriever.dim_trunc
    q = q_full[:dim] / max(float(np.linalg.norm(q_full[:dim])), 1e-12)
    t["embed"] = (clock() - t0) * 1000

    # breadth: paper-level abstracts
    t0 = clock()
    paper_level: dict[str, float] = {}
    if s.paper_index is not None:
        rows, scores = s.paper_index.search(q, k=req.limit * 8)
        paper_level = {
            s.paper_row_to_id[int(r)]: float(sc)
            for r, sc in zip(rows.tolist(), scores.tolist())
            if int(r) in s.paper_row_to_id
        }
    t["abstracts"] = (clock() - t0) * 1000

    # depth: full-text chunks, aggregated per paper
    t0 = clock()
    crows, cscores = s.retriever.dense.search(q, k=req.limit * 25)
    chunk_level = PS.aggregate_chunk_hits(crows, cscores, s.chunk_row_to_paper(crows))
    t["fulltext"] = (clock() - t0) * 1000

    scored = PS.fuse(paper_level, chunk_level)
    hits = PS.hydrate_papers(s.conn(), scored, req.limit * 3)

    # filters applied after scoring so they never change relative ranking
    if req.category:
        hits = [h for h in hits if req.category in h.categories]
    if req.year_from:
        hits = [h for h in hits if h.submitted[:4].isdigit() and int(h.submitted[:4]) >= req.year_from]
    if req.require_fulltext:
        hits = [h for h in hits if h.fulltext]
    hits = hits[: req.limit]

    # Induced subgraph over the results only. Edges to papers outside the result set are
    # deliberately dropped: a topic's structure is who-cites-whom *among the relevant
    # work*, and including every outbound citation would bury that in hundreds of edges
    # to unrelated background references.
    t0 = clock()
    ids = [h.arxiv_id for h in hits]
    edges: list[dict] = []
    indeg: dict[str, int] = {i: 0 for i in ids}
    outdeg: dict[str, int] = {i: 0 for i in ids}
    if len(ids) > 1:
        ph = ",".join("?" * len(ids))
        for r in s.conn().execute(
            f"SELECT src, dst FROM citations WHERE src IN ({ph}) AND dst IN ({ph})",
            ids + ids,
        ):
            if r["src"] == r["dst"]:
                continue
            edges.append({"src": r["src"], "dst": r["dst"]})
            outdeg[r["src"]] = outdeg.get(r["src"], 0) + 1
            indeg[r["dst"]] = indeg.get(r["dst"], 0) + 1
    t["graph"] = (clock() - t0) * 1000
    t["total"] = sum(t.values())

    return JSONResponse({
        "query": req.query,
        "timings_ms": {k: round(v, 1) for k, v in t.items()},
        "edges": edges,
        "results": [
            {
                "arxiv_id": h.arxiv_id, "title": h.title, "abstract": h.abstract,
                "authors": h.authors[:160], "categories": h.categories,
                "submitted": h.submitted, "score": round(h.score, 4),
                "n_chunks": h.n_chunks, "fulltext": h.fulltext,
                "cited_by": h.cited_by, "version": h.version,
                "evidence": {k: round(v, 4) for k, v in h.evidence.items()},
                "rank": i + 1,
                "in_degree": indeg.get(h.arxiv_id, 0),
                "out_degree": outdeg.get(h.arxiv_id, 0),
            }
            for i, h in enumerate(hits)
        ],
    })


@app.post("/api/reload")
def reload_index() -> JSONResponse:
    """Pick up vectors added since startup.

    The tier-2 mmap self-heals per request, but the tier-1 GPU tensor is a snapshot
    taken at load time, so chunks embedded since then are not *searchable* until it is
    rebuilt. That rebuild copies the whole matrix to VRAM, so it is explicit rather than
    automatic — call it after an embed run finishes.
    """
    s = _state()
    assert s.retriever is not None
    from lara.index import search as S

    before = s.retriever.dense.n
    papers_before = s.paper_index.n if s.paper_index is not None else 0
    s.retriever.dense = S.DenseIndex(s.store.load_int8(), device=s.device)
    s.retriever.fp16 = s.store.open_fp16()
    s.retriever.refresh_row_map()
    s._load_paper_index()
    return JSONResponse({
        "chunks_before": before, "chunks_now": s.retriever.dense.n,
        "papers_before": papers_before,
        "papers_now": s.paper_index.n if s.paper_index is not None else 0,
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


@app.get("/api/breadth")
def breadth_options() -> JSONResponse:
    """The speed<->accuracy spectrum, so the UI can render it without hardcoding."""
    from lara.serve import agent as AG

    return JSONResponse({
        "default": AG.DEFAULT,
        "options": [
            {
                "name": b.name, "label": b.label, "estimate": b.estimate,
                "max_rounds": b.max_rounds, "k": b.k,
                "expand_context": b.expand_context, "allow_clarify": b.allow_clarify,
            }
            for b in AG.SPECTRUM
        ],
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
    breadth: str | None = None           # instant | fast | balanced | thorough | exhaustive


@app.post("/api/ask")
async def ask(req: AskRequest) -> StreamingResponse:
    """Stream a grounded answer over SSE.

    Streaming is the difference between ~300 ms to first token and 5–10 s to a complete
    answer. If ``hits`` are supplied the client already retrieved them speculatively, so
    generation starts immediately.
    """
    s = _state()
    assert s.retriever is not None

    import time

    from lara.serve import agent as AG
    from lara.serve.generate import stream_answer

    breadth = AG.resolve(req.breadth)
    restrict = [req.paper] if (req.scope == "paper" and req.paper) else None

    def run_search(query: str, k: int) -> list[dict]:
        result = s.retriever.retrieve(
            query, papers=restrict, selection=req.selection, final_k=k
        )
        return [
            {"chunk_id": h.chunk_id, "arxiv_id": h.arxiv_id, "version": h.version,
             "url": h.fragment(), "anchor": h.anchor_start, "char_start": h.char_start,
             "char_end": h.char_end, "anchor_end": h.anchor_end,
             "section": h.section_title or h.section_anchor, "kind": h.kind,
             "score": round(h.score, 4), "paper_title": h.paper_title, "text": h.text}
            for h in result.hits
        ]

    async def events():
        started = time.time()
        cap = breadth.k * (breadth.max_rounds + 2)

        def step(kind: str, detail: str, **payload):
            return f"event: step\ndata: {json.dumps({'kind': kind, 'detail': detail, **payload})}\n\n"

        try:
            hits = req.hits
            if hits is None:
                yield step("search", f"Searching ({breadth.label})…")
                hits = await run_in_threadpool(run_search, req.query, breadth.k)
            yield f"event: hits\ndata: {json.dumps(hits)}\n\n"

            rounds = 0
            while rounds < breadth.max_rounds:
                if time.time() - started > breadth.budget_sec:
                    yield step("budget", "Time budget reached; answering with what I have.")
                    break
                # A single round needs no controller turn: there is nothing yet to judge
                # against, and the extra LLM call would double "instant" latency.
                if breadth.max_rounds == 1:
                    break

                yield step("decide", "Assessing whether the excerpts answer the question…")
                verdict = await AG.decide(
                    s.cfg, req.query, hits, breadth, rounds, req.model
                )
                action = verdict.get("action", "answer")

                if action == "clarify" and breadth.allow_clarify:
                    yield (
                        "event: clarify\ndata: "
                        + json.dumps({
                            "question": verdict.get("question")
                            or "Could you be more specific?",
                            "options": verdict.get("options", [])[:4],
                            "reason": verdict.get("reason", ""),
                        })
                        + "\n\n"
                    )
                    # Non-blocking by design: the reader still gets an answer from what we
                    # have, with the refinements offered alongside. A model that stops to
                    # ask a question and returns nothing is worse than one that guesses
                    # well and offers to narrow down.
                    break

                if action == "expand" and breadth.expand_context:
                    ids = [int(i) for i in verdict.get("chunk_ids", [])[:6] if str(i).isdigit()]
                    yield step("expand", f"Reading around {len(ids)} excerpt(s) for context…")
                    extra = await run_in_threadpool(AG.expand_chunks, s.conn(), ids)
                    hits, added = AG.merge_hits(hits, extra, cap)
                    yield f"event: hits\ndata: {json.dumps(hits)}\n\n"
                    if added == 0:
                        break
                    rounds += 1
                    continue

                if action == "search":
                    queries = [q for q in verdict.get("queries", []) if isinstance(q, str)][:3]
                    if not queries:
                        break
                    yield step("search", f"Searching again: {'; '.join(queries)[:110]}",
                               queries=queries)
                    total_added = 0
                    for q in queries:
                        more = await run_in_threadpool(run_search, q, breadth.k)
                        hits, added = AG.merge_hits(hits, more, cap)
                        total_added += added
                    yield f"event: hits\ndata: {json.dumps(hits)}\n\n"
                    # No new evidence means another round would ask the same question of
                    # the same excerpts. Stop rather than spend the budget.
                    if total_added == 0:
                        yield step("converged", "No new material found; answering.")
                        break
                    rounds += 1
                    continue

                break   # "answer"

            yield step("answer", f"Writing answer from {len(hits)} excerpts…", rounds=rounds)
            async for token in stream_answer(
                s.cfg, req.query, hits, selection=req.selection,
                model=req.model, temperature=req.temperature, max_tokens=req.max_tokens,
            ):
                yield f"event: token\ndata: {json.dumps(token)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps(str(exc)[:300])}\n\n"
        yield f"event: done\ndata: {json.dumps({'elapsed_ms': round((time.time()-started)*1000)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
