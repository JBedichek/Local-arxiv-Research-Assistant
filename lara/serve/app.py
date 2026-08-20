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
import time
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

# Installed at import time from the environment rather than from config, because uvicorn
# imports this module in its own process and the CLI's parsed config does not travel with
# it. `lara serve` exports LARA_TOKEN after deciding whether one is required; anything
# that imports the app directly — a test, an embedded runner — gets the same protection by
# setting the same variable, and no protection if it sets nothing and serves loopback.
_auth_token = os.environ.get("LARA_TOKEN", "").strip()
if _auth_token:
    from lara.serve.auth import TokenAuthMiddleware

    app.add_middleware(TokenAuthMiddleware, token=_auth_token)


@app.on_event("startup")
def _startup() -> None:
    global state
    state = AppState(os.environ.get("LARA_CONFIG"), load_models=os.environ.get("LARA_NO_MODELS") != "1")


def _state() -> AppState:
    if state is None or not state.ready:
        raise HTTPException(503, "still warming up")
    return state


# ── UI ────────────────────────────────────────────────────────────────────────────

# no-cache, not no-store: the browser still revalidates cheaply with ETag, but never
# serves a stale bundle after an edit. Debugging a fix that "did not work" because the
# browser kept yesterday's JavaScript is a bad afternoon.
_NOCACHE = {"Cache-Control": "no-cache, must-revalidate"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html", headers=_NOCACHE)


@app.get("/app.js")
def appjs() -> FileResponse:
    return FileResponse(WEB_ROOT / "app.js", media_type="application/javascript",
                        headers=_NOCACHE)


@app.get("/style.css")
def appcss() -> FileResponse:
    return FileResponse(WEB_ROOT / "style.css", media_type="text/css", headers=_NOCACHE)


@app.get("/p/{arxiv_id:path}")
def reader(arxiv_id: str) -> FileResponse:
    """Deep links land here; the client reads the id and fragment from the URL."""
    return FileResponse(WEB_ROOT / "index.html", headers=_NOCACHE)


# ── data ──────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    s = state
    sc = getattr(s, "scope", None) if s else None
    return {
        "ready": bool(s and s.ready),
        "warmup_ms": (s.warmup_ms if s else {}),
        "vectors": (s.store.rows() if s else 0),
        # D22: null when the whole corpus is resident. Surfaced so the UI can say what the
        # dense index actually covers — a scoped index is not a broken one, but a user
        # deserves to know that semantic recall is narrowed to their topics.
        "scope": None if sc is None else {
            "topics": sc.topics,
            "fraction": sc.fraction,
            "papers": sc.n_papers,
            "resident_chunks": sc.n_rows,
            "corpus_chunks": sc.corpus_chunks,
            "resident_gb": round(sc.resident_bytes() / 1e9, 2),
        },
    }


@app.get("/healthz")
def healthz() -> JSONResponse:
    """Liveness, reachable without a token so a tunnel can probe it without a credential."""
    return JSONResponse({"ok": True})


@app.get("/api/paper/{arxiv_id:path}")
def get_paper(arxiv_id: str) -> JSONResponse:
    s = _state()
    row = s.paper(arxiv_id)
    if row is None:
        raise HTTPException(404, f"{arxiv_id} not in corpus")

    version = row["fulltext_version"] or row["latest_version"] or 1
    path = s.raw_html_path(arxiv_id)
    body = papers_mod.render(str(path), arxiv_id) if path else ""

    try:
        from lara.serve import memory as MEM
        MEM.record_visit(_memory_root(), arxiv_id=arxiv_id,
                         version=version, title=row["title"] or "")
    except Exception:
        pass          # the library is a convenience; never fail a paper open over it

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


def _capture_judgements(s, query: str, hits, source: str) -> None:
    """Store the reranker's verdict on every retrieved passage.

    Free by construction — the scores were computed to rank these hits and are otherwise
    discarded. Failures are swallowed: harvesting training data must never be able to fail
    a user's search.
    """
    try:
        from lara.finetune import judgements as J
        from lara.store import db

        items = J.from_hits(query, [
            {"chunk_id": h.chunk_id, "score": h.score, "provenance": h.provenance}
            for h in hits
        ], source=source)
        if not items:
            return
        conn = db.connect(s.db_path)
        try:
            J.record(conn, items)
        finally:
            conn.close()
    except Exception:
        pass


def _capture_hit_dicts(s, query: str, hits: list[dict], source: str) -> None:
    try:
        from lara.finetune import judgements as J
        from lara.store import db

        items = J.from_hits(query, hits, source=source)
        if not items:
            return
        conn = db.connect(s.db_path)
        try:
            J.record(conn, items)
        finally:
            conn.close()
    except Exception:
        pass


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
    _capture_judgements(s, req.query, result.hits, source="search")
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


class ClickRequest(BaseModel):
    query: str
    chunk_id: int
    rank: int | None = None


@app.post("/api/click")
def record_click(req: ClickRequest) -> JSONResponse:
    """A citation the reader actually followed.

    The only relevance signal here that no model produced. It is rare — one click per
    answer at best — but it is the only label that can contradict the teacher rather than
    echo it, which makes it the natural held-out set for checking whether distillation is
    learning relevance or learning the reranker.
    """
    from lara.finetune import judgements as J
    from lara.store import db

    s = _state()
    conn = db.connect(s.db_path)
    try:
        n = J.record(conn, [J.Judgement(
            query=req.query, chunk_id=req.chunk_id, score=1.0, label=1,
            teacher="user_click", rank=req.rank, source="click",
        )])
    finally:
        conn.close()
    return JSONResponse({"stored": n})


# ── library: reading history, folders, and the system-prompt override ────────────────

def _memory_root() -> Path:
    """Where the library lives. Created on demand so a fresh install needs no setup step."""
    root = _state().cfg.get_path("paths.memory")
    root.mkdir(parents=True, exist_ok=True)
    return root


class VisitRequest(BaseModel):
    arxiv_id: str
    version: int = 1
    title: str = ""


class RememberRequest(BaseModel):
    question: str
    answer: str = ""
    arxiv_id: str | None = None
    version: int = 1
    title: str = ""
    scope: str = "corpus"
    selection: str | None = None


class FolderRequest(BaseModel):
    name: str = "New folder"
    parent: str | None = None


class FolderPatch(BaseModel):
    name: str | None = None
    parent: str | None = None
    reparent: bool = False       # distinguishes "move to root" from "leave where it is"


class EntryPatch(BaseModel):
    folder: str | None = None
    title: str | None = None
    note: str | None = None
    refile: bool = False         # same distinction: null folder means root, not no-op


class PromptRequest(BaseModel):
    text: str | None = None


@app.get("/api/memory")
def memory_list() -> JSONResponse:
    """The whole library. Small enough to send at once; the client builds the tree."""
    from lara.serve import memory as MEM

    data = MEM.load(_memory_root())
    # `_ts` is an internal coalescing clock, not something the UI should depend on.
    entries = [{k: v for k, v in e.items() if k != "_ts"} for e in data["entries"]]
    entries.sort(key=lambda e: e.get("updated_utc", ""), reverse=True)
    return JSONResponse({"folders": data["folders"], "entries": entries})


@app.post("/api/memory/visit")
def memory_visit(req: VisitRequest) -> JSONResponse:
    from lara.serve import memory as MEM

    e = MEM.record_visit(_memory_root(), arxiv_id=req.arxiv_id,
                         version=req.version, title=req.title)
    return JSONResponse({k: v for k, v in e.items() if k != "_ts"})


@app.post("/api/memory/question")
def memory_question(req: RememberRequest) -> JSONResponse:
    from lara.serve import memory as MEM

    e = MEM.record_question(
        _memory_root(), question=req.question, answer=req.answer,
        arxiv_id=req.arxiv_id, version=req.version, title=req.title,
        scope=req.scope, selection=req.selection,
    )
    return JSONResponse({k: v for k, v in e.items() if k != "_ts"})


@app.patch("/api/memory/entry/{entry_id}")
def memory_entry_patch(entry_id: str, req: EntryPatch) -> JSONResponse:
    from lara.serve import memory as MEM

    fields = {}
    if req.refile:
        fields["folder"] = req.folder
    if req.title is not None:
        fields["title"] = req.title
    if req.note is not None:
        fields["note"] = req.note
    e = MEM.update_entry(_memory_root(), entry_id, **fields)
    if e is None:
        raise HTTPException(404, f"no entry {entry_id}")
    return JSONResponse({k: v for k, v in e.items() if k != "_ts"})


@app.delete("/api/memory/entry/{entry_id}")
def memory_entry_delete(entry_id: str) -> JSONResponse:
    from lara.serve import memory as MEM

    if not MEM.delete_entry(_memory_root(), entry_id):
        raise HTTPException(404, f"no entry {entry_id}")
    return JSONResponse({"deleted": entry_id})


@app.post("/api/memory/folder")
def memory_folder_create(req: FolderRequest) -> JSONResponse:
    from lara.serve import memory as MEM

    return JSONResponse(MEM.create_folder(_memory_root(), name=req.name, parent=req.parent))


@app.patch("/api/memory/folder/{folder_id}")
def memory_folder_patch(folder_id: str, req: FolderPatch) -> JSONResponse:
    from lara.serve import memory as MEM

    fields = {}
    if req.name is not None:
        fields["name"] = req.name
    if req.reparent:
        fields["parent"] = req.parent
    f = MEM.update_folder(_memory_root(), folder_id, **fields)
    if f is None:
        raise HTTPException(404, f"no folder {folder_id}")
    return JSONResponse(f)


@app.delete("/api/memory/folder/{folder_id}")
def memory_folder_delete(folder_id: str) -> JSONResponse:
    from lara.serve import memory as MEM

    if not MEM.delete_folder(_memory_root(), folder_id):
        raise HTTPException(404, f"no folder {folder_id}")
    return JSONResponse({"deleted": folder_id})


@app.get("/api/settings/prompt")
def prompt_get() -> JSONResponse:
    """The active system prompt, the built-in default, and which one is in force."""
    from lara.serve import memory as MEM
    from lara.serve.generate import SYSTEM

    custom = MEM.get_prompt(_memory_root())
    return JSONResponse({
        "default": SYSTEM,
        "custom": custom,
        "active": custom or SYSTEM,
        "is_custom": custom is not None,
    })


@app.put("/api/settings/prompt")
def prompt_put(req: PromptRequest) -> JSONResponse:
    """Save an override, or clear it by sending empty text."""
    from lara.serve import memory as MEM
    from lara.serve.generate import SYSTEM

    MEM.set_prompt(_memory_root(), req.text)
    custom = MEM.get_prompt(_memory_root())
    return JSONResponse({
        "default": SYSTEM,
        "custom": custom,
        "active": custom or SYSTEM,
        "is_custom": custom is not None,
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


_fetching: set[str] = set()


@app.post("/api/fetch/{arxiv_id:path}")
async def fetch_fulltext(arxiv_id: str) -> JSONResponse:
    """Crawl, parse and index one paper on demand (D2's lazy fetch).

    The plan always said full text would be fetched when a paper is opened; until now it
    only ever arrived via the background crawler, so opening any of the ~84% not yet
    crawled showed an all-but-empty pane. Roughly a third of search results are in that
    state, including top hits.

    New chunks are embedded immediately, so the paper becomes searchable in the same
    breath — otherwise it would sit unretrievable until the next bulk embed run.
    """
    import asyncio

    from lara.index import embed as emb
    from lara.ingest import fulltext as ft
    from lara.store import db

    s = _state()
    row = s.paper(arxiv_id)
    if row is None:
        raise HTTPException(404, f"{arxiv_id} not in corpus")
    if row["fulltext_status"] == "ok":
        return JSONResponse({"status": "ok", "cached": True})
    if arxiv_id in _fetching:
        # Opening the same paper twice must not start two crawls of it.
        return JSONResponse({"status": "in_progress"})

    _fetching.add(arxiv_id)
    try:
        fcfg = s.cfg.get_in("ingest.fulltext")
        fetcher = ft.FullTextFetcher(
            user_agent=fcfg["user_agent"], rate_per_sec=float(fcfg.get("rate_per_sec", 3)),
            max_concurrency=3, raw_root=s.cfg.get_path("paths.raw_cache"),
        )
        try:
            version = row["latest_version"] or 1
            result = await fetcher.fetch(arxiv_id, version, list(fcfg.get("sources", [])))
        finally:
            await fetcher.close()

        if result.status != "ok" or result.parsed is None:
            def mark():
                conn = db.connect(s.db_path)
                try:
                    ft.persist(conn, result)
                finally:
                    conn.close()
            await run_in_threadpool(mark)
            return JSONResponse({"status": result.status, "error": result.error}, status_code=502)

        def store_and_embed() -> int:
            conn = db.connect(s.db_path)
            try:
                if result.raw and result.source:
                    fetcher.write_raw(arxiv_id, result.source, result.raw)
                n = ft.persist(conn, result)
                # Just this paper's chunks — an unfiltered run would drain the whole
                # crawl backlog and turn one click into a multi-minute request.
                emb.run(conn, s.store, s.retriever.embedder, batch_size=128,
                        slice_size=4096, only_paper=arxiv_id)
                return n
            finally:
                conn.close()

        n_chunks = await run_in_threadpool(store_and_embed)
        await run_in_threadpool(s.retriever.refresh_row_map)
        papers_mod.render.cache_clear()
        return JSONResponse({
            "status": "ok", "cached": False, "chunks": n_chunks, "source": result.source,
        })
    finally:
        _fetching.discard(arxiv_id)


@app.get("/api/dataset/manifest")
def dataset_manifest() -> JSONResponse:
    """What this node is publishing, if anything."""
    from lara.serve import dataset as DS

    s = _state()
    m = DS.load_manifest(s.cfg.get_path("disk.root"))
    if m is None:
        raise HTTPException(404, "nothing published — run `lara dataset publish` first")
    return JSONResponse(m)


@app.get("/api/dataset/file/{name:path}")
def dataset_file(name: str) -> FileResponse:
    """Serve one published artefact. Range requests are honoured, so fetches resume."""
    from lara.serve import dataset as DS

    s = _state()
    root = s.cfg.get_path("disk.root")
    m = DS.load_manifest(root)
    if m is None:
        raise HTTPException(404, "nothing published")
    # Serve only what the manifest names. Without this the endpoint is an arbitrary file
    # read on a machine that has no authentication.
    if name not in {f["name"] for f in m["files"]}:
        raise HTTPException(404, f"{name} is not published")
    path = (root / name).resolve()
    if not str(path).startswith(str(root.resolve())) or not path.is_file():
        raise HTTPException(404, "not a published file")
    return FileResponse(path, filename=Path(name).name)


@app.get("/api/device")
def device_info() -> JSONResponse:
    """What this machine is, and which generation backend suits it."""
    from lara.models import wants_format
    from lara.serve import devices as DV
    from lara.serve import generator as GEN

    d = DV.detect()
    # Written by `lara setup`, which is the only thing that knows what the index costs.
    # Absent until the wizard has run, so both keys can be null and the UI must cope.
    cfg = _state().cfg
    headroom = cfg.get_in("hardware.generator_headroom_gb")
    max_params = cfg.get_in("hardware.generator_max_params_4bit")
    return JSONResponse({
        "system": d.system, "machine": d.machine, "accelerator": d.accelerator,
        "unified_memory": d.unified_memory, "total_ram_gb": d.total_ram_gb,
        "usable_ram_gb": d.usable_ram_gb, "gpus": d.gpus,
        "total_vram_gb": d.total_vram_gb, "budget_gb": d.budget_gb,
        # d.backend is the advisory description; effective_backend is what will really
        # run, which differs on a Mac with mlx-lm installed and wants a different format.
        "backend": GEN.effective_backend(cfg, d.accelerator),
        "backend_advisory": d.backend,
        "backend_reason": d.backend_reason, "notes": d.notes,
        "generator_headroom_gb": headroom,
        "generator_max_params_4bit": max_params,
        # Which weight format this machine's runtime can actually load, so the download
        # dialog can say so before someone fetches 8 GB of the wrong one.
        "wants_format": wants_format(GEN.effective_backend(cfg, d.accelerator)),
    })


class ResolveRequest(BaseModel):
    query: str


@app.post("/api/model/resolve")
def model_resolve(req: ResolveRequest) -> JSONResponse:
    """Look a repo up without downloading it, and say whether it fits."""
    import os

    from lara.serve import devices as DV
    from lara.serve import downloads as DL

    s = _state()
    r = DL.resolve(req.query, token=os.environ.get("HF_TOKEN"))
    dev = DV.detect()
    body = {
        "repo": r.repo, "exists": r.exists, "gated": r.gated, "error": r.error,
        "params": r.params, "arch": r.arch, "quantization": r.quantization,
        "size_gb": r.size_gb, "n_safetensors": r.n_safetensors, "n_gguf": r.n_gguf,
        "pipeline": r.pipeline, "variants": r.variants, "pick": r.pick,
        "already_cached": _dl(s).cache_dir(r.repo).exists() if r.repo else False,
    }
    if r.exists and r.size_gb:
        body["fit"] = DV.fits(r.size_gb, dev)
    # Warn rather than block: an architecture an allowlist does not know may still be
    # servable, and refusing the download would be a worse failure than letting someone
    # try. Which warning applies depends entirely on the runtime that will load it.
    from lara.models import FORMAT_HELP, VLLM_ARCHS, wants_format
    fmt = wants_format(dev.backend)
    body["backend"] = dev.backend
    body["wants_format"] = fmt
    if fmt == "gguf":
        if not r.n_gguf:
            body["warning"] = (
                f"This repo has no GGUF weights, and {dev.backend} cannot load "
                f"safetensors. {FORMAT_HELP['gguf']}")
    elif fmt == "mlx":
        if not r.repo.startswith("mlx-community/"):
            body["warning"] = (f"This does not look like an MLX conversion. "
                               f"{FORMAT_HELP['mlx']}")
    else:
        if r.arch and r.arch not in VLLM_ARCHS:
            body["warning"] = (
                f"{r.arch} is not in the vLLM allowlist this build knows about. It may "
                "still work; the model picker will list it once downloaded.")
        if r.n_gguf and not r.n_safetensors:
            body["warning"] = ("GGUF-only repo, and this machine serves with vLLM, which "
                               "cannot read GGUF (D4). Use a safetensors build.")
    return JSONResponse(body)


def _dl(s):
    from lara.serve import downloads as DL
    if not hasattr(s, "_downloads"):
        s._downloads = DL.DownloadManager(s.cfg.get_path("huggingface.home"))
    return s._downloads


class DownloadRequest(BaseModel):
    repo: str
    size_gb: float = 0.0
    #: Exact file paths of one GGUF quantisation. Without it a GGUF repo pulls every
    #: quantisation it ships, which is 133 GB to obtain a 5 GB model.
    files: list[str] | None = None


@app.post("/api/model/download")
def model_download(req: DownloadRequest) -> JSONResponse:
    import os

    from lara.serve import downloads as DL

    s = _state()
    repo = DL.normalise(req.repo)
    if not repo:
        raise HTTPException(400, "not a Hugging Face id")
    free_gb = __import__("shutil").disk_usage(s.cfg.get_path("huggingface.home")).free / 1e9
    if req.size_gb and req.size_gb * 1.1 > free_gb:
        raise HTTPException(
            507, f"needs ~{req.size_gb:.0f} GB, only {free_gb:.0f} GB free on the model disk"
        )
    job = _dl(s).start(repo, req.size_gb, token=os.environ.get("HF_TOKEN"),
                       files=req.files)
    return JSONResponse({"repo": job.repo, "status": job.status})


@app.get("/api/model/download/{repo:path}")
def model_download_status(repo: str) -> JSONResponse:
    from lara.serve import downloads as DL

    s = _state()
    job = _dl(s).get(DL.normalise(repo) or repo)
    if job is None:
        raise HTTPException(404, "no such download")
    return JSONResponse({
        "repo": job.repo, "status": job.status, "pct": job.pct,
        "downloaded_gb": job.downloaded_gb, "total_gb": job.total_gb,
        "elapsed_s": round(time.time() - job.started),
        "error": job.error, "path": job.path,
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
    s.retriever.remap_vectors()
    s.retriever.refresh_row_map()
    s._load_paper_index()
    return JSONResponse({
        "chunks_before": before, "chunks_now": s.retriever.dense.n,
        "papers_before": papers_before,
        "papers_now": s.paper_index.n if s.paper_index is not None else 0,
    })


class HeatmapRequest(BaseModel):
    arxiv_id: str
    query: str | None = None
    anchor_chunk_id: int | None = None
    mode: str = "query"          # query | answer
    k: int = 5


@app.post("/api/heatmap")
def heatmap(req: HeatmapRequest) -> JSONResponse:
    """Score every chunk of one paper, for shading the most relevant passages.

    Two reference vectors, because they answer different questions:

    ``query``   similarity to what the reader asked. Surfaces passages that restate the
                question — good for "where else is this discussed".
    ``answer``  similarity to the top retrieved chunk. Surfaces the argument *around* the
                answer — the setup, the caveat, the ablation that qualifies it — which is
                usually what someone following a citation actually wants to read next.

    Scored at full 768-d precision from the mmap rather than the 256-d tier-1 vectors: a
    single paper is a few hundred rows, so exactness is free here.
    """
    import numpy as np

    from lara.index.embed import embed_queries

    s = _state()
    assert s.retriever is not None
    r = s.retriever
    r._ensure_fp16_current()

    rows = s.conn().execute(
        "SELECT chunk_id, vector_row, anchor_start, char_start, anchor_end, char_end, "
        "       section_anchor, kind, substr(text,1,160) AS preview "
        "FROM chunks WHERE arxiv_id=? AND vector_row IS NOT NULL ORDER BY ordinal",
        (req.arxiv_id,),
    ).fetchall()
    rows = [x for x in rows if 0 <= x["vector_row"] < r.n_vector_rows]
    if not rows:
        return JSONResponse({"mode": req.mode, "chunks": []})

    if req.mode == "answer" and req.anchor_chunk_id is not None:
        ref_row = s.conn().execute(
            "SELECT vector_row FROM chunks WHERE chunk_id=?", (req.anchor_chunk_id,)
        ).fetchone()
        if ref_row is None or ref_row["vector_row"] is None:
            return JSONResponse({"mode": req.mode, "chunks": [], "error": "anchor has no vector"})
        ref = r.vectors_for(np.asarray([ref_row["vector_row"]], dtype=np.int64))[0]
    else:
        if not (req.query or "").strip():
            return JSONResponse({"mode": req.mode, "chunks": []})
        # query_for keeps the query in the same space as vectors_for, which is 256-d
        # rather than 768-d on a core-tier install.
        ref = r.query_for(embed_queries(r.embedder, [req.query])[0])
    ref = ref / max(float(np.linalg.norm(ref)), 1e-12)

    idx = np.fromiter((x["vector_row"] for x in rows), dtype=np.int64, count=len(rows))
    scores = r.vectors_for(idx) @ ref
    k = max(1, min(req.k, 25))
    top = np.argsort(-scores)[:k]

    return JSONResponse({
        "mode": req.mode,
        "chunks": [
            {
                "chunk_id": rows[i]["chunk_id"], "anchor": rows[i]["anchor_start"],
                "char_start": rows[i]["char_start"], "char_end": rows[i]["char_end"],
                "anchor_end": rows[i]["anchor_end"], "section": rows[i]["section_anchor"],
                "kind": rows[i]["kind"], "preview": rows[i]["preview"],
                "score": round(float(scores[i]), 4), "rank": int(rank) + 1,
            }
            for rank, i in enumerate(top)
        ],
    })


# ── taste profile ────────────────────────────────────────────────────────────────────

#: How a set of taste vectors becomes one score per passage.
#:
#: ``sum``  is the centroid in disguise: sum_k (v_k . c) == (sum_k v_k) . c exactly, so
#:          adding similarities is arithmetically identical to searching once with the
#:          summed vector. Cheapest, and it blurs — a reader interested in optimisers *and*
#:          retrieval gets a profile matching the midpoint, which is neither.
#: ``max``  nearest single interest. Keeps distinct interests distinct, but one eccentric
#:          mark dominates every paper it is slightly related to.
#: ``lse``  LogSumExp: a soft maximum. Concentrates on the best-matching interests without
#:          letting a single one own the score. Same reduction, and for the same reason, as
#:          the bag pooling in lara/finetune/train.py.
TASTE_REDUCTIONS = ("lse", "max", "sum", "mean")


def _reduce_taste(sims, how: str, temp: float = 0.05):
    """(M, K) similarities -> (M,) scores."""
    import numpy as np

    if sims.size == 0:
        return np.zeros(sims.shape[0], dtype=np.float32)
    if how == "sum":
        return sims.sum(axis=1)
    if how == "mean":
        return sims.mean(axis=1)
    if how == "max":
        return sims.max(axis=1)
    # LogSumExp, shifted by the row max so exp() cannot overflow.
    m = sims.max(axis=1, keepdims=True)
    return (m[:, 0] + temp * np.log(np.exp((sims - m) / temp).sum(axis=1)))


def _taste_vectors(r, marks):
    """Unit-normalised reference vectors for the marks that still resolve to a row."""
    import numpy as np

    rows = np.asarray([m["vector_row"] for m in marks
                       if 0 <= m.get("vector_row", -1) < r.n_vector_rows], dtype=np.int64)
    if rows.size == 0:
        return np.zeros((0, 1), dtype=np.float32), rows
    V = r.vectors_for(rows)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return V, rows


class TasteRequest(BaseModel):
    chunk_id: int
    note: str = ""


@app.get("/api/taste")
def taste_list() -> JSONResponse:
    from lara.serve import memory as MEM

    data = MEM.load_taste(_memory_root())
    return JSONResponse({"marks": data["marks"], "n": len(data["marks"])})


@app.post("/api/taste")
def taste_add(req: TasteRequest) -> JSONResponse:
    """Mark a passage as interesting. The chunk carries its own vector row already."""
    from lara.serve import memory as MEM

    s = _state()
    row = s.conn().execute(
        """SELECT c.chunk_id, c.vector_row, c.arxiv_id, substr(c.text,1,400) AS text,
                  p.title
           FROM chunks c LEFT JOIN papers p USING(arxiv_id)
           WHERE c.chunk_id=?""", (req.chunk_id,)).fetchone()
    if row is None:
        raise HTTPException(404, f"no chunk {req.chunk_id}")
    if row["vector_row"] is None:
        raise HTTPException(409, "that passage has no embedding yet")
    mark = MEM.record_taste(
        _memory_root(), chunk_id=row["chunk_id"], vector_row=row["vector_row"],
        arxiv_id=row["arxiv_id"], title=row["title"] or "", text=row["text"] or "",
        note=req.note,
    )
    return JSONResponse(mark)


class TasteMarkRequest(BaseModel):
    arxiv_id: str
    text: str
    note: str = ""


@app.post("/api/taste/mark")
def taste_mark_selection(req: TasteMarkRequest) -> JSONResponse:
    """Mark the passage a reader highlighted, resolving the selection to a chunk.

    The browser hands back text, not a chunk id, and a highlight rarely lines up with a
    chunk boundary — it is usually a sentence inside one. Substring matching handles the
    common case and fails on the rest, because the rendered HTML normalises whitespace and
    drops markup that the stored text still carries. So the fallback embeds the selection
    and takes the nearest chunk *of that paper*, which is a few hundred rows and therefore
    exact and instant.
    """
    import numpy as np

    from lara.index.embed import embed_queries
    from lara.serve import memory as MEM

    s = _state()
    assert s.retriever is not None
    r = s.retriever
    text = " ".join((req.text or "").split())
    if len(text) < 12:
        raise HTTPException(400, "selection too short to locate")

    rows = s.conn().execute(
        """SELECT chunk_id, vector_row, text FROM chunks
           WHERE arxiv_id=? AND vector_row IS NOT NULL ORDER BY ordinal""",
        (req.arxiv_id,)).fetchall()
    rows = [x for x in rows if 0 <= x["vector_row"] < r.n_vector_rows]
    if not rows:
        raise HTTPException(404, f"no embedded passages for {req.arxiv_id}")

    probe = text[:80]
    hit = next((x for x in rows if probe and probe in " ".join(x["text"].split())), None)
    how = "substring"
    if hit is None:
        how = "nearest"
        q = r.query_for(embed_queries(r.embedder, [text[:800]])[0])
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        idx = np.fromiter((x["vector_row"] for x in rows), dtype=np.int64, count=len(rows))
        C = r.vectors_for(idx)
        C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        hit = rows[int(np.argmax(C @ q))]

    title_row = s.conn().execute(
        "SELECT title FROM papers WHERE arxiv_id=?", (req.arxiv_id,)).fetchone()
    mark = MEM.record_taste(
        _memory_root(), chunk_id=hit["chunk_id"], vector_row=hit["vector_row"],
        arxiv_id=req.arxiv_id, title=(title_row["title"] if title_row else "") or "",
        text=hit["text"][:400], note=req.note,
    )
    return JSONResponse({**mark, "resolved_by": how})


@app.delete("/api/taste/{mark_id}")
def taste_delete(mark_id: str) -> JSONResponse:
    from lara.serve import memory as MEM

    if not MEM.delete_taste(_memory_root(), mark_id):
        raise HTTPException(404, f"no mark {mark_id}")
    return JSONResponse({"deleted": mark_id})


@app.get("/api/taste/paper/{arxiv_id:path}")
def taste_paper(arxiv_id: str, reduce: str = Query(default="lse"),
                temp: float = Query(default=0.05), k: int = Query(default=8)) -> JSONResponse:
    """Score every passage of one paper against the taste profile.

    Exact 768-d, because one paper is a few hundred rows and the whole matrix is
    (rows x marks) — microseconds either way, so there is no reason to approximate.
    """
    import numpy as np

    from lara.serve import memory as MEM

    s = _state()
    assert s.retriever is not None
    r = s.retriever
    r._ensure_fp16_current()

    marks = MEM.load_taste(_memory_root())["marks"]
    if not marks:
        return JSONResponse({"chunks": [], "n_marks": 0, "reduce": reduce})
    V, _ = _taste_vectors(r, marks)
    if V.shape[0] == 0:
        return JSONResponse({"chunks": [], "n_marks": 0, "reduce": reduce})

    rows = s.conn().execute(
        "SELECT chunk_id, vector_row, anchor_start, char_start, anchor_end, char_end, "
        "       section_anchor, kind, ordinal, substr(text,1,160) AS preview "
        "FROM chunks WHERE arxiv_id=? AND vector_row IS NOT NULL ORDER BY ordinal",
        (arxiv_id,),
    ).fetchall()
    # A passage the reader already marked would otherwise score 1.0 and top its own list.
    marked = {m["chunk_id"] for m in marks}
    rows = [x for x in rows if 0 <= x["vector_row"] < r.n_vector_rows]
    if not rows:
        return JSONResponse({"chunks": [], "n_marks": len(marks), "reduce": reduce})

    idx = np.fromiter((x["vector_row"] for x in rows), dtype=np.int64, count=len(rows))
    C = r.vectors_for(idx)
    C = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    scores = _reduce_taste(C @ V.T, reduce if reduce in TASTE_REDUCTIONS else "lse", temp)

    order = np.argsort(-scores)
    top = [i for i in order if rows[i]["chunk_id"] not in marked][: max(1, min(k, 25))]
    return JSONResponse({
        "n_marks": len(marks),
        "reduce": reduce,
        # Every passage in reading order, for the scroll-bar profile.
        "profile": [
            {"ordinal": rows[i]["ordinal"], "chunk_id": rows[i]["chunk_id"],
             "score": round(float(scores[i]), 4),
             "marked": rows[i]["chunk_id"] in marked}
            for i in range(len(rows))
        ],
        # The jump list.
        "chunks": [
            {"chunk_id": rows[i]["chunk_id"], "anchor": rows[i]["anchor_start"],
             "char_start": rows[i]["char_start"], "char_end": rows[i]["char_end"],
             "anchor_end": rows[i]["anchor_end"], "section": rows[i]["section_anchor"],
             "kind": rows[i]["kind"], "preview": rows[i]["preview"],
             "score": round(float(scores[i]), 4), "rank": rank + 1}
            for rank, i in enumerate(top)
        ],
    })


@app.get("/api/taste/recommend")
def taste_recommend(reduce: str = Query(default="lse"), temp: float = Query(default=0.05),
                    k: int = Query(default=20), exclude_read: bool = Query(default=True),
                    ) -> JSONResponse:
    """Passages from anywhere in the corpus that match the profile.

    Scored on the resident tier-1 index. Measured on this corpus: streaming the 15.1 GB
    matrix costs ~11 ms and dominates, so 50 taste vectors cost 16 ms against 11 ms for
    one — the profile size is nearly free and the reduction is what carries the cost
    (LogSumExp at 50 marks, ~52 ms).
    """
    import numpy as np

    from lara.serve import memory as MEM

    s = _state()
    assert s.retriever is not None
    r = s.retriever
    marks = MEM.load_taste(_memory_root())["marks"]
    if not marks:
        return JSONResponse({"chunks": [], "n_marks": 0})
    V, _ = _taste_vectors(r, marks)
    if V.shape[0] == 0:
        return JSONResponse({"chunks": [], "n_marks": 0})

    how = reduce if reduce in TASTE_REDUCTIONS else "lse"
    if how in ("sum", "mean"):
        # Additive similarity IS the centroid: sum_k (v_k . c) == (sum_k v_k) . c. So this
        # collapses to a single ordinary search rather than K of them.
        ref = V.sum(axis=0)
        if how == "mean":
            ref = ref / V.shape[0]
        ref = ref / max(float(np.linalg.norm(ref)), 1e-12)
        rows, scores = r.dense.search(ref[: r.dim_trunc], k=k * 4)
    else:
        rows, scores = r.dense.search_multi(V[:, : r.dim_trunc], k=k * 4,
                                            reduce=how, temp=temp)

    marked_rows = {m["vector_row"] for m in marks}
    pairs = [(int(a), float(b)) for a, b in zip(rows, scores) if int(a) not in marked_rows]
    if not pairs:
        return JSONResponse({"chunks": [], "n_marks": len(marks)})

    ph = ",".join("?" * len(pairs))
    found = {x["vector_row"]: x for x in s.conn().execute(
        f"""SELECT c.chunk_id, c.vector_row, c.arxiv_id, c.anchor_start, c.char_start,
                   c.anchor_end, c.char_end, c.section_anchor,
                   substr(c.text,1,200) AS preview, p.title
            FROM chunks c LEFT JOIN papers p USING(arxiv_id)
            WHERE c.vector_row IN ({ph})""", [p_[0] for p_ in pairs])}

    seen_papers, out = set(), []
    for row_id, score in pairs:
        x = found.get(row_id)
        if x is None:
            continue
        # One passage per paper: a profile match usually lights up several chunks of the
        # same paper, and a recommendation list of one paper eight times is not a list.
        if x["arxiv_id"] in seen_papers:
            continue
        seen_papers.add(x["arxiv_id"])
        out.append({
            "chunk_id": x["chunk_id"], "arxiv_id": x["arxiv_id"], "title": x["title"] or "",
            "anchor": x["anchor_start"], "char_start": x["char_start"],
            "anchor_end": x["anchor_end"], "char_end": x["char_end"],
            "section": x["section_anchor"], "preview": x["preview"],
            "score": round(score, 4),
        })
        if len(out) >= k:
            break
    return JSONResponse({"chunks": out, "n_marks": len(marks), "reduce": how})


@app.get("/api/models")
def models() -> JSONResponse:
    """Generators available from the HF cache (R6, R7)."""
    s = _state()
    from lara.models import scan
    from lara.serve import devices as DV
    from lara.serve import generator as GEN

    backend = GEN.effective_backend(s.cfg, DV.detect().accelerator)
    found = scan(s.cfg.get_path("huggingface.home"), backend=backend)

    # Which model vLLM is actually serving. The picker lists everything in the cache, but
    # only one is loaded (D5, single_resident), and sending any other name to vLLM is a
    # 404 at generation time. Reporting it lets the UI default to the live one instead of
    # whichever happened to sort first.
    import httpx

    loaded: list[str] = []
    try:
        base = s.cfg.get_in("serving.vllm.base_url").rstrip("/")
        r = httpx.get(f"{base}/models", timeout=2.0)
        if r.status_code == 200:
            loaded = [m["id"] for m in r.json().get("data", [])]
    except Exception:
        pass

    serving = s.cfg.get_in("serving") or {}
    return JSONResponse({
        "loaded": loaded,
        "backend": backend,
        # Per backend: the three formats are not interchangeable, so each keeps its own
        # model setting. Reading only the vLLM one reported "no default" on every Mac.
        "configured_default": GEN.model_for(
            backend, {**(serving.get("generator") or {}), "vllm": serving.get("vllm") or {}}),
        "models": [
            {
                "repo": m.repo, "arch": m.arch, "size_gb": round(m.size_gb, 1),
                "quantization": m.quantization,
                "quant_options": m.runtime_quant_options(),
                "loaded": m.repo in loaded,
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

    import asyncio
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
                # A compound question embeds to a single vector sitting BETWEEN its parts,
                # retrieving well for none of them. Splitting first and searching each
                # part is the fix; the parts run concurrently, so the extra cost is one
                # decomposition call rather than N retrievals.
                split = {"kind": "atomic", "parts": []}
                if breadth.decompose:
                    split = await AG.decompose(s.cfg, req.query, req.model)

                if split["parts"]:
                    yield step(
                        "decompose",
                        f"Split into {len(split['parts'])} parts: "
                        + " | ".join(p[:48] for p in split["parts"]),
                        # not `kind=`: that is step()'s own positional parameter
                        parts=split["parts"], split_kind=split["kind"],
                    )
                    per_part = max(3, breadth.k // len(split["parts"]) + 2)
                    results = await asyncio.gather(*[
                        run_in_threadpool(run_search, part, per_part)
                        for part in split["parts"]
                    ])
                    hits = AG.merge_parts(
                        list(zip(split["parts"], results)), breadth.k * 2
                    )
                else:
                    yield step("search", f"Searching ({breadth.label})…")
                    hits = await run_in_threadpool(run_search, req.query, breadth.k)
            yield f"event: hits\ndata: {json.dumps(hits)}\n\n"
            _capture_hit_dicts(s, req.query, hits, "ask")

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

            # Coverage gate. Naming an insufficient result set explicitly is what stops
            # the model quietly filling the gap from parametric memory, which is the most
            # common way a RAG answer goes wrong while looking right.
            yield step("assess", "Checking whether the excerpts answer the question…")
            verdict = await AG.assess(s.cfg, req.query, hits, req.model)
            coverage = verdict.get("coverage", "full")
            yield f"event: coverage\ndata: {json.dumps(verdict)}\n\n"
            label = {"full": "Answering", "partial": "Answering what is supported",
                     "none": "Summarising what was found"}[coverage]
            yield step("answer", f"{label} from {len(hits)} excerpts…",
                       rounds=rounds, coverage=coverage)

            try:
                from lara.serve import memory as MEM
                custom_prompt = MEM.get_prompt(_memory_root())
            except Exception:
                custom_prompt = None

            answer = ""
            async for token in stream_answer(
                s.cfg, req.query, hits, selection=req.selection,
                model=req.model, temperature=req.temperature, max_tokens=req.max_tokens,
                coverage=coverage, system=custom_prompt,
            ):
                answer += token
                yield f"event: token\ndata: {json.dumps(token)}\n\n"

            try:
                from lara.serve import memory as MEM
                paper_title = next(
                    (h.get("paper_title", "") for h in hits
                     if req.paper and h.get("arxiv_id") == req.paper), "")
                MEM.record_question(
                    _memory_root(), question=req.query, answer=answer,
                    arxiv_id=req.paper, title=paper_title,
                    scope=req.scope, selection=req.selection,
                )
            except Exception:
                pass      # never lose a finished answer because the library write failed

            # Post-hoc grounding check: score every cited sentence against the excerpt it
            # actually cites. A citation marker makes a claim look verified, so an
            # unsupported one is worse than none at all.
            if answer.strip() and s.retriever.cross_encoder is not None:
                checks = await run_in_threadpool(
                    AG.check_grounding, s.retriever.cross_encoder, answer, hits
                )
                if checks:
                    yield f"event: grounding\ndata: {json.dumps(checks)}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps(str(exc)[:300])}\n\n"
        yield f"event: done\ndata: {json.dumps({'elapsed_ms': round((time.time()-started)*1000)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
