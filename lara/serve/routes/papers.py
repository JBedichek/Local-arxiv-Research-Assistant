"""One paper: its rendered text, its citation neighbourhood, and fetching it on demand."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from lara.serve import papers as papers_mod
from lara.serve.deps import memory_root, require_state

router = APIRouter()


@router.get("/api/paper/{arxiv_id:path}")
def get_paper(arxiv_id: str) -> JSONResponse:
    s = require_state()
    row = s.paper(arxiv_id)
    if row is None:
        raise HTTPException(404, f"{arxiv_id} not in corpus")

    version = row["fulltext_version"] or row["latest_version"] or 1
    path = s.raw_html_path(arxiv_id)
    body = papers_mod.render(str(path), arxiv_id, version) if path else ""

    try:
        from lara.serve import memory as MEM
        MEM.record_visit(memory_root(), arxiv_id=arxiv_id,
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

@router.get("/api/graph/{arxiv_id:path}")
def graph(arxiv_id: str, query: str = Query(default=""), limit: int = 60) -> JSONResponse:
    """Ego network with similarity shading (R8, R9)."""
    s = require_state()
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


_fetching: set[str] = set()


@router.post("/api/fetch/{arxiv_id:path}")
async def fetch_fulltext(arxiv_id: str) -> JSONResponse:
    """Crawl, parse and index one paper on demand (D2's lazy fetch).

    The plan always said full text would be fetched when a paper is opened; until now it
    only ever arrived via the background crawler, so opening any of the ~84% not yet
    crawled showed an all-but-empty pane. Roughly a third of search results are in that
    state, including top hits.

    New chunks are embedded immediately, so the paper becomes searchable in the same
    breath — otherwise it would sit unretrievable until the next bulk embed run.
    """
    from lara.index import embed as emb
    from lara.ingest import fulltext as ft
    from lara.store import db

    s = require_state()
    row = s.paper(arxiv_id)
    if row is None:
        raise HTTPException(404, f"{arxiv_id} not in corpus")

    # `fulltext_status='ok'` means the chunks were indexed, NOT that the HTML needed to
    # render the paper is on disk. Those are the same fact only on a machine that did
    # the crawl: the published `core` tier ships chunks and vectors while the raw HTML
    # lives in `archive` (~40 GB), so a normal install has 368k papers marked ok and no
    # renderable bytes for any of them. Returning early on the status alone meant the
    # reader asked for full text, was told "already have it", re-requested the paper,
    # got the same empty html, and sat on a spinner forever showing just the abstract.
    already_indexed = row["fulltext_status"] == "ok" and (row["n_chunks"] or 0) > 0
    if already_indexed and s.raw_html_path(arxiv_id) is not None:
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
                # The chunks and their vectors are already there; this fetch existed only
                # to put renderable HTML on disk. Re-persisting would duplicate rows and
                # re-embedding would spend GPU time reproducing vectors we already have.
                if already_indexed:
                    return int(row["n_chunks"] or 0)
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
