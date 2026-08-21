"""Search: passage retrieval, paper search, click feedback, heatmaps, index reload."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from lara.serve import passages as PSG
from lara.serve.deps import require_state

router = APIRouter()


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

class RetrieveRequest(BaseModel):
    query: str
    selection: str | None = None
    paper: str | None = None
    scope: str = "corpus"          # corpus | paper | neighbourhood
    k: int = 8


@router.post("/api/retrieve")
def retrieve(req: RetrieveRequest) -> JSONResponse:
    """Retrieval only — no generation.

    The client fires this speculatively the moment a selection is made, before the user
    has finished typing. By submit time the chunks are already in hand, which removes the
    retrieval leg from perceived latency entirely.
    """
    s = require_state()
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
            h.to_dict()
            for h in result.hits
        ],
    })


class ClickRequest(BaseModel):
    query: str
    chunk_id: int
    rank: int | None = None


@router.post("/api/click")
def record_click(req: ClickRequest) -> JSONResponse:
    """A citation the reader actually followed.

    The only relevance signal here that no model produced. It is rare — one click per
    answer at best — but it is the only label that can contradict the teacher rather than
    echo it, which makes it the natural held-out set for checking whether distillation is
    learning relevance or learning the reranker.
    """
    from lara.finetune import judgements as J
    from lara.store import db

    s = require_state()
    conn = db.connect(s.db_path)
    try:
        n = J.record(conn, [J.Judgement(
            query=req.query, chunk_id=req.chunk_id, score=1.0, label=1,
            teacher="user_click", rank=req.rank, source="click",
        )])
    finally:
        conn.close()
    return JSONResponse({"stored": n})


class SearchRequest(BaseModel):
    query: str
    limit: int = 20
    category: str | None = None
    year_from: int | None = None
    require_fulltext: bool = False


@router.post("/api/search")
def search_papers(req: SearchRequest) -> JSONResponse:
    """Semantic paper search — "show me papers on Muon learning rate adaptation".

    Fuses breadth (title+abstract vectors, which exist for every in-scope paper) with
    depth (full-text chunks aggregated per paper). Neither alone is adequate: abstracts
    miss methods introduced in section 4, and full text covers only the crawled minority.
    """
    import time

    from lara.index import paper_search as PS
    from lara.index.embed import embed_queries

    s = require_state()
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
        hits = [h for h in hits if h.submitted[:4].isdigit()
                and int(h.submitted[:4]) >= req.year_from]
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


@router.post("/api/reload")
def reload_index() -> JSONResponse:
    """Pick up vectors added since startup.

    The tier-2 mmap self-heals per request, but the tier-1 GPU tensor is a snapshot
    taken at load time, so chunks embedded since then are not *searchable* until it is
    rebuilt. That rebuild copies the whole matrix to VRAM, so it is explicit rather than
    automatic — call it after an embed run finishes.
    """
    s = require_state()
    assert s.retriever is not None
    from lara.index import backends as BK
    from lara.index import scope as SC

    before = s.retriever.dense.n
    papers_before = s.paper_index.n if s.paper_index is not None else 0
    # Rebuild through the same factory the retriever was constructed with. Calling
    # the index class directly dropped the D22 keep-set, the configured precision and the
    # backend choice, so one reload silently made the WHOLE corpus resident — on exactly
    # the machines that were scoped because they cannot hold it.
    icfg = s.cfg.get_in("index") or {}
    s.scope = SC.Scope.load(s.cfg.get_path("disk.root"))
    resident = s.scope.rows if s.scope is not None else None
    chosen = BK.choose_backend(icfg.get("backend", "auto"))
    s.retriever.dense = BK.make_index(
        s.store.load_int8(mmap=(chosen == "faiss" or resident is not None)),
        backend=chosen, precision=icfg.get("precision", "fp16"), device=s.device,
        row_ids=resident, faiss_cfg=BK.FaissConfig(**(icfg.get("faiss") or {})),
    )
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


@router.post("/api/heatmap")
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
    from lara.index.embed import embed_queries

    s = require_state()
    assert s.retriever is not None
    r = s.retriever
    r._ensure_fp16_current()

    rows = PSG.paper_chunks(s.conn(), r, req.arxiv_id)
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

    # Deliberately not re-normalised: these are the stored vectors, and the heatmap is
    # shading relative magnitudes within one paper rather than reporting a cosine.
    scores = r.vectors_for(PSG.vector_rows(rows)) @ ref
    top = np.argsort(-scores)[: max(1, min(req.k, 25))]

    return JSONResponse({
        "mode": req.mode,
        "chunks": [PSG.passage(rows[i], scores[i], rank) for rank, i in enumerate(top)],
    })
