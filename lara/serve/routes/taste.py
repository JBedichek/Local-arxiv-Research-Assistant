"""The taste profile: marked passages, and what in the corpus resembles them."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from lara.serve import passages as PSG
from lara.serve.deps import memory_root, require_state

router = APIRouter()


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


@router.get("/api/taste")
def taste_list() -> JSONResponse:
    from lara.serve import memory as MEM

    data = MEM.load_taste(memory_root())
    return JSONResponse({"marks": data["marks"], "n": len(data["marks"])})


@router.post("/api/taste")
def taste_add(req: TasteRequest) -> JSONResponse:
    """Mark a passage as interesting. The chunk carries its own vector row already."""
    from lara.serve import memory as MEM

    s = require_state()
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
        memory_root(), chunk_id=row["chunk_id"], vector_row=row["vector_row"],
        arxiv_id=row["arxiv_id"], title=row["title"] or "", text=row["text"] or "",
        note=req.note,
    )
    return JSONResponse(mark)


class TasteMarkRequest(BaseModel):
    arxiv_id: str
    text: str
    note: str = ""


@router.post("/api/taste/mark")
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

    s = require_state()
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
        memory_root(), chunk_id=hit["chunk_id"], vector_row=hit["vector_row"],
        arxiv_id=req.arxiv_id, title=(title_row["title"] if title_row else "") or "",
        text=hit["text"][:400], note=req.note,
    )
    return JSONResponse({**mark, "resolved_by": how})


@router.delete("/api/taste/{mark_id}")
def taste_delete(mark_id: str) -> JSONResponse:
    from lara.serve import memory as MEM

    if not MEM.delete_taste(memory_root(), mark_id):
        raise HTTPException(404, f"no mark {mark_id}")
    return JSONResponse({"deleted": mark_id})


@router.get("/api/taste/paper/{arxiv_id:path}")
def taste_paper(arxiv_id: str, reduce: str = Query(default="lse"),
                temp: float = Query(default=0.05), k: int = Query(default=8)) -> JSONResponse:
    """Score every passage of one paper against the taste profile.

    Exact 768-d, because one paper is a few hundred rows and the whole matrix is
    (rows x marks) — microseconds either way, so there is no reason to approximate.
    """
    import numpy as np

    from lara.serve import memory as MEM

    s = require_state()
    assert s.retriever is not None
    r = s.retriever
    r._ensure_fp16_current()

    marks = MEM.load_taste(memory_root())["marks"]
    if not marks:
        return JSONResponse({"chunks": [], "n_marks": 0, "reduce": reduce})
    V, _ = _taste_vectors(r, marks)
    if V.shape[0] == 0:
        return JSONResponse({"chunks": [], "n_marks": 0, "reduce": reduce})

    rows = PSG.paper_chunks(s.conn(), r, arxiv_id)
    # A passage the reader already marked would otherwise score 1.0 and top its own list.
    marked = {m["chunk_id"] for m in marks}
    if not rows:
        return JSONResponse({"chunks": [], "n_marks": len(marks), "reduce": reduce})

    C = r.vectors_for(PSG.vector_rows(rows))
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
        "chunks": [PSG.passage(rows[i], scores[i], rank) for rank, i in enumerate(top)],
    })


@router.get("/api/taste/recommend")
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

    s = require_state()
    assert s.retriever is not None
    r = s.retriever
    marks = MEM.load_taste(memory_root())["marks"]
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
