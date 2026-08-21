"""Scoring the passages of one open paper.

Two endpoints do this — ``/api/heatmap`` shades against the question or the answer,
``/api/taste/paper`` shades against the reader's marked passages — and they were written
twice: the same query, the same residency filter, the same argsort, and a byte-identical
result dict. Only the reference vector differs, which is the part that should differ.

What is NOT shared is the scoring itself. The heatmap takes the inner product against
stored vectors as they are; taste re-normalises them first. That is a real difference in
what the numbers mean, so each endpoint keeps its own line of arithmetic rather than
hiding it behind a flag.

Everything here is exact at full precision. One paper is a few hundred rows, so there is
no reason to approximate.
"""

from __future__ import annotations

import numpy as np

#: The columns a scored passage is rendered from. `ordinal` is what puts the profile in
#: reading order; the rest are what makes a passage addressable in the rendered HTML.
_COLUMNS = (
    "chunk_id, vector_row, anchor_start, char_start, anchor_end, char_end, "
    "section_anchor, kind, ordinal, substr(text,1,160) AS preview"
)


def paper_chunks(conn, retriever, arxiv_id: str) -> list:
    """Every embedded passage of one paper, in reading order, that can actually be scored.

    Rows whose ``vector_row`` falls outside the index are dropped rather than indexed:
    the row map is rebuilt on reload and a chunk embedded since the last rebuild has a
    row number the vector file does not have yet. Reading it would score a passage
    against whatever vector happens to occupy that slot.
    """
    rows = conn.execute(
        f"SELECT {_COLUMNS} FROM chunks "
        "WHERE arxiv_id=? AND vector_row IS NOT NULL ORDER BY ordinal",
        (arxiv_id,),
    ).fetchall()
    return [x for x in rows if 0 <= x["vector_row"] < retriever.n_vector_rows]


def vector_rows(rows: list) -> np.ndarray:
    """The vector-row indices of those passages, as one array for a single gather."""
    return np.fromiter((x["vector_row"] for x in rows), dtype=np.int64, count=len(rows))


def passage(row, score: float, rank: int) -> dict:
    """One scored passage, in the shape the reader's jump lists expect."""
    return {
        "chunk_id": row["chunk_id"], "anchor": row["anchor_start"],
        "char_start": row["char_start"], "char_end": row["char_end"],
        "anchor_end": row["anchor_end"], "section": row["section_anchor"],
        "kind": row["kind"], "preview": row["preview"],
        "score": round(float(score), 4), "rank": rank + 1,
    }
