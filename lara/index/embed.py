"""Embedding pipeline — EmbeddingGemma over the chunk table, resumable (PLAN.md §5).

**Contextual prefix.** Chunks are embedded as
``title: {paper title} > {section title} | text: {chunk}`` rather than bare text. A chunk
reading "we ablate this and lose 3 BLEU" is close to meaningless in isolation; prefixed
with "Attention Is All You Need > Model Ablations" it becomes retrievable. This is a free
approximation of contextual retrieval, which normally costs one LLM call per chunk and is
hopeless at 16 M chunks. EmbeddingGemma is trained with a ``title: ... | text: ...``
document format, so this rides its native prompt rather than fighting it.

**Measured throughput** (RTX PRO 6000 Blackwell, chunks averaging 219 tokens).

Sequence length, at bf16 eager::

    msl 512     452 chunks/s    9.8 GPU-h    1.9% truncated   <- chosen
    msl 384     528 chunks/s    8.4 GPU-h    9.2% truncated
    msl 256     713 chunks/s    6.2 GPU-h   31.5% truncated

512 wins: dropping to 384 buys 15% throughput for five times the truncation, and a
truncated chunk loses its tail, which is where a conclusion tends to live. (fp32 at msl
512 manages only 251 chunks/s — bf16 is worth ~1.8x on its own.)

Compilation, at bf16 msl 512::

    eager, length-sorted              469 chunks/s    9.5 GPU-h
    compile 'default', dynamic       1027 chunks/s    4.3 GPU-h   <- chosen, 2.19x
    eager, fixed pad 512              355 chunks/s   12.5 GPU-h
    max-autotune + fixed pad 512      101 chunks/s   44.1 GPU-h

Device count, at bf16 msl 512 compiled::

    1 GPU  (cuda:0)          1030 chunks/s    4.3 GPU-h
    3 GPUs (cuda:0,1,2)      2600 chunks/s    1.7 wall-h    <- chosen for bulk indexing

5.5x over the eager single-GPU baseline in total. Scaling is ~2.5x rather than 3x because
the parent process still tokenizes and marshals results, and the slice boundary is a
synchronization point.

``max-autotune`` is a trap here in both shape regimes. With dynamic shapes it re-tunes on
essentially every batch, because length-sorted batching hands it a new sequence length
each time — 51 chunks/s, nine times slower than eager. Forcing one static shape to stop
the churn does not rescue it either, and static padding is itself a loss: padding every
219-token chunk out to 512 costs more (355 chunks/s) than length-sorted batching. Plain
``default`` compiles once in ~100 s, keeps dynamic shapes, and more than doubles
throughput.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator

import numpy as np
import torch

from lara import device as dev
from lara.index.vectors import VectorStore

QUERY_PROMPT = "task: search result | query: "


def document_text(title: str | None, section_title: str | None, text: str) -> str:
    """Build the contextual document string. Mirrors EmbeddingGemma's native format."""
    head = (title or "none").strip()
    section = (section_title or "").strip()
    if section and section.lower() not in head.lower():
        head = f"{head} > {section}"
    return f"title: {head} | text: {text}"


#: The two document prefixes every trainer and evaluator needs, derived from the one
#: function that defines the corpus format rather than retyped. Four hand-written copies
#: had already drifted — kfold used "title: none", train used "title: {title}" — and
#: judgements.py and kfold.py both record what a train/serve format mismatch cost when it
#: last happened: 0.909 -> 0.741 cosine. Deriving them makes that drift impossible.
DOC_PREFIX_UNTITLED = document_text(None, None, "")      # "title: none | text: "


def doc_prefix_for(title: str | None) -> str:
    """The document prefix for a paper-level document with this title."""
    return document_text(title, None, "")


def load_model(
    name: str,
    device: str | int | None = None,
    max_seq_length: int = 512,
    compile_mode: str | None = None,
    compile_dynamic: bool = True,
):
    """Load the encoder in bf16, optionally torch.compile'd.

    ``compile_dynamic=True`` matters more than it looks. sentence-transformers sorts by
    length and pads per batch, so the encoder sees many distinct input shapes. Under
    static compilation each new shape triggers a fresh autotuning pass, and with
    ``max-autotune`` those passes are expensive enough to swamp the speedup they buy.
    Dynamic shapes compile once and hold.

    ``no-cudagraphs`` is the right variant here for the same reason: CUDA graphs want
    stable shapes and stable memory addresses, neither of which a length-sorted batcher
    provides.
    """
    from sentence_transformers import SentenceTransformer

    device = dev.resolve(device)
    # bf16 on CUDA, fp16 on MPS, fp32 on CPU — half precision is emulated on CPU and comes
    # out slower than fp32, so "smaller is faster" does not hold there.
    model = SentenceTransformer(name, device=device,
                                model_kwargs={"dtype": dev.model_dtype(device)})
    model.max_seq_length = max_seq_length
    if compile_mode:
        model[0].auto_model = torch.compile(
            model[0].auto_model, mode=compile_mode, dynamic=compile_dynamic
        )
    return model


class MultiGPUEncoder:
    """Fan encoding across several cards while keeping a single writer.

    The obvious alternative — one ``lara embed`` process per GPU — races on the
    append-only vector files: two processes appending concurrently interleave rows and
    hand out row numbers that point at each other's data. sentence-transformers' worker
    pool keeps encoding parallel but returns results to one parent, so exactly one process
    ever appends.

    Exposes the same ``.encode(...)`` shape as a plain SentenceTransformer so
    :func:`run` does not care which it was handed.

    Only ever constructed for multiple CUDA cards — see :func:`lara.device.can_fan_out`.
    On unified memory there is one GPU behind one pool of RAM, so N workers would multiply
    the footprint while contending for the same silicon.
    """

    def __init__(self, model, devices: list[str | int], chunk_size: int | None = None) -> None:
        self.model = model
        self.devices = dev.resolve_all(devices)
        self.chunk_size = chunk_size
        self.pool = model.start_multi_process_pool(self.devices)

    def encode(self, texts: list[str], batch_size: int = 256, **_: object) -> np.ndarray:
        return self.model.encode_multi_process(
            texts, self.pool, batch_size=batch_size,
            chunk_size=self.chunk_size, normalize_embeddings=True,
        )

    def close(self) -> None:
        self.model.stop_multi_process_pool(self.pool)


def embed_queries(model, queries: list[str], dim_trunc: int | None = None) -> np.ndarray:
    """Embed search queries. Uses the query prompt, which is *not* the document prompt —
    EmbeddingGemma is asymmetric and mixing them measurably degrades retrieval."""
    vecs = model.encode(
        queries, prompt=QUERY_PROMPT, batch_size=len(queries),
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False,
    ).astype(np.float32)
    # A non-finite embedding is not a degraded result, it is a broken one: every score
    # computed from it is NaN, and NaN survives all the way to JSON serialisation, where
    # it surfaces as a starlette stack trace naming neither the model nor the device.
    # Overflow in a too-narrow dtype is the usual cause — see lara.device.autocast_dtype.
    if not np.isfinite(vecs).all():
        dtype = getattr(getattr(model, "_first_module", lambda: None)(), "auto_model", None)
        dtype = getattr(getattr(dtype, "dtype", None), "__str__", lambda: "unknown")()
        raise RuntimeError(
            f"the embedder returned a non-finite vector on device "
            f"{getattr(model, 'device', 'unknown')} with dtype {dtype}. This is almost "
            f"always numeric overflow in half precision — check lara.device.autocast_dtype."
        )
    if dim_trunc:
        vecs = vecs[:, :dim_trunc]
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    return vecs


def pending_chunks(
    conn: sqlite3.Connection, limit: int, only_paper: str | None = None
) -> list[sqlite3.Row]:
    """Chunks with no vector yet, joined to the context their prefix needs.

    ``only_paper`` restricts to one paper. The on-demand fetch path needs it: without a
    filter it drains the entire backlog — a single paper's 62 chunks became a two-minute
    request because 1.5M crawled chunks were queued behind them.
    """
    where = "c.vector_row IS NULL" + (" AND c.arxiv_id = ?" if only_paper else "")
    params: tuple = (only_paper, limit) if only_paper else (limit,)
    return conn.execute(
        f"""
        SELECT c.chunk_id, c.text, p.title AS paper_title, s.title AS section_title
        FROM chunks c
        JOIN papers p ON p.arxiv_id = c.arxiv_id
        LEFT JOIN sections s
               ON s.arxiv_id = c.arxiv_id
              AND s.version  = c.version
              AND s.anchor   = c.section_anchor
        WHERE {where}
        ORDER BY c.chunk_id
        LIMIT ?
        """,
        params,
    ).fetchall()


def run(
    conn: sqlite3.Connection,
    store: VectorStore,
    model,
    *,
    batch_size: int = 256,
    slice_size: int = 8192,
    limit: int = 0,
    progress: Iterator | None = None,
    only_paper: str | None = None,
) -> dict[str, int | float]:
    """Embed all pending chunks. Resumable; each slice is durable before the next starts."""
    stats: dict[str, int | float] = {"chunks": 0, "slices": 0, "seconds": 0.0}
    started = time.time()

    while True:
        if limit and stats["chunks"] >= limit:
            break
        rows = pending_chunks(conn, slice_size, only_paper)
        if not rows:
            break
        if limit:
            # Trim to the exact remaining budget; otherwise --limit rounds up to a whole
            # slice, which is confusing when limits are used to sample a small run.
            rows = rows[: limit - int(stats["chunks"])]
            if not rows:
                break

        texts = [
            document_text(r["paper_title"], r["section_title"], r["text"]) for r in rows
        ]
        t0 = time.time()
        vectors = model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)
        elapsed = time.time() - t0

        # Vectors hit disk and are fsynced before any chunk row claims a row number.
        # The reverse order would publish row ids pointing at bytes that were never
        # written — silent corruption instead of a few wasted rows.
        start_row = store.append(vectors)
        with conn:
            conn.executemany(
                "UPDATE chunks SET vector_row=? WHERE chunk_id=?",
                [(start_row + i, r["chunk_id"]) for i, r in enumerate(rows)],
            )

        stats["chunks"] += len(rows)
        stats["slices"] += 1
        stats["seconds"] = time.time() - started
        if progress is not None:
            progress.send((len(rows), len(rows) / max(elapsed, 1e-6), start_row))  # type: ignore[union-attr]

    return stats


def pending_papers(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    """In-scope papers with no paper-level vector yet."""
    return conn.execute(
        """
        SELECT p.arxiv_id, p.title, p.abstract
        FROM papers p
        LEFT JOIN paper_vectors v ON v.arxiv_id = p.arxiv_id
        WHERE p.in_scope = 1 AND p.deleted = 0 AND v.arxiv_id IS NULL
          AND p.abstract IS NOT NULL AND length(p.abstract) > 40
        ORDER BY p.arxiv_id
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def run_papers(
    conn: sqlite3.Connection,
    store: VectorStore,
    model,
    *,
    batch_size: int = 256,
    slice_size: int = 8192,
    limit: int = 0,
    progress=None,
) -> dict[str, int | float]:
    """Embed title+abstract for every in-scope paper (breadth half of paper search).

    Cheap and worth doing regardless of crawl progress: every in-scope paper has an
    abstract, while only a minority has full text, so this is what makes search cover the
    whole corpus instead of the crawled slice.
    """
    stats: dict[str, int | float] = {"papers": 0, "seconds": 0.0}
    started = time.time()
    while True:
        if limit and stats["papers"] >= limit:
            break
        rows = pending_papers(conn, slice_size)
        if not rows:
            break
        if limit:
            rows = rows[: limit - int(stats["papers"])]
            if not rows:
                break

        texts = [document_text(r["title"], None, r["abstract"]) for r in rows]
        t0 = time.time()
        vectors = model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)
        elapsed = time.time() - t0

        start_row = store.append(vectors)
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO paper_vectors(arxiv_id, vector_row, n_chars) "
                "VALUES (?,?,?)",
                [
                    (r["arxiv_id"], start_row + i, len(texts[i]))
                    for i, r in enumerate(rows)
                ],
            )
        stats["papers"] += len(rows)
        stats["seconds"] = time.time() - started
        if progress is not None:
            progress.send((len(rows), len(rows) / max(elapsed, 1e-6), start_row))
    return stats


def rebuild_fts(conn: sqlite3.Connection) -> int:
    """(Re)populate the BM25 index. Cheap, and the lexical half of hybrid retrieval.

    Triggers keep ``chunks_fts`` in sync on insert, so this is only needed after a bulk
    load that bypassed them or a tokenizer change.
    """
    with conn:
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    return conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]


def refresh_df(conn: sqlite3.Connection) -> int:
    """Materialize document frequencies for query planning. See the chunk_df schema note."""
    with conn:
        conn.execute("DELETE FROM chunk_df")
        conn.execute("INSERT INTO chunk_df(term, doc) SELECT term, doc FROM chunks_vocab")
    return conn.execute("SELECT COUNT(*) FROM chunk_df").fetchone()[0]
