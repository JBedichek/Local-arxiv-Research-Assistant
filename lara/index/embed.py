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

from lara.index.vectors import VectorStore

QUERY_PROMPT = "task: search result | query: "


def document_text(title: str | None, section_title: str | None, text: str) -> str:
    """Build the contextual document string. Mirrors EmbeddingGemma's native format."""
    head = (title or "none").strip()
    section = (section_title or "").strip()
    if section and section.lower() not in head.lower():
        head = f"{head} > {section}"
    return f"title: {head} | text: {text}"


def load_model(
    name: str,
    device: str = "cuda:0",
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

    model = SentenceTransformer(name, device=device, model_kwargs={"dtype": torch.bfloat16})
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
    """

    def __init__(self, model, devices: list[int], chunk_size: int | None = None) -> None:
        self.model = model
        self.devices = devices
        self.chunk_size = chunk_size
        self.pool = model.start_multi_process_pool([f"cuda:{d}" for d in devices])

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
    if dim_trunc:
        vecs = vecs[:, :dim_trunc]
        vecs /= np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    return vecs


def pending_chunks(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    """Chunks with no vector yet, joined to the context their prefix needs."""
    return conn.execute(
        """
        SELECT c.chunk_id, c.text, p.title AS paper_title, s.title AS section_title
        FROM chunks c
        JOIN papers p ON p.arxiv_id = c.arxiv_id
        LEFT JOIN sections s
               ON s.arxiv_id = c.arxiv_id
              AND s.version  = c.version
              AND s.anchor   = c.section_anchor
        WHERE c.vector_row IS NULL
        ORDER BY c.chunk_id
        LIMIT ?
        """,
        (limit,),
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
) -> dict[str, int | float]:
    """Embed all pending chunks. Resumable; each slice is durable before the next starts."""
    stats: dict[str, int | float] = {"chunks": 0, "slices": 0, "seconds": 0.0}
    started = time.time()

    while True:
        if limit and stats["chunks"] >= limit:
            break
        rows = pending_chunks(conn, slice_size)
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


def rebuild_fts(conn: sqlite3.Connection) -> int:
    """(Re)populate the BM25 index. Cheap, and the lexical half of hybrid retrieval."""
    with conn:
        conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    return conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
