"""The retriever — assembles the stages in :mod:`lara.index.search` into one call.

Held open by the server process so the corpus tensor, the memory-mapped fp16 file, the
embedder and the reranker are all warm. Nothing here loads a model per request.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from lara import device as dev
from lara.index import search as S
from lara.index.embed import embed_queries
from lara.index.vectors import VectorStore


@dataclass
class RetrievalResult:
    hits: list[S.Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)
    n_candidates: int = 0


class Retriever:
    def __init__(
        self,
        conn: sqlite3.Connection | Callable[[], sqlite3.Connection],
        store: VectorStore,
        embedder,
        *,
        device: str | int | None = None,
        dim_trunc: int = 256,
        tier2_candidates: int = 200,
        rerank_candidates: int = 50,
        final_k: int = 8,
        rrf_k: int = 60,
        lexical: bool = True,
        max_terms: int = 3,
        df_ceiling_frac: float = 0.005,
        cross_encoder=None,
    ) -> None:
        # A connection may be passed directly (CLI, single-threaded) or as a factory
        # (server). It must be a factory under a server: Starlette dispatches sync
        # endpoints across a threadpool, and sqlite3 refuses to use a connection from a
        # thread other than the one that created it. Sharing one connection makes
        # requests fail or succeed depending on which worker thread they land on.
        # isinstance, NOT callable(): sqlite3.Connection defines __call__ (it compiles a
        # statement), so callable() reports True for a real connection and the "factory"
        # becomes the connection itself — which then raises
        # "function takes exactly 1 argument (0 given)" on first use.
        self._conn_factory: Callable[[], sqlite3.Connection] = (
            (lambda c=conn: c) if isinstance(conn, sqlite3.Connection) else conn
        )
        self.store = store
        self.embedder = embedder
        self.dim_trunc = dim_trunc
        self.tier2_candidates = tier2_candidates
        self.rerank_candidates = rerank_candidates
        self.final_k = final_k
        self.rrf_k = rrf_k
        self.lexical = lexical
        self.max_terms = max_terms
        self.df_ceiling_frac = df_ceiling_frac
        self.cross_encoder = cross_encoder
        self.corpus_size = (
            self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] or 1
        )

        self.device = dev.resolve(device)
        self.dense = S.DenseIndex(store.load_int8(), device=self.device)
        self.fp16 = store.open_fp16()          # mmap; the OS page cache does the rest
        self._row_to_chunk = self._load_row_map()

    @property
    def conn(self) -> sqlite3.Connection:
        """The connection belonging to the calling thread."""
        return self._conn_factory()

    def _load_row_map(self) -> dict[int, int]:
        return {
            r["vector_row"]: r["chunk_id"]
            for r in self.conn.execute(
                "SELECT vector_row, chunk_id FROM chunks WHERE vector_row IS NOT NULL"
            )
        }

    def rows_for_papers(self, papers: list[str]) -> np.ndarray:
        """Vector rows belonging to a set of papers — the tier-0 / scoped-search filter."""
        if not papers:
            return np.empty(0, dtype=np.int64)
        ph = ",".join("?" * len(papers))
        return np.fromiter(
            (
                r[0] for r in self.conn.execute(
                    f"SELECT vector_row FROM chunks "
                    f"WHERE arxiv_id IN ({ph}) AND vector_row IS NOT NULL",
                    papers,
                )
            ),
            dtype=np.int64,
        )

    def refresh_row_map(self) -> None:
        """Call after an incremental embed run extends the corpus."""
        self._row_to_chunk = self._load_row_map()

    # ── stages ────────────────────────────────────────────────────────────────────

    def _ensure_fp16_current(self) -> None:
        """Re-open the vector mmap if the file has grown since it was mapped.

        ``np.memmap`` fixes its length at open time, but the embedder appends to this
        file continuously while the server runs. Without this, any chunk embedded after
        startup raises ``IndexError: index N is out of bounds`` the moment BM25 surfaces
        it — a 500 that only appears once the corpus has moved on, and never in a test
        against older chunks.
        """
        try:
            on_disk = self.store.rows()
        except RuntimeError:
            return                      # torn write between the two files; try later
        if on_disk > self.fp16.shape[0]:
            # Re-map only. Rebuilding the row map here scans all 11M chunk rows and took
            # ~5 s, and because the embedder grows the file continuously it fired on
            # EVERY request while indexing was running — turning a 40 ms retrieval into a
            # 5 s one. It is also unnecessary: rows added since startup are not in the
            # GPU tier-1 index, so dense search cannot return them until /api/reload
            # rebuilds both together.
            self.fp16 = self.store.open_fp16()

    def _tier2_rescore(self, rows: np.ndarray, query_full: np.ndarray) -> np.ndarray:
        """Exact fp16 768-d scores for shortlisted rows, read from the mmap."""
        if rows.size == 0:
            return np.empty(0, dtype=np.float32)
        vectors = np.asarray(self.fp16[rows], dtype=np.float32)
        return vectors @ query_full

    def retrieve(
        self,
        query: str,
        *,
        papers: list[str] | None = None,
        rows: np.ndarray | None = None,
        selection: str | None = None,
        final_k: int | None = None,
    ) -> RetrievalResult:
        """Run the full pipeline.

        ``selection`` is the highlighted passage from the open paper (R4). It is appended
        to the embedded query so retrieval is anchored on what the user is looking at, but
        kept out of the BM25 term list, where a long passage's common words would swamp
        the question's distinctive ones.
        """
        t: dict[str, float] = {}
        clock = time.perf_counter

        t0 = clock()
        embed_text = f"{query}\n\n{selection}" if selection else query
        q_full = embed_queries(self.embedder, [embed_text])[0]
        q_trunc = q_full[: self.dim_trunc]
        q_trunc = q_trunc / max(float(np.linalg.norm(q_trunc)), 1e-12)
        t["embed"] = (clock() - t0) * 1000

        t0 = clock()
        if rows is None and papers:
            # A scope restriction has to reach the dense path too. Passing it only to
            # BM25 silently returns whole-corpus hits for "this paper" searches.
            rows = self.rows_for_papers(papers)
            if rows.size == 0:
                return RetrievalResult([], t, 0)
        dense_rows, _ = self.dense.search(q_trunc, k=self.tier2_candidates, rows=rows)
        dense_ids = [
            self._row_to_chunk[int(r)] for r in dense_rows if int(r) in self._row_to_chunk
        ]
        t["dense"] = (clock() - t0) * 1000

        t0 = clock()
        lexical_ids = (
            S.bm25_search(
                self.conn, query, k=self.tier2_candidates, papers=papers,
                max_terms=self.max_terms, df_ceiling_frac=self.df_ceiling_frac,
                corpus_size=self.corpus_size,
            )
            if self.lexical else []
        )
        t["bm25"] = (clock() - t0) * 1000

        t0 = clock()
        fused_ids, parts = S.reciprocal_rank_fusion(
            {"dense": dense_ids, "bm25": lexical_ids}, k=self.rrf_k
        )
        t["fuse"] = (clock() - t0) * 1000
        if not fused_ids:
            return RetrievalResult([], t, 0)

        shortlist = fused_ids[: self.rerank_candidates * 2]
        hits_by_id = S.hydrate(self.conn, shortlist)

        # Tier 2: exact full-precision rescore, which is what recovers the accuracy the
        # 256-d truncation gives up. Chunks BM25 surfaced that dense search missed have
        # vectors too, so they are rescored on the same footing.
        t0 = clock()
        self._ensure_fp16_current()
        # ids and rows are built in one pass and share the same bound, so the zip below
        # stays aligned. Filtering inside _tier2_rescore would silently misalign scores
        # with chunks — worse than the IndexError it was meant to avoid.
        n_rows = self.fp16.shape[0]
        pairs = [
            (c, hits_by_id[c].vector_row)
            for c in shortlist
            if c in hits_by_id and 0 <= hits_by_id[c].vector_row < n_rows
        ]
        ids_for_rescore = [c for c, _ in pairs]
        rows_for_rescore = np.fromiter((r for _, r in pairs), dtype=np.int64, count=len(pairs))
        exact = self._tier2_rescore(rows_for_rescore, q_full)
        for cid, score in zip(ids_for_rescore, exact):
            hits_by_id[cid].score = float(score)
            hits_by_id[cid].provenance = parts.get(cid, {})
            hits_by_id[cid].provenance["exact"] = float(score)
        t["tier2"] = (clock() - t0) * 1000

        ranked = sorted(
            (h for h in hits_by_id.values()), key=lambda h: -h.score
        )[: self.rerank_candidates]

        t0 = clock()
        if self.cross_encoder is not None and ranked:
            pairs = [
                (query, f"{h.paper_title} > {h.section_title}\n{h.text}") for h in ranked
            ]
            scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
            for hit, score in zip(ranked, scores):
                hit.provenance["cross_encoder"] = float(score)
                hit.score = float(score)
            ranked.sort(key=lambda h: -h.score)
        t["rerank"] = (clock() - t0) * 1000
        t["total"] = sum(v for k, v in t.items() if k != "total")

        return RetrievalResult(ranked[: (final_k or self.final_k)], t, len(fused_ids))


def load_cross_encoder(
    name: str, device: str | int | None = None, max_length: int = 512, verify: bool = True
):
    """Load the reranker.

    Qwen3-Reranker ships native sentence-transformers support, but as a decoder-derived
    model it defines no padding token — batching more than one pair raises
    ``Cannot handle batch sizes > 1 if no padding token is defined``. Since reranking is
    inherently batched (50 pairs at a time), pad has to be supplied. EOS is the
    conventional choice and is masked out anyway.
    """
    from sentence_transformers import CrossEncoder

    device = dev.resolve(device if device is not None else 1)
    ce = CrossEncoder(
        name, device=device, max_length=max_length,
        model_kwargs={"dtype": dev.model_dtype(device)},
    )
    tok = ce.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if getattr(ce.model.config, "pad_token_id", None) is None:
        ce.model.config.pad_token_id = tok.pad_token_id
    # Left padding keeps the final position — which a causal scorer reads — aligned
    # across a batch of differently-sized pairs.
    tok.padding_side = "left"
    if verify:
        assert_reranker_works(ce, name)
    return ce


_PROBE_QUERY = "How does multi-head self-attention avoid recurrence?"
_PROBE_RELEVANT = (
    "The Transformer replaces recurrence entirely with multi-head self-attention, letting "
    "every position attend to every other position in a single step."
)
_PROBE_IRRELEVANT = (
    "We fry the onions in olive oil for eight minutes, then add garlic and tomato paste."
)


def assert_reranker_works(ce, name: str, margin: float = 0.05) -> None:
    """Fail loudly if the scoring head is untrained.

    Loading a reranker whose head is randomly initialised produces no exception and no
    obviously wrong output — it returns well-formed floats and dutifully ranks documents,
    so retrieval quality degrades to noise while every test that only checks types and
    shapes still passes. ``Qwen/Qwen3-Reranker-4B`` does exactly this under
    ``CrossEncoder``: transformers emits a ``score.weight ... newly initialized`` warning
    among a wall of other logging, and the model then scored a self-attention passage
    0.865 and a recipe for frying onions 0.862.

    A behavioural probe is the only reliable check, so it runs at load time.
    """
    import numpy as np

    scores = np.asarray(
        ce.predict(
            [(_PROBE_QUERY, _PROBE_RELEVANT), (_PROBE_QUERY, _PROBE_IRRELEVANT)],
            show_progress_bar=False,
        ),
        dtype=float,
    ).ravel()
    if not scores[0] > scores[1] + margin:
        raise RuntimeError(
            f"reranker {name!r} does not discriminate: scored a relevant passage "
            f"{scores[0]:.3f} against an unrelated one {scores[1]:.3f}. Its scoring head "
            "is probably randomly initialised — sentence-transformers cannot serve models "
            "that score via LM logits (e.g. Qwen3-Reranker) and silently substitutes an "
            "untrained classification head."
        )
