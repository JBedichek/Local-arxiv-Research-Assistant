"""Process-wide warm state — everything expensive is loaded exactly once.

The single largest cause of a RAG server "feeling slow" is not throughput, it is doing
setup work inside the request path. Loading the embedder costs 8.5 s, the reranker ~5 s,
and moving the index to GPU another second; a first CUDA call costs ~290 ms against ~20 ms
warm. All of it happens here, at startup, before the first request is accepted.

SQLite connections are thread-local because Starlette runs sync endpoints in a threadpool
and a single connection shared across threads serializes every reader behind one lock.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

from lara import config as config_mod
from lara import device as dev
from lara.index import embed as emb
from lara.index import retrieve as R
from lara.index import scope as SC
from lara.index.vectors import VectorStore
from lara.store import db


class AppState:
    def __init__(self, config_path: str | None = None, *, load_models: bool = True) -> None:
        self.cfg = config_mod.load(config_path)
        self.db_path = self.cfg.get_path("paths.metadata_db")
        self.raw_root = self.cfg.get_path("paths.raw_cache")
        self._local = threading.local()
        self.ready = False
        self.warmup_ms: dict[str, float] = {}

        ecfg = self.cfg.get_in("embedding")
        icfg = self.cfg.get_in("index")
        # Config expresses devices as CUDA ordinals; lara.device maps that intent onto
        # whatever is actually here (see lara/device.py). On a Mac this becomes "mps".
        self.device = dev.resolve((ecfg.get("devices") or [None])[0])

        self.store = VectorStore(
            self.cfg.get_path("paths.vectors_fp16"),
            self.cfg.get_path("paths.vectors_int8"),
            dim_full=int(ecfg["dim_full"]), dim_trunc=int(ecfg["dim_truncated"]),
        )
        # Paper-level index for semantic paper search. Optional: it may not be built yet,
        # and search degrades to full-text-only rather than failing.
        self.paper_store = VectorStore(
            self.cfg.get_path("paths.papers_fp16"),
            self.cfg.get_path("paths.papers_int8"),
            dim_full=int(ecfg["dim_full"]), dim_trunc=int(ecfg["dim_truncated"]),
        )
        self.paper_index = None
        self.paper_row_to_id: dict[int, str] = {}

        self.retriever: R.Retriever | None = None
        self.scope: SC.Scope | None = None
        if load_models:
            self._load(ecfg, icfg)
            self._load_paper_index()

    def _load_paper_index(self) -> None:
        from lara.index import search as S

        try:
            if self.paper_store.rows() == 0:
                return
            self.paper_index = S.DenseIndex(self.paper_store.load_int8(), device=self.device)
            self.paper_row_to_id = {
                r["vector_row"]: r["arxiv_id"]
                for r in self.conn().execute("SELECT arxiv_id, vector_row FROM paper_vectors")
            }
            self.warmup_ms["paper_index"] = float(self.paper_index.n)
        except (RuntimeError, FileNotFoundError, ValueError):
            self.paper_index = None      # not built yet; search falls back to chunks

    def chunk_row_to_paper(self, rows) -> dict[int, str]:
        """Map chunk vector rows to their papers, for per-paper aggregation."""
        ids = [int(r) for r in rows.tolist()]
        if not ids:
            return {}
        ph = ",".join("?" * len(ids))
        return {
            r["vector_row"]: r["arxiv_id"]
            for r in self.conn().execute(
                f"SELECT vector_row, arxiv_id FROM chunks WHERE vector_row IN ({ph})", ids
            )
        }

    # ── connections ───────────────────────────────────────────────────────────────

    def conn(self) -> sqlite3.Connection:
        """Read-only, per-thread. WAL lets these run concurrently with the crawler."""
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA busy_timeout=5000")
            self._local.conn = c
        return c

    # ── startup ───────────────────────────────────────────────────────────────────

    def _load(self, ecfg: dict, icfg: dict) -> None:
        t0 = time.time()
        embedder = emb.load_model(
            ecfg["model"], device=self.device, max_seq_length=int(ecfg["max_seq_len"])
        )
        self.warmup_ms["embedder"] = (time.time() - t0) * 1000

        ccfg = (icfg["rerank"].get("cross_encoder") or {})
        cross = None
        if ccfg.get("enabled"):
            t0 = time.time()
            cross = R.load_cross_encoder(
                ccfg["model"], device=dev.resolve(ccfg.get("device", 1)),
                max_length=int(ccfg.get("max_length", 512)),
            )
            self.warmup_ms["reranker"] = (time.time() - t0) * 1000

        t0 = time.time()
        db.connect(self.db_path).close()          # ensure the schema exists, then let go
        lex = icfg["lexical"]
        # D22: if a keep-set exists, only those rows go into tier 1. BM25 and tier 2 stay
        # whole-corpus, so a scoped index narrows semantic recall rather than coverage.
        self.scope = SC.Scope.load(self.cfg.get_path("disk.root"))
        resident = self.scope.rows if self.scope is not None else None
        # Pass the thread-local factory, not a connection — see Retriever.__init__.
        self.retriever = R.Retriever(
            self.conn, self.store, embedder, device=self.device,
            dim_trunc=int(ecfg["dim_truncated"]),
            tier2_candidates=int(icfg["rerank"]["tier2_candidates"]),
            rerank_candidates=int(ccfg.get("candidates", 24)),
            final_k=int(icfg["rerank"].get("final_k", 8)),
            rrf_k=int(lex["rrf_k"]), lexical=bool(lex["enabled"]),
            max_terms=int(lex.get("max_terms", 3)),
            df_ceiling_frac=float(lex.get("df_ceiling_frac", 0.005)),
            cross_encoder=cross,
            resident_rows=resident,
        )
        self.warmup_ms["index"] = (time.time() - t0) * 1000

        # Burn in CUDA kernels and the SQLite page cache so the first real user request
        # is not the one that pays for them.
        t0 = time.time()
        for q in ("warmup query about attention", "gradient checkpointing memory"):
            try:
                self.retriever.retrieve(q)
            except Exception:
                pass
        self.warmup_ms["warmup_queries"] = (time.time() - t0) * 1000
        self.ready = True

    # ── data access ───────────────────────────────────────────────────────────────

    def paper(self, arxiv_id: str) -> sqlite3.Row | None:
        return self.conn().execute(
            "SELECT arxiv_id, title, abstract, authors, categories, submitted_utc, "
            "latest_version, fulltext_status, fulltext_source, fulltext_version, n_chunks "
            "FROM papers WHERE arxiv_id=?",
            (arxiv_id,),
        ).fetchone()

    def sections(self, arxiv_id: str, version: int) -> list[sqlite3.Row]:
        return self.conn().execute(
            "SELECT anchor, title, kind, depth FROM sections "
            "WHERE arxiv_id=? AND version=? ORDER BY ordinal",
            (arxiv_id, version),
        ).fetchall()

    def paper_rows(self, arxiv_ids: list[str]) -> dict[str, np.ndarray]:
        """vector_row ids grouped by paper — the input to heatmap scoring (R9)."""
        if not arxiv_ids:
            return {}
        ph = ",".join("?" * len(arxiv_ids))
        out: dict[str, list[int]] = {a: [] for a in arxiv_ids}
        for r in self.conn().execute(
            f"SELECT arxiv_id, vector_row FROM chunks "
            f"WHERE arxiv_id IN ({ph}) AND vector_row IS NOT NULL",
            arxiv_ids,
        ):
            out[r["arxiv_id"]].append(r["vector_row"])
        return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}

    def neighbours(self, arxiv_id: str, limit: int = 60) -> dict[str, list[str]]:
        c = self.conn()
        cites = [r[0] for r in c.execute(
            "SELECT dst FROM citations WHERE src=? LIMIT ?", (arxiv_id, limit))]
        cited_by = [r[0] for r in c.execute(
            "SELECT src FROM citations WHERE dst=? LIMIT ?", (arxiv_id, limit))]
        return {"cites": cites, "cited_by": cited_by}

    def raw_html_path(self, arxiv_id: str) -> Path | None:
        safe = arxiv_id.replace("/", "_")
        shard = self.raw_root / safe[:4]
        for source in ("arxiv_html", "ar5iv"):
            p = shard / f"{safe}.{source}.html.zst"
            if p.exists():
                return p
        return None
