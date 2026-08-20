"""Tier-1 search backends — pick one to match the hardware you actually have.

Until now there was exactly one implementation: convert the int8 file to fp16, put all of
it on a CUDA device, and matmul. That is the fastest option on this machine and impossible
on most others. 28.7 M chunks at 256 dims is 14.7 GB in fp16, which no laptop will hold,
and the code called ``torch.cuda`` unconditionally on the way there.

Four combinations are now selectable, along two axes the user is asked about at setup:

**Measured on 1–2 M real corpus vectors, recall@200 against exact fp16.** The full-corpus
memory column is the 2 M figure scaled by 14.8x:

==================  ========  ==========  ========  ==============================================
backend             p50       recall      full mem  verdict
==================  ========  ==========  ========  ==============================================
``torch`` fp16 cuda   1.1 ms     1.000     15.1 GB  fastest anywhere; the CUDA default
``torch`` int8 cuda   8.2 ms     0.996      7.5 GB  half the memory for 0.4 % recall
``torch`` fp16 cpu   12.3 ms     0.995     15.1 GB  **best exact CPU option**
``torch`` int8 cpu    282 ms     0.995      7.5 GB  dequant is not vectorised on CPU — avoid
``faiss`` flat       61.3 ms     0.996     30.3 GB  slower *and* larger than torch fp16
``faiss`` sq8         372 ms     0.991      7.5 GB  smallest, far too slow
``faiss`` hnsw        2.1 ms     0.979     37.9 GB  fastest at scale, but memory-hungry
==================  ========  ==========  ========  ==============================================

Two of these contradicted the assumptions this module was first written with, so they are
worth stating plainly:

**faiss is not automatically the right CPU backend.** A plain torch matmul on CPU beats
faiss flat by 5x and faiss sq8 by 30x, while using less memory than flat. Faiss earns its
place only through HNSW, whose sublinear search is genuinely in another class — and which
pays for it in memory and a slow build. So ``auto`` prefers torch everywhere, and faiss is
an opt-in for people who want HNSW's latency and can afford its footprint.

**int8 is a GPU optimisation, not a memory trick you can apply anywhere.** On CUDA it
halves memory for 0.4 % recall. On CPU the same code path costs 282 ms, because the
per-block dequantisation that a GPU absorbs trivially is the dominant cost without one.

**Precision costs less than it looks.** Tier 1 only has to produce a shortlist: every
survivor is re-scored at exact fp16-768 from the mmap in tier 2, and then reranked by a
cross-encoder. int8 residency loses a little ordering fidelity in a stage whose output is
re-ordered twice more downstream — 0.996 recall confirms it.

**No backend makes the full corpus fit on a laptop.** The smallest usable configuration is
7.5 GB and the fastest is 37.9 GB, against maybe 10 GB usable on a 16 GB Mac. Topic-scoped
residency (D22) is therefore not a nicety on that hardware, it is the enabling feature: at
``keep=0.05`` the same corpus is ~1.4 M chunks, where torch fp16 needs 0.7 GB and HNSW
1.8 GB and both answer in single-digit milliseconds.

**Subset search never goes through faiss.** Scoping to a paper or a citation neighbourhood
hands us explicit rows, and brute force over a gathered handful is exact, fast, and free of
the IDSelector support differences between faiss versions and index types. Faiss is used
only for whole-index search, which is what it is good at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from lara import device as dev


def _normalise(block: np.ndarray) -> np.ndarray:
    """int8 rows -> unit-norm float32. The stored bytes are a quantisation of unit
    vectors, so renormalising after the cast restores a true cosine."""
    out = block.astype(np.float32)
    np.divide(out, np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9), out=out)
    return out


class _Resident:
    """Shared global-row <-> local-index translation for partially resident indexes.

    Both backends need this and it is where a silent bug would live: returning a local
    index where a caller expects a global vector row points at the wrong passage without
    raising anything. Implemented once, tested once.
    """

    row_ids: np.ndarray | None

    @property
    def resident(self) -> bool:
        return self.row_ids is not None

    def to_local(self, rows: np.ndarray) -> np.ndarray:
        rows = np.ascontiguousarray(rows, dtype=np.int64)
        if self.row_ids is None:
            return rows
        if rows.size == 0 or self.row_ids.size == 0:
            return np.empty(0, dtype=np.int64)
        pos = np.searchsorted(self.row_ids, rows)
        clipped = np.minimum(pos, self.row_ids.size - 1)
        return pos[self.row_ids[clipped] == rows]

    def to_global(self, local: np.ndarray) -> np.ndarray:
        return local if self.row_ids is None else self.row_ids[local]


def _prepare_rows(int8_matrix: np.ndarray, row_ids: np.ndarray | None):
    """Normalise the residency argument and report how many rows the index will hold."""
    if row_ids is None:
        return None, int8_matrix.shape[0]
    row_ids = np.ascontiguousarray(row_ids, dtype=np.int64)
    if row_ids.size and not np.all(np.diff(row_ids) > 0):
        row_ids = np.unique(row_ids)          # sorted is a precondition of to_local
    return row_ids, int(row_ids.size)


# ── faiss ─────────────────────────────────────────────────────────────────────────


@dataclass
class FaissConfig:
    kind: str = "hnsw"        # flat | sq8 | hnsw — hnsw is the only reason to pick faiss
    m: int = 32               # hnsw graph degree
    ef_construction: int = 200
    # Measured recall@200 on 1 M real vectors: ef 64 -> 0.750, 128 -> 0.872, 256 -> 0.938,
    # 512 -> 0.979, 1024 -> 0.993, costing 0.3/0.6/1.1/2.1/4.1 ms respectively. The old
    # default of 64 discarded a quarter of the true top-200 to save 1.8 ms, which nobody
    # would choose knowingly. 512 is where recall stops being the limiting factor.
    ef_search: int = 512
    threads: int = 0          # 0 = leave faiss's default


class FaissBackend(_Resident):
    """CPU search via faiss. The portable path — no CUDA, no Metal, works anywhere.

    Vectors are added in blocks: reconstructing all 28.7 M rows as float32 at once would
    need 29 GB of transient, which defeats the point of choosing this backend.
    """

    def __init__(self, int8_matrix: np.ndarray, *, row_ids: np.ndarray | None = None,
                 cfg: FaissConfig | None = None, block: int = 500_000) -> None:
        import faiss

        self.cfg = cfg or FaissConfig()
        self.device = "cpu"
        self.row_ids, n = _prepare_rows(int8_matrix, row_ids)
        self.src = int8_matrix                 # retained for exact subset search
        dim = int(int8_matrix.shape[1])
        self.n, self.dim = n, dim

        if self.cfg.threads:
            faiss.omp_set_num_threads(int(self.cfg.threads))

        kind = self.cfg.kind
        if kind == "hnsw":
            index = faiss.IndexHNSWFlat(dim, int(self.cfg.m), faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = int(self.cfg.ef_construction)
            index.hnsw.efSearch = int(self.cfg.ef_search)
        elif kind == "sq8":
            index = faiss.IndexScalarQuantizer(
                dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
        elif kind == "flat":
            index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"unknown faiss kind {kind!r}; use flat, sq8 or hnsw")

        # A scalar quantiser needs its ranges before anything can be added. One block is a
        # large enough sample: these are unit vectors, so the per-dimension range is
        # already tightly bounded and does not drift across the corpus.
        if not index.is_trained:
            index.train(self._block(0, min(block, n)))

        for start in range(0, n, block):
            index.add(self._block(start, min(start + block, n)))

        self.index = index

    def _block(self, start: int, stop: int) -> np.ndarray:
        rows = (self.src[start:stop] if self.row_ids is None
                else self.src[self.row_ids[start:stop]])
        return np.ascontiguousarray(_normalise(rows))

    def memory_bytes(self) -> int:
        per = {"flat": 4 * self.dim, "sq8": self.dim,
               "hnsw": 4 * self.dim + 8 * self.cfg.m}[self.cfg.kind]
        return self.n * per

    def describe(self) -> str:
        extra = f", ef_search={self.cfg.ef_search}, M={self.cfg.m}" if self.cfg.kind == "hnsw" else ""
        return f"faiss {self.cfg.kind} on cpu ({self.n:,} vectors{extra})"

    def search(self, query: np.ndarray, k: int = 200, rows: np.ndarray | None = None):
        q = np.ascontiguousarray(np.atleast_2d(query).astype(np.float32))
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)

        if rows is None:
            scores, idx = self.index.search(q, min(k, self.n))
            keep = idx[0] >= 0                 # faiss pads short results with -1
            return self.to_global(idx[0][keep]), scores[0][keep]

        local = self.to_local(rows)
        if local.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        src_rows = local if self.row_ids is None else self.row_ids[local]
        sub = _normalise(self.src[src_rows])
        sims = sub @ q[0]
        top = np.argsort(-sims)[:min(k, sims.size)]
        return src_rows[top], sims[top].astype(np.float32)

    def search_multi(self, queries: np.ndarray, k: int = 200, reduce: str = "lse",
                     temp: float = 0.05):
        """K searches, merged on the host.

        faiss has no reduce-across-queries primitive, so this is approximate in a way the
        torch path is not: a row that scores moderately against every reference but tops
        none of their individual result lists is never seen. Widening each search by 4x
        limits that, and `sum`/`mean` should be collapsed to one vector by the caller
        anyway.
        """
        import numpy as _np

        qs = _np.atleast_2d(_np.asarray(queries, dtype=_np.float32))
        acc: dict[int, list[float]] = {}
        for i in range(qs.shape[0]):
            rows, scores = self.search(qs[i], k=k * 4)
            for rr, sc in zip(rows, scores):
                acc.setdefault(int(rr), []).append(float(sc))
        if not acc:
            return _np.empty(0, dtype=_np.int64), _np.empty(0, dtype=_np.float32)
        ids = _np.fromiter(acc.keys(), dtype=_np.int64, count=len(acc))
        if reduce == "max":
            vals = _np.array([max(v) for v in acc.values()], dtype=_np.float32)
        elif reduce in ("sum", "mean"):
            vals = _np.array([(sum(v) / (len(v) if reduce == "mean" else 1))
                              for v in acc.values()], dtype=_np.float32)
        else:
            vals = _np.array([temp * _np.log(_np.exp(_np.asarray(v) / temp).sum())
                              for v in acc.values()], dtype=_np.float32)
        order = _np.argsort(-vals)[:k]
        return ids[order], vals[order]

    def paper_scores(self, query: np.ndarray, rows_by_paper: dict[str, np.ndarray],
                     top_n: int = 3) -> dict[str, float]:
        q = np.ascontiguousarray(np.atleast_2d(query).astype(np.float32))[0]
        q /= max(float(np.linalg.norm(q)), 1e-9)
        out: dict[str, float] = {}
        for paper, rows in rows_by_paper.items():
            if rows.size == 0:
                continue
            local = self.to_local(rows)
            if local.size == 0:
                continue
            src_rows = local if self.row_ids is None else self.row_ids[local]
            sims = _normalise(self.src[src_rows]) @ q
            out[paper] = float(np.sort(sims)[-min(top_n, sims.size):].mean())
        return out


# ── torch ─────────────────────────────────────────────────────────────────────────


class TorchBackend(_Resident):
    """Exact search as a device matmul — CUDA, MPS or CPU.

    ``precision="int8"`` keeps the matrix in its stored form and dequantises per block
    inside the search, halving resident memory. That trades bandwidth for capacity: the
    dequant pass is what makes it slower, and the block size bounds the transient.
    """

    def __init__(self, int8_matrix: np.ndarray, device: str | None = None,
                 *, precision: str = "fp16", row_ids: np.ndarray | None = None,
                 block: int = 1_000_000, search_block: int = 4_000_000) -> None:
        if precision not in ("fp16", "int8"):
            raise ValueError(f"precision must be fp16 or int8, got {precision!r}")
        self.device = dev.resolve(device)
        if precision == "int8" and self.device == "cpu":
            import warnings
            warnings.warn(
                "index.precision=int8 on CPU is ~25x slower than fp16 (282 ms vs 12 ms per "
                "query, measured): the per-block dequantisation a GPU absorbs for free is "
                "the dominant cost without one. Use fp16 on CPU, and cut memory with a "
                "topic scope (`lara corpus scope`) instead.",
                RuntimeWarning, stacklevel=2)
        self.precision = precision
        self.search_block = search_block
        self.row_ids, n = _prepare_rows(int8_matrix, row_ids)
        dim = int(int8_matrix.shape[1])
        self.n, self.dim = n, dim

        dtype = torch.float16 if precision == "fp16" else torch.int8
        self.matrix = torch.empty((n, dim), dtype=dtype, device=self.device)
        for start in range(0, n, block):
            stop = min(start + block, n)
            src = (int8_matrix[start:stop] if self.row_ids is None
                   else int8_matrix[self.row_ids[start:stop]])
            # np.array(copy) rather than ascontiguousarray: a slice of a read-only memmap
            # stays read-only, and torch.from_numpy warns that writing through it is UB.
            t = torch.from_numpy(np.array(src, dtype=np.int8, order="C")).to(self.device)
            if precision == "fp16":
                self.matrix[start:stop] = torch.nn.functional.normalize(t.float(), dim=1).half()
            else:
                # Stored bytes are already a quantisation of unit vectors; keeping them
                # verbatim means the only cost is the per-row norm recovered at search.
                self.matrix[start:stop] = t
            del t
        dev.empty_cache(self.device)

    def memory_bytes(self) -> int:
        return self.matrix.element_size() * self.matrix.nelement()

    # Kept for callers that predate the rename.
    def vram_bytes(self) -> int:
        return self.memory_bytes()

    def describe(self) -> str:
        return f"torch {self.precision} on {self.device} ({self.n:,} vectors)"

    def _scores_for(self, q: torch.Tensor, idx: torch.Tensor | None) -> torch.Tensor:
        """Inner products against all rows, or a subset, dequantising if needed."""
        if self.precision == "fp16":
            m = self.matrix if idx is None else self.matrix[idx]
            return q @ m.T
        n = self.n if idx is None else int(idx.numel())
        out = torch.empty((1, n), dtype=torch.float32, device=self.device)
        for s in range(0, n, self.search_block):
            e = min(s + self.search_block, n)
            raw = self.matrix[s:e] if idx is None else self.matrix[idx[s:e]]
            blk = torch.nn.functional.normalize(raw.float(), dim=1)
            out[0, s:e] = (q.float() @ blk.T)[0]
            del blk
        return out

    def search(self, query: np.ndarray, k: int = 200, rows: np.ndarray | None = None):
        q = torch.from_numpy(np.ascontiguousarray(query, dtype=np.float16)).to(self.device)
        if q.ndim == 1:
            q = q.unsqueeze(0)

        if rows is None:
            scores = self._scores_for(q, None)
            top = torch.topk(scores, min(k, self.n), dim=1)
            return (self.to_global(top.indices[0].cpu().numpy()),
                    top.values[0].float().cpu().numpy())

        local = self.to_local(rows)
        if local.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        subset = torch.from_numpy(local).to(self.device)
        scores = self._scores_for(q, subset)
        top = torch.topk(scores, min(k, subset.numel()), dim=1)
        picked = subset[top.indices[0]].cpu().numpy()
        return self.to_global(picked), top.values[0].float().cpu().numpy()

    def search_multi(self, queries: np.ndarray, k: int = 200, reduce: str = "lse",
                     temp: float = 0.05):
        """Score every row against K reference vectors and reduce to one score per row.

        For a taste profile, `sum` and `mean` are not worth calling here: adding cosine
        similarities is identical to searching once with the summed vector, so the caller
        should collapse them and use `search`. What this exists for are the reductions that
        do NOT collapse — `max` and `lse` — where distinct interests must stay distinct.

        Measured on 29.6 M x 256 resident fp16: ~11 ms for one vector and ~16 ms for fifty,
        because streaming the 15.1 GB matrix dominates and the extra columns ride along
        almost free. LogSumExp costs more (~52 ms at fifty) — it is the exp/log over the
        (block x K) block, not the matmul.
        """
        q = torch.from_numpy(np.ascontiguousarray(queries, dtype=np.float16)).to(self.device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        q = torch.nn.functional.normalize(q.float(), dim=1).half()

        best = torch.empty(self.n, device=self.device, dtype=torch.float32)
        block = self.search_block
        for s0 in range(0, self.n, block):
            e0 = min(s0 + block, self.n)
            sims = self._block_sims(q, s0, e0)                    # (rows, K) float32
            if reduce == "max":
                best[s0:e0] = sims.max(dim=1).values
            elif reduce == "sum":
                best[s0:e0] = sims.sum(dim=1)
            elif reduce == "mean":
                best[s0:e0] = sims.mean(dim=1)
            else:
                m = sims.max(dim=1, keepdim=True).values
                best[s0:e0] = (m[:, 0] + temp * torch.log(
                    torch.exp((sims - m) / temp).sum(dim=1)))
            del sims
        top = torch.topk(best, min(k, self.n))
        return (self.to_global(top.indices.cpu().numpy()),
                top.values.float().cpu().numpy())

    def _block_sims(self, q: torch.Tensor, s0: int, e0: int) -> torch.Tensor:
        """(rows, K) similarities for one block.

        Dequantisation keys on `self.precision`, exactly as `_scores_for` does — the int8
        rows are stored unnormalised and must be renormalised after the cast, or every
        similarity comes out scaled by an arbitrary row norm.
        """
        raw = self.matrix[s0:e0]
        if self.precision == "fp16":
            return (raw @ q.T).float()
        blk = torch.nn.functional.normalize(raw.float(), dim=1)
        return blk @ q.T.float()

    def paper_scores(self, query: np.ndarray, rows_by_paper: dict[str, np.ndarray],
                     top_n: int = 3) -> dict[str, float]:
        q = torch.from_numpy(np.ascontiguousarray(query, dtype=np.float16)).to(self.device)
        if q.ndim == 1:
            q = q.unsqueeze(0)
        out: dict[str, float] = {}
        for paper, rows in rows_by_paper.items():
            if rows.size == 0:
                continue
            local = self.to_local(rows)
            if local.size == 0:
                continue
            subset = torch.from_numpy(local).to(self.device)
            scores = self._scores_for(q, subset)[0]
            out[paper] = float(torch.topk(scores, min(top_n, scores.numel())).values.mean())
        return out


# ── selection ─────────────────────────────────────────────────────────────────────


def faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False


def choose_backend(requested: str | None) -> str:
    """Resolve ``auto``. Always torch — faiss is opt-in.

    This deliberately does *not* prefer faiss off CUDA, which is what an earlier version
    did on the assumption that faiss owns CPU search. Measured, a torch matmul on CPU beats
    faiss flat by 5x and faiss sq8 by 30x at equal or lower memory. The one case where
    faiss wins outright is HNSW, and that trades 2.5x the memory and a slow build for its
    latency — a real choice with real costs, not a default anyone should be opted into.
    """
    requested = (requested or "auto").lower()
    return "torch" if requested == "auto" else requested


def make_index(int8_matrix: np.ndarray, *, backend: str | None = None,
               precision: str = "fp16", device: str | None = None,
               row_ids: np.ndarray | None = None, faiss_cfg: FaissConfig | None = None):
    """Build the tier-1 index the config asks for, falling back loudly rather than dying."""
    chosen = choose_backend(backend)
    if chosen == "faiss":
        if not faiss_available():
            import warnings
            warnings.warn(
                "index.backend=faiss but faiss is not installed "
                "(pip install 'lara[cpu]'); falling back to the torch matmul",
                RuntimeWarning, stacklevel=2)
        else:
            cfg = faiss_cfg or FaissConfig()
            if precision == "int8" and cfg.kind == "flat":
                cfg = FaissConfig(**{**cfg.__dict__, "kind": "sq8"})
            return FaissBackend(int8_matrix, row_ids=row_ids, cfg=cfg)
    return TorchBackend(int8_matrix, device=device, precision=precision, row_ids=row_ids)
