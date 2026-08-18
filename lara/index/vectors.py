"""Append-only flat vector files backing the tiered index (PLAN.md §5).

Two files, written together, one row per chunk at the same row index:

``fp16.bin``  ``(N, 768)`` float16 — tier 2. Memory-mapped, read only for the ~200
              candidates that survive tier 1, and rescored exactly.
``int8.bin``  ``(N, 256)`` int8    — tier 1. Held resident for the whole corpus. At 16 M
              chunks that is 4.1 GB, versus 24 GB for the same vectors in fp16.

The 768 -> 256 reduction is Matryoshka truncation, which EmbeddingGemma is trained for:
take the leading 256 dimensions and renormalize. Accuracy lost at the shortlist stage is
recovered by the exact fp16 rescore in tier 2, so the cheap index only has to be good
enough to keep the right chunks in the top 200.

**Crash behaviour.** Vectors are appended and fsynced *before* the owning chunk rows are
stamped with their ``vector_row``. A crash in between leaves orphan rows at the tail of
the file — wasted bytes, never wrong answers — and those chunks simply get re-embedded on
the next run. The reverse order would hand out row numbers pointing at data that was never
written, which is silent corruption.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

INT8_SCALE = 127.0


def to_int8_mrl(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Truncate to ``dim`` dimensions, renormalize, and quantize symmetrically to int8.

    Inputs are L2-normalized, so every component lies in [-1, 1] and a single global scale
    is exact enough — no per-vector scale factor to store or apply at search time, which
    keeps the tier-1 inner product a plain integer dot product.
    """
    truncated = vectors[:, :dim].astype(np.float32)
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    truncated /= norms
    return np.rint(truncated * INT8_SCALE).clip(-127, 127).astype(np.int8)


class VectorStore:
    def __init__(
        self,
        fp16_path: str | Path,
        int8_path: str | Path,
        *,
        dim_full: int = 768,
        dim_trunc: int = 256,
    ) -> None:
        self.fp16_path = Path(fp16_path)
        self.int8_path = Path(int8_path)
        self.dim_full = dim_full
        self.dim_trunc = dim_trunc
        self.fp16_path.parent.mkdir(parents=True, exist_ok=True)
        self.int8_path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self) -> int:
        """Row count implied by file size, and a consistency check between the two files."""
        n_fp16 = self._rows_of(self.fp16_path, self.dim_full, 2)
        n_int8 = self._rows_of(self.int8_path, self.dim_trunc, 1)
        if n_fp16 != n_int8:
            raise RuntimeError(
                f"vector files disagree: fp16 has {n_fp16} rows, int8 has {n_int8}. "
                "Truncate both to the shorter length and re-embed the difference."
            )
        return n_fp16

    @staticmethod
    def _rows_of(path: Path, dim: int, itemsize: int) -> int:
        if not path.exists():
            return 0
        size = path.stat().st_size
        stride = dim * itemsize
        if size % stride:
            raise RuntimeError(f"{path} is {size} bytes, not a multiple of {stride}")
        return size // stride

    def append(self, vectors: np.ndarray) -> int:
        """Append a batch, returning the row index the batch starts at."""
        if vectors.ndim != 2 or vectors.shape[1] != self.dim_full:
            raise ValueError(f"expected (n, {self.dim_full}), got {vectors.shape}")
        start = self.rows()
        fp16 = np.ascontiguousarray(vectors, dtype=np.float16)
        int8 = to_int8_mrl(vectors, self.dim_trunc)

        # fsync both before the caller records the row numbers in SQLite.
        for path, payload in ((self.fp16_path, fp16), (self.int8_path, int8)):
            with open(path, "ab") as fh:
                fh.write(payload.tobytes())
                fh.flush()
                os.fsync(fh.fileno())
        return start

    def open_fp16(self) -> np.memmap:
        return np.memmap(self.fp16_path, dtype=np.float16, mode="r").reshape(-1, self.dim_full)

    def load_int8(self, mmap: bool = False) -> np.ndarray:
        """The tier-1 matrix — 7.3 GB at 28.7 M chunks.

        ``mmap=True`` maps it instead of reading it, which matters in two cases that are
        the norm on a small machine: a **scoped** index (D22) touches only the pages of the
        rows it keeps, and a **faiss** build streams the matrix in blocks and never needs
        all of it at once. Reading 7.3 GB into RAM first, only to gather 5 % of it, is the
        difference between working and swapping on a 16 GB laptop.

        The default stays eager: the CUDA path copies the whole thing to the device
        immediately, where a memmap would only add a page-fault detour.
        """
        if mmap:
            return np.memmap(self.int8_path, dtype=np.int8, mode="r").reshape(-1, self.dim_trunc)
        return np.fromfile(self.int8_path, dtype=np.int8).reshape(-1, self.dim_trunc)
