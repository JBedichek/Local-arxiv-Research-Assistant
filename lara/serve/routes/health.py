"""Readiness, liveness, and where the resident bytes actually went."""

from __future__ import annotations

import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from lara.serve.deps import current_state, require_state

router = APIRouter()


@router.get("/api/health")
def health() -> dict:
    s = current_state()
    sc = getattr(s, "scope", None) if s else None
    return {
        "ready": bool(s and s.ready),
        "warmup_ms": (s.warmup_ms if s else {}),
        "vectors": (s.store.rows() if s else 0),
        # D22: null when the whole corpus is resident. Surfaced so the UI can say what the
        # dense index actually covers — a scoped index is not a broken one, but a user
        # deserves to know that semantic recall is narrowed to their topics.
        "scope": None if sc is None else {
            "topics": sc.topics,
            "fraction": sc.fraction,
            "papers": sc.n_papers,
            "resident_chunks": sc.n_rows,
            "corpus_chunks": sc.corpus_chunks,
            "resident_gb": round(sc.resident_bytes() / 1e9, 2),
        },
    }


def _parse_size(text: str) -> float | None:
    """vmmap's '  7.8G' or '892.3M' -> gigabytes."""
    t = text.strip().rstrip("B").strip()
    if not t:
        return None
    mult = 1.0
    scale = {"K": 1e-6, "M": 1e-3, "G": 1.0, "T": 1e3}
    if t[-1].upper() in scale:
        mult, t = scale[t[-1].upper()], t[:-1]
    try:
        return round(float(t) * mult, 2)
    except ValueError:
        return None


# NOT /api/memory: that is the reading library, and registering this there shadowed it,
# because FastAPI matches the first route declared. The library pane silently received
# memory statistics instead of its entries.
@router.get("/api/meminfo")
def memory_breakdown() -> JSONResponse:
    """Where the resident bytes actually are.

    Added because the process was 8.9 GB against a planner that budgeted 3.8, and there
    was no way to ask it why. Reports what can be measured directly rather than what the
    plan assumed: torch's own accounting for device tensors, real sizes for the index and
    the row map, and the process RSS to compare them against.
    """
    import subprocess
    import sys

    s = require_state()
    out: dict = {}

    pid = os.getpid()
    try:
        rss_kb = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                                capture_output=True, text=True, timeout=5).stdout.strip()
        out["process_rss_gb"] = round(int(rss_kb) / 1048576, 2)
    except Exception:
        out["process_rss_gb"] = None

    # On macOS `ps` RSS is not the number anyone means. Metal buffers live in
    # IOAccelerator regions that it does not count, so it reported 0.04 GB for a process
    # whose real footprint was 7.8 GB -- off by two orders of magnitude, and precisely on
    # the memory this endpoint exists to explain. `vmmap --summary` agrees with Activity
    # Monitor. It walks the address space, so it costs a second or two; this is a
    # diagnostic endpoint and that is the right trade.
    if sys.platform == "darwin":
        try:
            vm = subprocess.run(["vmmap", "--summary", str(pid)],
                                capture_output=True, text=True, timeout=30).stdout
            for line in vm.splitlines():
                if "Physical footprint:" in line and "peak" not in line.lower():
                    out["process_footprint_gb"] = _parse_size(line.split(":", 1)[1])
                elif "Physical footprint (peak):" in line:
                    out["process_footprint_peak_gb"] = _parse_size(line.split(":", 1)[1])
        except Exception:
            out["process_footprint_gb"] = None
        out["process_rss_note"] = ("ps RSS excludes Metal/IOAccelerator memory on macOS; "
                                   "use process_footprint_gb")

    import torch
    if s.device == "mps" and hasattr(torch, "mps"):
        # Unified memory: these Metal buffers live in this process's footprint, so they
        # are part of the RSS above rather than separate from it.
        out["torch_mps_allocated_gb"] = round(torch.mps.current_allocated_memory() / 1e9, 2)
        try:
            out["torch_mps_driver_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
        except Exception:
            pass
    elif s.device.startswith("cuda"):
        out["torch_cuda_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        out["torch_cuda_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)

    r = s.retriever
    if r is not None:
        dense = getattr(r, "dense", None)
        if dense is not None and hasattr(dense, "memory_bytes"):
            out["tier1_index_gb"] = round(dense.memory_bytes() / 1e9, 2)
            out["tier1_rows"] = getattr(dense, "n", None)
            out["tier1_precision"] = getattr(dense, "precision", None)
            out["tier1_search_block"] = getattr(dense, "search_block", None)
        rowmap = getattr(r, "_row_to_chunk", None)
        if rowmap is not None:
            # Whole-corpus, so it does not shrink when the corpus is scoped: the one
            # resident cost the keep fraction cannot move.
            out["row_map_entries"] = int(getattr(rowmap, "size", len(rowmap)))
            out["row_map_gb"] = round(
                (rowmap.nbytes if hasattr(rowmap, "nbytes")
                 else sys.getsizeof(rowmap)) / 1e9, 3)
            out["row_map_dtype"] = str(getattr(rowmap, "dtype", "dict"))
        out["tier2_source"] = "fp16 mmap" if getattr(r, "fp16", None) is not None else "int8 mmap"
        out["cross_encoder_loaded"] = getattr(r, "cross_encoder", None) is not None
    if getattr(s, "paper_index", None) is not None:
        out["paper_index_gb"] = round(s.paper_index.memory_bytes() / 1e9, 2)

    # The planner reserves this, but nothing in the serving path allocates it.
    out["hot_tier_configured_gb"] = round(
        float(s.cfg.get_in("hot_tier.max_bytes", 0) or 0) / 1e9, 2)
    out["hot_tier_allocated"] = False
    return JSONResponse(out)


@router.get("/healthz")
def healthz() -> JSONResponse:
    """Liveness, reachable without a token so a tunnel can probe it without a credential."""
    return JSONResponse({"ok": True})
