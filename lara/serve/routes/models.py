"""What this machine is, which generator suits it, and fetching one."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from lara.serve.deps import require_state

router = APIRouter()


@router.get("/api/device")
def device_info() -> JSONResponse:
    """What this machine is, and which generation backend suits it."""
    from lara.models import wants_format
    from lara.serve import devices as DV
    from lara.serve import generator as GEN

    d = DV.detect()
    # Written by `lara setup`, which is the only thing that knows what the index costs.
    # Absent until the wizard has run, so both keys can be null and the UI must cope.
    cfg = require_state().cfg
    headroom = cfg.get_in("hardware.generator_headroom_gb")
    max_params = cfg.get_in("hardware.generator_max_params_4bit")
    return JSONResponse({
        "system": d.system, "machine": d.machine, "accelerator": d.accelerator,
        "unified_memory": d.unified_memory, "total_ram_gb": d.total_ram_gb,
        "usable_ram_gb": d.usable_ram_gb, "gpus": d.gpus,
        "total_vram_gb": d.total_vram_gb, "budget_gb": d.budget_gb,
        # d.backend is the advisory description; resolve_backend is what will really
        # run, which differs on a Mac with mlx-lm installed and wants a different format.
        "backend": GEN.resolve_backend(cfg, d.accelerator, cfg.get_path("huggingface.home")),
        "backend_advisory": d.backend,
        "backend_reason": d.backend_reason, "notes": d.notes,
        "generator_headroom_gb": headroom,
        "generator_max_params_4bit": max_params,
        # Which weight format this machine's runtime can actually load, so the download
        # dialog can say so before someone fetches 8 GB of the wrong one.
        "wants_format": wants_format(
            GEN.resolve_backend(cfg, d.accelerator, cfg.get_path("huggingface.home"))),
    })


class ResolveRequest(BaseModel):
    query: str


@router.post("/api/model/resolve")
def model_resolve(req: ResolveRequest) -> JSONResponse:
    """Look a repo up without downloading it, and say whether it fits."""
    import os

    from lara.serve import devices as DV
    from lara.serve import downloads as DL

    s = require_state()
    r = DL.resolve(req.query, token=os.environ.get("HF_TOKEN"))
    dev = DV.detect()
    body = {
        "repo": r.repo, "exists": r.exists, "gated": r.gated, "error": r.error,
        "params": r.params, "arch": r.arch, "quantization": r.quantization,
        "size_gb": r.size_gb, "n_safetensors": r.n_safetensors, "n_gguf": r.n_gguf,
        # The quantisation is treated as a property of the repo, not something to choose:
        # the server picks the 4-bit build and reports only that. `pick_files` still has
        # to travel, because the repo physically contains every other quantisation and
        # downloading without naming files fetches all of them.
        "pipeline": r.pipeline, "pick": r.pick,
        "pick_files": next((v["files"] for v in r.variants if v["quant"] == r.pick), None),
        "already_cached": _dl(s).cache_dir(r.repo).exists() if r.repo else False,
    }
    if r.exists and r.size_gb:
        body["fit"] = DV.fits(r.size_gb, dev)
    # Warn rather than block: an architecture an allowlist does not know may still be
    # servable, and refusing the download would be a worse failure than letting someone
    # try. Which warning applies depends entirely on the runtime that will load it.
    from lara.models import FORMAT_HELP, VLLM_ARCHS, wants_format
    fmt = wants_format(dev.backend)
    body["backend"] = dev.backend
    body["wants_format"] = fmt
    if fmt == "gguf":
        if not r.n_gguf:
            body["warning"] = (
                f"This repo has no GGUF weights, and {dev.backend} cannot load "
                f"safetensors. {FORMAT_HELP['gguf']}")
    elif fmt == "mlx":
        if not r.repo.startswith("mlx-community/"):
            body["warning"] = (f"This does not look like an MLX conversion. "
                               f"{FORMAT_HELP['mlx']}")
    else:
        if r.arch and r.arch not in VLLM_ARCHS:
            body["warning"] = (
                f"{r.arch} is not in the vLLM allowlist this build knows about. It may "
                "still work; the model picker will list it once downloaded.")
        if r.n_gguf and not r.n_safetensors:
            body["warning"] = ("GGUF-only repo, and this machine serves with vLLM, which "
                               "cannot read GGUF (D4). Use a safetensors build.")
    return JSONResponse(body)


def _dl(s):
    from lara.serve import downloads as DL
    if not hasattr(s, "_downloads"):
        s._downloads = DL.DownloadManager(s.cfg.get_path("huggingface.home"))
    return s._downloads


class DownloadRequest(BaseModel):
    repo: str
    size_gb: float = 0.0
    #: Exact file paths of one GGUF quantisation. Without it a GGUF repo pulls every
    #: quantisation it ships, which is 133 GB to obtain a 5 GB model.
    files: list[str] | None = None


@router.post("/api/model/download")
def model_download(req: DownloadRequest) -> JSONResponse:
    import os

    from lara.serve import downloads as DL

    s = require_state()
    repo = DL.normalise(req.repo)
    if not repo:
        raise HTTPException(400, "not a Hugging Face id")
    free_gb = __import__("shutil").disk_usage(s.cfg.get_path("huggingface.home")).free / 1e9
    if req.size_gb and req.size_gb * 1.1 > free_gb:
        raise HTTPException(
            507, f"needs ~{req.size_gb:.0f} GB, only {free_gb:.0f} GB free on the model disk"
        )
    job = _dl(s).start(repo, req.size_gb, token=os.environ.get("HF_TOKEN"),
                       files=req.files)
    return JSONResponse({"repo": job.repo, "status": job.status})


@router.get("/api/model/download/{repo:path}")
def model_download_status(repo: str) -> JSONResponse:
    from lara.serve import downloads as DL

    s = require_state()
    job = _dl(s).get(DL.normalise(repo) or repo)
    if job is None:
        raise HTTPException(404, "no such download")
    return JSONResponse({
        "repo": job.repo, "status": job.status, "pct": job.pct,
        "downloaded_gb": job.downloaded_gb, "total_gb": job.total_gb,
        "elapsed_s": round(time.time() - job.started),
        "error": job.error, "path": job.path,
    })


@router.get("/api/models")
def models() -> JSONResponse:
    """Generators available from the HF cache (R6, R7)."""
    s = require_state()
    from lara.models import survey
    from lara.serve import devices as DV
    from lara.serve import generator as GEN

    backend = GEN.resolve_backend(s.cfg, DV.detect().accelerator,
                                 s.cfg.get_path("huggingface.home"))
    found = survey(s.cfg.get_path("huggingface.home"), preferred=backend)

    # Which model vLLM is actually serving. The picker lists everything in the cache, but
    # only one is loaded (D5, single_resident), and sending any other name to vLLM is a
    # 404 at generation time. Reporting it lets the UI default to the live one instead of
    # whichever happened to sort first.
    import httpx

    loaded: list[str] = []
    try:
        base = s.cfg.get_in("serving.vllm.base_url").rstrip("/")
        r = httpx.get(f"{base}/models", timeout=2.0)
        if r.status_code == 200:
            loaded = [m["id"] for m in r.json().get("data", [])]
    except Exception:
        pass

    serving = s.cfg.get_in("serving") or {}
    return JSONResponse({
        "loaded": loaded,
        "backend": backend,
        # Per backend: the three formats are not interchangeable, so each keeps its own
        # model setting. Reading only the vLLM one reported "no default" on every Mac.
        "configured_default": GEN.model_for(
            backend, {**(serving.get("generator") or {}), "vllm": serving.get("vllm") or {}}),
        # Everything a backend could serve, including ones whose backend is not
        # installed: those carry `needs_install` and a hint naming what to do. Omitting
        # them is what made a freshly downloaded model vanish with no explanation.
        "models": [
            {**m, "loaded": m["repo"] in loaded}
            for m in found if m["backend"]
        ],
        "rejected": len([m for m in found if not m["backend"]]),
    })


@router.get("/api/breadth")
def breadth_options() -> JSONResponse:
    """The speed<->accuracy spectrum, so the UI can render it without hardcoding."""
    from lara.serve import agent as AG

    return JSONResponse({
        "default": AG.DEFAULT,
        "options": [
            {
                "name": b.name, "label": b.label, "estimate": b.estimate,
                "max_rounds": b.max_rounds, "k": b.k,
                "expand_context": b.expand_context, "allow_clarify": b.allow_clarify,
            }
            for b in AG.SPECTRUM
        ],
    })
