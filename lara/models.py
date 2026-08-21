"""Scan the Hugging Face cache for models *this machine's backend* can serve (R7).

**Which runtime is asking matters.** vLLM reads safetensors, llama.cpp reads GGUF, and
MLX reads its own conversions; none of the three can load another's files. Judging the
cache against vLLM unconditionally meant that on Apple Silicon — where the detected
backend is llama.cpp, because vLLM has no Metal support — the picker hid every model
the machine could actually run and listed the ones it could not.


A naive listing of ``~/.cache/huggingface/hub`` is badly misleading. On this machine 390
``models--*`` repos are present but only 5 hold real weights — the rest are metadata-only
snapshots, partial downloads, GGUF-only repos, or ``tiny-random`` architecture stubs from
MoE research. A snapshot directory full of dangling symlinks looks identical to a complete
one unless you follow the links into ``blobs/`` and stat them, which is what this does.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from lara.serve.runtimes import BACKENDS

# Architectures vLLM serves that we would plausibly use as a generator. Not exhaustive —
# vLLM's registry is the authority; this is a conservative allowlist for the picker.
VLLM_ARCHS = {
    "LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM",
    "GraniteForCausalLM", "GraniteMoeForCausalLM", "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM",
    "Qwen3_5ForConditionalGeneration", "Qwen3_5MoeForConditionalGeneration",
    "Phi3ForCausalLM", "PhiMoEForCausalLM", "Gemma2ForCausalLM", "Gemma3ForCausalLM",
    "OlmoeForCausalLM", "GptOssForCausalLM", "DeepseekV2ForCausalLM",
}

# Below this, a repo is a test stub or a partial download rather than a usable generator.
MIN_WEIGHT_BYTES = 1_000_000_000

#: Backends that read GGUF and nothing else. On Apple Silicon this is the detected
#: default, because vLLM has no Metal backend (see lara/serve/devices.py).
GGUF_BACKENDS = {"llama.cpp", "llamacpp", "ollama"}

#: Where to find weights in each format, for error messages that end somewhere useful.
FORMAT_HELP = {
    "gguf": ("GGUF builds are at https://huggingface.co/models?library=gguf — "
             "`bartowski` and `unsloth` publish them for most popular models."),
    "mlx": ("MLX conversions are at https://huggingface.co/mlx-community."),
    "safetensors": ("Standard Hugging Face repos are safetensors; most models ship "
                    "this format by default."),
}


def wants_format(backend: str) -> str:
    """The weight format a backend can actually load.

    The three are not interchangeable and never have been: a repo vLLM serves is not a
    GGUF file and is not MLX-converted. Scanning the cache without knowing which runtime
    will read it produced a picker that, on a Mac, hid every model the machine could run
    and listed the ones it could not.
    """
    b = (backend or "").lower()
    if b in GGUF_BACKENDS:
        return "gguf"
    if b == "mlx":
        return "mlx"
    return "safetensors"


@dataclass
class CachedModel:
    repo: str
    path: Path
    arch: str | None
    weight_bytes: int
    quantization: str | None        # baked into the checkpoint; not runtime-selectable
    n_safetensors: int
    n_gguf: int
    servable: bool
    reasons: list[str] = field(default_factory=list)
    #: For a GGUF repo: one entry per quantisation present locally, smallest first.
    #: A publisher ships many, llama.cpp loads exactly one, so summing them describes
    #: nothing that will ever be in memory at once.
    quants: list[dict] = field(default_factory=list)
    #: The quantisation this entry is sized against, and what `model` should name.
    quant_pick: str | None = None

    @property
    def spec(self) -> str:
        """What to put in ``serving.generator.llamacpp.model``.

        A GGUF repo needs the quantisation as well as the name -- "Qwen/Qwen3-8B-GGUF"
        alone does not identify a file.
        """
        return f"{self.repo}:{self.quant_pick}" if self.quant_pick else self.repo

    @property
    def size_gb(self) -> float:
        return self.weight_bytes / 1e9

    def runtime_quant_options(self) -> list[str]:
        """What the UI may legitimately offer for this checkpoint.

        Quantization is mostly *not* a free-floating dial: AWQ/GPTQ/compressed-tensors are
        baked into the weights at conversion time. What vLLM can genuinely apply at load
        time is weight-only FP8 (well supported on Blackwell) and bitsandbytes. Offering a
        generic "pick any quantization" dropdown would be lying to the user.
        """
        if self.quantization:
            return [self.quantization]          # fixed by the checkpoint
        return ["none", "fp8", "bitsandbytes"]


def _snapshot_dir(repo_dir: Path) -> Path | None:
    snaps = sorted((repo_dir / "snapshots").glob("*")) if (repo_dir / "snapshots").is_dir() else []
    return snaps[-1] if snaps else None


def _reasons_for(fmt: str, repo: str, arch: str | None,
                 n_safet: int, n_gguf: int) -> list[str]:
    """Why this repo cannot be served *by the runtime that will actually load it*."""
    if fmt == "gguf":
        # llama.cpp has its own architecture support matrix and GGUF repos frequently
        # ship no config.json at all, so the vLLM allowlist says nothing useful here.
        if not n_gguf:
            return [f"no GGUF weights — this backend cannot load safetensors. "
                    f"{FORMAT_HELP['gguf']}"]
        return []
    if fmt == "mlx":
        # Repo naming is the only cheap signal: an MLX conversion is safetensors plus a
        # weight layout mlx-lm understands, indistinguishable from a vLLM checkpoint by
        # file listing alone. mlx-community is where essentially all of them live.
        if not repo.startswith("mlx-community/"):
            return [f"not an MLX conversion. {FORMAT_HELP['mlx']}"]
        return []
    reasons: list[str] = []
    if arch is None:
        reasons.append("no config.json / no architectures entry")
    elif arch not in VLLM_ARCHS:
        reasons.append(f"architecture {arch} not in the vLLM allowlist")
    if not n_safet and n_gguf:
        reasons.append("GGUF-only — vLLM cannot serve these (D4)")
    if repo.startswith("mlx-community/"):
        # These are safetensors with a stock architecture, so they pass every check
        # above while being MLX-quantised weights vLLM cannot read.
        reasons.append("MLX conversion — safetensors in a layout only mlx-lm reads")
    return reasons


def scan(hf_home: str | Path, min_bytes: int = MIN_WEIGHT_BYTES,
         backend: str | None = None) -> list[CachedModel]:
    """List cached models, judged against the backend that will serve them.

    ``backend`` defaults to whatever this machine detects — llama.cpp on Apple Silicon,
    vLLM on CUDA. Pass it explicitly to ask "what could I run *if* I switched runtime".
    """
    if backend is None:
        from lara.serve import devices as DV
        backend = DV.detect().backend
    fmt = wants_format(backend)

    hub = Path(os.path.expanduser(str(hf_home))) / "hub"
    found: list[CachedModel] = []
    if not hub.is_dir():
        return found

    for repo_dir in sorted(hub.iterdir()):
        if not repo_dir.name.startswith("models--"):
            continue
        snap = _snapshot_dir(repo_dir)
        if snap is None:
            continue

        files = list(snap.iterdir())
        safet = [f for f in files if f.name.endswith(".safetensors")]
        gguf = [f for f in files if f.name.endswith(".gguf")]
        bins = [f for f in files if f.name.endswith(".bin")]

        # Resolve symlinks into blobs/; dangling links contribute zero bytes, which is
        # exactly how partial downloads are detected.
        weight_bytes = 0
        for f in safet + bins + gguf:
            real = Path(os.path.realpath(f))
            if real.exists():
                weight_bytes += real.stat().st_size

        arch = quant = None
        cfg_file = snap / "config.json"
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text())
                arch = (cfg.get("architectures") or [None])[0]
                quant = (cfg.get("quantization_config") or {}).get("quant_method")
            except (json.JSONDecodeError, OSError):
                pass

        repo = repo_dir.name[len("models--"):].replace("--", "/")

        # A GGUF repo commonly holds Q4_K_M through Q8_0 of the same model. llama.cpp
        # loads one of them, so `weight_bytes` summed over all of them said 32 GB for an
        # 8B model and the wizard concluded it needed 43 GB of an 11 GB budget -- for a
        # file that is 4.7 GB. Size against the quantisation that would actually be used.
        quants: list[dict] = []
        quant_pick: str | None = None
        if gguf:
            from lara.serve.downloads import gguf_variants, pick_4bit

            local = [{"path": f.name,
                      "size": (Path(os.path.realpath(f)).stat().st_size
                               if Path(os.path.realpath(f)).exists() else 0)}
                     for f in gguf]
            quants = gguf_variants(local)
            quant_pick = pick_4bit(quants) or (quants[0]["quant"] if quants else None)
            chosen = next((q for q in quants if q["quant"] == quant_pick), None)
            if chosen:
                weight_bytes = int(chosen["size_gb"] * 1e9)

        reasons = _reasons_for(fmt, repo, arch, len(safet), len(gguf))
        if weight_bytes < min_bytes:
            reasons.append(
                f"only {weight_bytes / 1e9:.2f} GB of resolved weights — "
                "metadata-only, partial download, or a tiny-random test stub"
            )

        found.append(CachedModel(
            repo=repo, path=snap, arch=arch, weight_bytes=weight_bytes, quantization=quant,
            n_safetensors=len(safet), n_gguf=len(gguf),
            servable=not reasons, reasons=reasons,
            quants=quants, quant_pick=quant_pick,
        ))

    found.sort(key=lambda m: (-m.servable, -m.weight_bytes))
    return found


def servable(hf_home: str | Path, backend: str | None = None) -> list[CachedModel]:
    return [m for m in scan(hf_home, backend=backend) if m.servable]


#: Backends worth testing a cached model against. Ordered by how much work it is to make
#: one usable, so the first hit is the best suggestion.
SURVEY_BACKENDS = ("vllm", "mlx", "llamacpp")


def survey(hf_home: str | Path, preferred: str | None = None) -> list[dict]:
    """Every cached model, with the backends that could serve it.

    Filtering to one backend hid models the machine already had. A Mac with mlx-lm
    installed resolves to MLX, so a freshly downloaded GGUF vanished from the picker
    entirely — no entry, no explanation, and the download dialog still cheerfully
    reporting it as cached. Whether a runtime is *installed* is a separate question from
    whether the weights are the right format, and conflating them removed the one piece
    of information that would have explained the gap.

    Each entry names the backend that would serve it, whether that backend is installed,
    and why it is unusable if it is not.
    """
    by_repo: dict[str, dict] = {}
    order = [b for b in ((preferred,) if preferred else ()) if b] + \
            [b for b in SURVEY_BACKENDS if b != preferred]
    for backend in order:
        for m in scan(hf_home, backend=backend):
            entry = by_repo.setdefault(m.repo, {
                "repo": m.repo, "arch": m.arch, "size_gb": round(m.size_gb, 1),
                "quantization": m.quantization,
                "quant_options": m.runtime_quant_options(),
                "backends": [], "reasons": m.reasons,
            })
            if m.servable:
                entry["backends"].append(backend)

    out: list[dict] = []
    for entry in by_repo.values():
        backends = entry.pop("backends")
        installed = [b for b in backends if BACKENDS[b].available()]
        # Prefer a backend that is both able and installed; fall back to naming one that
        # would work if installed, which is actionable in a way that silence is not.
        chosen = (installed or backends or [None])[0]
        entry["backend"] = chosen
        entry["servable"] = bool(installed)
        entry["needs_install"] = bool(backends) and not installed
        if entry["needs_install"]:
            b = BACKENDS[backends[0]]
            entry["hint"] = f"needs {b.label} — {b.install_hint}"
        elif not backends:
            entry["hint"] = "; ".join(entry["reasons"]) or "no backend can serve this"
        else:
            entry["hint"] = ""
        # `reasons` came from whichever backend was scanned first and contradicts
        # `backend` whenever a later one succeeded. It has been folded into `hint`.
        entry.pop("reasons", None)
        out.append(entry)
    out.sort(key=lambda e: (not e["servable"], -e["size_gb"]))
    return out
