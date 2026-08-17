"""Scan the Hugging Face cache for models vLLM can actually serve (requirement R7).

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

# Architectures vLLM serves that we would plausibly use as a generator. Not exhaustive —
# vLLM's registry is the authority; this is a conservative allowlist for the picker.
VLLM_ARCHS = {
    "LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM",
    "GraniteForCausalLM", "GraniteMoeForCausalLM", "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM",
    "Phi3ForCausalLM", "PhiMoEForCausalLM", "Gemma2ForCausalLM", "Gemma3ForCausalLM",
    "OlmoeForCausalLM", "GptOssForCausalLM", "DeepseekV2ForCausalLM",
}

# Below this, a repo is a test stub or a partial download rather than a usable generator.
MIN_WEIGHT_BYTES = 1_000_000_000


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


def scan(hf_home: str | Path, min_bytes: int = MIN_WEIGHT_BYTES) -> list[CachedModel]:
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

        reasons: list[str] = []
        if arch is None:
            reasons.append("no config.json / no architectures entry")
        elif arch not in VLLM_ARCHS:
            reasons.append(f"architecture {arch} not in the vLLM allowlist")
        if not safet and gguf:
            reasons.append("GGUF-only (D4: vLLM is the sole backend, GGUF is excluded)")
        if weight_bytes < min_bytes:
            reasons.append(
                f"only {weight_bytes / 1e9:.2f} GB of resolved weights — "
                "metadata-only, partial download, or a tiny-random test stub"
            )

        found.append(CachedModel(
            repo=repo_dir.name[len("models--"):].replace("--", "/"),
            path=snap, arch=arch, weight_bytes=weight_bytes, quantization=quant,
            n_safetensors=len(safet), n_gguf=len(gguf),
            servable=not reasons, reasons=reasons,
        ))

    found.sort(key=lambda m: (-m.servable, -m.weight_bytes))
    return found


def servable(hf_home: str | Path) -> list[CachedModel]:
    return [m for m in scan(hf_home) if m.servable]
