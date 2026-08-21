"""Writing config.local.yaml: what goes in it, and how it is explained.

The other half of `lara setup`. lara/setup.py works out what this machine can run;
this turns that plan into a file, with a header saying where the numbers came from and
a backup of whatever was there before.

**Comments do not survive a write.** yaml.safe_dump cannot preserve them, so any
explanation a user added to their own config.local.yaml is lost when the wizard runs
again. Round-tripping would mean a new dependency (ruamel.yaml); a timestamped backup
plus saying so out loud is the cheap honest alternative. carry_forward() exists for the
same reason from the other direction: settings the wizard does not manage are copied
across rather than silently dropped, and the caller is told which ones.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import yaml

from lara.serve import devices as DV
from lara.setup import Plan


HEADER = """\
# Machine-specific configuration — NOT tracked in git.
#
# Written by `lara setup` on {when}. This file deep-merges over config.yaml,
# which holds portable defaults; anything true of *this computer* rather than of the
# project belongs here. Safe to delete — deleting it returns you to the defaults.
#
# Detected: {machine}
{notes}
"""


def render(overrides: dict, device: DV.Device, plan: Plan | None = None) -> str:
    """Serialise overrides with a header explaining where they came from."""
    notes = []
    if plan is not None:
        notes.append(f"# Index:    {plan.option.label} — {plan.index_gb:.1f} GB for "
                     f"{plan.n_chunks / 1e6:.1f}M chunks, p50 ~{plan.option.p50_ms:.1f}ms, "
                     f"recall {plan.option.recall:.3f}")
        notes.append(f"# Budget:   {plan.budget_gb:.0f} GB usable, "
                     f"{plan.overhead_gb:.1f} GB for models and cache")
        if plan.scope != "unnecessary":
            notes.append(f"# Scoping:  keeping {plan.effective_keep:.0%} resident — "
                         f"built automatically from corpus.scope on first start")
    head = HEADER.format(
        when=time.strftime("%Y-%m-%d %H:%M:%S"),
        machine=f"{device.system}/{device.machine}, {device.accelerator}, "
                f"{device.total_ram_gb:.0f} GB RAM"
                + (f", {device.total_vram_gb:.0f} GB VRAM" if device.total_vram_gb else ""),
        notes="\n".join(notes),
    )
    return head + "\n" + yaml.safe_dump(overrides, sort_keys=False, default_flow_style=False)


def carry_forward(existing: dict, overrides: dict) -> list[str]:
    """Report which existing settings the wizard is NOT overwriting.

    The merge itself keeps everything: overrides win where they exist, and every other key
    in the current file survives untouched. Two earlier designs were worse. Rewriting the
    file wholesale silently dropped ``disk.forbid_paths`` and ``min_free_gb`` — hand-written
    safety pins whose entire job is to stop 30 GB landing on a full disk. Stripping a fixed
    list of "managed" keys instead cleared ``default_quantization`` whenever the wizard
    adopted an already-running server and therefore never chose one.

    The rule that survives both: **replace what you write, preserve what you do not.** For
    that to be safe the wizard must write every key it could meaningfully change, which is
    why :func:`overrides_for` always emits the cross-encoder flag and the hot-tier size
    rather than only emitting them when they differ from the default.
    """
    out: list[str] = []

    def walk(node: dict, over: dict, prefix: str = "") -> None:
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else k
            ov = over.get(k) if isinstance(over, dict) else None
            if isinstance(v, dict):
                walk(v, ov if isinstance(ov, dict) else {}, key)
            elif not (isinstance(over, dict) and k in over):
                out.append(key)

    walk(existing or {}, overrides or {})
    return sorted(out)


def write_local(path: Path, overrides: dict, device: DV.Device,
                plan: Plan | None = None, existing: dict | None = None
                ) -> tuple[Path, Path | None, list[str]]:
    """Write ``config.local.yaml``, backing up any existing one.

    Returns the path, the backup path, and the dotted keys carried over untouched.
    """
    from lara.config import deep_merge

    backup = None
    if path.exists():
        backup = path.with_suffix(f".yaml.bak.{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)

    existing = existing or {}
    kept = carry_forward(existing, overrides)
    path.write_text(render(deep_merge(existing, overrides), device, plan))
    return path, backup, kept


def overrides_for(plan: Plan, *, model: str | None = None,
                  quantization: str | None = None,
                  base_url: str | None = None,
                  disk_root: str | None = None,
                  devices: list[int] | str | None = None,
                  topics: list[str] | None = None,
                  backend: str = "vllm",
                  ctx_size: int | None = None,
                  kv_quant: bool = False,
                  slots: int | None = None) -> dict:
    """Build the override dict the wizard writes. Only non-default keys are included."""
    index: dict = {"backend": plan.option.backend, "precision": plan.option.precision}
    if plan.option.faiss_kind:
        index["faiss"] = {"kind": plan.option.faiss_kind}
    # Always emitted, not only when they differ from the default: "replace what you write,
    # preserve what you do not" is only safe if the wizard writes everything it decides.
    index["rerank"] = {"cross_encoder": {"enabled": not plan.disable_cross_encoder}}

    # Saved so the reader can tell you what will fit without re-deriving it. The UI has
    # no idea what the index costs, and asking a user to remember "about 8B at 4-bit"
    # from a wizard they ran last month is not a plan.
    out: dict = {
        "index": index,
        "hot_tier": {"max_bytes": int(plan.hot_tier_bytes)},
        "hardware": {
            "generator_headroom_gb": round(plan.generator_headroom_gb, 1),
            "generator_max_params_4bit": int(plan.generator_params_4bit),
        },
    }
    # Scoping is a load-time decision (see lara/index/scope.py), so it belongs in the
    # config rather than in a command the user is told to go and run. The server builds
    # the keep-set from these on first start and caches it.
    if topics and plan.effective_keep < 1.0:
        out["corpus"] = {"scope": {
            "topics": list(topics),
            "keep": round(plan.effective_keep, 4),
            "expand_min_citations": 3,
        }}
    if disk_root:
        out["disk"] = {"root": disk_root}
    if devices is not None:
        out["embedding"] = {"devices": devices}

    # The chosen model has to be written under the key the *serving backend* reads.
    # generator.model_for looks at serving.vllm.default_model for vLLM and
    # serving.<backend>.model for everything else, so writing it under vllm on a Mac
    # left model_for returning None, from_config returning None, and no generator ever
    # starting -- while the picker cheerfully listed the model as present but not loaded.
    serving: dict = {}
    generator: dict = {}
    vllm: dict = {}
    if base_url:
        vllm["base_url"] = base_url             # the reader talks here whatever serves
    if backend:
        generator["backend"] = backend
    if model:
        if backend == "vllm":
            vllm["default_model"] = model
            if quantization:
                vllm["default_quantization"] = quantization
        else:
            # Matches the schema generator.model_for reads: serving.generator.<name>.model,
            # a sibling of `backend` rather than of `generator`.
            generator[backend] = {"model": model}
            # The KV cache is sized from this, and at 32k it outweighs the weights. It is
            # a per-backend setting, so it rides along with the model it applies to.
            if backend in ("llamacpp", "ollama"):
                if ctx_size:
                    generator[backend]["ctx_size"] = int(ctx_size)
                if slots:
                    generator[backend]["parallel"] = int(slots)
                # Written explicitly either way: null means fp16, and leaving the key
                # absent would let a previously-quantised cache persist unnoticed.
                generator[backend]["cache_type_k"] = "q8_0" if kv_quant else None
                generator[backend]["cache_type_v"] = "q8_0" if kv_quant else None
    if vllm:
        serving["vllm"] = vllm
    if generator:
        serving["generator"] = generator
    if serving:
        out["serving"] = serving
    return out
