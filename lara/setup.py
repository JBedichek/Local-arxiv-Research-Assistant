"""First-run planning — decide what this machine can actually run, then write it down.

The wizard in ``lara setup`` is a thin shell over this module. Everything here is pure
computation against measured constants, so the same recommendations can be produced
non-interactively, tested, and explained.

Two ideas do the work.

**The binding constraint is the index, not the model.** Tier 1 is loaded before anything
else and has no graceful degradation: if it does not fit, nothing runs. A wizard that only
checked whether the *generator* fits would happily configure a machine that then dies
building its search index.

**Nothing here is a default.** Topic scoping in particular is *recommended*, never applied:
plenty of Macs have 32, 64 or 128 GB and should use the whole corpus. The recommendation
fires on the measured budget, not on the platform — a 16 GB machine gets a strong nudge, a
64 GB machine is left alone.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from lara.serve import devices as DV

#: Corpus the published dataset ships with, used for planning before vectors exist locally.
REFERENCE_CHUNKS = 28_723_432

#: Steady-state cost of everything that is not the tier-1 index, in GB.
EMBEDDER_GB = 1.2
RERANKER_GB = 1.2

#: Below this much *total* RAM, a full-corpus index stops being realistic on any backend.
SMALL_MACHINE_GB = 16.0


@dataclass(frozen=True)
class IndexOption:
    """One selectable tier-1 configuration, with its measured cost.

    Latencies are p50 at k=200 over 2 M real vectors; recall is against exact fp16. See
    ``lara/index/backends.py`` and reproduce with ``lara bench-index``.
    """
    key: str
    backend: str
    precision: str
    faiss_kind: str | None
    bytes_per_vector: int          # at dim 256
    p50_ms: float
    recall: float
    needs_cuda: bool
    note: str
    #: Whether auto-selection may pick this. Dominated options stay listed and manually
    #: selectable, but are never chosen for you: faiss flat is slower *and* larger than
    #: torch fp16, and sq8 at 372 ms/query is worse than scoping the corpus and using a
    #: fast backend. "It fits" is not sufficient grounds to recommend something.
    recommendable: bool = True

    @property
    def label(self) -> str:
        tail = f" {self.faiss_kind}" if self.faiss_kind else f" {self.precision}"
        return f"{self.backend}{tail}"

    def index_gb(self, n_chunks: int, dim: int = 256) -> float:
        return n_chunks * self.bytes_per_vector * (dim / 256) / 1e9


# Ordered best-first for a machine that can afford them. `bytes_per_vector` is at dim 256:
# fp16 = 2*dim, int8 = 1*dim, faiss flat = 4*dim, hnsw = 4*dim + 8*M.
OPTIONS: tuple[IndexOption, ...] = (
    IndexOption("torch-fp16", "torch", "fp16", None, 512, 1.1, 1.000, False,
                "exact; fastest on CUDA and the best exact option on CPU too"),
    # needs_cuda is False despite the name: int8 needs *a GPU*, not specifically CUDA, and
    # backends.py only rejects it on CPU. Flagging it CUDA-only hid a working option from
    # every Apple Silicon machine, contradicting this row's own note.
    IndexOption("torch-int8", "torch", "int8", None, 256, 8.2, 0.996, False,
                "half the memory for 0.4% recall — CUDA/MPS only, 25x slower on CPU"),
    IndexOption("faiss-hnsw", "faiss", "fp16", "hnsw", 1280, 2.1, 0.979, False,
                "approximate; fastest at scale but the largest, and slow to build"),
    IndexOption("faiss-flat", "faiss", "fp16", "flat", 1024, 61.3, 0.996, False,
                "exact CPU search, but slower and larger than torch fp16", False),
    IndexOption("faiss-sq8", "faiss", "int8", "sq8", 256, 371.8, 0.991, False,
                "smallest faiss option; too slow to recommend", False),
)

OPTIONS_BY_KEY = {o.key: o for o in OPTIONS}


@dataclass
class Plan:
    device: DV.Device
    n_chunks: int
    dim: int
    option: IndexOption
    budget_gb: float
    overhead_gb: float
    scope: str                      # "unnecessary" | "recommended" | "required"
    scope_keep: float = 0.05
    reasons: list[str] = field(default_factory=list)
    alternatives: list[tuple[IndexOption, float, bool]] = field(default_factory=list)
    #: Changes to the non-index resident cost. Scoping cannot help when the models and
    #: cache alone exceed the budget, so those have to be named separately.
    overhead_advice: list[str] = field(default_factory=list)
    disable_cross_encoder: bool = False
    hot_tier_bytes: int = 2_000_000_000

    @property
    def index_gb(self) -> float:
        return self.option.index_gb(self.n_chunks, self.dim)

    @property
    def total_gb(self) -> float:
        return self.index_gb + self.overhead_gb

    @property
    def fits(self) -> bool:
        return self.total_gb <= self.budget_gb

    @property
    def scoped_chunks(self) -> int:
        return int(self.n_chunks * self.scope_keep)

    @property
    def scoped_index_gb(self) -> float:
        return self.option.index_gb(self.scoped_chunks, self.dim)


def overhead_gb(hot_tier_bytes: int = 2_000_000_000, cross_encoder: bool = True) -> float:
    """Everything resident that is not the tier-1 index."""
    return EMBEDDER_GB + (RERANKER_GB if cross_encoder else 0.0) + hot_tier_bytes / 1e9


def keep_fraction_for(budget_gb: float, overhead: float, option: IndexOption,
                      n_chunks: int, dim: int = 256, headroom: float = 0.8) -> float:
    """Largest keep fraction whose index still fits, rounded down to something legible."""
    room = max(0.0, budget_gb * headroom - overhead)
    full = option.index_gb(n_chunks, dim)
    if full <= 0:
        return 1.0
    frac = min(1.0, room / full)
    for step in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01):
        if frac >= step:
            return step
    return 0.01


def plan_index(device: DV.Device | None = None, n_chunks: int = 0, dim: int = 256,
               *, hot_tier_bytes: int = 2_000_000_000, cross_encoder: bool = True,
               prefer: str = "balanced") -> Plan:
    """Choose a tier-1 configuration for this machine.

    ``prefer`` is "balanced" (exact where affordable), "speed" (accept HNSW's approximation
    and memory for its latency) or "memory" (smallest that is still usable).
    """
    device = device or DV.detect()
    n_chunks = n_chunks or REFERENCE_CHUNKS
    # single_device_gb, not budget_gb: the index is one tensor on one card and is
    # never sharded, so the sum across GPUs is the wrong number to plan against.
    budget = device.single_device_gb
    over = overhead_gb(hot_tier_bytes, cross_encoder)
    cuda = device.accelerator == "cuda"
    reasons: list[str] = []

    usable = [o for o in OPTIONS if not (o.needs_cuda and not cuda)]
    # int8 on plain CPU is 282 ms/query — technically available, never advisable.
    if device.accelerator == "cpu":
        usable = [o for o in usable if o.key not in ("torch-int8", "faiss-sq8")]

    def fits(o: IndexOption) -> bool:
        return o.index_gb(n_chunks, dim) + over <= budget * 0.8

    if prefer == "speed":
        order = sorted(usable, key=lambda o: (o.p50_ms, o.index_gb(n_chunks, dim)))
    elif prefer == "memory":
        order = sorted(usable, key=lambda o: (o.index_gb(n_chunks, dim), o.p50_ms))
    else:
        order = sorted(usable, key=lambda o: (not fits(o), -o.recall, o.p50_ms))

    order = [o for o in order if o.recommendable]
    chosen = next((o for o in order if fits(o)), None)
    if chosen is None:
        # Nothing fits whole. Pick the best option on its merits and scope the corpus,
        # which is a far better trade than a 372 ms/query index that technically fits.
        chosen = order[0]
        reasons.append(
            f"No backend holds all {n_chunks / 1e6:.1f}M chunks in {budget:.0f} GB: the "
            f"smallest usable index is "
            f"{min(o.index_gb(n_chunks, dim) for o in usable):.1f} GB before "
            f"{over:.1f} GB of models and cache."
        )

    total = chosen.index_gb(n_chunks, dim) + over
    if total > budget * 0.8:
        scope = "required"
        reasons.append(
            f"{chosen.label} needs {total:.1f} GB against a {budget:.0f} GB budget.")
    elif device.total_ram_gb and device.total_ram_gb <= SMALL_MACHINE_GB:
        scope = "recommended"
        reasons.append(
            f"{device.total_ram_gb:.0f} GB of RAM is enough to load this, but it leaves "
            f"little for anything else — the display server and the generator draw from "
            f"the same pool on unified memory.")
    elif total > budget * 0.5:
        scope = "recommended"
        reasons.append(
            f"The index would use {total:.1f} GB of a {budget:.0f} GB budget, over half "
            f"of it.")
    else:
        scope = "unnecessary"
        reasons.append(
            f"{total:.1f} GB of a {budget:.0f} GB budget — the whole corpus fits "
            f"comfortably.")

    # A *required* scope should use whatever room exists; a *recommended* one has to leave
    # real headroom or it is not recommending anything. Computing both against the same
    # target produced "scoping recommended (keep 100%)", which is a contradiction.
    keep = keep_fraction_for(budget, over, chosen, n_chunks, dim,
                             headroom=0.8 if scope == "required" else 0.4)
    if keep >= 1.0 and scope == "recommended":
        scope = "unnecessary"
        reasons = [f"{total:.1f} GB of a {budget:.0f} GB budget — it fits, and scoping "
                   f"would not free enough to be worth the narrowed recall."]

    alts = [(o, o.index_gb(n_chunks, dim) + over, fits(o)) for o in usable]
    alts.sort(key=lambda t: t[1])

    # Scoping shrinks the index and nothing else. When the fixed costs alone crowd the
    # budget, no keep fraction rescues the machine and the honest advice is to cut them.
    advice: list[str] = []
    drop_ce = False
    hot = hot_tier_bytes
    if over > budget * 0.5:
        advice.append(
            f"Models and cache alone need {over:.1f} GB of a {budget:.0f} GB budget, which "
            f"no keep fraction can reduce.")
        if cross_encoder and budget < 8:
            drop_ce = True
            advice.append(
                "Disable the cross-encoder reranker (-1.2 GB). Ranking falls back to the "
                "dense+BM25 fusion, which is measurably worse but entirely usable.")
        if hot > 512_000_000:
            hot = 512_000_000
            advice.append(
                "Shrink the tier-0 hot cache from 2.0 GB to 0.5 GB. It only prefetches the "
                "open paper's neighbourhood, so this costs latency on citation follows, "
                "not correctness.")
        if drop_ce or hot != hot_tier_bytes:
            new_over = overhead_gb(hot, cross_encoder and not drop_ce)
            advice.append(f"Together those bring fixed costs to {new_over:.1f} GB.")
            over = new_over
            keep = keep_fraction_for(budget, over, chosen, n_chunks, dim,
                                     headroom=0.8 if scope == "required" else 0.4)

    return Plan(device=device, n_chunks=n_chunks, dim=dim, option=chosen,
                budget_gb=budget, overhead_gb=over, scope=scope, scope_keep=keep,
                reasons=reasons, alternatives=alts, overhead_advice=advice,
                disable_cross_encoder=drop_ce, hot_tier_bytes=hot)


# ── writing the config ────────────────────────────────────────────────────────────


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
            notes.append(f"# Scoping:  {plan.scope} — see `lara corpus scope`")
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
                  devices: list[int] | str | None = None) -> dict:
    """Build the override dict the wizard writes. Only non-default keys are included."""
    index: dict = {"backend": plan.option.backend, "precision": plan.option.precision}
    if plan.option.faiss_kind:
        index["faiss"] = {"kind": plan.option.faiss_kind}
    # Always emitted, not only when they differ from the default: "replace what you write,
    # preserve what you do not" is only safe if the wizard writes everything it decides.
    index["rerank"] = {"cross_encoder": {"enabled": not plan.disable_cross_encoder}}

    out: dict = {"index": index, "hot_tier": {"max_bytes": int(plan.hot_tier_bytes)}}
    if disk_root:
        out["disk"] = {"root": disk_root}
    if devices is not None:
        out["embedding"] = {"devices": devices}

    vllm: dict = {}
    if model:
        vllm["default_model"] = model
    if quantization:
        vllm["default_quantization"] = quantization
    if base_url:
        vllm["base_url"] = base_url
    if vllm:
        out["serving"] = {"vllm": vllm}
    return out
