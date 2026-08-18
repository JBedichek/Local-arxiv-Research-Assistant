"""One place that decides what "device 1" means on this machine.

Every accelerator reference in this codebase used to be an f-string: ``f"cuda:{n}"`` in
eight call sites, ``torch.cuda.empty_cache()`` in five more, ``autocast(device_type="cuda")``
in two. That is fine on the machine it was written for and fatal everywhere else — on a Mac
the first one raises before any useful error message, and on a single-GPU box a config
saying ``devices: [0, 1, 2]`` fails on the second card rather than degrading to the first.

The config keeps expressing devices the way a CUDA user thinks about them — bare integers
(``cross_encoder.device: 1``, ``embedding.devices: [0, 1, 2]``) — because that is the right
mental model when you have several cards. This module translates that intent onto whatever
hardware is actually present:

    resolve(1)          cuda:1 with 3 cards | cuda:0 with 1 card | mps on a Mac | cpu
    resolve("auto")     best available
    resolve("cpu")      cpu, always — an explicit string is an instruction, not a hint

The asymmetry is deliberate. An **integer** is a preference and gets clamped to what exists;
an explicit **string** like ``"cuda:1"`` or ``"cpu"`` is honoured or, if impossible, falls
back loudly via :func:`warnings`. Silently running on CPU because a typo asked for a GPU
that is not there is a 50x slowdown that looks like a hang.
"""

from __future__ import annotations

import functools
import warnings

import torch

# Preference order when nothing specific is asked for. ROCm reports itself as `cuda`
# through torch's HIP shim, so it needs no separate branch here.
_ORDER = ("cuda", "mps", "cpu")


@functools.lru_cache(maxsize=1)
def _probe() -> tuple[str, int]:
    """(best available backend, number of visible devices). Cached — it cannot change
    within a process, and ``torch.cuda.is_available()`` is not free on first call."""
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda", torch.cuda.device_count()
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps", 1
    return "cpu", 1


def backend() -> str:
    """``cuda`` | ``mps`` | ``cpu`` — what this machine can actually use."""
    return _probe()[0]


def count() -> int:
    """Number of usable accelerators. Always 1 for mps and cpu."""
    return _probe()[1]


def available() -> list[str]:
    """Every usable device, as torch device strings."""
    kind, n = _probe()
    return [f"cuda:{i}" for i in range(n)] if kind == "cuda" else [kind]


def resolve(spec: str | int | None = None) -> str:
    """Map a config device spec onto a real torch device string.

    ``None`` and ``"auto"`` pick the best available. Integers are treated as a *preference*
    for a CUDA ordinal and clamped into range. Explicit strings are honoured where possible
    and warned about where not.
    """
    kind, n = _probe()

    if spec is None or spec == "auto" or spec == "":
        return f"{kind}:0" if kind == "cuda" else kind

    # Bare integer, or a string that is just digits: a CUDA ordinal preference.
    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        idx = int(spec)
        if kind != "cuda":
            return kind                      # no ordinals to speak of on mps/cpu
        if idx >= n:
            warnings.warn(
                f"device {idx} requested but only {n} CUDA device(s) visible; using cuda:0",
                RuntimeWarning, stacklevel=2,
            )
            return "cuda:0"
        return f"cuda:{idx}"

    spec = str(spec)
    want = spec.split(":", 1)[0]
    if want == "cpu":
        return "cpu"                          # explicit, always honoured
    if want == kind:
        if want != "cuda":
            return kind
        idx = int(spec.split(":", 1)[1]) if ":" in spec else 0
        if idx >= n:
            warnings.warn(
                f"{spec} requested but only {n} CUDA device(s) visible; using cuda:0",
                RuntimeWarning, stacklevel=2,
            )
            return "cuda:0"
        return f"cuda:{idx}"

    warnings.warn(
        f"{spec} requested but this machine has {kind}; using {kind}. "
        f"Generation and embedding will be much slower if this was not intended.",
        RuntimeWarning, stacklevel=2,
    )
    return f"{kind}:0" if kind == "cuda" else kind


def resolve_all(specs: "list[str | int] | str | int | None") -> list[str]:
    """Resolve a device list, de-duplicated and order-preserving.

    Accepts what config files actually contain: a list (``[0, 1, 2]``), the string
    ``"auto"``, a bare scalar, or nothing. A scalar is *not* iterated — ``"auto"[0]`` is
    ``"a"``, and silently resolving that would land on a fallback device with a confusing
    warning instead of using every card.

    ``[0, 1, 2]`` on a one-GPU box collapses to ``["cuda:0"]`` rather than resolving to the
    same card three times, which would start three worker processes contending for one
    device.
    """
    if specs is None or specs == "auto" or specs == "" or specs == []:
        return available()
    if not isinstance(specs, (list, tuple)):
        specs = [specs]
    out: list[str] = []
    for s in specs:
        d = resolve(s)
        if d not in out:
            out.append(d)
    return out


def first(specs: "list[str | int] | str | int | None") -> str:
    """The single device to use for a config field that may name several."""
    return resolve_all(specs)[0]


def empty_cache(device: str | None = None) -> None:
    """Release cached allocations on whichever backend is in use.

    ``torch.cuda.empty_cache()`` raises on a machine without CUDA, and the calls that
    matter here are all "we just freed something large, give it back" — advisory, never
    load-bearing. So failure is swallowed rather than propagated.
    """
    kind = (device or backend()).split(":", 1)[0]
    try:
        if kind == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif kind == "mps" and getattr(torch, "mps", None) is not None:
            torch.mps.empty_cache()
    except Exception:
        pass


def autocast_dtype(device: str | None = None) -> torch.dtype | None:
    """Mixed-precision dtype for a device, or ``None`` where autocast is not worth it.

    bf16 on CUDA. **fp16 on MPS**, not bf16: Metal's bf16 support lands unevenly across
    torch versions and silently falls back to fp32 in places, so fp16 is the honest choice.
    ``None`` on CPU — autocast there is supported but routinely slower than plain fp32 for
    models this size, and correctness beats a speedup that is not real.
    """
    kind = (device or backend()).split(":", 1)[0]
    if kind == "cuda":
        return torch.bfloat16
    if kind == "mps":
        return torch.float16
    return None


def autocast(device: str | None = None):
    """Context manager for mixed precision, a no-op where it does not apply."""
    import contextlib

    kind = (device or backend()).split(":", 1)[0]
    dtype = autocast_dtype(kind)
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=kind, dtype=dtype)


def model_dtype(device: str | None = None) -> torch.dtype:
    """Weight dtype for loading an encoder. fp32 on CPU, where half precision is emulated
    and therefore slower rather than faster."""
    return autocast_dtype(device) or torch.float32


def can_fan_out(devices: list[str]) -> bool:
    """Whether a multi-process encoder pool is worth starting.

    Only ever true for multiple CUDA cards. On unified memory there is one GPU sharing one
    pool of RAM, so N workers would multiply the memory footprint while contending for the
    same silicon — strictly worse than a single process.
    """
    return len(devices) > 1 and all(d.startswith("cuda") for d in devices)


def describe() -> str:
    """One line for logs and preflight output."""
    kind, n = _probe()
    if kind == "cuda":
        names = {torch.cuda.get_device_name(i) for i in range(n)}
        return f"{n} x CUDA ({', '.join(sorted(names))})"
    return {"mps": "Apple Silicon GPU (MPS, unified memory)"}.get(kind, "CPU only")
