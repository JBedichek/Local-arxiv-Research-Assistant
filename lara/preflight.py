"""Startup checks — refuse to run if data would land on the wrong disk.

This machine has three filesystems and only ``/`` is usable::

    /                        nvme0n1p5   949G, 245G free   <-- OK
    /data                    nvme0n1p3   883G,   0G free   <-- FULL
    /media/user/Extreme SSD  sda1        3.7T,  31G free   <-- nearly full, external

A symlink anywhere in a configured path can quietly move a 30 GB index onto a disk
with no room. Every check here resolves through ``os.path.realpath`` first, so it sees
where bytes actually land rather than what the config string says.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lara.config import Config, load


@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    @property
    def symbol(self) -> str:
        return "PASS" if self.ok else "FAIL"


def _device_for(path: Path) -> str:
    """Backing device for the nearest existing ancestor of ``path``."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        out = subprocess.run(
            ["df", "--output=source", str(probe)],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.splitlines()
        return out[1].strip() if len(out) > 1 else "unknown"
    except Exception:
        return "unknown"


def _free_gb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9


def check_config(cfg: Config) -> list[Check]:
    """Report which config layers are in play (D16).

    Never fails. Running on defaults alone is a legitimate state — it is what a fresh
    clone looks like — but it is worth saying out loud, because the defaults deliberately
    pin nothing to this machine and someone debugging a path or a device wants to know
    whether their overrides were picked up at all.
    """
    names = ", ".join(p.name for p in getattr(cfg, "sources", []) or ["config.yaml"])
    if getattr(cfg, "has_local", False):
        return [Check("config", True, f"{names}")]
    return [Check("config", True,
                  f"{names} only — no config.local.yaml, using portable defaults. "
                  f"Run `lara setup` to pin this machine's disks, devices and model.")]


def check_disks(cfg: Config) -> list[Check]:
    checks: list[Check] = []
    required = cfg.get_in("disk.required_device")
    min_free = float(cfg.get_in("disk.min_free_gb", 0))
    forbidden = [Path(os.path.realpath(p)) for p in (cfg.get_in("disk.forbid_paths") or [])]

    for address, resolved in cfg.all_paths().items():
        raw = str(cfg.get_in(address))
        note = "" if os.path.realpath(raw) == str(Path(raw).expanduser()) else " (via symlink)"

        under_forbidden = next(
            (f for f in forbidden if resolved == f or f in resolved.parents), None
        )
        if under_forbidden:
            checks.append(Check(
                f"path {address}", False,
                f"resolves to {resolved}{note}, which is under forbidden {under_forbidden}",
            ))
            continue

        device = _device_for(resolved)
        if required and device != required and device != "unknown":
            checks.append(Check(
                f"path {address}", False,
                f"resolves to {resolved}{note} on {device}, expected {required}",
            ))
            continue

        checks.append(Check(f"path {address}", True, f"{resolved} on {device}{note}"))

    root = cfg.get_path("disk.root")
    free = _free_gb(root)
    checks.append(Check(
        "free space", free >= min_free,
        f"{free:.0f} GB free at {root}, minimum {min_free:.0f} GB",
    ))

    if os.environ.get("HF_HOME"):
        env_hf = Path(os.path.realpath(os.environ["HF_HOME"]))
        cfg_hf = cfg.get_path("huggingface.home")
        checks.append(Check(
            "HF_HOME override", env_hf == cfg_hf,
            f"environment HF_HOME={env_hf} vs config {cfg_hf}",
        ))
    return checks


def check_gpu() -> list[Check]:
    """Report the accelerator, and fail only on a *broken* one.

    Having no GPU is not a failure — retrieval runs on CPU (~7 ms at 96 % recall) and
    generation can point at any OpenAI-compatible endpoint, so a CPU-only machine is a
    supported configuration, not a misconfigured one. What *is* a failure is a CUDA stack
    that is present but non-functional: an NVML/kernel-module mismatch leaves
    ``nvidia-smi`` failing while the config still asks for ``cuda:0``, and every job dies
    at model-load time with a far less obvious error.
    """
    from lara import device as dev

    backend = dev.backend()
    if backend == "mps":
        return [Check("gpu", True, "Apple Silicon GPU via MPS (unified memory). "
                                   "vLLM has no Metal backend — see serving notes.")]
    if backend == "cpu" and not shutil.which("nvidia-smi"):
        return [Check("gpu", True, "no accelerator — CPU only. Retrieval works; "
                                   "embedding is ~50x slower and generation needs an "
                                   "external endpoint.")]

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except FileNotFoundError:
        return [Check("gpu", False, "nvidia-smi not found")]

    if r.returncode != 0:
        err = (r.stderr or r.stdout).strip().splitlines()
        hint = ""
        if any("mismatch" in line.lower() for line in err):
            hint = (
                " — driver/library version mismatch. Fix with: sudo rmmod nvidia_uvm "
                "nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia   (or reboot)"
            )
        return [Check("gpu", False, (err[0] if err else "nvidia-smi failed") + hint)]

    gpus = [line.strip() for line in r.stdout.splitlines() if line.strip()]
    if gpus and backend != "cuda":
        # The driver is healthy but torch cannot reach the cards — a CPU-only torch build,
        # or CUDA_VISIBLE_DEVICES emptied in the environment. Everything would silently
        # run on CPU at ~50x the cost, which reads as a hang rather than an error.
        return [Check(
            "gpu", False,
            f"nvidia-smi sees {len(gpus)} GPU(s) but torch reports '{backend}'. "
            f"Check CUDA_VISIBLE_DEVICES and that torch was built with CUDA "
            f"(torch.cuda.is_available() is False).",
        )]
    return [Check("gpu", bool(gpus), f"{len(gpus)} visible: " + "; ".join(gpus))]


def run(cfg: Config | None = None) -> tuple[list[Check], bool]:
    cfg = cfg or load()
    checks = check_config(cfg) + check_disks(cfg) + check_gpu()
    return checks, all(c.ok for c in checks)
