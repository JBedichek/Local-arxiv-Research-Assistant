"""Detect the machine and pick a generation backend.

**vLLM is not portable to Apple Silicon.** Its platform list is cpu, cuda, rocm, tpu, xpu
and zen_cpu — there is no Metal or MPS backend, so on a Mac vLLM runs CPU-only and leaves
the GPU entirely idle. The right local runtime there is llama.cpp (or Ollama / LM Studio /
MLX), all of which are Metal-accelerated.

That costs us nothing, because the reader talks to the generator over an OpenAI-compatible
HTTP endpoint and never imports vLLM. Swapping the runtime is a URL change, not a port.

Memory is reported as *usable*, not installed. On unified-memory Macs the GPU shares system
RAM with everything else, so quoting the installed figure would tell someone with 16 GB
that a 14 GB model fits.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class Device:
    system: str                     # Linux | Darwin | Windows
    machine: str                    # x86_64 | arm64
    accelerator: str                # cuda | rocm | mps | cpu
    unified_memory: bool
    total_ram_gb: float
    usable_ram_gb: float            # what a model may realistically claim
    gpus: list[dict] = field(default_factory=list)
    total_vram_gb: float = 0.0
    backend: str = "vllm"           # vllm | llama.cpp | ollama
    backend_reason: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def budget_gb(self) -> float:
        """Memory a generator may use: VRAM on discrete GPUs, shared RAM if unified.

        This is the *aggregate*, which is right for a generator — vLLM can shard one model
        across cards with tensor parallelism. It is wrong for anything that lives on a
        single device; use :attr:`single_device_gb` for that.
        """
        return self.total_vram_gb if (self.total_vram_gb and not self.unified_memory) \
            else self.usable_ram_gb

    @property
    def single_device_gb(self) -> float:
        """Memory available to one device — the budget for the tier-1 index.

        The index is a single tensor on a single card; it is not sharded. Planning it
        against the sum of three GPUs would report 287 GB of headroom on a machine where
        no individual card has more than 96 GB.
        """
        if self.gpus and not self.unified_memory:
            return max(g["vram_gb"] for g in self.gpus)
        return self.usable_ram_gb


def _total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        pass
    try:  # macOS
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        return int(out) / 1e9
    except Exception:
        return 0.0


def _cuda_gpus() -> list[dict]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            name, mem = [x.strip() for x in line.split(",")]
            gpus.append({"name": name, "vram_gb": round(float(mem) / 1024, 1)})
        return gpus
    except Exception:
        return []


def detect() -> Device:
    system = platform.system()
    machine = platform.machine()
    ram = _total_ram_gb()
    gpus = _cuda_gpus()
    notes: list[str] = []

    apple_silicon = system == "Darwin" and machine in ("arm64", "aarch64")
    if gpus:
        accel, unified = "cuda", False
    elif apple_silicon:
        accel, unified = "mps", True
    else:
        accel, unified = "cpu", system == "Darwin"

    # Leave headroom for the OS, the reader process, the embedder and the reranker. On
    # unified memory that headroom is not optional: the display server draws from the same
    # pool, and overcommitting it swaps or wedges the machine rather than OOMing cleanly.
    reserve = 6.0 if unified else 4.0
    usable = max(0.0, ram - reserve)

    if accel == "cuda":
        backend = "vllm"
        reason = f"{len(gpus)} CUDA GPU(s) detected"
    elif accel == "mps":
        backend = "llama.cpp"
        reason = ("vLLM has no Metal backend — its platforms are cpu/cuda/rocm/tpu/xpu, "
                  "so on Apple Silicon it would run CPU-only and ignore the GPU")
        notes.append(
            "Install a Metal runtime that exposes an OpenAI-compatible server: "
            "`brew install llama.cpp` then `llama-server -hf <repo> --port 8000`, "
            "or Ollama, or LM Studio. Point serving.vllm.base_url at it."
        )
        notes.append("Prefer GGUF builds on Apple Silicon; llama.cpp loads them directly.")
    else:
        backend = "llama.cpp"
        reason = "no supported accelerator; CPU inference will be slow"
        notes.append("Generation on CPU is workable for small models but slow. "
                     "Retrieval alone still runs well without a GPU.")

    if unified:
        notes.append(
            f"Unified memory: the GPU shares system RAM, so the generator budget is "
            f"~{usable:.0f} GB of {ram:.0f} GB, not the full figure."
        )

    return Device(
        system=system, machine=machine, accelerator=accel, unified_memory=unified,
        total_ram_gb=round(ram, 1), usable_ram_gb=round(usable, 1),
        gpus=gpus, total_vram_gb=round(sum(g["vram_gb"] for g in gpus), 1),
        backend=backend, backend_reason=reason, notes=notes,
    )


def fits(size_gb: float, dev: Device, kv_overhead: float = 1.35) -> dict:
    """Will a checkpoint of this size run here?

    ``kv_overhead`` covers the KV cache and activations, which are not optional and are
    routinely forgotten — a 20 GB checkpoint does not run in 20 GB.
    """
    need = size_gb * kv_overhead
    budget = dev.budget_gb
    return {
        "needed_gb": round(need, 1),
        "budget_gb": round(budget, 1),
        "fits": need <= budget,
        "margin_gb": round(budget - need, 1),
        "where": "VRAM" if (dev.total_vram_gb and not dev.unified_memory) else "unified RAM",
    }
