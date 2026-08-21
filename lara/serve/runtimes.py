"""What each generation runtime is, and the command line that starts it.

A leaf: it imports nothing from lara. That is the point. `lara.models` has to ask whether
a runtime is installed in order to say which cached checkpoints are servable, and
`lara.serve.generator` has to ask `lara.models` what is servable in order to pick a
runtime worth starting. Those two imported each other, and the cycle was held together
only by doing both imports inside function bodies -- which works until someone moves one
to the top of its file.

Choosing between these, launching one and waiting for it to answer is generator.py's job.
This file only describes them.

**The three weight formats are not interchangeable.** vLLM reads HF safetensors,
llama.cpp reads GGUF, MLX reads MLX conversions, so each runtime carries its own `model`
setting rather than sharing one. Pointing llama.cpp at a safetensors repo fails at load
time with an error about the file type, which is a confusing way to discover the rule.

**vLLM has no Metal backend** -- its platforms are cpu/cuda/rocm/tpu/xpu, so on a Mac it
runs CPU-only and leaves the GPU idle. That is why Apple Silicon gets two purpose-built
options rather than a degraded vLLM.

Flag surfaces drift between releases of all three. Everything here uses long-stable flags
and exposes `extra_args` for the rest.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _flash_attn_takes_value() -> bool:
    """Whether this llama-server wants ``-fa on`` rather than a bare ``-fa``.

    Read from ``--help`` rather than from a version number: the binary can come from
    Homebrew, a manual build or a container, and its own help text is the only account
    of its flags that is guaranteed to match the binary being launched. Defaults to the
    current syntax if help cannot be read at all.
    """
    exe = shutil.which("llama-server")
    if not exe:
        return True
    try:
        out = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return True
    text = (out.stdout or "") + (out.stderr or "")
    for line in text.splitlines():
        if "--flash-attn" in line:
            return "on|off|auto" in line.replace(" ", "")
    return True



REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Ports commonly used by local OpenAI-compatible servers, in probe order.
KNOWN_PORTS = ((8000, "vLLM"), (11434, "Ollama"), (1234, "LM Studio"), (8080, "llama.cpp"))


@dataclass
class Backend:
    name: str
    label: str
    model_format: str
    platforms: tuple[str, ...]          # accelerators it is worth running on
    install_hint: str
    notes: str = ""

    def available(self) -> bool:
        raise NotImplementedError

    def command(self, model: str, port: int, cfg: dict) -> list[str]:
        raise NotImplementedError

    def env(self, cfg: dict) -> dict:
        return dict(os.environ)


class VllmBackend(Backend):
    def __init__(self) -> None:
        super().__init__(
            "vllm", "vLLM", "HF safetensors", ("cuda", "rocm"),
            "pip install vllm",
            "Fastest option with a discrete GPU. No Metal backend, so not useful on a Mac.",
        )

    def _binary(self) -> str:
        # vLLM is often pinned to a different torch than the reader's environment, so an
        # isolated venv beside the repo wins if one exists. It is a separate process
        # reached over HTTP, so isolating it costs nothing.
        isolated = REPO_ROOT / ".venv-vllm" / "bin" / "vllm"
        return str(isolated) if isolated.exists() else "vllm"

    def available(self) -> bool:
        b = self._binary()
        return Path(b).exists() if os.sep in b else shutil.which(b) is not None

    def command(self, model: str, port: int, cfg: dict) -> list[str]:
        v = cfg.get("vllm") or {}
        cmd = [
            self._binary(), "serve", model,
            "--port", str(port),
            "--served-model-name", model,
            "--gpu-memory-utilization", str(v.get("gpu_memory_utilization", 0.5)),
            "--max-model-len", str(v.get("max_model_len", 32768)),
            "--kv-cache-dtype", str(v.get("kv_cache_dtype", "auto")),
            "--max-num-seqs", str(v.get("max_num_seqs", 64)),
        ]
        if v.get("enable_prefix_caching", True):
            cmd.append("--enable-prefix-caching")
        return cmd + [str(a) for a in (v.get("extra_args") or [])]

    def env(self, cfg: dict) -> dict:
        env = dict(os.environ)
        v = cfg.get("vllm") or {}
        if v.get("gpu_devices"):
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in v["gpu_devices"])
        # CUDA orders devices by compute capability by default, not by slot, so "device 1"
        # here and "device 1" in the reader can be different cards on a mixed machine.
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        return env


class LlamaCppBackend(Backend):
    def __init__(self) -> None:
        super().__init__(
            "llamacpp", "llama.cpp", "GGUF", ("mps", "cpu", "cuda"),
            "brew install llama.cpp   (or build from ggml-org/llama.cpp)",
            "Runs everywhere and is the most mature local server. Metal was its original "
            "target. Quantised KV cache and slot-based prefix reuse are supported.",
        )

    def available(self) -> bool:
        return shutil.which("llama-server") is not None

    def command(self, model: str, port: int, cfg: dict) -> list[str]:
        c = cfg.get("llamacpp") or {}
        # `-hf` resolves and caches a GGUF from the Hub; `-m` takes a local file. The
        # distinction cannot be "does it contain a slash": a Hub id like
        # `Qwen/Qwen3-8B-GGUF:Q4_K_M` contains one and is not a path. Decide on the
        # filename suffix and on whether the thing actually exists on disk.
        looks_local = model.endswith(".gguf") or Path(model).expanduser().exists()
        source = ["-m", str(Path(model).expanduser())] if looks_local else ["-hf", model]
        cmd = [
            "llama-server", *source,
            "--port", str(port),
            "--host", str(c.get("host", "127.0.0.1")),
            "-c", str(c.get("ctx_size", 32768)),
            # On Metal and CUDA this offloads every layer it can; it is ignored on a
            # CPU-only build, so one value is safe across platforms.
            "-ngl", str(c.get("n_gpu_layers", 999)),
            "--parallel", str(c.get("parallel", 2)),
        ]
        # `-fa` became a valued option: older builds took it as a bare boolean, current
        # ones require on|off|auto. Emitting the bare form against a current build makes
        # it swallow the NEXT argument as its value -- the observed failure was
        # `error: unknown value for --flash-attn: '--cache-reuse'`, which names neither
        # the real problem nor the flag that caused it.
        want_fa = c.get("flash_attn", True)
        if _flash_attn_takes_value():
            cmd += ["-fa", "on" if want_fa else "off"]
        elif want_fa:
            cmd.append("-fa")
        # Prefix reuse is what keeps follow-up questions about the same paper fast, which
        # is the whole reason the prompt is laid out stable-prefix-first.
        if c.get("cache_reuse", 256):
            cmd += ["--cache-reuse", str(c.get("cache_reuse", 256))]
        for key, flag in (("cache_type_k", "--cache-type-k"), ("cache_type_v", "--cache-type-v")):
            if c.get(key):
                cmd += [flag, str(c[key])]
        return cmd + [str(a) for a in (c.get("extra_args") or [])]


class MlxBackend(Backend):
    def __init__(self) -> None:
        super().__init__(
            "mlx", "MLX", "MLX (mlx-community/*)", ("mps",),
            "pip install mlx-lm",
            "Apple's own framework. Uses unified memory directly and is often faster than "
            "llama.cpp on Apple Silicon, but the server is younger: simpler prompt caching "
            "and no continuous batching. Fine for a single-user reader.",
        )

    def available(self) -> bool:
        if shutil.which("mlx_lm.server"):
            return True
        try:                                  # installed as a module but not on PATH
            import importlib.util
            return importlib.util.find_spec("mlx_lm") is not None
        except Exception:
            return False

    def command(self, model: str, port: int, cfg: dict) -> list[str]:
        c = cfg.get("mlx") or {}
        base = ([shutil.which("mlx_lm.server")] if shutil.which("mlx_lm.server")
                else ["python", "-m", "mlx_lm.server"])
        cmd = [*base, "--model", model, "--port", str(port),
               "--host", str(c.get("host", "127.0.0.1"))]
        if c.get("max_tokens"):
            cmd += ["--max-tokens", str(c["max_tokens"])]
        return cmd + [str(a) for a in (c.get("extra_args") or [])]


class OllamaBackend(Backend):
    """Ollama — llama.cpp underneath, with model management on top.

    Two things make it behave unlike the others. The server is **not** told which model to
    load: ``ollama serve`` starts a daemon and models are loaded on demand by whatever name
    a request asks for, so ``model`` here is a tag like ``qwen3:8b`` that must already have
    been pulled. And the port comes from ``OLLAMA_HOST`` rather than a flag, because
    ``serve`` takes no address argument.

    On macOS the desktop app usually has a daemon running already, in which case the probe
    adopts it and nothing is launched.
    """

    def __init__(self) -> None:
        super().__init__(
            "ollama", "Ollama", "Ollama tags (e.g. qwen3:8b)", ("mps", "cpu", "cuda"),
            "https://ollama.com/download  (or `brew install ollama`)",
            "Easiest to run and usually already installed on a Mac. llama.cpp underneath, "
            "so expect llama.cpp's speed with less to configure. Pull models with "
            "`ollama pull <tag>`; lara cannot pull them for you.",
        )

    def available(self) -> bool:
        return shutil.which("ollama") is not None

    def command(self, model: str, port: int, cfg: dict) -> list[str]:
        # No model argument: the daemon serves whatever has been pulled, chosen per request.
        return ["ollama", "serve"]

    def env(self, cfg: dict) -> dict:
        env = dict(os.environ)
        c = cfg.get("ollama") or {}
        host = c.get("host", "127.0.0.1")
        env["OLLAMA_HOST"] = f"{host}:{c.get('port', 11434)}"
        return env


class ExternalBackend(Backend):
    def __init__(self) -> None:
        super().__init__(
            "external", "external server", "whatever it serves", ("cuda", "mps", "cpu"),
            "start it yourself",
            "Nothing is launched or stopped; the reader just uses base_url. Use this for "
            "Ollama, LM Studio, or a generator on another machine.",
        )

    def available(self) -> bool:
        return True

    def command(self, model: str, port: int, cfg: dict) -> list[str]:
        return []


BACKENDS: dict[str, Backend] = {
    b.name: b for b in (VllmBackend(), LlamaCppBackend(), MlxBackend(),
                        OllamaBackend(), ExternalBackend())
}


def installed() -> list[Backend]:
    return [b for b in BACKENDS.values() if b.name != "external" and b.available()]
