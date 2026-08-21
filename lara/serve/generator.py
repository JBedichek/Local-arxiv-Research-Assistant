"""Generation backends and the process that runs them.

The reader speaks only OpenAI-compatible HTTP and never imports an inference library, so
swapping runtimes is a URL change rather than a port. That is what makes three backends
cheap: the difference between them is a command line and a health check, not a code path
through retrieval.

===========  ==================  ==============================  ===========================
backend      platform            model format                    why you would pick it
===========  ==================  ==============================  ===========================
``vllm``     CUDA / ROCm         HF safetensors                  fastest with a real GPU
``llamacpp`` anywhere, Metal     GGUF                            most mature; runs anywhere
``mlx``      Apple Silicon only  MLX (``mlx-community/*``)       unified memory, often
                                                                 faster than llama.cpp
``external`` anything            whatever it already serves      you started it yourself
===========  ==================  ==============================  ===========================

**The formats are not interchangeable**, which is the one thing that makes this more than a
flag. A repo that vLLM serves happily is not a GGUF file and is not MLX-converted, so each
backend carries its own ``model`` setting. Pointing llama.cpp at a safetensors repo fails at
load time with an error about the file type, and that is a confusing way to discover the
rule.

**vLLM has no Metal backend** — its platforms are cpu/cuda/rocm/tpu/xpu, so on a Mac it runs
CPU-only and leaves the GPU idle. That is why Apple Silicon gets two purpose-built options
rather than a degraded vLLM.

Flag surfaces drift between releases of all three. Everything here uses long-stable flags
and exposes ``extra_args`` for the rest, and readiness is established by *probing the
endpoint* rather than by trusting that the process started correctly.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

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
        if c.get("flash_attn", True):
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
    b.name: b for b in (VllmBackend(), LlamaCppBackend(), MlxBackend(), ExternalBackend())
}


def installed() -> list[Backend]:
    return [b for b in BACKENDS.values() if b.name != "external" and b.available()]


def choose(accelerator: str, requested: str | None = None) -> Backend:
    """Resolve ``auto`` against the platform and what is actually installed.

    Order is by platform fit first and installation second, so the answer explains itself:
    on Apple Silicon MLX and llama.cpp are both right and vLLM never is, regardless of
    whether vLLM happens to be installed.
    """
    requested = (requested or "auto").lower()
    if requested != "auto":
        return BACKENDS.get(requested, BACKENDS["external"])

    if accelerator == "mps":
        order = ("mlx", "llamacpp")
    elif accelerator in ("cuda", "rocm"):
        order = ("vllm", "llamacpp")
    else:
        order = ("llamacpp",)

    for name in order:
        if BACKENDS[name].available():
            return BACKENDS[name]
    # Nothing installed. Return the platform's best choice anyway, rather than silently
    # falling back to `external` and doing nothing: the caller checks `available()` and can
    # then say "brew install llama.cpp", which is actionable. `external` is what you get by
    # asking for it, or on a platform with no options at all.
    return BACKENDS[order[0]] if order else BACKENDS["external"]


def resolve_backend(cfg, accelerator: str, hf_home=None) -> str:
    """The backend to use, taking into account which models are actually present.

    :func:`choose` ranks by platform fit and installation, which is right until both
    candidates are installed and only one of them can read anything you own. On a Mac
    with mlx-lm present it returns MLX; a user whose only cached generator is a GGUF
    then gets a backend that can serve none of their models, a wizard that scans for
    MLX conversions and finds nothing, and no generator configured at all -- while the
    picker lists the GGUF as perfectly servable.

    An explicit ``serving.generator.backend`` always wins: this only breaks ties that
    ``auto`` left open, and only in favour of a backend that can serve something.
    """
    requested = ((cfg.get_in("serving.generator") or {}) or {}).get("backend")
    if requested and str(requested).lower() != "auto":
        return choose(accelerator, requested).name

    default = choose(accelerator, None).name
    if hf_home is None:
        return default

    from lara.models import servable as _servable

    order = [default] + [n for n in ("llamacpp", "mlx", "vllm") if n != default]
    for name in order:
        if not BACKENDS[name].available():
            continue
        try:
            if _servable(hf_home, backend=name):
                return name
        except Exception:       # a cache we cannot read is not a reason to fail startup
            continue
    return default


def effective_backend(cfg, accelerator: str) -> str:
    """The backend that will actually serve, as a canonical name.

    ``devices.Device.backend`` is advisory prose written before anything is installed —
    it says "llama.cpp" on any Mac. :func:`choose` is the real decision and takes
    installation into account, so on a Mac with ``mlx-lm`` present the two disagree and
    name different weight formats. Anything that has to match the runtime — which models
    are listed as servable, which config key the chosen model is written under — has to
    ask this, not the Device.
    """
    requested = ((cfg.get_in("serving.generator") or {}) or {}).get("backend")
    return choose(accelerator, requested).name


def model_for(backend: str, cfg: dict) -> str | None:
    """The model configured for one backend. Formats differ, so these cannot be shared."""
    if backend == "vllm":
        return (cfg.get("vllm") or {}).get("default_model")
    return (cfg.get(backend) or {}).get("model")


# ── the process ───────────────────────────────────────────────────────────────────


def port_of(base_url: str) -> int:
    tail = base_url.rstrip("/").rsplit(":", 1)[-1]
    digits = "".join(c for c in tail.split("/")[0] if c.isdigit())
    return int(digits) if digits else 8000


def probe(base_url: str, timeout: float = 1.0) -> list[str] | None:
    """Model ids served at ``base_url``, or None if nothing answers."""
    import httpx

    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
        if r.status_code == 200:
            return [m.get("id", "?") for m in (r.json().get("data") or [])]
    except Exception:
        pass
    return None


@dataclass
class GeneratorProcess:
    """Supervises a generation server for the lifetime of the reader.

    Deliberately conservative about ownership: if something is already answering on the
    configured URL, this attaches to it and will not stop it on exit. Killing a server the
    user started themselves — or worse, one shared with another tool — is not a decision a
    reader process should make.
    """

    backend: Backend
    model: str
    base_url: str
    cfg: dict
    log_path: Path | None = None
    proc: subprocess.Popen | None = None
    adopted: bool = False
    _log: object = field(default=None, repr=False)

    @property
    def port(self) -> int:
        return port_of(self.base_url)

    def start(self) -> "GeneratorProcess":
        existing = probe(self.base_url)
        if existing is not None:
            self.adopted = True
            return self
        if self.backend.name == "external" or not self.backend.available():
            # Not installed: leave proc as None so the caller reports the install hint
            # rather than dying on FileNotFoundError from Popen.
            return self

        cmd = self.backend.command(self.model, self.port, self.cfg)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            cmd, env=self.backend.env(self.cfg),
            stdout=self._log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT if self._log else subprocess.DEVNULL,
            # Own process group, so stopping the reader stops the whole server tree rather
            # than orphaning workers that keep the GPU allocated.
            start_new_session=True,
        )
        return self

    def wait_ready(self, timeout: float = 300.0, interval: float = 1.0,
                   on_wait=None) -> bool:
        """Poll until the endpoint answers. Returns False on timeout or early exit."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if probe(self.base_url) is not None:
                return True
            if self.proc is not None and self.proc.poll() is not None:
                return False                  # died during startup; log has the reason
            if on_wait:
                on_wait(deadline - time.time())
            time.sleep(interval)
        return False

    def stop(self, grace: float = 10.0) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            self.proc.terminate()
        try:
            self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self.proc.kill()
        finally:
            if self._log:
                self._log.close()

    def __enter__(self) -> "GeneratorProcess":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()


def from_config(cfg, accelerator: str, model_override: str | None = None,
                backend_override: str | None = None) -> GeneratorProcess | None:
    """Build a supervisor from the ``serving`` config, or None if nothing is configured.

    ``backend_override`` travels with ``model_override``: the two are a pair, because a
    model only means anything to the runtime whose format it is. Overriding the model
    alone would hand a GGUF spec to MLX.
    """
    serving = cfg.get_in("serving") or {}
    gen = serving.get("generator") or {}
    backend = choose(accelerator, backend_override or gen.get("backend"))
    merged = {**gen, "vllm": serving.get("vllm") or {}}
    model = model_override or model_for(backend.name, merged)
    base_url = (serving.get("vllm") or {}).get("base_url", "http://127.0.0.1:8000/v1")
    if backend.name != "external" and not model:
        return None
    logs = cfg.get_path("paths.logs") if cfg.get_in("paths.logs") else None
    return GeneratorProcess(
        backend=backend, model=model or "", base_url=base_url, cfg=merged,
        log_path=(logs / f"generator-{backend.name}.log") if logs else None,
    )
