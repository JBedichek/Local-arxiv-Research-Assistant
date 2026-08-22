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
``ollama``   anywhere, Metal     Ollama tags (``qwen3:8b``)      easiest; usually already
                                                                 installed on a Mac
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
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from lara.models import servable as _servable
from lara.serve.runtimes import (  # noqa: F401 — re-exported
    BACKENDS,
    Backend,
    installed,
)

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
        # llama.cpp first. It is the most mature of the three, it reads GGUF -- the format
        # most local models are actually published in -- and it is what lara/serve/devices.py
        # has always advised on this hardware. Ranking MLX ahead of it contradicted that
        # advice, so the same machine was told two different things depending on which
        # module you asked.
        #
        # MLX stays as the fallback rather than the default, and the ordering is
        # deliberate in both directions: MLX is often faster, but it arrives automatically
        # with `pip install -e '.[mac]'` while llama.cpp is a brew binary the user has to
        # install on purpose. Defaulting to the thing that installs itself is how the
        # default stopped matching the documentation. Preferring llama.cpp and falling
        # back to MLX means a pip-only install still generates instead of finding nothing
        # available. Ollama sits between them for the reason it always did: on a Mac it is
        # far more often already running, and it is llama.cpp underneath anyway.
        order = ("llamacpp", "ollama", "mlx")
    elif accelerator in ("cuda", "rocm"):
        order = ("vllm", "llamacpp", "ollama")
    else:
        order = ("llamacpp", "ollama")

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
    with llama.cpp installed it returns llama.cpp; a user whose only cached generator is
    an MLX conversion then gets a backend that can serve none of their models, a wizard
    that scans for GGUF and finds nothing, and no generator configured at all -- while
    the picker lists the MLX model as perfectly servable. The roles here swap with the
    platform default; the tie-break is what stops either of them being a dead end.

    An explicit ``serving.generator.backend`` always wins: this only breaks ties that
    ``auto`` left open, and only in favour of a backend that can serve something.
    """
    requested = ((cfg.get_in("serving.generator") or {}) or {}).get("backend")
    if requested and str(requested).lower() != "auto":
        return choose(accelerator, requested).name

    default = choose(accelerator, None).name
    if hf_home is None:
        return default

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


def model_for(backend: str, cfg: dict) -> str | None:
    """The model configured for one backend. Formats differ, so these cannot be shared."""
    if backend == "vllm":
        return (cfg.get("vllm") or {}).get("default_model")
    return (cfg.get(backend) or {}).get("model")


def generator_cfg(cfg) -> dict:
    """`serving.generator` with `serving.vllm` folded in under its own key.

    vLLM's settings live beside `generator` in config.yaml rather than under it, for the
    good reason that `lara serve-llm` manages a vLLM process whether or not it is the
    configured generator. Every caller that wants "the generator settings" therefore has
    to graft the two together, and four of them were doing it inline -- so reading only
    `serving.generator` reported "no default model" on every Mac, where the model lives
    under the mlx or llamacpp key instead.
    """
    serving = cfg.get_in("serving") or {}
    return {**(serving.get("generator") or {}), "vllm": serving.get("vllm") or {}}


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
    merged = generator_cfg(cfg)
    model = model_override or model_for(backend.name, merged)
    base_url = (serving.get("vllm") or {}).get("base_url", "http://127.0.0.1:8000/v1")
    if backend.name != "external" and not model:
        return None
    logs = cfg.get_path("paths.logs") if cfg.get_in("paths.logs") else None
    return GeneratorProcess(
        backend=backend, model=model or "", base_url=base_url, cfg=merged,
        log_path=(logs / f"generator-{backend.name}.log") if logs else None,
    )
