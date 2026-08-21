# Generation backends

Which runtime writes the answers, how `lara serve` starts it, and why it will not stop one
it did not start.

Implementation: [`lara/serve/generator.py`](../../lara/serve/generator.py). Commands:
`lara backends`, `lara serve`, `lara serve-llm`, `lara bench-generate`.

---

## 1. Why five backends is cheap

The reader speaks only OpenAI-compatible HTTP and never imports an inference library. Every
generation call goes through `stream_answer` to `{base_url}/chat/completions`, so swapping
runtimes is a URL change rather than a port. The difference between backends is a command
line and a health check, not a code path through retrieval.

| backend | platform | model format | why you would pick it |
|---|---|---|---|
| `vllm` | CUDA / ROCm | HF safetensors | fastest with a real GPU |
| `llamacpp` | anywhere, Metal | GGUF | most mature; runs anywhere |
| `mlx` | Apple Silicon only | MLX (`mlx-community/*`) | unified memory, often faster than llama.cpp |
| `ollama` | anywhere, Metal | Ollama tags (`qwen3:8b`) | easiest; usually already installed on a Mac |
| `external` | anything | whatever it already serves | you started it yourself |

## 2. The formats are not interchangeable

This is the one thing that makes the choice more than a flag. A repo vLLM serves happily is
not a GGUF file and is not MLX-converted, so **each backend carries its own `model`
setting**. Pointing llama.cpp at a safetensors repo fails at load time with an error about
the file type, which is a confusing way to discover the rule.

```yaml
serving:
  vllm:
    default_model: Qwen/Qwen3.8-27B-FP8      # vLLM reads THIS key
  generator:
    llamacpp:
      model: null                            # everything else reads serving.generator.<name>.model
    mlx:
      model: null
    ollama:
      model: null
```

`model_for(backend, cfg)` encodes exactly that asymmetry: `serving.vllm.default_model` for
vLLM, `serving.generator.<name>.model` for the rest. It is also why
`serve-llm --model` and `--backend` travel together — overriding the model alone would hand
a GGUF spec to MLX.

**vLLM has no Metal backend.** Its platforms are cpu/cuda/rocm/tpu/xpu, so on a Mac it runs
CPU-only and leaves the GPU idle. That is why Apple Silicon gets two purpose-built options
rather than a degraded vLLM.

## 3. Choosing

`choose(accelerator, requested)` resolves `auto` **by platform fit first, installation
second**, so the answer explains itself:

```
mps          →  mlx, ollama, llamacpp
cuda / rocm  →  vllm, llamacpp, ollama
anything else→  llamacpp, ollama
```

On Apple Silicon MLX and llama.cpp are both right and vLLM never is, regardless of whether
vLLM happens to be installed. Ollama comes before llama.cpp on a Mac because it is far more
often already there — and it is llama.cpp underneath anyway.

If nothing in the order is installed, it returns the platform's **best** choice anyway
rather than silently falling back to `external`. The caller then checks `available()` and
can say *"brew install llama.cpp"*, which is actionable. `external` is what you get by
asking for it, or on a platform with no options at all.

### `resolve_backend` breaks ties by what you actually own

`choose` is right until two candidates are both installed and only one can read anything in
your cache. On a Mac with `mlx-lm` present it returns MLX; a user whose only cached
generator is a GGUF then gets a backend that can serve none of their models, a wizard that
scans for MLX conversions and finds nothing, and no generator configured — while the picker
lists the GGUF as perfectly servable.

`resolve_backend(cfg, accelerator, hf_home)` therefore tries `[default, llamacpp, mlx,
vllm]` and returns the first installed backend that `lara.models.servable()` finds
something for. An explicit `serving.generator.backend` always wins; this only breaks ties
`auto` left open.

`effective_backend()` is the canonical name for anything that must match the runtime.
Note that `devices.Device.backend` is **advisory prose** written before anything is
installed — it says "llama.cpp" on any Mac — and the two disagree on a Mac with mlx-lm
present. Anything deciding which models are servable or which config key to write must ask
`effective_backend`, not the `Device`.

## 4. `lara backends`

```bash
lara backends
```

Prints every backend with its weight format, whether it is installed (with the install hint
if not), the model configured for it, and a one-line note. Backends whose platform does not
match the detected accelerator are dimmed. `→` marks what `lara serve` would start.

The last line probes `serving.vllm.base_url` and reports whether something is already
answering there, with the model ids it lists. On the machine these docs were written on:

```
detected cuda; → marks what `lara serve` would start
already running at http://127.0.0.1:8000/v1: yes — Qwen/Qwen3.8-27B-FP8
```

## 5. Autostart with `lara serve`

`lara serve` starts the generator alongside the reader when
`serving.generator.autostart` is true (the default). `--no-llm` skips it entirely —
retrieval works, answers do not.

The sequence:

1. **Probe first.** `GeneratorProcess.start()` calls `probe(base_url)` — a `GET /models`
   with a 1 s timeout. If anything answers, it sets `adopted = True` and launches nothing.
2. **Nothing configured?** Fall back to the cache. `resolve_backend` picks a backend, and
   the largest cached model that fits (`DV.fits`) is used *for this run only*, with a notice
   that `lara setup` makes it permanent. Adopting a model beats printing "run lara setup" at
   someone who just downloaded one and reasonably expects to ask a question.
3. **Launch.** `subprocess.Popen(..., start_new_session=True)` — its **own process group**,
   so stopping the reader stops the whole server tree rather than orphaning workers that
   keep the GPU allocated. stdout and stderr go to `paths.logs/generator-<backend>.log`.
4. **Wait on the endpoint, not the process.** `wait_ready()` polls `probe()` every second up
   to `serving.generator.startup_timeout_sec` (300), returning early and false if the
   process exits during startup. Readiness is established by probing rather than by trusting
   that the process started correctly, because flag surfaces drift between releases of all
   three runtimes.
5. **Announce when ready.** The reader's own URL is printed from a watcher thread that waits
   for `GET /api/health` to report `ready`. Model loading takes tens of seconds and during it
   every endpoint answers 503, so a URL printed before that is an invitation to click
   something broken.

Every failure is reported and non-fatal: not installed (with the install hint), configured
but nothing running, or did not come up (with the log path). Retrieval still works in all
three cases.

## 6. Adoption, and what `stop` will not do

> Deliberately conservative about ownership: if something is already answering on the
> configured URL, this attaches to it and will not stop it on exit. Killing a server the
> user started themselves — or worse, one shared with another tool — is not a decision a
> reader process should make.

`stop()` returns immediately unless `self.proc` is a live process this instance spawned, and
`lara serve`'s `finally` block only calls it when `gen.proc is not None`. An adopted server
survives the reader.

Shutdown is `SIGTERM` to the process group, `wait(grace=10.0)`, then `SIGKILL` to the group.

## 7. `lara serve-llm`

Starts the generator on its own — for running it on a different machine, or keeping it up
across reader restarts, since model loads are slow.

```bash
lara serve-llm --show                       # print the command, run nothing
lara serve-llm --backend llamacpp --model ~/models/qwen3-8b-Q4_K_M.gguf
```

It refuses clearly rather than failing later: `external` says there is nothing to start and
names the `base_url` the reader will use; no model configured names the exact config key
(`serving.generator.<name>.model`); not installed prints the install hint.

Unlike `lara serve`, this does **not** probe first — it runs the command in the foreground.

## 8. Per-backend flags

Everything below is long-stable syntax, with `extra_args` for the rest.

### vLLM

`--port`, `--served-model-name`, `--gpu-memory-utilization` (0.5), `--max-model-len`
(32768), `--kv-cache-dtype` (auto), `--max-num-seqs` (64), `--enable-prefix-caching` when
`enable_prefix_caching` is true.

`gpu_devices` becomes `CUDA_VISIBLE_DEVICES`, and `CUDA_DEVICE_ORDER=PCI_BUS_ID` is always
set — CUDA orders devices by compute capability by default, not by slot, so "device 1" here
and "device 1" in the reader can be different cards on a mixed machine.

The binary is `.venv-vllm/bin/vllm` beside the repo if it exists, otherwise `vllm` on PATH.
vLLM is often pinned to a different torch than the reader's environment, and it is a
separate process reached over HTTP, so isolating it costs nothing.

### llama.cpp

`-m <path>` or `-hf <hub id>`, `--port`, `--host`, `-c` (ctx_size, 32768), `-ngl`
(n_gpu_layers, 999), `--parallel` (2), `--cache-reuse` (256), and `--cache-type-k` /
`--cache-type-v` when set.

Two traps handled explicitly:

- **Local file vs Hub id.** The distinction cannot be "does it contain a slash": a Hub id
  like `Qwen/Qwen3-8B-GGUF:Q4_K_M` contains one and is not a path. It is decided on the
  `.gguf` suffix and on whether the path exists.
- **`-fa` changed shape.** Older builds took it as a bare boolean, current ones require
  `on|off|auto`. Emitting the bare form against a current build makes it swallow the *next*
  argument — the observed failure was
  `error: unknown value for --flash-attn: '--cache-reuse'`, which names neither the real
  problem nor the flag that caused it. `_flash_attn_takes_value()` reads the binary's own
  `--help` (cached, 15 s timeout) rather than a version number, because the binary can come
  from Homebrew, a manual build or a container. It defaults to the current syntax if help
  cannot be read.

`-ngl 999` offloads every layer it can on Metal and CUDA and is ignored on a CPU-only build,
so one value is safe across platforms. `--cache-reuse` is what keeps follow-up questions
about the same paper fast, which is the whole reason the prompt is laid out
stable-prefix-first.

### MLX

`--model`, `--port`, `--host`, `--max-tokens` if set. Runs `mlx_lm.server` from PATH, or
`python -m mlx_lm.server` if the module is importable but not on PATH.

The server is younger than llama.cpp's: simpler prompt caching and no continuous batching.
Fine for a single-user reader.

### Ollama

Two things make it behave unlike the others.

**The server is not told which model to load.** `ollama serve` starts a daemon and models
are loaded on demand by whatever name a request asks for, so `model` here is a tag like
`qwen3:8b` that must already have been pulled. **lara cannot pull it for you** — use
`ollama pull <tag>`.

**The port comes from `OLLAMA_HOST`**, not a flag, because `serve` takes no address
argument. `OLLAMA_HOST` is set from `serving.generator.ollama.host` and `.port` (11434).

On macOS the desktop app usually has a daemon running already, in which case the probe
adopts it and nothing is launched.

### external

Nothing is launched or stopped; the reader just uses `base_url`. Use this for LM Studio, or
a generator on another machine.

## 9. `lara bench-generate`

```bash
lara bench-generate --runs 5 --max-tokens 512
```

Backend-agnostic on purpose: it speaks the same OpenAI-compatible API the reader uses, so
llama.cpp and MLX can be compared on one machine by pointing this at each in turn (change
`serving.vllm.base_url`, which is the single URL the reader talks to whatever serves).

It probes `base_url` first and refuses with *"nothing answering at … — start one with
`lara serve-llm`"*. The model benchmarked is **the first id the server lists**, not the
configured one.

One warm-up run precedes the timed ones — warming the prefix cache and the kernels. Per
run it reports TTFT, total time, tokens and tok/s, then the median TTFT and median tok/s.

**tok/s excludes the first token**: `(n - 1) / (total - ttft)`. That is a decode rate;
including the prefill would make a long prompt look like a slow model. The same convention
is used for the per-answer figure in the UI.

## 10. Configuration

```yaml
serving:
  generator:
    backend: auto              # auto | vllm | llamacpp | mlx | ollama | external
    autostart: true
    startup_timeout_sec: 300
    llamacpp:
      model: null
      ctx_size: 32768
      n_gpu_layers: 999
      parallel: 2
      flash_attn: true
      cache_reuse: 256
      cache_type_k: null       # q8_0 halves the KV cache
      cache_type_v: null
      extra_args: []
    mlx:   {model: null, extra_args: []}
    ollama:{model: null, host: 127.0.0.1, port: 11434, extra_args: []}
  vllm:
    base_url: http://127.0.0.1:8000/v1
    gpu_devices: [1]
    gpu_memory_utilization: 0.5
    enable_prefix_caching: true
    kv_cache_dtype: auto
    max_model_len: 32768
    max_num_seqs: 64
    default_model: Qwen/Qwen3.8-27B-FP8
    default_quantization: fp8
```

```bash
lara config set serving.generator.backend llamacpp     # enum-validated
lara config set serving.generator.llamacpp.ctx_size 8192
```

## 11. Things worth knowing

- **`serving.vllm.base_url` is the reader's URL regardless of backend.** The key is named
  for vLLM for historical reasons; llama.cpp, MLX and Ollama all read their port from it via
  `port_of()`, which takes the trailing digits.
- **`port_of` parses digits out of the last colon-separated segment.** A `base_url` with no
  port defaults to 8000.
- **The context length decision is a memory decision.** At 32 k the KV cache outweighs the
  weights it serves — see
  [`configuration.md`](configuration.md#5-the-wizard-screen-by-screen) for the arithmetic
  and what `lara setup` offers.
- **Only vLLM's config has `policy: single_resident`.** Switching models means restarting
  the generator; the reader cannot load two.
- **Nothing here downloads models.** The reader UI does (`/api/model/download`), and Ollama
  needs `ollama pull`.
