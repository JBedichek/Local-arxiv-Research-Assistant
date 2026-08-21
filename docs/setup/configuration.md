# Configuration: `lara setup` and `lara config`

How the machine plans what it can run, what it writes down, and how to change a setting
afterwards without editing YAML.

Implementation: [`lara/setup.py`](../../lara/setup.py) (the planner),
`setup()` in [`lara/cli.py`](../../lara/cli.py) (the wizard),
`config_app` in the same file (`get`/`set`/`unset`/`show`),
[`lara/config.py`](../../lara/config.py) (loading and layering).

The two-layer model — `config.yaml` tracked, `config.local.yaml` machine-local and
gitignored — is covered in [`device_detection.md`](device_detection.md). This document is
about the tools that write and read it.

---

# Part 1 — `lara setup`

## 1. What it is

A thin shell over `lara/setup.py`, which is pure computation against measured constants. The
same recommendations can therefore be produced non-interactively, tested, and explained.

```bash
lara setup                    # interactive
lara setup --show             # print the plan and the file it would write; write nothing
lara setup --non-interactive  # accept every recommendation
lara setup --prefer speed     # balanced (default) | speed | memory
```

`--show` skips the Hugging Face access check, so it can report on a machine that has no
access yet. It previews exactly what `--non-interactive` would do, including which
already-running generator it would adopt and which cached model it would pick — a `--show`
that previewed a config with no model would not be a preview.

## 2. The binding constraint is the index, not the model

Tier 1 is loaded before anything else and has no graceful degradation: if it does not fit,
nothing runs. A wizard that only checked whether the *generator* fits would happily
configure a machine that then dies building its search index.

So the planner budgets in this order:

```
budget      device.single_device_gb        (largest single GPU — the index is never sharded)
overhead    embedder + reranker + row map  (fixed; no keep fraction can reduce it)
index       n_chunks × bytes_per_vector    (the only part the slider moves)
headroom    budget − index − overhead      → the largest 4-bit generator that fits
```

`single_device_gb`, not the sum across GPUs: the index is one tensor on one card. The
generator budget is reported separately because vLLM *can* shard.

### The fixed costs, spelled out

`overhead_gb()` counts the embedder, the cross-encoder reranker and the whole-corpus row
map. Parameter counts come from the model cards — `google/embeddinggemma-300m` at 308 M and
`tomaarsen/Qwen3-Reranker-0.6B-seq-cls` at 596 M — and resident cost is params × dtype
width, so **a model costs twice as much on CPU as on a GPU**: encoders load at bf16 on CUDA,
fp16 on MPS and fp32 on CPU, and neither is int-quantised.

These used to be flat 1.2 GB constants, which happened to be the fp16 figure for the
reranker and the *fp32* figure for the embedder — overstating the embedder by 2x on a GPU
and understating the reranker by the same factor on CPU.

The row map is `vector_row → chunk_id` for the whole corpus as an int32 table, 4 bytes per
row. It is worth counting because it does **not** shrink when the corpus is scoped, but at
0.12 GB for the current corpus it is no longer material — as a dict it measured ~23 bytes
per entry and 0.67 GB.

`hot_tier_bytes` defaults to **zero**. `hot_tier.max_bytes` is read by this planner and by
nothing that serves, so reserving 2 GB for it made every machine look 2 GB smaller than it
is and pushed scoping harder than the evidence supports.

## 3. The index options

Five, from `OPTIONS` in `lara/setup.py`, all with `bytes_per_vector` at dim 256 and their
measured p50 and recall. See [`../retrieval/search_backends.md`](../retrieval/search_backends.md)
for the full table and how to reproduce it with `lara bench-index`.

**Every option is selectable; two are never *recommended*.** `recommendable=False` on faiss
flat (slower *and* larger than torch fp16) and faiss sq8 (372 ms/query). *"It fits" is not
sufficient grounds to recommend something.*

Two further filters apply per platform:

- On plain CPU, `torch-int8` and `faiss-sq8` are removed from the list entirely.
- Off CUDA/ROCm, **any int8 option stops being recommendable**. The per-block
  dequantisation a CUDA card absorbs is the dominant cost everywhere else, and it is the
  only path that returns corrupt scores under memory pressure on Metal. Still listed, still
  selectable, never the recommendation.
- `P50_BY_DEVICE` overrides the CUDA p50 where a figure was measured elsewhere: on Metal,
  `torch-int8` is 403 ms and `torch-fp16` is 28.9 ms (2.76 M rows, dim 256). Quoting the
  CUDA figure on a Mac promised 8.2 ms for something measured at 403.

## 4. Three-state fit guarding

`Plan.scope` is not a boolean.

| state | condition | consequence |
|---|---|---|
| `required` | index + overhead > 80 % of budget | scoping is the only way this machine runs |
| `recommended` | total RAM ≤ 16 GB, **or** index + overhead > 50 % of budget | it fits, but leaves little for anything else |
| `unnecessary` | otherwise | the whole corpus fits comfortably |

**Nothing here is a default.** Topic scoping in particular is *recommended*, never applied:
plenty of Macs have 32, 64 or 128 GB and should use the whole corpus. The recommendation
fires on the measured budget, not on the platform.

The keep fraction is solved against a *different* target depending on the state —
`headroom=0.8` for `required`, `0.4` for `recommended`. Computing both against the same
target produced *"scoping recommended (keep 100 %)"*, which is a contradiction. If a
`recommended` scope solves to `keep >= 1.0`, the state is downgraded to `unnecessary` with
an honest reason: *"it fits, and scoping would not free enough to be worth the narrowed
recall."*

`effective_keep` exists for the same reason: `scope_keep` is left at whatever the solver
last computed even when scoping turns out to be unnecessary, so reading it directly would
shrink an index that is never going to be shrunk.

### When no keep fraction can help

Scoping shrinks the index and nothing else. When the fixed costs alone exceed half the
budget, `overhead_advice` says so and names what to cut instead:

- disable the cross-encoder reranker (−1.2 GB) — *"ranking falls back to the dense+BM25
  fusion, which is measurably worse but entirely usable"* — only offered below an 8 GB
  budget;
- shrink the tier-0 hot cache from 2.0 GB to 0.5 GB — *"it only prefetches the open paper's
  neighbourhood, so this costs latency on citation follows, not correctness."*

Both are re-costed and the keep fraction re-solved against the new overhead.

## 5. The wizard, screen by screen

**1. Hardware** — platform, accelerator and GPU names, RAM/VRAM, the index budget (with
"largest of N GPUs; the index is not sharded" when there is more than one), the generator
budget, and the advisory backend with its reason.

**2. Corpus** — vectors present, or a note that planning is against the published corpus
(`REFERENCE_CHUNKS = 28,723,432`) and `lara dataset fetch` will get it.

**3. Retrieval backend** — the option table *and the keep slider on one screen*.

> Backend and keep fraction are one decision, not two: the whole reason to shrink the corpus
> is to afford a backend and still have room to generate, and that trade is unreadable if
> you pick the backend on one screen and the fraction on the next.

Every column is quoted **at the size this machine will actually build** — mixing a
full-corpus index with leftover memory computed after scoping made two columns that could
not both be true at once. `↑/↓` chooses an engine, `◀/▶` moves the slider through
`0.01 0.02 0.05 0.10 0.15 0.20 0.25 0.33 0.50 0.66 0.75 0.90 1.00`, enter confirms, escape
restores the recommendation *wholesale* (engine and fraction together).

The legend names every addend rather than saying "+models": *"+models" hid the fact that a
third of the fixed cost is cache, not a model, and named no figure you could check.*

Outside a terminal — piped input, CI, `TERM=dumb` — there is no slider, so both halves get a
typed prompt. A typo in the backend name is **rejected and re-asked**: silently falling back
to the default meant a typo picked a backend you did not ask for, and nothing said so.

**4. Your interests** — only if the fraction is below 1.0. Topics are prompted for
repeatedly until a blank line. Giving none is handled honestly: *"No topics given — keeping
the whole corpus resident instead"*, and the plan is reset to `keep=1.0`. Without topics
there is nothing to score against, so a keep fraction cannot be honoured, and writing a
config that silently keeps everything would be worse.

The screen also repeats that nothing is deleted — dropped papers stay searchable through
BM25 and open normally, only their dense vectors leave RAM. See
[`../retrieval/corpus_scoping.md`](../retrieval/corpus_scoping.md).

**5. Generator** — the backend from `resolve_backend` (not `Device.backend`, see
[`generation_backends.md`](generation_backends.md#3-choosing)), then a probe of
127.0.0.1 on **8000 (vLLM), 11434 (Ollama), 1234 (LM Studio), 8080 (llama.cpp)**. Anything
already running is offered first, because *asking someone to re-download a model they have
is the fastest way to lose them*. Otherwise the HF cache is scanned for models this backend
can serve, listed with size and a fit verdict (`fits`, or `needs X of Y GB`).

**5b. Context length** (llama.cpp and Ollama only) — the largest generator-side memory
decision, and nothing used to surface it.

KV cache is 2 (K and V) × layers × kv_heads × head_dim × dtype_bytes **per token**. At long
context it outweighs the weights it serves: an 8B at 32 k in fp16 needs 4.8 GB of cache
against 4.7 GB of Q4_K_M weights, and `ctx_size: 32768` sat in the config quietly doubling
the generator. The constant assumes an 8B-class GQA shape — 36 layers, 8 KV heads, 128 head
dim, which is Qwen3-8B — so it is **sizing guidance, not a measurement**; llama.cpp prints
the exact number at load, and the wizard says so.

Choices are 4096 / 8192 / 16384 / 32768 / 65536, each with its cache size, cache + weights,
and whether it fits the space left after retrieval. `◀/▶` toggles the KV cache between fp16
and `q8_0`, which halves it. Both `cache_type_k` and `cache_type_v` are written
**explicitly either way** — null means fp16, and leaving the key absent would let a
previously-quantised cache persist unnoticed.

**5c. Concurrent requests** — 1, 2 or 4 slots. `-c` is the *total* context shared across
slots in llama.cpp, so slots do not multiply memory; each extra slot halves what one request
can read. It is a throughput/window trade, which is why it comes after the memory decision
rather than tangled into it.

**6. Write** — `config.local.yaml`, then a preflight run reporting any failures, then the
next commands.

## 6. What it writes

From `overrides_for()`:

```yaml
index:
  backend: torch
  precision: fp16
  faiss: {kind: hnsw}          # only when the chosen option is a faiss one
  rerank: {cross_encoder: {enabled: true}}
hot_tier: {max_bytes: 0}
hardware:
  generator_headroom_gb: 78.5
  generator_max_params_4bit: 116360042488
corpus:
  scope: {topics: [...], keep: 0.05, expand_min_citations: 3}   # only when keep < 1.0
disk: {root: ..., required_device: /dev/nvme0n1p5}
embedding: {devices: [0, 1, 2]}
serving:
  generator: {backend: vllm}
  vllm: {base_url: ..., default_model: ..., default_quantization: ...}
```

Four choices in there are deliberate:

- **`hardware.*` is saved so the reader can tell you what will fit** without re-deriving it.
  The UI has no idea what the index costs, and asking a user to remember "about 8B at 4-bit"
  from a wizard they ran last month is not a plan.
- **`corpus.scope` is written, not printed as a command.** Scoping is a load-time decision,
  so the server builds the keep-set from these on first start and caches it. Printing a
  command left every scoped machine one forgotten step away from loading a corpus it could
  not fit.
- **The model goes under the key the serving backend reads.** `model_for` looks at
  `serving.vllm.default_model` for vLLM and `serving.generator.<backend>.model` for
  everything else. Writing it under vllm on a Mac left `model_for` returning None,
  `from_config` returning None, and no generator ever starting — while the picker cheerfully
  listed the model as present but not loaded.
- **`disk.required_device` pins the filesystem the corpus actually lives on.** Detectable,
  and the check that catches a symlink quietly redirecting 30 GB onto the wrong disk.

### Replace what you write, preserve what you do not

`write_local()` backs the existing file up to `config.local.yaml.bak.<timestamp>`,
deep-merges the overrides over what is there, and reports every dotted key it did **not**
touch.

Two earlier designs were worse. Rewriting the file wholesale silently dropped
`disk.forbid_paths` and `min_free_gb` — hand-written safety pins whose entire job is to stop
30 GB landing on a full disk. Stripping a fixed list of "managed" keys instead cleared
`default_quantization` whenever the wizard adopted an already-running server and therefore
never chose one.

For the surviving rule to be safe, the wizard must **write every key it could meaningfully
change**, which is why `overrides_for` always emits the cross-encoder flag and the hot-tier
size rather than only when they differ from the default.

A real run on the machine these docs were written on carried over
`disk.forbid_paths, disk.min_free_gb, huggingface.home, serving.auth.mode,
serving.auth.token, serving.vllm.default_quantization, serving.vllm.gpu_devices,
serving.vllm.gpu_memory_utilization, serving.vllm.max_num_seqs`.

The file gets a header naming when it was written, the detected machine, the chosen index
with its size and measured figures, the budget, and — when scoping applies — the keep
fraction and the note that it is built automatically on first start.

---

# Part 2 — `lara config`

## 7. The four subcommands

```bash
lara config show                       # everything, and which files it came from
lara config show index                 # one subtree
lara config get retrieval.hierarchy.paper_frac
lara config set index.backend faiss
lara config unset index.backend
```

`show` and `get` take `--config PATH`; `set` and `unset` do not, because they always write
`config.local.yaml`.

`show` prints the layers first:

```
layers: /path/config.yaml, /path/config.local.yaml
```

`get` exits 1 with *"not set: KEY"* if the key does not resolve. Dicts and lists are printed
as JSON.

## 8. `set` writes the local layer only

> Writes to the machine-local layer rather than `config.yaml`, so your change is not a
> pending edit to a tracked file and does not travel to other machines.

It reports the transition, so you can see whether the change did anything:

```
index.backend: 'auto' -> 'faiss'   in /path/config.local.yaml
```

and adds *"restart the reader for this to take effect"* for any key under `index.` or
`embedding.`.

Values are parsed as YAML, so `512` becomes an int, `[0, 1]` a list, `true` a bool.

## 9. Enum validation

Six keys have a fixed set of legal values:

| key | choices |
|---|---|
| `index.backend` | `auto`, `torch`, `faiss` |
| `index.precision` | `fp16`, `int8` |
| `index.faiss.kind` | `flat`, `sq8`, `hnsw` |
| `serving.generator.backend` | `auto`, `vllm`, `llamacpp`, `mlx`, `ollama`, `external` |
| `serving.auth.mode` | `auto`, `always`, `off` |
| `embedding.compile` | `default`, `reduce-overhead`, `max-autotune`, `null` |

> A typo here otherwise surfaces much later — `index.backend: fiass` does not fail at
> startup, it silently falls through to the default and you wonder why the benchmark you ran
> does not match what you are running.

An invalid value exits 1 and lists the choices. These keys also **bypass YAML parsing and
stay strings**, which is what stops `lara config set serving.auth.mode off` from writing the
boolean `False` — see [`authentication.md`](authentication.md#4-the-yaml-off-trap).

## 10. `unset` prunes

Removes the key from `config.local.yaml` and reports the value that now applies from the
defaults. Emptied branches are pruned, so unsetting `index.faiss.kind` does not leave
`index: {faiss: {}}` behind to puzzle whoever reads the file next.

Unsetting a key that is not set locally says so rather than failing: *"KEY is not set
locally; the default applies already."*

## 11. Comments are not preserved

`yaml.safe_dump` cannot round-trip comments. Rather than adding a dependency
(`ruamel.yaml`), both `set` and `unset` take a timestamped backup **when the existing file
contains any comment line** and say so out loud:

```
comments in that file were not preserved — previous version saved as config.local.yaml.bak.20260821-113700
```

`lara setup` writes its own header comments; a subsequent `lara config set` will strip them,
leaving the backup as the record.

## 12. Things worth knowing

- **`--config PATH` disables layering.** An explicit `--config` selects that file alone —
  layering a machine's overrides onto a config the caller deliberately named would make the
  flag mean something other than "use this configuration".
- **Interpolation runs after the merge.** Overriding `disk.root` in the local file carries
  every `${disk.root}/…` path with it. Interpolating each layer separately would freeze the
  defaults' paths against the default root — the override would appear to apply while the
  data quietly went somewhere else.
- **Relative paths resolve against the config file's own directory**, not the process
  working directory, so `./data` means the same thing however you invoke `lara`. The one
  exception is the packaged default inside site-packages, where the working directory is the
  only sane anchor.
- **Lists replace, they do not concatenate.** `forbid_paths: []` in the local file has to be
  able to mean "nothing is forbidden here", and an appending merge could never express that.
- **`lara config set` does not validate anything outside the enum table.** A nonsense value
  for a numeric key is accepted and fails at load.
- **`hot_tier.max_bytes` is read by the planner and by nothing that serves.** Setting it
  changes what the wizard reports and nothing else.
