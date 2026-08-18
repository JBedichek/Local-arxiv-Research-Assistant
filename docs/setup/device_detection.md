# Devices: what runs where

How the project decides which accelerator to use, what hardware it supports, and what to
change when it guesses wrong.

Short version: **you should not have to configure anything.** The config expresses devices
the way a CUDA user thinks about them — bare integers — and `lara/device.py` translates
that onto whatever is actually present. If you have one GPU, or a Mac, or nothing at all,
the same config works.

---

## 1. Supported hardware

| accelerator | retrieval | embedding | generation | notes |
|---|---|---|---|---|
| **NVIDIA CUDA** | ✅ full | ✅ full, multi-GPU | ✅ vLLM | the reference configuration |
| **AMD ROCm** | ✅ | ✅ | ✅ vLLM | reports as `cuda` through torch's HIP shim |
| **Apple Silicon (MPS)** | ✅ | ✅ single GPU | ⚠️ external | vLLM has no Metal backend — see §5 |
| **CPU only** | ✅ | 🐢 ~50× slower | ⚠️ external | a supported configuration, not a broken one |

**Having no GPU is not an error.** Retrieval on CPU measures ~7 ms at 96 % recall, which is
a perfectly good reader. Only *building* the index is painful without a GPU, and you can
skip that entirely by fetching a prebuilt corpus (see `scraping_from_scratch.md` §4).

Check what was detected:

```bash
lara preflight
```

---

## 2. How a config device becomes a real device

`config.yaml` says things like `embedding.devices: [0, 1, 2]` and
`cross_encoder.device: 1`. Those are CUDA ordinals, which is the right mental model when
you have several cards and meaningless when you do not. `device.resolve()` maps intent
onto reality:

| config says | 3× CUDA | 1× CUDA | Apple Silicon | CPU only |
|---|---|---|---|---|
| `1` | `cuda:1` | `cuda:0` ⚠️ | `mps` | `cpu` |
| `[0, 1, 2]` | all three | `[cuda:0]` | `[mps]` | `[cpu]` |
| `"cuda:1"` | `cuda:1` | `cuda:0` ⚠️ | `mps` ⚠️ | `cpu` ⚠️ |
| `"cpu"` | `cpu` | `cpu` | `cpu` | `cpu` |
| `auto` / unset | `cuda:0` | `cuda:0` | `mps` | `cpu` |

⚠️ = falls back and emits a `RuntimeWarning` saying what it did.

**Integers and strings are treated differently, deliberately.** An integer is a
*preference* and gets clamped silently — asking for card 1 on a one-card machine is an
ordinary thing for a shared config to do. An explicit string like `"cuda:1"` is an
*instruction*, so if it cannot be honoured you get a warning. Silently running on CPU
because a typo asked for an absent GPU is a 50× slowdown that presents as a hang rather
than an error, and that is worth being noisy about.

A list of devices is de-duplicated: `[0, 1, 2]` on a single-GPU box becomes `[cuda:0]`,
not the same card three times. Three worker processes contending for one device is slower
than one.

---

## 3. What else is derived from the device

Three things used to be hardcoded and are now decided per-machine.

### Precision

| device | model weights | autocast |
|---|---|---|
| CUDA | bf16 | bf16 |
| MPS | fp16 | fp16 |
| CPU | **fp32** | **off** |

**fp16 rather than bf16 on Metal**, because MPS bf16 support lands unevenly across torch
versions and silently degrades to fp32 in places — fp16 is the honest choice.

**fp32 on CPU**, because half precision there is emulated: it is *slower* than fp32, not
faster. The usual "smaller is faster" intuition is simply false on CPU, and autocast is
disabled for the same reason.

### Multi-GPU fan-out

Only ever used for **multiple CUDA cards**. On unified memory there is one GPU behind one
pool of RAM, so N worker processes would multiply the memory footprint while contending
for the same silicon — strictly worse than a single process. `embed` checks this and
quietly runs single-process on a Mac even if the config lists three devices.

### Cache release

`torch.cuda.empty_cache()` raises on a machine without CUDA. Calls now route through
`device.empty_cache()`, which dispatches to CUDA or MPS and is a no-op elsewhere. These
calls are advisory — "we just freed something large, give it back" — so failure is
swallowed rather than propagated.

---

## 4. Overriding the choice

Every command that touches an accelerator takes `--device`:

```bash
lara embed --device cuda:2          # pin to one card
lara embed --device cpu             # force CPU, e.g. to free GPUs for serving
lara explore --device mps
lara fit-check --mode kfold --device cuda:0
```

The flag beats the config. Omitting it auto-detects. There is deliberately no global
environment variable for this — `CUDA_VISIBLE_DEVICES` already exists and works, and a
second overlapping mechanism is a debugging trap.

To change the default for every run, edit `config.yaml`:

```yaml
embedding:
  devices: [0, 1, 2]        # or `auto`, or ["cpu"]
index:
  rerank:
    cross_encoder:
      device: 1
```

---

## 5. Generation is the one thing that is not portable

**vLLM has no Metal backend.** Its platform list is cpu, cuda, rocm, tpu, xpu and zen_cpu.
On Apple Silicon it runs CPU-only and leaves the GPU idle, which is worse than not using
it at all.

This costs the project nothing, because the reader speaks to the generator over an
**OpenAI-compatible HTTP endpoint and never imports vLLM**. Swapping the runtime is a URL
change, not a port:

```bash
# macOS — any of these expose an OpenAI-compatible server
brew install llama.cpp && llama-server -hf <repo> --port 8000
# or Ollama (:11434), or LM Studio (:1234), or MLX
```

```yaml
serving:
  vllm:
    base_url: http://127.0.0.1:11434/v1     # point at whatever you are running
```

Prefer GGUF builds on Apple Silicon; llama.cpp loads them directly and they are what the
Metal kernels are tuned for.

The same escape hatch covers CPU-only machines and anyone who wants generation to happen
on a different box entirely — the reader does not care whether the endpoint is local.

---

## 6. Memory: the number people get wrong

The obvious question is "does my model fit". The more important one is **does my index
fit**, because the index is loaded before anything else and has no graceful degradation.

Resident tier-1 cost is `n_chunks × dim_truncated × bytes_per_element`:

| configuration | 23.9 M chunks | full 28.7 M |
|---|---|---|
| fp16, dim 256 *(current default)* | 12.2 GB | 14.7 GB |
| fp16, dim 128 | 6.1 GB | 7.3 GB |
| int8, dim 256 | 6.1 GB | 7.3 GB |
| int8, dim 128 | 3.1 GB | 3.7 GB |

Plus the embedder (~1.2 GB), the reranker (~1.2 GB), the tier-0 hot cache (2 GB by
default) and the generator (checkpoint size × 1.35 for KV cache and activations).

On unified memory **all of these draw from the same pool as the display server**, which is
why the reported budget reserves 6 GB there against 4 GB on discrete GPUs. `lara`
reports *usable* memory, not installed — quoting the installed figure would tell someone
with 16 GB that a 14 GB model fits.

Reducing the index footprint (int8 residency, and keeping only the topically relevant
fraction of the corpus in RAM) is in progress — see `PLAN.md` D19 and D22.

---

## 7. Things that will bite you

- **`nvidia-smi` works but nothing uses the GPU.** `lara preflight` now fails explicitly
  on this: the driver is healthy but torch cannot reach the cards, usually a CPU-only
  torch build or an emptied `CUDA_VISIBLE_DEVICES`. Without the check everything runs on
  CPU at ~50× the cost and looks like a hang.
- **Driver/library version mismatch.** `nvidia-smi` fails outright and no CUDA process can
  start. Preflight prints the fix:
  ```bash
  sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia
  # or reboot if a display server holds the modules
  ```
- **Config lists more GPUs than you have.** Harmless — extra entries collapse to the cards
  that exist, with a warning. You do not need to edit the config to run the same repo on a
  smaller machine.
- **A Mac will not serve a generator through `lara serve-llm`.** That command starts vLLM.
  Run llama.cpp/Ollama yourself and point `base_url` at it; the reader is unchanged.
- **`faiss-gpu` is not used and would not help.** There are no sm_120 kernels for this
  hardware, and tier 1 is an exact GPU matmul with no index build — which matters because
  the crawler appends continuously.
