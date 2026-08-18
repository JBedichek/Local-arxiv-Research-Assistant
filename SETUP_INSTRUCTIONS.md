# Setup

From nothing to a running reader. Four commands do the work; the rest of this file explains
what they do and what to change if something is unusual about your machine.

**You do not enter any system information.** Hardware is detected — operating system, CPU
architecture, whether there is an NVIDIA GPU or Apple Silicon, how much RAM and VRAM, and
whether that memory is unified. The wizard asks only for *preferences*: which search backend
(it recommends one), which topics matter to you (only if your machine needs the corpus
trimmed), and which model to generate with.

---

## The short version

```bash
git clone https://github.com/JBedichek/Local-arxiv-Research-Assistant.git
cd Local-arxiv-Research-Assistant

python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e '.[cpu]'                                 # see step 2 for the GPU variant

lara dataset pull --tiers core                          # ~51 GB, resumable, no account
lara setup                                              # interactive; writes config.local.yaml
lara serve                                              # reader on http://127.0.0.1:8080
```

Everything below is detail.

---

## 1. Requirements

- **Python 3.12+**
- **Disk**: 51 GB for `core`, 96 GB with `full`. Check with `df -h .` before starting.
- **RAM**: 8 GB works with a trimmed corpus; 32 GB+ runs everything untouched. Step 4
  measures your machine and tells you which case you are in.
- **A GPU is optional.** Search runs fine on CPU (~12 ms). Only *building* an index from
  scratch really wants one, and pulling the prebuilt corpus skips that entirely.

---

## 2. Install

```bash
git clone https://github.com/JBedichek/Local-arxiv-Research-Assistant.git
cd Local-arxiv-Research-Assistant
python3 -m venv .venv && source .venv/bin/activate
```

Pick the extra that matches your hardware:

```bash
pip install -e '.[cpu]'      # macOS, or any machine without an NVIDIA GPU
pip install -e '.[gpu]'      # NVIDIA, CUDA 12
```

The only difference is which faiss build comes along. **Both work everywhere** — the default
search backend is a plain torch matmul, and faiss is an opt-in alternative (step 4). If you
are unsure, `[cpu]` is the safe choice.

### Generating answers

Retrieval works immediately after step 3. To get written answers you also need a local LLM
server. This is the one place the platform matters:

**NVIDIA** — vLLM, installed separately because it is large and CUDA-version-sensitive:

```bash
pip install vllm
lara serve-llm                       # reads your configured model
```

**Apple Silicon** — vLLM has no Metal backend; on a Mac it would run CPU-only and leave the
GPU idle. Use any Metal-accelerated runtime that speaks the OpenAI API:

```bash
brew install llama.cpp
llama-server -hf Qwen/Qwen3-8B-GGUF --port 8000
```

Ollama and LM Studio work equally well. **You do not need to configure this** — step 4 probes
the usual ports and adopts whatever it finds. Prefer GGUF builds on a Mac; that is what the
Metal kernels are tuned for.

---

## 3. Download the corpus

```bash
lara dataset pull --tiers core
```

No account, no token, no rate limit. Resumable — re-run after an interruption and it picks up
where it stopped.

| tier | size | what it adds |
|---|---|---|
| `core` | ~51 GB | paper text, BM25 index, citations, and the search vectors. **Start here.** |
| `full` | +45 GB | fp16 vectors for exact rescoring — slightly better ranking |
| `archive` | +38 GB | raw crawled HTML; only useful if you intend to re-parse |

```bash
lara dataset pull --tiers core,full          # both
lara dataset pull --list                     # see what is in the repo without downloading
```

The corpus is **1,011,039 papers harvested / 377,093 in scope, 28.7 M chunks, 7.2 M citation
edges**, covering `cs.LG`, `cs.CL`, `stat.ML` and `cs.NE` from 2015 on.

> The upload is still in progress at the time of writing. If `pull` reports *"nothing matches
> tier core"*, the files are not published yet — `--list` shows what is actually there.

Building it yourself instead is documented in [`docs/setup/scraping_from_scratch.md`](docs/setup/scraping_from_scratch.md).
It takes about a day and needs a GPU.

---

## 4. Run the wizard

```bash
lara setup
```

It walks five steps and writes `config.local.yaml`. The server reads that file on **every**
start, so this is a one-time step.

**Nothing about your hardware is asked for.** Detected automatically:

| detected | how |
|---|---|
| OS and CPU architecture | `platform` |
| NVIDIA GPUs, and each card's VRAM | `nvidia-smi` |
| Apple Silicon / Metal | `torch.backends.mps` |
| total and usable RAM | `sysconf` / `sysctl` |
| unified vs discrete memory | inferred from the above |
| a generator already running | probes `:8000`, `:11434`, `:1234`, `:8080` |

What it actually asks you:

1. **Search backend** — a table of every option with *your* memory cost, measured latency and
   recall, one row marked as the recommendation. Press enter to accept it.
2. **Topics** — only if your machine needs the corpus trimmed (see below). Otherwise skipped.
3. **Model** — cached models that fit, or the running server it found.

Useful variants:

```bash
lara setup --show                # print the plan and the config it would write; changes nothing
lara setup --non-interactive     # accept every recommendation
lara setup --prefer speed        # or: memory, balanced (default)
```

Re-running is safe. It **replaces only the settings it manages and preserves everything else**,
and backs up the previous file first.

### Search backends

Measured on real corpus vectors — reproduce with `lara bench-index`:

| option | p50 | recall | full corpus |
|---|---|---|---|
| torch fp16 (CUDA) | 1.1 ms | 1.000 | 15.1 GB |
| torch int8 (CUDA/MPS) | 8.2 ms | 0.996 | 7.5 GB |
| torch fp16 (CPU) | 12.3 ms | 0.995 | 15.1 GB |
| faiss hnsw | 2.1 ms | 0.979 | 37.9 GB |

`int8` halves memory for 0.4 % recall **on a GPU**; on CPU the same option is 25× slower, so
the wizard will not offer it there. faiss is worth choosing only for HNSW's latency, and only
if you can spare the memory.

### If your machine is small

**No backend fits all 28.7 M chunks on a laptop** — the smallest usable index is 7.5 GB before
models. So for machines under about 24 GB the wizard recommends keeping only the topics you
care about resident:

```bash
lara corpus scope -t "data selection for language models" \
                  -t "optimizers and learning rate schedules" \
                  --keep 0.1 --preview      # shows the cut line, writes nothing
lara corpus scope -t "..." --keep 0.1 --apply
```

At `keep=0.1` that is **1.5 GB instead of 15 GB**. Nothing is deleted: dropped papers stay
fully searchable through BM25 and open normally — only their dense vectors leave RAM. Papers
heavily cited by the ones you kept are pulled in automatically, so foundational work does not
fall through the cracks.

**This is a recommendation, never automatic.** A 32 GB or 64 GB Mac is told it does not need
it, and keeps the whole corpus.

---

## 5. Start

```bash
lara serve                        # http://127.0.0.1:8080
lara serve --host 0.0.0.0         # reachable from your network — read the warning below
```

Open a paper by arXiv id, or type anything else to search. Highlight a passage and click
**Ask about this** for the core interaction.

> **No endpoint is authenticated.** Binding to `0.0.0.0` exposes the corpus, the model picker
> and the generator to anything that can reach the host. Fine on a home network; put a reverse
> proxy with auth in front on a shared one.

---

## Verifying and fixing

```bash
lara preflight        # disks, paths, GPU, and which config layers loaded
lara status           # corpus counts
lara bench-index      # measure the search backends yourself
```

`lara preflight` is the first thing to run when something is wrong. It reports what it
checked and, where it can, how to fix it.

**Configuration lives in two files.** `config.yaml` is tracked in git and holds portable
defaults; `config.local.yaml` is written by `lara setup`, describes *your* machine, and is
gitignored. The second merges over the first, so editing it is how you change anything
permanently. Deleting it returns you to the defaults.

| symptom | cause |
|---|---|
| `nvidia-smi` works but everything is slow | torch cannot see the GPU — preflight fails on this explicitly |
| `Driver/library version mismatch` | reload the kernel module, or reboot; preflight prints the command |
| Out of memory at startup | the index does not fit — re-run `lara setup`, or scope the corpus |
| Answers 404 | the selected model is not the one the generator has loaded |
| `lara serve-llm` fails on a Mac | expected; vLLM has no Metal backend — use llama.cpp or Ollama |

More detail: [`docs/setup/device_detection.md`](docs/setup/device_detection.md) for hardware
and backends, [`docs/UI_guide/guide_layman.md`](docs/UI_guide/guide_layman.md) for the reader
itself.
