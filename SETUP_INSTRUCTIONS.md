# Setup

> **Copyright © 2026 James Bedichek. All rights reserved.** This is proprietary software,
> not open source. These instructions describe how to run it; they do not grant permission
> to use it. Copying, modifying, redistributing, or commercial use require prior written
> permission — see [`LICENSE`](LICENSE).


From nothing to a running reader. Four commands do the work; the rest of this file explains
what they do and what to change if something is unusual about your machine.

**You do not enter any system information.** Hardware is detected — operating system, CPU
architecture, whether there is an NVIDIA GPU or Apple Silicon, how much RAM and VRAM, and
whether that memory is unified. The wizard asks only for *preferences*: which search backend
(it recommends one), which topics matter to you (only if your machine needs the corpus
trimmed), and which model to generate with.

---

## Getting access (do this first)

**Two separate accounts are involved, and neither is optional**: Hugging Face for the
embedding model, GitHub for this repository. Start with Hugging Face — that one is approved
by someone else, so it is the step you cannot rush.

### Hugging Face — the embedding model is gated

The query encoder is [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m),
a **gated** repository: you must accept Google's terms and be granted access before it can
be downloaded. There is no way around this and no substitute model configured — **every
search embeds its query with this model**, so until access is granted the reader cannot
search at all. It is not only needed for building an index.

1. Create an account: <https://huggingface.co/join>
2. Open the model card: <https://huggingface.co/google/embeddinggemma-300m>
3. Click **Acknowledge license** and complete the form.

The Hub reports this repository's gating as `manual` — requests are reviewed rather than
auto-approved on accepting the terms. It is usually quick, but **start it before you need
it** rather than discovering it at a prompt. The model card reads "You have been granted
access" once it is done, and you get an email.

Authenticating the CLI comes later, in [step 2](#2-install) — the `hf` command ships with
the project's dependencies, so it does not exist until after `pip install`. (If you want to
do it sooner, `brew install hf` installs it standalone.)

### GitHub — this repository

This repository is **private**. On a new machine, two commands are all you need — **no SSH
key, no Personal Access Token to manage by hand**:

```bash
gh auth login --git-protocol https --web
gh repo clone JBedichek/Local-arxiv-Research-Assistant
```

The first opens a browser, you approve once, and the token is saved to your system
credential store. The second clones using it.

You need [GitHub CLI](https://cli.github.com) (`brew install gh`, `winget install
GitHub.cli`, or your package manager) and access granted by James Bedichek.

> **The one thing that catches people:** `git clone https://github.com/…` fails with
> `could not read Username` unless git has a credential helper. Being signed in to
> github.com **in a browser does nothing for git**, and neither does `gh auth login` on its
> own if you chose `ssh` at the protocol prompt. `--git-protocol https` above is what wires
> git; `gh auth setup-git` fixes it afterwards if you already logged in with `ssh`. Since
> 2021 GitHub rejects account passwords for git, so a password prompt wants a Personal
> Access Token.

Already using SSH keys, or on a headless machine? See [Access](#access) below.

---

## The short version

Pick your platform. Only the install lines differ — everything from `hf auth login` onward
is identical everywhere.

**Every block on this page is comment-free and safe to paste whole.** What each line does is
in the table underneath.

**macOS — zsh**

```zsh
gh repo clone JBedichek/Local-arxiv-Research-Assistant
cd Local-arxiv-Research-Assistant

python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[mac]'

hf auth login
lara dataset pull --tiers core
lara setup
lara serve
```

**Linux — bash**

```bash
gh repo clone JBedichek/Local-arxiv-Research-Assistant
cd Local-arxiv-Research-Assistant

python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[cuda]'

hf auth login
lara dataset pull --tiers core
lara setup
lara serve
```

**Windows — PowerShell**

```powershell
gh repo clone JBedichek/Local-arxiv-Research-Assistant
cd Local-arxiv-Research-Assistant

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e '.[cpu]'

hf auth login
lara dataset pull --tiers core
lara setup
lara serve
```

| line | what it does |
|---|---|
| `pip install -e '.[mac]'` | Apple Silicon — adds faiss-cpu and mlx-lm. **The quotes are required**: zsh expands `[...]` as a glob |
| `pip install -e '.[cuda]'` | NVIDIA, CUDA 12. No GPU? use the CPU install in [step 2](#2-install) |
| `pip install -e '.[cpu]'` | Linux without a GPU, and **all** Windows — yes, even with an NVIDIA card; see the note below |
| `.\.venv\Scripts\Activate.ps1` | Refuses to run? That is PowerShell's execution policy — see the note below |
| `hf auth login` | Hugging Face token for the gated embedder. Access must already be granted |
| `lara dataset pull --tiers core` | ~50 GB, resumable, no account needed for the corpus itself |
| `lara setup` | Interactive; writes `config.local.yaml` |
| `lara serve` | Reader on <http://127.0.0.1:8080> |

**`dataset pull` and `setup` both stop immediately if the embedder is out of reach**, so a
missing Hugging Face signup costs you a second rather than a 50 GB download. `lara preflight`
reports the same thing on demand.

> **Why no `#` comments in any of these blocks.** zsh does not treat `#` as a comment
> interactively — `interactive_comments` is off by default, unlike bash. A pasted line with a
> trailing `# …` hands those words to the command as arguments, which surfaces as errors like
> `fatal: Too many arguments.` If you want commented recipes from elsewhere to paste cleanly:
>
> ```zsh
> echo 'setopt interactive_comments' >> ~/.zshrc && exec zsh
> ```
>
> Full diagnosis in [`SETUP_TROUBLESHOOTING.md`](SETUP_TROUBLESHOOTING.md).

> **Three things only Windows users hit.**
>
> 1. **`Activate.ps1` refuses to run.** That is PowerShell's execution policy, not a broken
>    venv. `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` lifts it for the
>    current session and nothing wider.
> 2. **`'.[cpu]'` is right even on an NVIDIA machine** — but on its own it leaves you on the
>    CPU. The `cuda` extra exists only to add `faiss-gpu-cu12`, which ships Linux wheels
>    alone, and skipping faiss costs nothing because the default search backend is torch and
>    faiss is opt-in everywhere. The catch is torch itself: **PyPI's default wheel is
>    CPU-only on Windows**, where on Linux it bundles CUDA. To use your GPU, install torch
>    from PyTorch's CUDA index *before* the extra — the selector at
>    <https://pytorch.org/get-started/locally/> gives the exact URL for your driver:
>    ```powershell
>    pip install torch --index-url https://download.pytorch.org/whl/cu124
>    pip install -e '.[cpu]'
>    ```
> 3. **Generation is the real gap.** vLLM is Linux-only and MLX is Apple-only, so nothing
>    autostarts. Run Ollama or LM Studio yourself and `lara setup` adopts whichever is
>    already answering.

Everything below is detail.

---

## 1. Requirements

### Access

The two-command path is at the top of this file. The alternatives, if it does not suit you:

| situation | what to run |
|---|---|
| You already use SSH keys | `git clone git@github.com:JBedichek/Local-arxiv-Research-Assistant.git` |
| Logged in with `ssh` and want HTTPS too | `gh auth setup-git` |
| Headless box, no browser | `gh auth login --with-token < token.txt` |
| No `gh`, no SSH | clone over HTTPS and paste a Personal Access Token at the password prompt |

A Personal Access Token needs the `repo` scope to reach a private repository.

- **A Hugging Face account with access to `google/embeddinggemma-300m`** — gated, granted by
  request, and required for search to work at all. Start this first; see
  [Getting access](#getting-access-do-this-first).
- **Python 3.12+**
- **Disk**: 50 GB for `core`, 95 GB with `full`. Check with `df -h .`, or
  `Get-PSDrive C` in PowerShell, before starting.
- **RAM**: 8 GB works with a trimmed corpus; 32 GB+ runs everything untouched. Step 4
  measures your machine and tells you which case you are in.
- **A GPU is optional.** Search runs fine on CPU (~12 ms). Only *building* an index from
  scratch really wants one, and pulling the prebuilt corpus skips that entirely.

---

## 2. Install

```bash
gh repo clone JBedichek/Local-arxiv-Research-Assistant
cd Local-arxiv-Research-Assistant
python3 -m venv .venv && source .venv/bin/activate
```

Pick exactly one platform extra:

```zsh
pip install -e '.[mac]'
pip install -e '.[cuda]'
pip install -e '.[cpu]'
```

| extra | for |
|---|---|
| `mac` | Apple Silicon — adds faiss-cpu and mlx-lm |
| `cuda` | NVIDIA, CUDA 12 |
| `cpu` | Linux with no GPU, and all Windows — see the note above |

> **On macOS, the quotes are load-bearing.** zsh has been the default shell since Catalina,
> and it expands `[...]` as a glob before pip ever sees it. Unquoted, `pip install -e .[mac]`
> dies with `zsh: no matches found: .[mac]` — where bash would have passed the string
> through untouched. Quote every extra on this page, and the command works in both shells.

**Linux/Windows without a GPU: install torch from the CPU index first.** PyPI's default
torch wheel drags in **~5.4 GB of CUDA runtime libraries** that a machine with no NVIDIA
card can never use:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[cpu]'
```

macOS is unaffected — the arm64 wheels carry no CUDA payload.

Optional extras, only if you need them:

```bash
pip install -e '.[ingest]'
pip install -e '.[vllm]'
```

| extra | for |
|---|---|
| `ingest` | crawling and PDF parsing — only needed to **build** a corpus |
| `vllm` | NVIDIA generation backend; large, and sensitive to your CUDA version |

The base install is deliberately just what it takes to **search and read** a corpus that
already exists, which is what most people do.

### Authenticate with Hugging Face

You requested access to [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m)
back in [Getting access](#getting-access-do-this-first). Now log in so the download can use
it. `hf` was installed with the dependencies above, so it is on your PATH inside the venv:

```bash
hf auth login
```

Paste an access token when prompted — create one at
<https://huggingface.co/settings/tokens>. A **Read** token is sufficient; the fine-grained
kind works too provided it has *Read access to contents of public gated repos*. The token is
written to `~/.cache/huggingface/token` and every later download picks it up.

> **The command is `hf`, not `huggingface-cli`.** From `huggingface-hub` 1.0 the old
> entrypoint is deprecated and **no longer functional** — it exits with
> `` `huggingface-cli` is deprecated and no longer works. Use `hf` instead. `` Guides
> written before mid-2025 all use the old name. This project pins `huggingface-hub>=0.26`,
> so which one you get depends on what pip resolved; `hf --version` settles it.

Confirm the login and that gating actually cleared:

```bash
hf auth whoami
hf download google/embeddinggemma-300m
```

`hf auth whoami` prints your username; "Not logged in" means the token did not stick.
`hf download` pulls ~1.2 GB into `~/.cache/huggingface`.

That download is optional — `lara` fetches the model on first use anyway, and
`lara preflight` verifies access without downloading anything. Pulling it now simply gets
the bytes out of the way while you are paying attention.

| what you see | what it means |
|---|---|
| `401 Client Error` / `Not logged in` | no token — run `hf auth login` |
| `GatedRepoError` / `403 Forbidden` | logged in, but access not granted yet — check the model card |
| `Repository Not Found` | typo in the model id, or a token with no gated-repo scope |
| `You have been granted access` on the model card | you are clear to proceed |

**Access is per-account, not per-machine.** Once Google grants it, any machine you log into
with that account can pull the model; you do not repeat the request.

> **`lara preflight` does not currently check this.** It verifies disks, paths and GPUs but
> not Hugging Face auth, so a missing token surfaces later as a failure on the first search
> rather than up front.

### Generating answers

Retrieval works immediately after step 3. Written answers need an inference server, and
**`lara serve` starts and stops it for you** — you do not run it by hand.

| backend | platform | model format | install |
|---|---|---|---|
| **vLLM** | NVIDIA | HF safetensors | `pip install -e '.[vllm]'` |
| **MLX** | Apple Silicon only | `mlx-community/*` | comes with `'.[mac]'` |
| **llama.cpp** | anywhere | GGUF | `brew install llama.cpp` |
| external | anywhere | whatever it serves | run Ollama/LM Studio yourself |

**vLLM has no Metal backend** — on a Mac it runs CPU-only and leaves the GPU idle. That is
why Apple Silicon gets two purpose-built options instead.

**On a Mac, try both.** MLX uses unified memory directly and is often faster; llama.cpp is
more mature and has richer KV-cache options. Compare them on your own hardware:

`lara backends` lists what is installed and what would be used. Then start one backend and,
**in a second shell**, benchmark it — `bench-generate` reports median TTFT and tok/s:

```zsh
lara backends
lara serve-llm --backend mlx
lara bench-generate
lara serve-llm --backend llamacpp
lara bench-generate
```

> **The three formats are not interchangeable.** A repo vLLM serves is not a GGUF file and
> is not MLX-converted, so each backend has its own `model` setting in `config.local.yaml`.
> Pointing llama.cpp at a safetensors repo fails at load time.

**Where to find each format.** `lara models` prints which one your machine needs, and the
reader's download dialog links to these directly:

| format | for | where |
|---|---|---|
| **GGUF** | llama.cpp, Ollama — **the default on Apple Silicon** | [all GGUF models](https://huggingface.co/models?library=gguf), or [bartowski](https://huggingface.co/bartowski) and [unsloth](https://huggingface.co/unsloth), who build them for most popular models. Take a `Q4_K_M` file to match the 4-bit sizing the wizard quotes |
| **MLX** | Apple Silicon, opt-in via `--backend mlx` | [mlx-community](https://huggingface.co/mlx-community) — look for a `-4bit` suffix |
| **safetensors** | vLLM (NVIDIA) | ordinary model pages; this is what most repos ship |

On a Mac, **a plain safetensors repo will not load at all** — that includes the model page
you get from searching Hugging Face for a model by name. Look for the GGUF build of it.

If you already run Ollama or LM Studio, do nothing: the wizard probes the usual ports and
adopts whatever is already answering. A server lara did not start is never stopped by it.

---

## 3. Download the corpus

```bash
lara dataset pull --tiers core
```

No account, no token, no rate limit. Resumable — re-run after an interruption and it picks up
where it stopped.

| tier | size | what it adds |
|---|---|---|
| `core` | ~50 GB | paper text, BM25 index, citations, and the search vectors. **Start here.** |
| `full` | +45 GB | fp16 vectors for exact rescoring — slightly better ranking |
| `archive` | +40 GB | raw crawled HTML; only useful if you intend to re-parse |

**`core` runs on its own.** It ships the int8 vectors only, so tier-2 rescore falls back to
those instead of the 768-d fp16 file. Measured against a `full` install on the same queries,
that costs about 15 % of the top-8 result set and no latency — adding `full` sharpens the
ranking rather than switching search on.

**`archive` needs ~80 GB transiently.** The raw HTML ships as one tar per year (368 k loose
files is past the Hub's per-repo limit), and `pull` unpacks them into the layout the parser
reads. Both forms are on disk until you remove the tars, which the command tells you how to
do; `--no-extract` skips unpacking and prints the manual command instead.

```bash
lara dataset pull --tiers core,full
lara dataset pull --list
```

The first pulls both tiers. The second lists what is in the repo without downloading it —
and, being read-only, is the one `dataset pull` variant that does **not** require Hugging
Face access to the embedder first.

The corpus is **1,011,039 papers harvested / 377,093 in scope, 28.7 M chunks, 7.2 M citation
edges**, covering `cs.LG`, `cs.CL`, `stat.ML` and `cs.NE` from 2015 on.

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
lara setup --show
lara setup --non-interactive
lara setup --prefer speed
```

| variant | effect |
|---|---|
| `--show` | print the plan and the config it would write; changes nothing, and skips the Hugging Face gate so it works before access is granted |
| `--non-interactive` | accept every recommendation |
| `--prefer speed` | or `memory`, or `balanced` (the default) |

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
                  --keep 0.1 --preview
lara corpus scope -t "..." --keep 0.1 --apply
```

`--preview` shows you the cut line and writes nothing; `--apply` commits it.

At `keep=0.1` that is **1.5 GB instead of 15 GB**. Nothing is deleted: dropped papers stay
fully searchable through BM25 and open normally — only their dense vectors leave RAM. Papers
heavily cited by the ones you kept are pulled in automatically, so foundational work does not
fall through the cracks.

**This is a recommendation, never automatic.** A 32 GB or 64 GB Mac is told it does not need
it, and keeps the whole corpus.

---

## 5. Start

```bash
lara serve
lara serve --no-llm
lara serve --host 0.0.0.0
```

| variant | effect |
|---|---|
| *(no flags)* | starts the generator too, then the reader on `:8080` |
| `--no-llm` | retrieval only; skip the generator |
| `--host 0.0.0.0` | reachable from your network — **read the warning below first** |

The generation server starts alongside the reader and is shut down with it. If one is
already running, lara adopts it and leaves it alone on exit — it may be shared with
something else, and that is not the reader's call to make.

Open a paper by arXiv id, or type anything else to search. Highlight a passage and click
**Ask about this** for the core interaction.

> **No endpoint is authenticated.** Binding to `0.0.0.0` exposes the corpus, the model picker
> and the generator to anything that can reach the host. Fine on a home network; put a reverse
> proxy with auth in front on a shared one.

---

## Verifying and fixing

```bash
lara preflight
lara status
lara bench-index
```

| command | reports |
|---|---|
| `lara preflight` | disks, paths, GPU, Hugging Face access, and which config layers loaded |
| `lara status` | corpus counts |
| `lara bench-index` | measures the search backends on your own hardware |

`lara preflight` is the first thing to run when something is wrong. It reports what it
checked and, where it can, how to fix it.

**Configuration lives in two files.** `config.yaml` is tracked in git and holds portable
defaults; `config.local.yaml` is written by `lara setup`, describes *your* machine, and is
gitignored. The second merges over the first, so editing it is how you change anything
permanently. Deleting it returns you to the defaults.

| symptom | cause |
|---|---|
| `GatedRepoError` on the first search | access to `embeddinggemma-300m` not granted yet — see [Authenticate with Hugging Face](#authenticate-with-hugging-face) |
| `401` fetching the embedder | not logged in — `hf auth login` |
| `huggingface-cli: deprecated and no longer works` | use `hf`; the old entrypoint was removed in `huggingface-hub` 1.0 |
| `fatal: Too many arguments.` pasting a command | zsh does not strip `#` comments — see [`SETUP_TROUBLESHOOTING.md`](SETUP_TROUBLESHOOTING.md) |
| `nvidia-smi` works but everything is slow | torch cannot see the GPU — preflight fails on this explicitly |
| `Driver/library version mismatch` | reload the kernel module, or reboot; preflight prints the command |
| Out of memory at startup | the index does not fit — re-run `lara setup`, or scope the corpus |
| Answers 404 | the selected model is not the one the generator has loaded |
| `lara serve-llm` fails on a Mac | expected; vLLM has no Metal backend — use llama.cpp or Ollama |

More detail: [`docs/setup/device_detection.md`](docs/setup/device_detection.md) for hardware
and backends, [`docs/UI_guide/guide_layman.md`](docs/UI_guide/guide_layman.md) for the reader
itself.
