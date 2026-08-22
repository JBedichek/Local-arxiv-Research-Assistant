# Setup

> **Copyright © 2026 James Bedichek. All rights reserved.** This is proprietary software,
> not open source. These instructions describe how to run it; they do not grant permission
> to use it. Copying, modifying, redistributing, or commercial use require prior written
> permission — see [`LICENSE`](LICENSE).

From nothing to a running reader in four commands. **You never enter system information** —
OS, CPU, GPU, RAM and whether memory is unified are all detected. The wizard asks only for
preferences: search backend, topics (only if your machine needs the corpus trimmed), and
which model generates.

> **Brand-new Mac?** It has no Homebrew, no Python 3.12 and no usable `git`, and it hides
> this well — `/usr/bin/git` and `/usr/bin/python3` both answer `which` while being installer
> stubs. [`SETUP_MACOS_BOOTSTRAP.md`](SETUP_MACOS_BOOTSTRAP.md) covers that layer and hands
> back here at [step 2](#2-install). Everything below assumes it is done.

---

## Getting access (do this first)

Two accounts, neither optional. **Start with Hugging Face** — someone else approves it, so it
is the one you cannot rush.

### Hugging Face — the embedder is gated

The query encoder [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m)
is **gated**, with no substitute configured. **Every search embeds its query with it**, so
until access is granted the reader cannot search at all — this is not only an indexing
requirement.

1. Sign up: <https://huggingface.co/join>
2. Open the [model card](https://huggingface.co/google/embeddinggemma-300m)
3. Click **Acknowledge license** and complete the form.

Gating is `manual` — reviewed, not auto-approved. Usually quick, but **start it before you
need it**. You get an email, and the card reads "You have been granted access".

Logging in comes later, in [step 2](#2-install): `hf` ships with the dependencies, so it does
not exist until after `pip install`.

### GitHub — clone and push

**This repository is public** — cloning needs no account, token or key. You need credentials
only to **push**. Two commands set that up, with no SSH key or Personal Access Token to
manage by hand:

```zsh
brew install gh
gh auth login --git-protocol https --web
```

`gh auth login` opens a browser; approve once and the token goes to your system credential
store. Confirm with `gh auth status`.

On other platforms `gh` is `winget install GitHub.cli`, or your package manager — see
[cli.github.com](https://cli.github.com).

**Install `gh` even if you prefer SSH keys**: every clone line in this file is written as
`gh repo clone`, and on a new machine that command does not exist.

> **`--git-protocol https` is load-bearing.** Choosing `ssh` at that prompt leaves git with
> no credential helper, and HTTPS clones then fail with `could not read Username` even though
> `gh` reports you as logged in. Being signed in to github.com **in a browser does nothing for
> git**. `gh auth setup-git` repairs it after the fact. Since 2021 GitHub rejects account
> passwords, so a git password prompt is asking for a token.

| instead of the above | run |
|---|---|
| You use SSH keys | `git clone git@github.com:JBedichek/Local-arxiv-Research-Assistant.git` |
| Logged in with `ssh`, want HTTPS too | `gh auth setup-git` |
| Headless, no browser | `gh auth login --with-token < token.txt` |
| No `gh`, no SSH, read-only | `git clone https://github.com/JBedichek/Local-arxiv-Research-Assistant.git` |

A Personal Access Token used to push needs the `repo` scope.

---

## The short version

Only the venv and install lines differ by platform; everything from `hf auth login` on is
identical. **Every block here is comment-free and safe to paste whole.**

**macOS — zsh**

```zsh
gh repo clone JBedichek/Local-arxiv-Research-Assistant
cd Local-arxiv-Research-Assistant
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e '.[mac]'
brew install llama.cpp

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
```

**Windows — PowerShell**

```powershell
gh repo clone JBedichek/Local-arxiv-Research-Assistant
cd Local-arxiv-Research-Assistant
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e '.[cpu]'
```

| line | what it does |
|---|---|
| `python3.12 -m venv` | **macOS: use the version by name.** System `python3` is 3.9.6 and Homebrew does not displace it |
| `pip install -e '.[mac]'` | Apple Silicon — faiss-cpu and mlx-lm. **Quotes required**: zsh globs `[...]` |
| `brew install llama.cpp` | **the default generator on Apple Silicon.** A binary, so pip cannot supply it |
| `pip install -e '.[cuda]'` | NVIDIA, CUDA 12 |
| `pip install -e '.[cpu]'` | Linux without a GPU, and **all** Windows — even with an NVIDIA card |
| `hf auth login` | token for the gated embedder; access must already be granted |
| `lara dataset pull --tiers core` | ~50 GB, resumable, no account needed |
| `lara setup` | interactive; writes `config.local.yaml` |
| `lara serve` | reader on <http://127.0.0.1:8080> |

**`dataset pull` and `setup` both stop immediately if the embedder is out of reach**, so a
missing signup costs a second rather than a 50 GB download.

> **Why no `#` comments in any block.** zsh does not treat `#` as a comment interactively —
> `interactive_comments` is off by default, unlike bash. A pasted trailing `# …` becomes
> arguments, surfacing as `fatal: Too many arguments.` To make commented recipes paste
> cleanly: `echo 'setopt interactive_comments' >> ~/.zshrc && exec zsh`. Full diagnosis in
> [`SETUP_TROUBLESHOOTING.md`](SETUP_TROUBLESHOOTING.md).

> **Three things only Windows hits.**
> 1. **`Activate.ps1` refuses to run** — execution policy, not a broken venv.
>    `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` lifts it for that session.
> 2. **`'.[cpu]'` is right even on an NVIDIA machine.** The `cuda` extra only adds
>    `faiss-gpu-cu12`, which is Linux-only, and skipping faiss costs nothing since torch is
>    the default backend. But **PyPI's torch wheel is CPU-only on Windows**, so for GPU use
>    install torch from the CUDA index *first* — get the URL from
>    <https://pytorch.org/get-started/locally/>:
>    `pip install torch --index-url https://download.pytorch.org/whl/cu124`
> 3. **Generation is the real gap.** vLLM is Linux-only, MLX Apple-only, so nothing
>    autostarts. Run Ollama or LM Studio and `lara setup` adopts it.

Everything below is detail.

---

## 1. Requirements

| | |
|---|---|
| **Hugging Face** | account with access to `google/embeddinggemma-300m`. Required for search to work at all — [start it first](#getting-access-do-this-first) |
| **Python** | **3.12+**. macOS ships 3.9.6, which is not close enough to limp along on |
| **Disk** | 50 GB for `core`, 95 GB with `full`. Check with `df -h .` |
| **RAM** | 8 GB works with a trimmed corpus; 32 GB+ runs everything untouched |
| **GPU** | optional. Search runs fine on CPU (~12 ms); only *building* an index wants one |
| **llama.cpp** | macOS: `brew install llama.cpp`. The default generator here, and pip cannot install it |

---

## 2. Install

```bash
gh repo clone JBedichek/Local-arxiv-Research-Assistant
cd Local-arxiv-Research-Assistant
```

Then create the venv with the interpreter **named by version**:

| platform | command | why |
|---|---|---|
| **macOS** | `python3.12 -m venv .venv` | bare `python3` is the system **3.9.6** — too old, and it fails confusingly |
| **Linux** | `python3 -m venv .venv` | distro `python3` is normally already 3.12+. Check with `python3 --version`; if it is older, name the version — `python3.12` |
| **Windows** | `py -3.12 -m venv .venv` | the `py` launcher selects the version explicitly |

Activate it — `source .venv/bin/activate` on macOS and Linux,
`.\.venv\Scripts\Activate.ps1` on Windows.

Then **verify before installing** — one command, and it saves the whole detour below:

```zsh
python --version
```

It must print **3.12 or newer**. If it prints 3.9, delete `.venv` and rebuild it with
`python3.12` by name:

```zsh
deactivate
rm -rf .venv
python3.12 -m venv .venv && source .venv/bin/activate
```

> **Why bare `python3` is wrong on macOS.** `brew install python@3.12` deliberately does not
> displace the system 3.9.6, so `python3` keeps resolving to the old one and builds the venv
> from an interpreter this project cannot use.
>
> **The error you get does not mention Python versions at all.** It lands one command later,
> at `pip install -e`, and blames the *project*:
>
> ```
> ERROR: File "setup.py" or "setup.cfg" not found. Directory cannot be installed in
> editable mode: /path/to/Local-arxiv-Research-Assistant
> (A "pyproject.toml" file was found, but editable mode currently requires a
> setuptools-based build.)
> ```
>
> Nothing is wrong with the project. Python 3.9 ships **pip 21.2.4**, which predates
> [PEP 660](https://peps.python.org/pep-0660/) — editable installs for non-setuptools
> backends arrived in pip 21.3. This project builds with hatchling, so that pip cannot
> install it editable and reaches for `setup.py`, which correctly does not exist. It never
> gets far enough to check `requires-python`.
>
> **The version in the message is the tell**: a `WARNING: You are using pip version 21.2.4`
> means the venv is 3.9, whatever you thought you ran. Do not fix this by upgrading pip —
> that patches the symptom onto an interpreter still too old for the project. Rebuild the
> venv. After activating, `python --version` must say 3.12 or newer.
>
> **`deactivate` first if any venv is active.** Inside an activated 3.9 venv, `python3` *is*
> that venv's 3.9, so a nested `python3 -m venv` inherits it and the mistake propagates.

Pick exactly one platform extra:

| extra | for |
|---|---|
| `pip install -e '.[mac]'` | Apple Silicon — adds faiss-cpu and mlx-lm |
| `pip install -e '.[cuda]'` | NVIDIA, CUDA 12 |
| `pip install -e '.[cpu]'` | Linux with no GPU, and all Windows |

> **On macOS the quotes are load-bearing.** zsh expands `[...]` as a glob before pip sees it,
> so unquoted `pip install -e .[mac]` dies with `zsh: no matches found: .[mac]`. Quoting works
> in both shells.

**Linux/Windows without a GPU: install torch from the CPU index first**, or PyPI's default
wheel drags in **~5.4 GB of CUDA runtime** the machine can never use. macOS is unaffected —
arm64 wheels carry no CUDA payload.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[cpu]'
```

Optional extras: `'.[ingest]'` for crawling and PDF parsing (only to **build** a corpus), and
`'.[vllm]'` for the NVIDIA generation backend (large, sensitive to your CUDA version). The
base install is deliberately just enough to **search and read** an existing corpus.

### Authenticate with Hugging Face

`hf` is on your PATH inside the venv now:

```bash
hf auth login
hf auth whoami
```

Paste a token from <https://huggingface.co/settings/tokens>. **Read** is sufficient;
fine-grained works with *Read access to contents of public gated repos*. It is written to
`~/.cache/huggingface/token`. `hf auth whoami` printing "Not logged in" means it did not
stick.

Optionally pull the ~1.2 GB model now rather than on first use:
`hf download google/embeddinggemma-300m`.

> **The command is `hf`, not `huggingface-cli`.** From `huggingface-hub` 1.0 the old
> entrypoint is **no longer functional**. Guides written before mid-2025 all use the old
> name; `hf --version` settles which you have.

| what you see | what it means |
|---|---|
| `401` / `Not logged in` | no token — run `hf auth login` |
| `GatedRepoError` / `403` | logged in, access not granted yet — check the model card |
| `Repository Not Found` | typo, or a token with no gated-repo scope |

**Access is per-account, not per-machine** — once granted, any machine you log into can pull
the model. `lara preflight` verifies access without downloading anything.

### Generating answers

Retrieval works immediately after step 3. Written answers need an inference server, and
**`lara serve` starts and stops it for you**.

| backend | platform | format | install |
|---|---|---|---|
| **llama.cpp** | anywhere, Metal — **the default on Apple Silicon** | GGUF | `brew install llama.cpp` |
| **vLLM** | NVIDIA | safetensors | `pip install -e '.[vllm]'` |
| **MLX** | Apple Silicon | `mlx-community/*` | comes with `'.[mac]'` |
| external | anywhere | whatever it serves | run Ollama/LM Studio yourself |

> **On a Mac, llama.cpp is the default and is the one piece pip cannot install.** Without
> `brew install llama.cpp` the resolver falls back to MLX, which works — it ships with
> `'.[mac]'` — but is not what the wizard, this file, or `lara backends` describe. If
> `lara setup` reports `backend mlx` on a machine where you wanted llama.cpp, the binary is
> missing: check with `which llama-server`, since availability is decided on that alone.

**vLLM has no Metal backend** — on a Mac it runs CPU-only and leaves the GPU idle, which is
why Apple Silicon gets two purpose-built options.

**On a Mac, still try both.** llama.cpp is the default because it is the most mature and
reads GGUF, the format most local models are actually published in — but MLX uses unified
memory directly and is often faster on this hardware, so the default is a starting point
rather than a verdict. Benchmark on your own machine — start a backend, then in a **second
shell** run `lara bench-generate` for median TTFT and tok/s:

```zsh
lara backends
lara serve-llm --backend mlx
lara serve-llm --backend llamacpp
```

> **The three formats are not interchangeable.** Each backend has its own `model` setting in
> `config.local.yaml`; pointing llama.cpp at a safetensors repo fails at load time.

| format | for | where |
|---|---|---|
| **GGUF** | llama.cpp, Ollama — **default on Apple Silicon** | [GGUF models](https://huggingface.co/models?library=gguf), or [bartowski](https://huggingface.co/bartowski) / [unsloth](https://huggingface.co/unsloth). Take `Q4_K_M` to match the wizard's 4-bit sizing |
| **MLX** | Apple Silicon, via `--backend mlx` | [mlx-community](https://huggingface.co/mlx-community) — look for `-4bit` |
| **safetensors** | vLLM (NVIDIA) | ordinary model pages |

On a Mac **a plain safetensors repo will not load at all** — including the page you land on
searching Hugging Face by name. Look for the GGUF build. `lara models` prints which format
your machine needs.

Already running Ollama or LM Studio? Do nothing — the wizard probes the usual ports and
adopts it, and never stops a server it did not start.

---

## 3. Download the corpus

```bash
lara dataset pull --tiers core
```

No account, no token, no rate limit. Resumable — re-run and it picks up where it stopped.

| tier | size | what it adds |
|---|---|---|
| `core` | ~50 GB | paper text, BM25 index, citations, search vectors. **Start here.** |
| `full` | +45 GB | fp16 vectors for exact rescoring — slightly better ranking |
| `archive` | +40 GB | raw crawled HTML; only if you intend to re-parse |

**`core` runs on its own.** It ships int8 vectors only, so tier-2 rescore falls back to those.
Measured against `full` on the same queries that costs ~15 % of the top-8 result set and no
latency — `full` sharpens ranking rather than switching search on.

**`archive` needs ~80 GB transiently**: it ships as one tar per year and `pull` unpacks them,
so both forms are on disk until you remove the tars. `--no-extract` skips unpacking.

`lara dataset pull --list` shows what is in the repo without downloading — the one variant
that does **not** require Hugging Face access first.

The corpus is **1,011,039 papers harvested / 377,093 in scope, 28.7 M chunks, 7.2 M citation
edges**, covering `cs.LG`, `cs.CL`, `stat.ML` and `cs.NE` from 2015 on. Building it yourself
takes about a day and needs a GPU — see
[`docs/setup/scraping_from_scratch.md`](docs/setup/scraping_from_scratch.md).

---

## 4. Run the wizard

```bash
lara setup
```

Writes `config.local.yaml`, which the server reads on **every** start, so this is a one-time
step. Re-running is safe: it replaces only the settings it manages, preserves everything
else, and backs up the previous file.

**Nothing about your hardware is asked for.** OS and architecture come from `platform`,
NVIDIA cards and VRAM from `nvidia-smi`, Metal from `torch.backends.mps`, RAM from
`sysconf`/`sysctl`, and a running generator from probing `:8000`, `:11434`, `:1234`, `:8080`.

It asks only: **search backend** (recommended row pre-selected), **topics** (only if your
machine needs trimming), and **model**.

| variant | effect |
|---|---|
| `--show` | print the plan and config it would write; changes nothing, and skips the Hugging Face gate |
| `--non-interactive` | accept every recommendation |
| `--prefer speed` | or `memory`, or `balanced` (default) |

### Search backends

Measured on real corpus vectors — reproduce with `lara bench-index`:

| option | p50 | recall | full corpus |
|---|---|---|---|
| torch fp16 (CUDA) | 1.1 ms | 1.000 | 15.1 GB |
| torch int8 (CUDA/MPS) | 8.2 ms | 0.996 | 7.5 GB |
| torch fp16 (CPU) | 12.3 ms | 0.995 | 15.1 GB |
| faiss hnsw | 2.1 ms | 0.979 | 37.9 GB |

`int8` halves memory for 0.4 % recall **on a GPU**; on CPU it is 25× slower, so the wizard
will not offer it there. faiss is worth it only for HNSW's latency, and only if you can spare
the memory.

### If your machine is small

**No backend fits all 28.7 M chunks on a laptop** — the smallest usable index is 7.5 GB before
models. Under about 24 GB the wizard recommends keeping only the topics you care about
resident:

```bash
lara corpus scope -t "data selection for language models" \
                  -t "optimizers and learning rate schedules" \
                  --keep 0.1 --preview
```

`--preview` shows the cut line and writes nothing; `--apply` commits it. At `keep=0.1` that is
**1.5 GB instead of 15 GB**. Nothing is deleted — dropped papers stay fully searchable through
BM25 and open normally, only their dense vectors leave RAM. Papers heavily cited by the ones
you kept are pulled in automatically.

**This is a recommendation, never automatic.** A 32 GB or 64 GB Mac keeps the whole corpus.

---

## 5. Start

```bash
lara serve
```

| variant | effect |
|---|---|
| *(no flags)* | starts the generator, then the reader on `:8080` |
| `--no-llm` | retrieval only |
| `--host 0.0.0.0` | reachable from your network — **read the warning below** |

The generator starts alongside the reader and shuts down with it. If one is already running,
lara adopts it and leaves it alone on exit — it may be shared, and that is not the reader's
call.

Open a paper by arXiv id, or type anything else to search. Highlight a passage and click
**Ask about this** for the core interaction.

> **No endpoint is authenticated.** Binding to `0.0.0.0` exposes the corpus, model picker and
> generator to anything that can reach the host. Fine at home; put an authenticating reverse
> proxy in front on a shared network.

---

## Verifying and fixing

```bash
lara preflight
```

**The first thing to run when something is wrong.** It checks disks, paths, GPU, Hugging Face
access to the embedder, and which config layers loaded, reporting how to fix what it can.
`lara status` gives corpus counts; `lara bench-index` measures backends on your hardware.

**Configuration lives in two files.** `config.yaml` is tracked in git with portable defaults;
`config.local.yaml` is written by `lara setup`, describes *your* machine, and is gitignored.
The second merges over the first, so edit it to change anything permanently — deleting it
returns you to the defaults.

| symptom | cause |
|---|---|
| `GatedRepoError` on the first search | access not granted yet — see [Authenticate with Hugging Face](#authenticate-with-hugging-face) |
| `401` fetching the embedder | not logged in — `hf auth login` |
| `huggingface-cli: deprecated` | use `hf`; removed in `huggingface-hub` 1.0 |
| `command not found: gh` | `brew install gh` — see [GitHub](#github--clone-and-push) |
| `could not read Username` | git has no credential helper — `gh auth setup-git` |
| `fatal: Too many arguments.` when pasting | zsh does not strip `#` comments — see [`SETUP_TROUBLESHOOTING.md`](SETUP_TROUBLESHOOTING.md) |
| `zsh: no matches found: .[mac]` | quote the extra: `'.[mac]'` |
| `File "setup.py" or "setup.cfg" not found` on `pip install -e` | venv built from system 3.9, whose pip 21.2.4 predates PEP 660 — rebuild with `python3.12 -m venv` |
| `WARNING: You are using pip version 21.2.4` | same cause; that pip only ships with Python 3.9 |
| `nvidia-smi` works but everything is slow | torch cannot see the GPU — preflight fails on this explicitly |
| `Driver/library version mismatch` | reload the kernel module or reboot; preflight prints the command |
| Out of memory at startup | index does not fit — re-run `lara setup`, or scope the corpus |
| Answers 404 | the selected model is not the one the generator loaded |
| `lara serve-llm` fails on a Mac | expected; vLLM has no Metal backend — use llama.cpp or Ollama |
| `lara setup` says `backend mlx` when you wanted llama.cpp | `llama-server` is not on PATH — `brew install llama.cpp`. An explicit `serving.generator.backend` in `config.local.yaml` also wins over the default, so re-run `lara setup` after installing |

More detail: [`docs/setup/device_detection.md`](docs/setup/device_detection.md) for hardware
and backends, [`docs/UI_guide/guide_layman.md`](docs/UI_guide/guide_layman.md) for the reader.
