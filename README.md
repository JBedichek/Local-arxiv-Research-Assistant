# Local arXiv Research Assistant

Read arXiv ML papers with a local LLM at your elbow. Open a paper by its arXiv number,
highlight a passage, ask a question about it, and get a grounded answer whose citations are
links that scroll the paper to the exact paragraph they came from. A citation graph beside
the paper shades each neighbour by how relevant it is to what you just asked.

Everything runs locally: retrieval over a vector index of arXiv ML papers, generation on
[vLLM](https://github.com/vllm-project/vllm) with models from your own Hugging Face cache.

See [`PLAN.md`](PLAN.md) for the full design — requirements, sizing math, latency budget,
and the decisions taken so far.

**Status: pre-alpha.** Configuration, preflight checks, and HF-cache model discovery work.
Ingest, indexing and the UI are not built yet.

---

## Requirements

- Python ≥ 3.12
- An NVIDIA GPU with a working driver (`nvidia-smi` must succeed — see troubleshooting)
- ~60 GB of free disk on a single filesystem
- ~16 GB of RAM for the resident index, plus whatever vLLM needs for your model

## Setup

### Option A — fresh environment (recommended for new contributors)

```bash
git clone git@github.com:JBedichek/Local-arxiv-Research-Assistant.git
cd Local-arxiv-Research-Assistant

python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[cpu,dev]"      # or ".[gpu,dev]" if you have faiss-gpu-cu12 working
pip install vllm                 # heavy and CUDA-sensitive; install separately
```

`uv` works too and is considerably faster:

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[cpu,dev]"
```

### Option B — reuse an existing environment

If you already have a venv with `vllm`, `torch` and `faiss` installed, install this project
into it without touching its dependency versions:

```bash
source /path/to/existing/venv/bin/activate
pip install -e . --no-deps
pip install pyyaml typer rich httpx aiolimiter tenacity lxml zstandard sse-starlette
```

### Configure

Copy nothing — edit [`config.yaml`](config.yaml) in place, or drop overrides in
`config.local.yaml` (gitignored). The paths at the top are the ones that matter.

### Verify before ingesting anything

```bash
lara preflight     # disks, symlink resolution, free space, GPU
lara models        # which cached models vLLM can actually serve
lara models --all  # ...and why the others can't
```

`preflight` is not ceremony. It resolves every configured path through `realpath` and
refuses to run if data would land on a filesystem other than the intended one — a symlink
in the middle of a path is otherwise an easy way to discover, 30 GB in, that your index is
being written to a disk with no room.

---

## Troubleshooting

**`nvidia-smi` reports "Driver/library version mismatch".** The NVML userspace library has
been upgraded but the old kernel module is still loaded, usually after an unattended
driver update. Nothing CUDA will start until they agree:

```bash
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia
```

If the modules are busy (a display server or a stray Python process is holding them),
reboot instead.

**`lara models` shows almost nothing.** That is usually correct rather than a bug. A
snapshot directory of dangling symlinks is indistinguishable from a complete download until
you follow the links into `blobs/`, which is what the scan does. Run `lara models --all` to
see every cached repo along with the reason it was rejected.

---

## License

MIT
