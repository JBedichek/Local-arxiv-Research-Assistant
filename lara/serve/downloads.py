"""Resolve and download Hugging Face models from the UI.

Two stages on purpose. ``resolve`` is a metadata-only lookup that reports what a repo
actually is — parameter count, architecture, quantisation, download size, and whether it
fits this machine — before anything is written to disk. A model download is tens of
gigabytes; committing to one because a name looked right is an expensive way to find out
it is a 70B checkpoint or a vision model.

Downloads run in a thread and report progress by watching the cache directory grow, which
is crude but robust: ``snapshot_download`` has no progress callback that survives its
internal retries, and inferring bytes-on-disk cannot get out of sync with reality.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

HF_API = "https://huggingface.co/api/models"
_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")

# Weight-bearing files. Excludes .bin deliberately: most repos ship both, and pulling
# safetensors plus pickled duplicates doubles the download for nothing.
ALLOW = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.gguf"]

#: Everything in a repo that is not model weights. Small, and llama.cpp wants the
#: tokenizer/template files alongside a GGUF it did not build itself.
ALLOW_METADATA = ["*.json", "*.model", "*.txt", "*.jinja"]

#: A GGUF publisher ships one file per quantisation of the same model — 25 of them,
#: 133 GB, for an 8B — so ``allow_patterns=["*.gguf"]`` fetches every alternative to get
#: the one you wanted. Quantisation is part of the *filename*, not repo metadata, so it
#: has to be parsed out of the file listing.
QUANT_RE = re.compile(
    r"(?:^|[-_.])((?:UD-)?(?:IQ|Q)\d+(?:_[A-Z0-9]+)*|MXFP4|BF16|FP16|F16|F32)(?=[-.]|$)",
    re.IGNORECASE,
)

#: Multi-part quantisations: Qwen3-8B-Q4_K_M-00001-of-00002.gguf
SHARD_RE = re.compile(r"-\d{5}-of-\d{5}$")

#: Preference among 4-bit variants, best first. Q4_K_M is the near-universal default —
#: the quality/size knee, and what "4-bit" means in practice for llama.cpp. K_S trades a
#: little quality for size; the IQ (importance-matrix) forms are smaller again but slower
#: on some hardware, so they rank below the plain K-quants.
GGUF_4BIT_PREFERENCE = ["Q4_K_M", "Q4_K_S", "Q4_1", "Q4_0", "IQ4_NL", "IQ4_XS", "MXFP4"]
#: MXFP4 sits last because it is not a choice a publisher makes alongside the K-quants:
#: repos that ship it usually ship *only* it (gpt-oss), so it never competes with Q4_K_M
#: in practice, and where it somehow does the K-quant is the better-understood default.
#: Leaving it out entirely was the bug -- `ggml-org/gpt-oss-20b-GGUF` reports three GGUF
#: files, of which the real 20B model is `gpt-oss-20b-MXFP4.gguf` and the other two are an
#: `eagle3` draft model. Unrecognised, the model itself grouped as "unlabelled", no 4-bit
#: build was found, and the dialog offered nothing while telling the user to pick.


def quant_of(path: str) -> str | None:
    """The quantisation label in a GGUF filename, e.g. ``Q4_K_M``.

    Takes the *last* match: repo and model names contain digits and underscores of their
    own, and the quant tag is conventionally the final component before any shard suffix.
    """
    stem = path.rsplit("/", 1)[-1]
    if stem.lower().endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    stem = SHARD_RE.sub("", stem)
    found = QUANT_RE.findall(stem)
    return found[-1].upper() if found else None


def gguf_variants(files: list[dict]) -> list[dict]:
    """Group a repo's GGUF files by quantisation, summing sharded ones.

    ``files`` is the Hub tree listing: dicts with ``path`` and a size, which for LFS
    objects lives under ``lfs.size`` rather than ``size``.
    """
    groups: dict[str, dict] = {}
    for f in files:
        path = f.get("path") or ""
        if not path.lower().endswith(".gguf"):
            continue
        label = quant_of(path) or "unlabelled"
        size = int((f.get("lfs") or {}).get("size") or f.get("size") or 0)
        g = groups.setdefault(label, {"quant": label, "files": [], "size_gb": 0.0})
        g["files"].append(path)
        g["size_gb"] = round(g["size_gb"] + size / 1e9, 2)
    out = list(groups.values())
    out.sort(key=lambda g: g["size_gb"])
    return out


#: Unsloth publishes "Unsloth Dynamic" builds as ``UD-Q4_K_XL`` and the like: the same
#: quantisations with a per-tensor recipe on top. For many of its repos they are the *only*
#: 4-bit builds -- unsloth/Qwen3.6-35B-A3B-GGUF ships twenty-six quantisations, every 4-bit
#: one of them ``UD-`` prefixed -- so matching the plain names alone found nothing and left
#: the download dialog with no default on the publisher SETUP_INSTRUCTIONS.md recommends.
_UD_PREFIX = "UD-"


def _is_four_bit(quant: str) -> bool:
    """Whether a label names some 4-bit build, dynamic prefix or not."""
    base = quant.upper().removeprefix(_UD_PREFIX)
    return base.startswith(("Q4", "IQ4")) or base == "MXFP4"


def pick_4bit(variants: list[dict]) -> str | None:
    """The 4-bit quantisation to default to, or None if the repo ships no 4-bit build.

    Three passes, narrowing from "the exact thing everyone means by 4-bit" to "any 4-bit
    build at all", because a repo that ships one is always better served by defaulting to
    it than by presenting no default and downloading everything.
    """
    have = [v["quant"] for v in variants]
    by_upper = {q.upper(): q for q in have}
    for want in GGUF_4BIT_PREFERENCE:
        if want in by_upper:
            return by_upper[want]
    for want in GGUF_4BIT_PREFERENCE:
        if f"{_UD_PREFIX}{want}" in by_upper:
            return by_upper[f"{_UD_PREFIX}{want}"]
    # Only a dynamic build with no plain equivalent, e.g. UD-Q4_K_XL. `gguf_variants`
    # sorted these smallest first, so this takes the least demanding one.
    return next((q for q in have if _is_four_bit(q)), None)


def tree(repo: str, token: str | None = None) -> list[dict]:
    """The repo's file listing with sizes. Empty on any failure — callers degrade."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = httpx.get(f"https://huggingface.co/api/models/{repo}/tree/main",
                      headers=headers, timeout=20, follow_redirects=True,
                      params={"recursive": "true"})
    except httpx.RequestError:
        return []
    if r.status_code != 200:
        return []
    data = r.json()
    return data if isinstance(data, list) else []


def normalise(query: str) -> str | None:
    """Accept a bare id, a full URL, or a paste with surrounding whitespace."""
    q = (query or "").strip()
    q = re.sub(r"^https?://(www\.)?huggingface\.co/", "", q)
    q = q.split("?")[0].split("#")[0].strip("/")
    q = re.sub(r"/(tree|blob)/[^/]+.*$", "", q)
    return q if _REPO_RE.match(q) else None


@dataclass
class Resolved:
    repo: str
    exists: bool
    gated: bool = False
    params: int = 0
    arch: str | None = None
    quantization: str | None = None
    size_gb: float = 0.0
    n_safetensors: int = 0
    n_gguf: int = 0
    pipeline: str | None = None
    error: str | None = None
    #: One entry per quantisation the repo ships, smallest first, with the file(s) that
    #: make it up. Empty for safetensors repos.
    variants: list[dict] = field(default_factory=list)
    #: The 4-bit variant to default to, if the repo has one.
    pick: str | None = None


def resolve(query: str, token: str | None = None) -> Resolved:
    repo = normalise(query)
    if not repo:
        return Resolved(repo=query, exists=False,
                        error="Not a Hugging Face id. Expected owner/name.")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        r = httpx.get(f"{HF_API}/{repo}", headers=headers, timeout=20,
                      follow_redirects=True)
    except httpx.RequestError as exc:
        return Resolved(repo=repo, exists=False, error=f"network error: {exc}")

    if r.status_code == 404:
        return Resolved(repo=repo, exists=False, error="No such repo on Hugging Face.")
    if r.status_code in (401, 403):
        # HF answers 401/403 for private AND nonexistent repos alike, deliberately, so it
        # does not leak which private repos exist. We cannot tell them apart, and saying
        # "gated" outright sends someone with a typo off to accept a licence that is not
        # the problem.
        return Resolved(
            repo=repo, exists=False, gated=True,
            error="Not accessible: either the name is wrong, or it is gated/private. "
                  "Check the spelling first; if the repo is real, accept its licence on "
                  "huggingface.co and set HF_TOKEN.",
        )
    if r.status_code != 200:
        return Resolved(repo=repo, exists=False, error=f"HTTP {r.status_code}")

    d = r.json()
    siblings = [f.get("rfilename", "") for f in (d.get("siblings") or [])]
    st = [f for f in siblings if f.endswith(".safetensors")]
    gg = [f for f in siblings if f.endswith(".gguf")]
    cfg = d.get("config") or {}
    safet = d.get("safetensors") or {}

    params = int(safet.get("total") or 0)
    variants: list[dict] = []
    pick: str | None = None
    if gg and not st:
        # A GGUF repo ships every quantisation of the same model, so usedStorage is the
        # sum of a dozen alternatives — it reported 707 GB for an 8B model. Size is only
        # meaningful once a quantisation is chosen, so resolve them individually and
        # quote the one we would actually download.
        variants = gguf_variants(tree(repo, token))
        pick = pick_4bit(variants)
        chosen = next((v for v in variants if v["quant"] == pick), None)
        size = chosen["size_gb"] if chosen else 0.0
    else:
        # Weigh the actual files rather than inferring from the parameter count.
        # `safetensors.total` counts *stored elements*, which is not the parameter count
        # for a packed checkpoint: MLX stores eight 4-bit weights per uint32, so
        # mlx-community/Qwen3-8B-4bit reports 1.28B and the old `params * 1 byte` sized it
        # at 1.3 GB against a real 4.61 GB file. Undersizing is the dangerous direction —
        # it is what both the "does it fit" verdict and the free-disk guard are computed
        # from. Summing `*.safetensors` also matches what is actually fetched, since ALLOW
        # takes all of them.
        size = sum(f.get("size") or 0 for f in tree(repo, token)
                   if str(f.get("path", "")).endswith(".safetensors")) / 1e9
        if not size:
            # tree() degrades to [] on any failure, and reporting 0 GB reads as "free" to
            # every caller. Fall back to the old estimate, wrong but not silently zero.
            dtype_bytes = 1 if (cfg.get("quantization_config") or {}) else 2
            size = (params * dtype_bytes / 1e9 if params
                    else (d.get("usedStorage") or 0) / 1e9)

    return Resolved(
        repo=repo, exists=True, params=params,
        arch=(cfg.get("architectures") or [None])[0],
        quantization=(cfg.get("quantization_config") or {}).get("quant_method"),
        size_gb=round(size, 1), n_safetensors=len(st), n_gguf=len(gg),
        pipeline=d.get("pipeline_tag"), variants=variants, pick=pick,
    )


@dataclass
class Job:
    repo: str
    status: str = "starting"        # starting | downloading | done | error
    downloaded_gb: float = 0.0
    total_gb: float = 0.0
    error: str | None = None
    started: float = field(default_factory=time.time)
    path: str | None = None

    @property
    def pct(self) -> float:
        # Clamped: total_gb is an estimate from parameter count, and the real download is
        # often a little larger (tokenizer, configs, an extra shard). Showing 126% makes a
        # working download look broken.
        if not self.total_gb:
            return 0.0
        return round(min(100.0, 100 * self.downloaded_gb / self.total_gb), 1)


class DownloadManager:
    def __init__(self, hf_home: Path) -> None:
        self.hf_home = Path(hf_home)
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def cache_dir(self, repo: str) -> Path:
        return self.hf_home / "hub" / ("models--" + repo.replace("/", "--"))

    def has_files(self, repo: str, files: list[str] | None) -> bool:
        """Whether what we would download is already present *and complete*.

        ``cache_dir().exists()`` is not that question, and answering it with that was
        wrong in three ways at once. The directory appears the moment a download starts,
        it survives one that was abandoned, and -- now that a GGUF download fetches one
        quantisation rather than the whole repo -- it says nothing about whether *this*
        quantisation is there. A repo holding Q4_1 reported every other quantisation as
        already cached, which hid the download button for a model that was not
        downloaded.

        Completeness comes free from the cache layout: huggingface_hub writes to
        ``blobs/<sha>.incomplete`` and only links the file into ``snapshots/<rev>/`` once
        it has all of it, so a partial download has no snapshot entry to find.
        """
        snaps = self.cache_dir(repo) / "snapshots"
        if not snaps.is_dir():
            return False
        revs = [p for p in snaps.iterdir() if p.is_dir()]
        if not files:
            # Safetensors: the whole snapshot is the model, so the question is whether any
            # revision has weights in it rather than which files were named.
            return any(any(rev.rglob(pat)) for rev in revs
                       for pat in ("*.safetensors", "*.gguf"))
        return all(any((rev / f).is_file() for rev in revs) for f in files)

    def _dir_gb(self, path: Path) -> float:
        if not path.exists():
            return 0.0
        total = 0
        for p in path.rglob("*"):
            try:
                if p.is_file() and not p.is_symlink():
                    total += p.stat().st_size
            except OSError:
                continue
        return total / 1e9

    def start(self, repo: str, total_gb: float, token: str | None = None,
              files: list[str] | None = None) -> Job:
        """Fetch ``repo``, or only ``files`` from it.

        ``files`` carries the exact paths of one GGUF quantisation. Without it a GGUF
        repo pulls every quantisation it ships — 133 GB to obtain the 5 GB you asked
        for — because the wanted one is distinguishable only by filename.
        """
        with self._lock:
            existing = self.jobs.get(repo)
            if existing and existing.status in ("starting", "downloading"):
                return existing
            job = Job(repo=repo, total_gb=total_gb)
            self.jobs[repo] = job
        patterns = (ALLOW_METADATA + list(files)) if files else ALLOW

        def work() -> None:
            from huggingface_hub import snapshot_download
            stop = threading.Event()

            def watch() -> None:
                d = self.cache_dir(repo)
                while not stop.wait(2.0):
                    job.downloaded_gb = round(self._dir_gb(d), 2)

            watcher = threading.Thread(target=watch, daemon=True)
            watcher.start()
            try:
                job.status = "downloading"
                path = snapshot_download(
                    repo, allow_patterns=patterns, max_workers=8,
                    token=token, cache_dir=str(self.hf_home / "hub"),
                )
                job.path = path
                job.status = "done"
            except Exception as exc:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {str(exc)[:300]}"
            finally:
                stop.set()
                job.downloaded_gb = round(self._dir_gb(self.cache_dir(repo)), 2)

        threading.Thread(target=work, daemon=True).start()
        return job

    def get(self, repo: str) -> Job | None:
        return self.jobs.get(repo)

    def active(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.status in ("starting", "downloading")]
