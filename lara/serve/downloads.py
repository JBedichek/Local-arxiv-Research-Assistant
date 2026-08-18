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
    dtype_bytes = 1 if (cfg.get("quantization_config") or {}) else 2
    if params:
        size = params * dtype_bytes / 1e9
    elif gg and not st:
        # A GGUF repo usually ships every quantisation of the same model, so usedStorage
        # is the sum of a dozen alternatives — it reported 707 GB for an 8B model. There
        # is no single honest number without picking a quant, so report none.
        size = 0.0
    else:
        size = (d.get("usedStorage") or 0) / 1e9

    return Resolved(
        repo=repo, exists=True, params=params,
        arch=(cfg.get("architectures") or [None])[0],
        quantization=(cfg.get("quantization_config") or {}).get("quant_method"),
        size_gb=round(size, 1), n_safetensors=len(st), n_gguf=len(gg),
        pipeline=d.get("pipeline_tag"),
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

    def start(self, repo: str, total_gb: float, token: str | None = None) -> Job:
        with self._lock:
            existing = self.jobs.get(repo)
            if existing and existing.status in ("starting", "downloading"):
                return existing
            job = Job(repo=repo, total_gb=total_gb)
            self.jobs[repo] = job

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
                    repo, allow_patterns=ALLOW, max_workers=8,
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
