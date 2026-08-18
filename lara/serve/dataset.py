"""Publish and fetch the built corpus over the LAN.

The cheapest distribution that needs no account, no fee and no third party is the HTTP
server already running: Starlette's ``FileResponse`` honours Range requests, so downloads
resume, and any machine that can reach the reader can pull the index.

**Publishing takes a snapshot on purpose.** The corpus is written to continuously — the
crawler appends chunks, the embedder appends vectors — so hashing a live file yields a
digest that is wrong by the time it is read. ``publish`` records sizes and SHA-256 at one
instant and serves only what it recorded; a client that downloads more bytes than the
manifest promises is reading a file that grew underneath it, and truncating to the
published length is the correct repair.

Tiers exist because the artefacts differ by two orders of magnitude in usefulness per byte:

``core``    metadata, chunk text, BM25 index, citation graph, tier-1 vectors. Search and
            answers work. Tier-2 exact rescore degrades to the truncated vectors.
``full``    adds the fp16 768-d vectors, restoring exact rescore.
``archive`` adds the raw compressed HTML — only needed to re-parse after a parser change.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

TIERS = {
    "core": ["meta.sqlite", "vectors/int8.bin", "vectors/papers_int8.bin",
             "vectors/papers_fp16.bin"],
    "full": ["vectors/fp16.bin"],
    "archive": ["raw"],
}
MANIFEST_NAME = "dataset_manifest.json"

#: Published corpus on the Hub. A dataset repo, so `repo_type="dataset"` is required —
#: the default model-repo lookup 404s against it.
HF_REPO = "JamesBedichek/lara-corpus-ML-08-17-26"

#: Which repo paths belong to which tier. Matched as prefixes so the sharding inside
#: `vectors/` or `raw/` does not have to be known ahead of time.
HF_TIER_PREFIXES = {
    "core": ("meta.sqlite", "vectors/int8.bin", "vectors/papers_"),
    "full": ("vectors/fp16.bin",),
    "archive": ("raw/",),
}


def hf_patterns(tiers: tuple[str, ...]) -> list[str]:
    """`allow_patterns` for the requested tiers.

    Prefixes become globs so a file that was uploaded in parts (``vectors/fp16.bin.001``)
    is still matched. Downloading the whole repo when someone asked for `core` would mean
    an unwanted 38 GB of raw HTML.
    """
    out: list[str] = []
    for tier in tiers:
        for prefix in HF_TIER_PREFIXES.get(tier, ()):
            out.append(prefix if prefix.endswith("/") else f"{prefix}*")
            if prefix.endswith("/"):
                out[-1] = f"{prefix}**"
    return out


@dataclass
class Entry:
    name: str
    size: int
    sha256: str
    tier: str


def _sha256(path: Path, chunk: int = 8 << 20, limit: int | None = None) -> str:
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as fh:
        while True:
            want = chunk if limit is None else min(chunk, limit - read)
            if want <= 0:
                break
            b = fh.read(want)
            if not b:
                break
            h.update(b)
            read += len(b)
    return h.hexdigest()


def publish(root: Path, tiers: tuple[str, ...] = ("core", "full"),
            progress=None) -> dict:
    """Record sizes and digests for the chosen tiers. Slow but done once."""
    entries: list[Entry] = []
    for tier in tiers:
        for rel in TIERS.get(tier, []):
            path = root / rel
            if not path.exists():
                continue
            if path.is_dir():
                # Directory tiers (raw HTML) are described, not hashed file by file:
                # 26 GB across 200k files would take longer than the download.
                size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                entries.append(Entry(rel, size, "", tier))
                continue
            size = path.stat().st_size
            if progress:
                progress.send({"file": rel, "size": size, "stage": "hashing"})
            entries.append(Entry(rel, size, _sha256(path, limit=size), tier))

    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tiers": list(tiers),
        "files": [asdict(e) for e in entries],
        "total_bytes": sum(e.size for e in entries),
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1))
    return manifest


def load_manifest(root: Path) -> dict | None:
    p = root / MANIFEST_NAME
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def verify(root: Path, entry: dict) -> tuple[bool, str]:
    """Check a downloaded file against the manifest."""
    path = root / entry["name"]
    if not path.exists():
        return False, "missing"
    size = path.stat().st_size
    if size < entry["size"]:
        return False, f"short by {entry['size'] - size} bytes"
    if size > entry["size"]:
        # The publisher's file grew after the manifest was written; the extra bytes are
        # not covered by the digest, so hash only the published prefix.
        pass
    if not entry["sha256"]:
        return True, "size ok (no digest recorded)"
    got = _sha256(path, limit=entry["size"])
    return (got == entry["sha256"],
            "ok" if got == entry["sha256"] else "sha256 mismatch")
