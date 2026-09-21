"""Bringing courses over from another store -- autoresearch keeps its own under
~/.autoresearch/courses -- into lara's. Copy only, never overwrite: what lara already has wins,
and the source is left as it was, so this is safe to run again."""

from __future__ import annotations

import shutil
from pathlib import Path

from lara.learn import store


def copy_courses(src: Path, dst: Path | None = None) -> dict:
    """{"copied": [...], "skipped": [...]} -- course directories (and the shared concept cache)
    present in `src` and absent from `dst`."""
    dst = dst or store.ROOT
    out: dict = {"copied": [], "skipped": []}
    if not src.is_dir():
        return out
    for path in sorted(src.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        target = dst / path.name
        if target.exists():
            out["skipped"].append(path.name)
            continue
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(path, target)
        out["copied"].append(path.name)
    return out
