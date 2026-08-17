"""Configuration loading with ``${a.b}`` interpolation and path resolution.

Every filesystem path in the config is resolved through ``realpath`` before use so a
symlink cannot silently redirect data onto a full disk. See ``lara.preflight``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_INTERP = re.compile(r"\$\{([a-zA-Z0-9_.]+)\}")

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def _lookup(tree: dict[str, Any], dotted: str) -> Any:
    node: Any = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"config interpolation ${{{dotted}}} does not resolve")
        node = node[part]
    return node


def _interpolate(node: Any, root: dict[str, Any], depth: int = 0) -> Any:
    if depth > 10:
        raise ValueError("config interpolation nested too deeply (cycle?)")
    if isinstance(node, dict):
        return {k: _interpolate(v, root, depth + 1) for k, v in node.items()}
    if isinstance(node, list):
        return [_interpolate(v, root, depth + 1) for v in node]
    if isinstance(node, str) and "${" in node:
        def sub(m: re.Match[str]) -> str:
            return str(_interpolate(_lookup(root, m.group(1)), root, depth + 1))

        return _INTERP.sub(sub, node)
    return node


class Config(dict):
    """Dict with dotted access, e.g. ``cfg.get_path("paths.metadata_db")``."""

    def get_in(self, dotted: str, default: Any = None) -> Any:
        try:
            return _lookup(self, dotted)
        except KeyError:
            return default

    def get_path(self, dotted: str) -> Path:
        """Return a config path resolved through realpath (symlinks collapsed)."""
        raw = self.get_in(dotted)
        if raw is None:
            raise KeyError(f"no path configured at {dotted}")
        return Path(os.path.realpath(os.path.expanduser(str(raw))))

    def all_paths(self) -> dict[str, Path]:
        """Every configured path that will be written to, keyed by its config address."""
        out: dict[str, Path] = {"disk.root": self.get_path("disk.root")}
        for key in self.get_in("paths", {}) or {}:
            out[f"paths.{key}"] = self.get_path(f"paths.{key}")
        out["huggingface.home"] = self.get_path("huggingface.home")
        return out


def load(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p) as fh:
        raw = yaml.safe_load(fh)
    return Config(_interpolate(raw, raw))
