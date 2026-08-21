"""``lara config`` -- read and change settings without editing YAML."""

from __future__ import annotations

import typer

from lara import config as config_mod
from lara.cli._base import app, console

config_app = typer.Typer(help="Read and change settings without editing YAML")
app.add_typer(config_app, name="config")


#: Settings with a fixed set of legal values. A typo here otherwise surfaces much later —
#: `index.backend: fiass` does not fail at startup, it silently falls through to the
#: default and you wonder why the benchmark you ran does not match what you are running.
_CHOICES: dict[str, tuple[str, ...]] = {
    "index.backend": ("auto", "torch", "faiss"),
    "index.precision": ("fp16", "int8"),
    "index.faiss.kind": ("flat", "sq8", "hnsw"),
    "serving.generator.backend": ("auto", "vllm", "llamacpp", "mlx", "ollama",
                                 "external"),
    "serving.auth.mode": ("auto", "always", "off"),
    "embedding.compile": ("default", "reduce-overhead", "max-autotune", "null"),
}


def _set_in(tree: dict, dotted: str, value) -> dict:
    node = tree
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value
    return tree


def _unset_in(tree: dict, dotted: str) -> bool:
    parts = dotted.split(".")
    chain = [tree]
    for part in parts[:-1]:
        nxt = chain[-1].get(part)
        if not isinstance(nxt, dict):
            return False
        chain.append(nxt)
    if chain[-1].pop(parts[-1], _MISSING) is _MISSING:
        return False
    # Prune branches the removal emptied, so unsetting index.faiss.kind does not leave
    # `index: {faiss: {}}` behind to puzzle whoever reads the file next.
    for parent, key in zip(reversed(chain[:-1]), reversed(parts[:-1])):
        if parent.get(key) == {}:
            parent.pop(key)
    return True


def _write_local(path, tree) -> "Path | None":
    """Write config.local.yaml, backing up first.

    yaml.safe_dump cannot preserve comments, so any explanation in the file is lost on
    write. Round-tripping would mean a new dependency (ruamel.yaml); a timestamped backup
    is the cheap honest alternative, and the caller says so out loud.
    """
    import shutil
    import time
    from pathlib import Path

    import yaml as _yaml

    backup = None
    if path.is_file() and any(line.lstrip().startswith("#")
                              for line in path.read_text().splitlines()):
        backup = Path(f"{path}.bak.{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(path, backup)
    path.write_text(_yaml.safe_dump(tree, sort_keys=False, default_flow_style=False))
    return backup


_MISSING = object()


@config_app.command("show")
def config_show(
    config: str = typer.Option(None, help="Path to config.yaml"),
    section: str = typer.Argument(None, help="Only this subtree, e.g. index"),
) -> None:
    """Print the resolved configuration and which files it came from."""
    import yaml as _yaml

    cfg = config_mod.load(config)
    console.print(f"[dim]layers: {', '.join(str(p) for p in cfg.sources)}[/dim]\n")
    tree = cfg.get_in(section) if section else dict(cfg)
    if tree is None:
        console.print(f"[red]no such section[/red] {section}")
        raise typer.Exit(1)
    console.print(_yaml.safe_dump(tree, sort_keys=False, default_flow_style=False).rstrip())


@config_app.command("get")
def config_get(
    key: str = typer.Argument(..., help="Dotted key, e.g. index.backend"),
    config: str = typer.Option(None, help="Path to config.yaml"),
) -> None:
    """Print one resolved value."""
    cfg = config_mod.load(config)
    val = cfg.get_in(key, _MISSING)
    if val is _MISSING:
        console.print(f"[red]not set[/red]: {key}")
        raise typer.Exit(1)
    console.print(val if not isinstance(val, (dict, list)) else __import__("json").dumps(val))


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Dotted key, e.g. index.backend"),
    value: str = typer.Argument(..., help="New value; parsed as YAML (512 -> int)"),
) -> None:
    """Change a setting in config.local.yaml, leaving the tracked defaults alone.

    Writes to the machine-local layer rather than config.yaml, so your change is not a
    pending edit to a tracked file and does not travel to other machines.
    """
    import yaml as _yaml

    if key in _CHOICES and value not in _CHOICES[key]:
        console.print(f"[red]{value!r} is not valid for {key}[/red] — "
                      f"choose from {', '.join(_CHOICES[key])}")
        raise typer.Exit(1)

    # YAML-typed so `512` is an int and `[0, 1]` a list, but a value with a fixed set of
    # legal strings stays a string: YAML 1.1 would turn `off` into False.
    parsed = value if key in _CHOICES else _yaml.safe_load(value)

    path = config_mod.local_config_path()
    existing = _yaml.safe_load(path.read_text()) if path.is_file() else {}
    existing = existing or {}
    before = config_mod.load().get_in(key, _MISSING)
    _set_in(existing, key, parsed)
    backup = _write_local(path, existing)

    after = config_mod.load().get_in(key, _MISSING)
    was = "unset" if before is _MISSING else repr(before)
    console.print(f"[green]{key}[/green]: {was} -> {after!r}   [dim]in {path}[/dim]")
    if backup:
        console.print(f"[yellow]comments in that file were not preserved[/yellow] — "
                      f"previous version saved as {backup.name}")
    if key.startswith(("index.", "embedding.")):
        console.print("[dim]restart the reader for this to take effect[/dim]")


@config_app.command("unset")
def config_unset(
    key: str = typer.Argument(..., help="Dotted key to remove from config.local.yaml"),
) -> None:
    """Drop a local override, falling back to the tracked default."""
    import yaml as _yaml

    path = config_mod.local_config_path()
    if not path.is_file():
        console.print("no config.local.yaml — nothing to unset")
        return
    tree = _yaml.safe_load(path.read_text()) or {}
    if not _unset_in(tree, key):
        console.print(f"[yellow]{key} is not set locally[/yellow]; the default applies already")
        return
    backup = _write_local(path, tree)
    console.print(f"[green]removed[/green] {key} — now {config_mod.load().get_in(key)!r} "
                  f"(from the defaults)")
    if backup:
        console.print(f"[yellow]comments in that file were not preserved[/yellow] — "
                      f"previous version saved as {backup.name}")
