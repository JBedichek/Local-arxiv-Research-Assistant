"""The Typer application and the console every command prints through.

Separate from ``lara.cli`` itself so that a command module can import them without
importing the package that imports it. Nothing that belongs to one command lives here.
"""

from __future__ import annotations

import typer
from rich.console import Console

from lara import config as config_mod
from lara import preflight as preflight_mod

app = typer.Typer(add_completion=False, help="Local arXiv Research Assistant")
console = Console()


def _require_hf(cfg: config_mod.Config) -> None:
    """Stop before anything expensive if the gated embedder is out of reach.

    The embedder is needed to encode every query, so without it the reader cannot
    search at all. Checking here costs one HTTP call; not checking costs a 50 GB
    download followed by a confusing failure at first use.
    """
    try:
        preflight_mod.require_hf_access(cfg)
    except preflight_mod.HFAccessError as e:
        console.print(f"\n[red]Hugging Face access required.[/red] {e}")
        console.print("Re-check with [bold]lara preflight[/bold] once access is granted. "
                      "To bypass deliberately: [bold]LARA_SKIP_HF_CHECK=1[/bold].")
        raise typer.Exit(1)
