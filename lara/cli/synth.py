"""``lara synth`` -- goal-graph synthesis."""

from __future__ import annotations

from pathlib import Path

import typer

from lara.cli._base import app, console

synth_app = typer.Typer(help="Synthesis -- goal-graph research runs")
app.add_typer(synth_app, name="synth")


@synth_app.command("import-autoresearch")
def import_autoresearch(
    source: Path = typer.Argument(Path("~/.autoresearch"), help="autoresearch's data directory"),
) -> None:
    """Copy synthesis runs, distilled facts, goal embeddings, follow-up history and the
    interest profile into lara's stores (~/.lara). Never overwrites; safe to repeat."""
    from lara.serve import synthimport as SI

    src = source.expanduser()
    memory = SI.import_memory(src)
    runs = SI.import_runs(src)
    for name in ("facts", "goal_embeddings", "suggestions"):
        console.print(f"{name}: copied {memory[name]['copied']}, already present {memory[name]['skipped']}")
    console.print(f"interest profile: {memory['profile']}")
    console.print(f"runs: copied {len(runs['copied'])}, already present {len(runs['skipped'])}")
    if runs["failed"]:
        console.print(f"runs that could not be written: {', '.join(runs['failed'])}")
        raise typer.Exit(1)
