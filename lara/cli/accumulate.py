"""Fold finished research back into the corpus it was drawn from."""

from __future__ import annotations

import typer

from lara import config as config_mod
from lara.cli._base import app, console


@app.command()
def accumulate(
    config: str = typer.Option(None, help="Path to config.yaml"),
    run: str = typer.Option(None, help="One synthesis run id (default: every new one)"),
    limit: int = typer.Option(0, help="Stop after N runs (0 = all)"),
    force: bool = typer.Option(False, "--force", help="Re-promote a run already done"),
) -> None:
    """Turn deep-research output into retrievable chunks.

    Every synthesis run leaves behind a summary and a set of extracted claims. This makes
    them findable: each claim becomes a chunk attributed to the paper it came from, each
    run's summaries become chunks under a pseudo-paper named for the question.

    Nothing is embedded here — the new chunks land with no vector, which is exactly what
    `lara embed` looks for. Run that next, then `lara serve` (or POST /api/reload) to make
    them searchable.
    """
    from lara.index import accumulate as acc
    from lara.store import db

    cfg = config_mod.load(config)
    conn = db.connect(cfg.get_path("paths.metadata_db"))
    try:
        out = acc.promote_run(conn, run, force=force) if run else acc.promote_all(conn, limit=limit)
        console.print(
            f"[bold]{out.runs}[/bold] run(s) promoted — "
            f"{out.claim_chunks:,} claim, {out.synthesis_chunks:,} synthesis chunk(s)"
        )
        if out.duplicates:
            console.print(f"  {out.duplicates:,} already known, not duplicated")
        if out.skipped:
            console.print(f"  {out.skipped} run(s) already promoted")
        pending = acc.pending_vectors(conn)
        if pending:
            console.print(f"\n{pending:,} chunk(s) awaiting a vector — run [bold]lara embed[/bold]")
    finally:
        conn.close()
