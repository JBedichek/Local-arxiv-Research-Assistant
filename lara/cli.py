"""``lara`` command line entry point."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from lara import config as config_mod
from lara import models as models_mod
from lara import preflight as preflight_mod

app = typer.Typer(add_completion=False, help="Local arXiv Research Assistant")
console = Console()


@app.command()
def preflight(config: str = typer.Option(None, help="Path to config.yaml")) -> None:
    """Verify disks, paths and GPUs before anything writes to disk."""
    cfg = config_mod.load(config)
    checks, ok = preflight_mod.run(cfg)

    table = Table(show_header=True, header_style="bold")
    table.add_column("")
    table.add_column("check")
    table.add_column("detail", overflow="fold")
    for c in checks:
        table.add_row(
            f"[green]{c.symbol}[/green]" if c.ok else f"[red]{c.symbol}[/red]",
            c.name, c.detail,
        )
    console.print(table)

    if not ok:
        console.print("\n[red]Preflight failed.[/red] Fix the above before ingesting.")
        raise typer.Exit(1)
    console.print("\n[green]Preflight passed.[/green]")


@app.command("models")
def list_models(
    config: str = typer.Option(None, help="Path to config.yaml"),
    show_all: bool = typer.Option(False, "--all", help="Include unservable repos"),
) -> None:
    """List HF-cached models vLLM can serve (R7)."""
    cfg = config_mod.load(config)
    hf_home = cfg.get_path("huggingface.home")
    found = models_mod.scan(hf_home)
    usable = [m for m in found if m.servable]

    table = Table(show_header=True, header_style="bold")
    table.add_column("GB", justify="right")
    table.add_column("repo")
    table.add_column("arch")
    table.add_column("quant options")
    table.add_column("why not", overflow="fold")

    for m in found:
        if not m.servable and not show_all:
            continue
        table.add_row(
            f"{m.size_gb:.1f}", m.repo, m.arch or "-",
            ", ".join(m.runtime_quant_options()) if m.servable else "-",
            "" if m.servable else "; ".join(m.reasons),
        )
    console.print(table)
    console.print(
        f"\n[bold]{len(usable)}[/bold] servable of [bold]{len(found)}[/bold] cached repos "
        f"in {hf_home}"
    )
    if not usable:
        console.print("[yellow]No servable generator in cache.[/yellow]")


@app.command()
def harvest(
    config: str = typer.Option(None, help="Path to config.yaml"),
    sets: str = typer.Option("cs,stat", help="OAI sets to harvest"),
    restart: bool = typer.Option(False, "--restart", help="Ignore saved resumption tokens"),
    max_pages: int = typer.Option(0, help="Stop after N pages (0 = until complete)"),
) -> None:
    """Harvest arXiv metadata over OAI-PMH. Resumable; checkpoints every page."""
    from lara.ingest import oai
    from lara.store import db

    cfg = config_mod.load(config)
    conn = db.connect(cfg.get_path("paths.metadata_db"))
    in_scope = oai.make_scope_filter(
        cfg.get_in("corpus.categories"), cfg.get_in("corpus.date_floor")
    )
    harvester = oai.Harvester(
        cfg.get_in("ingest.oai_endpoint"),
        cfg.get_in("ingest.fulltext.user_agent"),
        min_interval_sec=float(cfg.get_in("ingest.oai_min_interval_sec", 3.0)),
    )

    try:
        for set_spec in [s.strip() for s in sets.split(",") if s.strip()]:
            if db.get_state(conn, f"oai_complete:{set_spec}") and not restart:
                console.print(f"[dim]set {set_spec}: already complete, skipping[/dim]")
                continue

            token = None if restart else db.get_state(conn, f"oai_token:{set_spec}")
            n = 0 if restart or not token else int(db.get_state(conn, f"oai_requests:{set_spec}") or 0)
            console.print(
                f"[bold]set {set_spec}[/bold]: {'resuming at page ' + str(n) if token else 'starting fresh'}"
            )

            for records, next_token in harvester.pages(set_spec, token):
                n += 1
                kept = oai.write_page(
                    conn, records, in_scope,
                    set_spec=set_spec, next_token=next_token, request_n=n,
                )
                c = db.counts(conn)
                console.print(
                    f"  page {n:>5}  +{len(records):>5} records  (+{kept} in scope)  "
                    f"total {c['papers']:,} / in-scope {c['in_scope']:,}"
                )
                if next_token is None:
                    console.print(f"[green]set {set_spec} complete[/green]")
                    break
                if max_pages and n >= max_pages:
                    console.print(f"[yellow]stopping at --max-pages {max_pages}; resumable[/yellow]")
                    break
    finally:
        harvester.close()
        conn.close()


@app.command()
def status(config: str = typer.Option(None, help="Path to config.yaml")) -> None:
    """Corpus counts and per-stage progress."""
    from lara.store import db

    cfg = config_mod.load(config)
    path = cfg.get_path("paths.metadata_db")
    if not path.exists():
        console.print("[yellow]No database yet. Run `lara harvest`.[/yellow]")
        raise typer.Exit(1)
    conn = db.connect(path)
    table = Table(show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("count", justify="right")
    for k, v in db.counts(conn).items():
        table.add_row(k, f"{v:,}")
    for row in conn.execute(
        "SELECT key, value FROM harvest_state WHERE key LIKE 'oai_%' ORDER BY key"
    ):
        val = row["value"] or "-"
        table.add_row(f"[dim]{row['key']}[/dim]", f"[dim]{val[:40]}[/dim]")
    console.print(table)
    conn.close()


if __name__ == "__main__":
    app()
