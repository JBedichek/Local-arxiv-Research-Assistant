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
    from_date: str = typer.Option(
        None, "--from", help="OAI datestamp floor (YYYY-MM-DD). Controls ORDERING, not scope."
    ),
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
            # Streams are keyed by (set, from) so a scope-first pass and a full backfill
            # can each keep their own resumption token without clobbering one another.
            stream = f"{set_spec}@{from_date or 'all'}"
            if db.get_state(conn, f"oai_complete:{stream}") and not restart:
                console.print(f"[dim]stream {stream}: already complete, skipping[/dim]")
                continue

            token = None if restart else db.get_state(conn, f"oai_token:{stream}")
            n = 0 if restart or not token else int(db.get_state(conn, f"oai_requests:{stream}") or 0)
            console.print(
                f"[bold]stream {stream}[/bold]: "
                f"{'resuming at page ' + str(n) if token else 'starting fresh'}"
            )

            for records, next_token in harvester.pages(set_spec, token, from_date=from_date):
                n += 1
                kept = oai.write_page(
                    conn, records, in_scope,
                    set_spec=stream, next_token=next_token, request_n=n,
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
def crawl(
    config: str = typer.Option(None, help="Path to config.yaml"),
    limit: int = typer.Option(0, help="Stop after N papers (0 = drain the queue)"),
    batch: int = typer.Option(200, help="Queue slice per round"),
) -> None:
    """Fetch and parse full text for in-scope papers. Resumable; commits per paper."""
    import asyncio

    from lara.ingest import fulltext as ft
    from lara.store import db

    cfg = config_mod.load(config)
    conn = db.connect(cfg.get_path("paths.metadata_db"))
    fcfg = cfg.get_in("ingest.fulltext")
    sources = list(fcfg.get("sources", ["arxiv_html", "ar5iv", "pdf"]))

    fetcher = ft.FullTextFetcher(
        user_agent=fcfg["user_agent"],
        rate_per_sec=float(fcfg.get("rate_per_sec", 3.0)),
        max_concurrency=int(fcfg.get("max_concurrency", 6)),
        raw_root=cfg.get_path("paths.raw_cache"),
        adaptive_throttle=bool(fcfg.get("backoff", {}).get("adaptive_throttle", True)),
        on_429_initial_sec=float(fcfg.get("backoff", {}).get("on_429_initial_sec", 60)),
        backoff_multiplier=float(fcfg.get("backoff", {}).get("multiplier", 2.0)),
        backoff_max_sec=float(fcfg.get("backoff", {}).get("max_sec", 3600)),
    )

    async def run() -> None:
        try:
            await _drain()
        finally:
            # Must close on the loop the client was created on, not a fresh one.
            await fetcher.close()

    async def _drain() -> None:
        done = 0
        while True:
            queue = ft.pending_queue(conn, batch)
            if not queue:
                console.print("[green]queue empty[/green]")
                return
            if limit:
                queue = queue[: max(0, limit - done)]
                if not queue:
                    console.print(f"[yellow]stopped at --limit {limit}; resumable[/yellow]")
                    return

            # as_completed, not gather: each paper is persisted the moment it lands, so a
            # kill loses at most the few in flight rather than the whole batch. Keeps the
            # DB continuously usable while a long crawl runs.
            tasks = [
                asyncio.create_task(fetcher.fetch(aid, ver, sources)) for aid, ver in queue
            ]
            for coro in asyncio.as_completed(tasks):
                res = await coro
                if res.status == "ok" and res.raw and res.source:
                    fetcher.write_raw(res.arxiv_id, res.source, res.raw)
                n = ft.persist(conn, res)
                fetcher.stats[res.status] = fetcher.stats.get(res.status, 0) + 1
                done += 1
                if res.status == "ok":
                    console.print(
                        f"  [green]ok[/green] {res.arxiv_id:16} {res.source:11} {n:4} chunks"
                    )
                else:
                    console.print(
                        f"  [red]{res.status}[/red] {res.arxiv_id:16} {(res.error or '')[:52]}"
                    )
                if done % 25 == 0:
                    console.file.flush()
            s = fetcher.stats
            console.print(
                f"[bold]{done} done[/bold] — ok {s['ok']} / failed {s['failed']} / "
                f"unavailable {s['unavailable']} / 429s {s['429']} / rate {fetcher.rate:.1f}/s"
            )

    try:
        asyncio.run(run())
    finally:
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
