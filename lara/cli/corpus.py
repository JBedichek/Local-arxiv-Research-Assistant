"""Building the corpus: harvest, crawl, embed, citations, and what state it is in.

The commands here run for hours and are all resumable, which is why each one reports
progress through a primed generator rather than returning at the end -- see
:mod:`lara.cli._progress`.
"""

from __future__ import annotations

import typer
from rich.table import Table

from lara import config as config_mod
from lara.cli._base import app, console
from lara.cli._progress import reporter


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
        chunking=cfg.get_in("chunking") or {},
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
            try:
                for coro in asyncio.as_completed(tasks):
                    # One bad paper must not end a multi-hour crawl. A transient
                    # `database is locked` (an index rebuild holding an exclusive lock
                    # longer than busy_timeout) previously escaped here, closed the HTTP
                    # client in the finally block, and produced 304 cascading
                    # "client has been closed" errors that buried the real cause.
                    try:
                        res = await coro
                        if res.status == "ok" and res.raw and res.source:
                            fetcher.write_raw(res.arxiv_id, res.source, res.raw)
                        n = ft.persist(conn, res)
                    except Exception as exc:
                        fetcher.stats["failed"] = fetcher.stats.get("failed", 0) + 1
                        console.print(f"  [red]error[/red] {type(exc).__name__}: {str(exc)[:70]}")
                        continue
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
            finally:
                # Never leave tasks in flight when the client may close underneath them.
                for task in tasks:
                    if not task.done():
                        task.cancel()
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
def embed(
    config: str = typer.Option(None, help="Path to config.yaml"),
    limit: int = typer.Option(0, help="Stop after N chunks (0 = drain)"),
    device: str = typer.Option(None, help="Override device, e.g. cuda:0"),
    no_compile: bool = typer.Option(False, "--no-compile", help="Disable torch.compile"),
    fts: bool = typer.Option(True, help="Rebuild the BM25 index afterwards"),
) -> None:
    """Embed pending chunks into the tier-1/tier-2 vector files. Resumable."""
    from lara import device as ldev
    from lara.index import embed as emb
    from lara.index.vectors import VectorStore
    from lara.store import db

    cfg = config_mod.load(config)
    ecfg = cfg.get_in("embedding")
    conn = db.connect(cfg.get_path("paths.metadata_db"))
    store = VectorStore(
        cfg.get_path("paths.vectors_fp16"), cfg.get_path("paths.vectors_int8"),
        dim_full=int(ecfg["dim_full"]), dim_trunc=int(ecfg["dim_truncated"]),
    )

    pending = conn.execute("SELECT COUNT(*) FROM chunks WHERE vector_row IS NULL").fetchone()[0]
    console.print(
        f"[bold]{pending:,}[/bold] chunks pending, {store.rows():,} already embedded"
    )
    if not pending:
        conn.close()
        return

    devices = ldev.resolve_all(device if device else ecfg.get("devices"))
    mode = None if no_compile else ecfg.get("compile")
    console.print(
        f"loading {ecfg['model']} on {', '.join(devices)} (compile={mode or 'off'})…"
    )
    model = emb.load_model(
        ecfg["model"], device=devices[0], max_seq_length=int(ecfg["max_seq_len"]),
        compile_mode=mode, compile_dynamic=bool(ecfg.get("compile_dynamic", True)),
    )
    encoder = model
    # Only fan out across several CUDA cards; one unified-memory GPU gains nothing from
    # N worker processes and pays N times the memory. See lara.device.can_fan_out.
    if ldev.can_fan_out(devices):
        encoder = emb.MultiGPUEncoder(
            model, devices, chunk_size=int(ecfg.get("slice_size", 8192)) // len(devices)
        )

    done = 0

    def line(rec):
        nonlocal done
        n, rate, start_row = rec
        done += n
        return (f"  +{n:>5} chunks  {rate:6.0f}/s  rows {start_row:,}–{start_row + n:,}  "
                f"({done:,}/{pending:,})")

    progress = reporter(line)

    try:
        stats = emb.run(
            conn, store, encoder,
            batch_size=int(ecfg["batch_size"]),
            slice_size=int(ecfg.get("slice_size", 8192)),
            limit=limit, progress=progress,
        )
        console.print(
            f"[green]embedded {stats['chunks']:,} chunks[/green] in "
            f"{stats['seconds']/60:.1f} min "
            f"({stats['chunks']/max(stats['seconds'],1e-6):.0f}/s)"
        )
        if fts:
            n = emb.rebuild_fts(conn)
            console.print(f"BM25 index rebuilt: {n:,} rows")
            console.print(f"document frequencies refreshed: {emb.refresh_df(conn):,} terms")
    finally:
        if isinstance(encoder, emb.MultiGPUEncoder):
            encoder.close()
        conn.close()

@app.command()
def citations(
    config: str = typer.Option(None, help="Path to config.yaml"),
    limit: int = typer.Option(0, help="Stop after N papers (0 = drain)"),
) -> None:
    """Enrich the citation graph from Semantic Scholar. Resumable per batch."""
    from lara.ingest import citations as cit
    from lara.store import db

    cfg = config_mod.load(config)
    conn = db.connect(cfg.get_path("paths.metadata_db"))
    todo = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE in_scope=1 AND deleted=0 AND s2_status='pending'"
    ).fetchone()[0]
    have = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
    console.print(f"[bold]{todo:,}[/bold] papers pending · {have:,} edges already")

    p = reporter(lambda s: (
        f"  batch {s.batches:>5}  papers {s.papers:,}  +{s.edges:,} edges  "
        f"missing {s.missing:,}  retries {s.retries}  errors {s.errors}"))
    try:
        stats = cit.run(conn, cfg, limit=limit, progress=p)
        console.print(
            f"[green]enriched {stats.papers:,} papers[/green], "
            f"{stats.edges:,} new edges, {stats.missing:,} unknown to S2"
        )
    finally:
        conn.close()

@app.command("embed-papers")
def embed_papers(
    config: str = typer.Option(None, help="Path to config.yaml"),
    limit: int = typer.Option(0, help="Stop after N papers (0 = drain)"),
    device: str = typer.Option(None, help="Override device, e.g. cuda:2"),
) -> None:
    """Embed title+abstract for every in-scope paper (powers semantic paper search)."""
    from lara import device as ldev
    from lara.index import embed as emb
    from lara.index.vectors import VectorStore
    from lara.store import db

    cfg = config_mod.load(config)
    ecfg = cfg.get_in("embedding")
    conn = db.connect(cfg.get_path("paths.metadata_db"))
    store = VectorStore(
        cfg.get_path("paths.papers_fp16"), cfg.get_path("paths.papers_int8"),
        dim_full=int(ecfg["dim_full"]), dim_trunc=int(ecfg["dim_truncated"]),
    )
    pending = conn.execute(
        "SELECT COUNT(*) FROM papers p LEFT JOIN paper_vectors v ON v.arxiv_id=p.arxiv_id "
        "WHERE p.in_scope=1 AND p.deleted=0 AND v.arxiv_id IS NULL "
        "AND p.abstract IS NOT NULL AND length(p.abstract) > 40"
    ).fetchone()[0]
    console.print(f"[bold]{pending:,}[/bold] papers pending, {store.rows():,} embedded")
    if not pending:
        conn.close()
        return

    devices = ldev.resolve_all(device if device else ecfg.get("devices"))
    model = emb.load_model(
        ecfg["model"], device=devices[0], max_seq_length=int(ecfg["max_seq_len"]),
        compile_mode=ecfg.get("compile"), compile_dynamic=True,
    )
    encoder = model
    if len(devices) > 1:
        encoder = emb.MultiGPUEncoder(model, devices, chunk_size=2048)

    done = 0

    def line(rec):
        nonlocal done
        n, rate, _start_row = rec
        done += n
        return f"  +{n:>5} papers  {rate:6.0f}/s  ({done:,}/{pending:,})"

    progress = reporter(line)
    try:
        stats = emb.run_papers(
            conn, store, encoder, batch_size=int(ecfg["batch_size"]),
            limit=limit, progress=progress,
        )
        console.print(
            f"[green]embedded {stats['papers']:,} papers[/green] in "
            f"{stats['seconds']/60:.1f} min"
        )
    finally:
        if isinstance(encoder, emb.MultiGPUEncoder):
            encoder.close()
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
