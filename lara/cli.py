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

    devices = [int(d) for d in (ecfg.get("devices") or [0])]
    if device:
        devices = [int(device.rsplit(":", 1)[-1])]
    mode = None if no_compile else ecfg.get("compile")
    console.print(
        f"loading {ecfg['model']} on cuda:{devices} (compile={mode or 'off'})…"
    )
    model = emb.load_model(
        ecfg["model"], device=f"cuda:{devices[0]}", max_seq_length=int(ecfg["max_seq_len"]),
        compile_mode=mode, compile_dynamic=bool(ecfg.get("compile_dynamic", True)),
    )
    encoder = model
    if len(devices) > 1:
        encoder = emb.MultiGPUEncoder(
            model, devices, chunk_size=int(ecfg.get("slice_size", 8192)) // len(devices)
        )

    def reporter():
        total = 0
        while True:
            n, rate, start_row = yield
            total += n
            console.print(
                f"  +{n:>5} chunks  {rate:6.0f}/s  rows {start_row:,}–{start_row + n:,}  "
                f"({total:,}/{pending:,})"
            )

    progress = reporter()
    next(progress)

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
def search(
    query: str = typer.Argument(..., help="Question to retrieve for"),
    config: str = typer.Option(None, help="Path to config.yaml"),
    k: int = typer.Option(8, help="Results to show"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip the cross-encoder"),
    no_bm25: bool = typer.Option(False, "--no-bm25", help="Dense only"),
    paper: str = typer.Option(None, help="Restrict to one arXiv id"),
) -> None:
    """Retrieve chunks for a query and show anchored citations with timings."""
    from lara.index import embed as emb
    from lara.index import retrieve as R
    from lara.index.vectors import VectorStore
    from lara.store import db

    cfg = config_mod.load(config)
    ecfg, icfg = cfg.get_in("embedding"), cfg.get_in("index")
    conn = db.connect(cfg.get_path("paths.metadata_db"))
    store = VectorStore(
        cfg.get_path("paths.vectors_fp16"), cfg.get_path("paths.vectors_int8"),
        dim_full=int(ecfg["dim_full"]), dim_trunc=int(ecfg["dim_truncated"]),
    )
    dev = f"cuda:{(ecfg.get('devices') or [0])[0]}"
    console.print(f"loading index ({store.rows():,} vectors) and embedder on {dev}…")
    embedder = emb.load_model(ecfg["model"], device=dev, max_seq_length=int(ecfg["max_seq_len"]))

    ce = None
    ccfg = icfg["rerank"].get("cross_encoder") or {}
    if ccfg.get("enabled") and not no_rerank:
        console.print(f"loading reranker {ccfg['model']}…")
        ce = R.load_cross_encoder(
            ccfg["model"], device=f"cuda:{ccfg.get('device', 1)}",
            max_length=int(ccfg.get("max_length", 512)),
        )

    retr = R.Retriever(
        conn, store, embedder, device=dev, dim_trunc=int(ecfg["dim_truncated"]),
        tier2_candidates=int(icfg["rerank"]["tier2_candidates"]),
        rerank_candidates=int(ccfg.get("candidates", 50)),
        final_k=k, rrf_k=int(icfg["lexical"]["rrf_k"]),
        lexical=bool(icfg["lexical"]["enabled"]) and not no_bm25,
        cross_encoder=ce,
    )
    console.print(f"index VRAM: {retr.dense.vram_bytes()/1e9:.2f} GB\n")

    result = retr.retrieve(query, papers=[paper] if paper else None)
    for i, hit in enumerate(result.hits, 1):
        console.print(
            f"[bold]{i}.[/bold] [cyan]{hit.fragment()}[/cyan]  "
            f"score {hit.score:.3f}  [{hit.kind}]"
        )
        console.print(f"   [dim]{hit.paper_title[:70]} › {hit.section_title[:34]}[/dim]")
        console.print(f"   {hit.text[:200].strip()}…\n")
    console.print(
        "[bold]timings[/bold]: "
        + "  ".join(f"{k}={v:.1f}ms" for k, v in result.timings_ms.items())
        + f"   candidates={result.n_candidates}"
    )
    conn.close()


@app.command()
def serve(
    config: str = typer.Option(None, help="Path to config.yaml"),
    host: str = typer.Option(None),
    port: int = typer.Option(None),
) -> None:
    """Run the reader server. Models load once at startup, before the first request."""
    import os

    import uvicorn

    cfg = config_mod.load(config)
    if config:
        os.environ["LARA_CONFIG"] = config
    host = host or cfg.get_in("serving.host", "127.0.0.1")
    port = port or int(cfg.get_in("serving.port", 8080))
    console.print(f"[bold]http://{host}:{port}[/bold]  (warming models — first start is slow)")
    # One worker: the GPU index, embedder and reranker are process-local and would be
    # duplicated per worker, tripling VRAM for no throughput gain on a single-user tool.
    uvicorn.run("lara.serve.app:app", host=host, port=port, workers=1, log_level="info")


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
