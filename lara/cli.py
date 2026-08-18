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
def pairs(
    config: str = typer.Option(None, help="Path to config.yaml"),
    limit: int = typer.Option(0, help="Stop after N papers (0 = all cached)"),
) -> None:
    """Extract citation-context training pairs from the raw HTML cache."""
    from lara.finetune import pairs as P
    from lara.store import db

    cfg = config_mod.load(config)
    conn = db.connect(cfg.get_path("paths.metadata_db"))

    def reporter():
        while True:
            s = yield
            console.print(
                f"  {s['files']:,} papers · {s['contexts']:,} contexts · {s['skipped']} skipped"
            )

    p = reporter()
    next(p)
    try:
        stats = P.build(conn, cfg.get_path("paths.raw_cache"), limit=limit, progress=p)
        total = conn.execute("SELECT COUNT(*) FROM citation_contexts").fetchone()[0]
        console.print(f"[green]{stats['contexts']:,} new contexts[/green] · {total:,} total")
    finally:
        conn.close()


@app.command()
def explore(
    config: str = typer.Option(None, help="Path to config.yaml"),
    n: int = typer.Option(50, help="Questions to generate"),
    k: int = typer.Option(20, help="Passages retrieved per question"),
    device: str = typer.Option("cuda:0"),
    seed: int = typer.Option(0),
) -> None:
    """Generate questions, retrieve, and judge — harvesting training pairs."""
    import asyncio

    from lara.finetune import explore as EX
    from lara.finetune import judgements as J
    from lara.index import embed as emb
    from lara.index import retrieve as R
    from lara.index.vectors import VectorStore
    from lara.store import db

    cfg = config_mod.load(config)
    ecfg, icfg = cfg.get_in("embedding"), cfg.get_in("index")
    conn = db.connect(cfg.get_path("paths.metadata_db"))
    store = VectorStore(cfg.get_path("paths.vectors_fp16"), cfg.get_path("paths.vectors_int8"),
                        dim_full=int(ecfg["dim_full"]), dim_trunc=int(ecfg["dim_truncated"]))
    ccfg = icfg["rerank"]["cross_encoder"]

    console.print("loading embedder, reranker and index…")
    embedder = emb.load_model(ecfg["model"], device=device, max_seq_length=512)
    ce = R.load_cross_encoder(ccfg["model"], device=f"cuda:{ccfg.get('device', 1)}",
                              max_length=int(ccfg.get("max_length", 512)))
    lex = icfg["lexical"]
    retr = R.Retriever(conn, store, embedder, device=device,
                       dim_trunc=int(ecfg["dim_truncated"]),
                       tier2_candidates=int(icfg["rerank"]["tier2_candidates"]),
                       rerank_candidates=int(ccfg.get("candidates", 24)), final_k=k,
                       rrf_k=int(lex["rrf_k"]), lexical=bool(lex["enabled"]),
                       max_terms=int(lex.get("max_terms", 3)),
                       df_ceiling_frac=float(lex.get("df_ceiling_frac", 0.005)),
                       cross_encoder=ce)

    def reporter():
        while True:
            s = yield
            mark = "miss" if s["source_rank"] is None else f"#{s['source_rank']}"
            console.print(
                f"  [{s['cycles']:>4}] {s['style']:16} src {mark:5} "
                f"+{s['positives']}/-{s['negatives']}  {s['last_q'][:58]}"
            )

    p = reporter()
    next(p)
    stats = asyncio.run(EX.run_cycles(cfg, conn, retr, ce, n=n, k=k, seed=seed, progress=p))
    console.print(
        f"\n[green]{stats['cycles']} cycles[/green] · {stats['stored']:,} new judgements "
        f"({stats['positives']} pos / {stats['negatives']} neg) · "
        f"{stats['rejected']} questions rejected"
    )
    console.print(
        f"  source-chunk recall@{k}: {stats['source_recall']:.1%} "
        f"(MRR {stats['source_mrr']:.3f}) — misses are the most valuable examples"
    )
    console.print(f"  totals: {J.stats(conn)}")
    conn.close()


@app.command()
def finetune(
    config: str = typer.Option(None, help="Path to config.yaml"),
    out: str = typer.Option(None, help="Where to save the tuned model"),
    epochs: int = typer.Option(1),
    batch_size: int = typer.Option(16, help="Citation edges (bags) per step"),
    chunks_per_paper: int = typer.Option(4, help="Chunks sampled from each side of a bag"),
    max_edges: int = typer.Option(200000, help="Cap citation edges used (0 = all)"),
    device: str = typer.Option("cuda:0"),
    no_compile: bool = typer.Option(False, "--no-compile"),
    eval_only: bool = typer.Option(False, "--eval-only", help="Just measure the baseline"),
) -> None:
    """Fine-tune the embedder on citation contexts with Muon. Evaluates before and after."""
    from pathlib import Path as _P

    from lara.finetune import evaluate as EV
    from lara.finetune import train as T
    from lara.index.embed import load_model
    from lara.store import db

    cfg = config_mod.load(config)
    conn = db.connect(cfg.get_path("paths.metadata_db"))
    from lara.finetune import bags as BG
    st = BG.edge_stats(conn, min_chunks=8)
    console.print(
        f"[bold]{st['usable_edges']:,}[/bold] usable citation edges "
        f"(of {st['citation_edges']:,}) across {st['papers_with_fulltext']:,} crawled papers"
    )

    console.print("\n[bold]baseline[/bold] (before training)")
    base_model = load_model(cfg.get_in("embedding.model"), device=device, max_seq_length=512)
    before = EV.run(base_model, conn, n=400, pool_size=1500)
    console.print(EV.format_report(before))
    del base_model
    import torch as _t
    _t.cuda.empty_cache()
    if eval_only:
        conn.close()
        return

    tc = T.TrainConfig(model=cfg.get_in("embedding.model"), device=device,
                       epochs=epochs, batch_size=batch_size,
                       chunks_per_paper=chunks_per_paper, max_edges=max_edges,
                       compile_mode=None if no_compile else "default")

    def reporter():
        while True:
            s = yield
            console.print(
                f"  step {s['step']:>5}/{s['steps']}  loss {s['loss']:.4f}  "
                f"lr {s['lr']:.2e}  {s['elapsed']/60:.1f} min"
            )

    p = reporter()
    next(p)
    model, stats = T.train(conn, tc, progress=p)
    console.print(
        f"[green]trained[/green] {stats['steps']} steps on {stats['bags']:,} bags "
        f"in {stats['seconds']/60:.1f} min ({stats['held_out_papers']:,} papers held out)"
    )

    console.print("\n[bold]after training[/bold]")
    after = EV.run(model, conn, n=400, pool_size=1500)
    console.print(EV.format_report(before, after))

    # Only keep a fine-tune that wins where it should without losing where it must not.
    won = after["citation"]["mrr"] > before["citation"]["mrr"]
    kept = after["paraphrase"]["mrr"] > before["paraphrase"]["mrr"] - 0.02
    if not (won and kept):
        console.print(
            "\n[yellow]not saved[/yellow]: citation MRR must improve and paraphrase MRR "
            "must not fall more than 0.02. Re-embedding the corpus with a worse encoder "
            "costs hours and degrades every search."
        )
        conn.close()
        return

    dest = _P(out or (cfg.get_path("disk.root") / "models" / "embeddinggemma-citation"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(dest))
    console.print(f"[green]saved[/green] {dest}")
    console.print("Point embedding.model at it and re-run `lara embed --restart` to adopt.")
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

    def reporter():
        while True:
            s = yield
            console.print(
                f"  batch {s.batches:>5}  papers {s.papers:,}  +{s.edges:,} edges  "
                f"missing {s.missing:,}  retries {s.retries}  errors {s.errors}"
            )

    p = reporter()
    next(p)
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

    devices = [int(d) for d in (ecfg.get("devices") or [0])]
    if device:
        devices = [int(device.rsplit(":", 1)[-1])]
    model = emb.load_model(
        ecfg["model"], device=f"cuda:{devices[0]}", max_seq_length=int(ecfg["max_seq_len"]),
        compile_mode=ecfg.get("compile"), compile_dynamic=True,
    )
    encoder = model
    if len(devices) > 1:
        encoder = emb.MultiGPUEncoder(model, devices, chunk_size=2048)

    def reporter():
        total = 0
        while True:
            n, rate, start_row = yield
            total += n
            console.print(f"  +{n:>5} papers  {rate:6.0f}/s  ({total:,}/{pending:,})")

    progress = reporter()
    next(progress)
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


dataset_app = typer.Typer(help="Publish or fetch the built corpus over the LAN")
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("publish")
def dataset_publish(
    config: str = typer.Option(None, help="Path to config.yaml"),
    tiers: str = typer.Option("core,full", help="core | full | archive"),
) -> None:
    """Snapshot sizes and SHA-256 so other machines can verify what they download."""
    from lara.serve import dataset as DS

    cfg = config_mod.load(config)
    root = cfg.get_path("disk.root")
    chosen = tuple(t.strip() for t in tiers.split(",") if t.strip())

    def reporter():
        while True:
            e = yield
            console.print(f"  hashing {e['file']} ({e['size']/1e9:.1f} GB)…")

    p = reporter()
    next(p)
    console.print("[yellow]Ingest should be stopped first[/yellow] — the corpus is written "
                  "continuously, and a digest taken over a growing file is wrong on arrival.")
    m = DS.publish(root, chosen, progress=p)
    console.print(f"[green]published[/green] {len(m['files'])} files, "
                  f"{m['total_bytes']/1e9:.1f} GB total")
    for f in m["files"]:
        console.print(f"  {f['tier']:8} {f['size']/1e9:8.2f} GB  {f['name']}")
    host = cfg.get_in("serving.host", "127.0.0.1")
    port = cfg.get_in("serving.port", 8080)
    console.print(
        f"\nOn another machine:\n  lara dataset fetch http://<this-host>:{port}"
        + ("" if host == "0.0.0.0" else "   (set serving.host: 0.0.0.0 to allow LAN access)")
    )


@dataset_app.command("fetch")
def dataset_fetch(
    base_url: str = typer.Argument(..., help="http://host:8080 of the publishing node"),
    config: str = typer.Option(None, help="Path to config.yaml"),
    tiers: str = typer.Option("core", help="core | full | archive"),
    verify: bool = typer.Option(True, help="Check SHA-256 after each file"),
) -> None:
    """Download a published corpus. Resumes partial files; verifies digests."""
    import httpx

    from lara.serve import dataset as DS

    cfg = config_mod.load(config)
    root = cfg.get_path("disk.root")
    root.mkdir(parents=True, exist_ok=True)
    want = {t.strip() for t in tiers.split(",") if t.strip()}
    base = base_url.rstrip("/")

    m = httpx.get(f"{base}/api/dataset/manifest", timeout=30).json()
    files = [f for f in m["files"] if f["tier"] in want and f["sha256"]]
    total = sum(f["size"] for f in files)
    console.print(f"{len(files)} files · {total/1e9:.1f} GB · published {m['created']}")

    free = __import__("shutil").disk_usage(root).free
    if total * 1.02 > free:
        console.print(f"[red]not enough disk[/red]: need {total/1e9:.0f} GB, "
                      f"{free/1e9:.0f} GB free at {root}")
        raise typer.Exit(1)

    for f in files:
        dest = root / f["name"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        have = dest.stat().st_size if dest.exists() else 0
        if have >= f["size"]:
            console.print(f"  [dim]{f['name']} already complete[/dim]")
        else:
            # Range resume: a 49 GB file over a home network will be interrupted, and
            # restarting from zero each time makes the transfer effectively impossible.
            headers = {"Range": f"bytes={have}-"} if have else {}
            mode = "ab" if have else "wb"
            console.print(f"  {f['name']} — {(f['size']-have)/1e9:.1f} GB to go"
                          + (f" (resuming at {have/1e9:.1f} GB)" if have else ""))
            with httpx.stream("GET", f"{base}/api/dataset/file/{f['name']}",
                              headers=headers, timeout=None) as r:
                r.raise_for_status()
                got = have
                last = 0.0
                with open(dest, mode) as fh:
                    for block in r.iter_bytes(8 << 20):
                        fh.write(block)
                        got += len(block)
                        if got - last > 2e9:
                            last = got
                            console.print(f"    {got/1e9:.1f}/{f['size']/1e9:.1f} GB")
        if verify:
            ok, why = DS.verify(root, f)
            console.print(f"    {'[green]verified[/green]' if ok else '[red]FAILED[/red]'} {why}")
            if not ok:
                raise typer.Exit(1)

    console.print("\n[green]done[/green]. Run `lara preflight`, then `lara serve`.")


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


@app.command("serve-llm")
def serve_llm(
    config: str = typer.Option(None, help="Path to config.yaml"),
    model: str = typer.Option(None, help="Override the model repo"),
    show: bool = typer.Option(False, "--show", help="Print the command instead of running"),
) -> None:
    """Start vLLM for the selected generator (D4: one model resident at a time)."""
    import os
    import subprocess

    cfg = config_mod.load(config)
    v = cfg.get_in("serving.vllm")
    repo = model or v["default_model"]
    port = v["base_url"].rstrip("/").rsplit(":", 1)[-1].split("/")[0]

    # vLLM lives in its own venv. The reader's environment is pinned to torch 2.9.1 /
    # transformers 4.57 by other projects that share it, which is far too old for recent
    # checkpoints (Qwen3.8 needs a `qwen3_5` implementation that vLLM 0.14 does not
    # have). Since vLLM is a separate process reached over HTTP, isolating it costs
    # nothing and leaves the shared environment untouched.
    from pathlib import Path as _Path

    isolated = _Path(__file__).resolve().parent.parent / ".venv-vllm" / "bin" / "vllm"
    vllm_bin = str(isolated) if isolated.exists() else "vllm"

    cmd = [
        vllm_bin, "serve", repo,
        "--port", str(port),
        "--served-model-name", repo,
        "--gpu-memory-utilization", str(v.get("gpu_memory_utilization", 0.5)),
        "--max-model-len", str(v.get("max_model_len", 32768)),
        "--kv-cache-dtype", str(v.get("kv_cache_dtype", "auto")),
        "--max-num-seqs", str(v.get("max_num_seqs", 64)),
    ]
    if v.get("enable_prefix_caching", True):
        cmd.append("--enable-prefix-caching")

    env = dict(os.environ)
    devices = v.get("gpu_devices") or [0]
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)
    # The three cards are not identical — 0 and 1 are Max-Q, 2 is the full Workstation
    # Edition — and CUDA's default ordering is by compute capability, not slot. Without
    # this, "device 1" here and "device 1" in the reader can be different cards.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    console.print(f"[dim]CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}[/dim] {' '.join(cmd)}")
    if show:
        return
    subprocess.run(cmd, env=env, check=False)


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
