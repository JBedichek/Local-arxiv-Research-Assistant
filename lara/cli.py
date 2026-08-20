"""``lara`` command line entry point."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from lara import config as config_mod
from lara import device as ldev
from lara import models as models_mod
from lara import preflight as preflight_mod
from lara import prompt as prompt_mod

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


@app.command()
def preflight(config: str = typer.Option(None, help="Path to config.yaml")) -> None:
    """Verify disks, paths, GPUs and Hugging Face access before anything writes to disk."""
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
    """List HF-cached models this machine's generation backend can serve (R7)."""
    from lara.serve import devices as DV

    cfg = config_mod.load(config)
    hf_home = cfg.get_path("huggingface.home")
    dev = DV.detect()
    found = models_mod.scan(hf_home, backend=dev.backend)
    usable = [m for m in found if m.servable]
    console.print(f"[dim]backend [bold]{dev.backend}[/bold], which loads "
                  f"[bold]{models_mod.wants_format(dev.backend)}[/bold] weights[/dim]")

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
    device: str = typer.Option(None, help="Override device; default auto-detects"),
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
    edev = ldev.resolve(device) if device else ldev.first(ecfg.get("devices"))
    console.print(f"loading index ({store.rows():,} vectors) and embedder on {edev}…")
    embedder = emb.load_model(ecfg["model"], device=edev,
                              max_seq_length=int(ecfg["max_seq_len"]))

    ce = None
    ccfg = icfg["rerank"].get("cross_encoder") or {}
    if ccfg.get("enabled") and not no_rerank:
        console.print(f"loading reranker {ccfg['model']}…")
        ce = R.load_cross_encoder(
            ccfg["model"], device=ldev.resolve(ccfg.get("device", 1)),
            max_length=int(ccfg.get("max_length", 512)),
        )

    retr = R.Retriever(
        conn, store, embedder, device=edev, dim_trunc=int(ecfg["dim_truncated"]),
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
    device: str = typer.Option(None, help="Override device; default auto-detects"),
    seed: int = typer.Option(0),
    topics: str = typer.Option(None, help="Semicolon-separated topics to focus sampling on"),
) -> None:
    """Generate questions, retrieve, and judge — harvesting training pairs.

    `--topics` seeds passages from papers about those subjects instead of uniformly at
    random, which is how you deepen coverage of a specific area rather than thinning it
    across the whole corpus.
    """
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
    ce = R.load_cross_encoder(ccfg["model"], device=ldev.resolve(ccfg.get("device", 1)),
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
    topic_list = [t.strip() for t in (topics or "").split(";") if t.strip()]
    if topic_list:
        console.print(f"focusing on {len(topic_list)} topics: " + ", ".join(topic_list))
    stats = asyncio.run(EX.run_cycles(cfg, conn, retr, ce, n=n, k=k, seed=seed,
                                      topics=topic_list or None, progress=p))
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


@app.command("finetune-pairs")
def finetune_pairs(
    config: str = typer.Option(None, help="Path to config.yaml"),
    out: str = typer.Option(None, help="Where to save the tuned model"),
    lr_muon: float = typer.Option(3e-5),
    batch_size: int = typer.Option(512, help="Optimiser batch; the negative supply"),
    micro_batch: int = typer.Option(64, help="Sequences per forward pass"),
    epochs: int = typer.Option(4),
    max_per_query: int = typer.Option(32),
    max_seq_length: int = typer.Option(512, help="Match what the corpus was embedded at"),
    compile_mode: str = typer.Option("default", help="torch.compile mode; 'none' to disable"),
    patience: int = typer.Option(4),
    eval_every: int = typer.Option(10),
    n_eval: int = typer.Option(800, help="Queries per independent eval task"),
    device: str = typer.Option(None),
    force: bool = typer.Option(False, "--force", help="Save even if the guard rejects it"),
) -> None:
    """Train once on ALL judgement pairs, then judge it on the independent eval.

    k-fold measures whether a *recipe* generalises and deliberately throws every model
    away; each fold trains on different data, so no fold's weights are the artifact. This
    trains a single model on everything and keeps it — but only if it earns its place.

    The gate is `lara/finetune/evaluate.py`, not the pairwise metrics. Those are agreement
    with the cross-encoder that produced the labels, so a model can improve them by
    learning its teacher's habits. Citation retrieval is scored against what human authors
    actually cited, and paraphrase retrieval is the canary for catastrophic forgetting: a
    model that wins on citations while losing there has traded general retrieval for a
    narrow skill and would make the reader worse.
    """
    from pathlib import Path as _P

    from lara.finetune import evaluate as EV
    from lara.finetune import kfold as KF
    from lara.index.embed import load_model
    from lara.store import db

    cfg = config_mod.load(config)
    conn = db.connect(cfg.get_path("paths.metadata_db"))
    model_name = cfg.get_in("embedding.model")

    triples = KF.make_triples(conn, max_per_query=max_per_query, contextual=True)
    n_q = len({t.query_hash for t in triples})
    if len(triples) < batch_size * 4:
        console.print(f"[red]only {len(triples):,} triples[/red] — run `lara explore` first")
        raise typer.Exit(1)
    rec = KF.Recipe(lr_muon=lr_muon, lr_adam=lr_muon / 5, batch_size=batch_size,
                    micro_batch=micro_batch, epochs=epochs, max_seq_length=max_seq_length,
                    patience=patience, eval_every=eval_every,
                    compile_mode=None if compile_mode.lower() == "none" else compile_mode)
    console.print(f"[bold]{len(triples):,}[/bold] triples from {n_q:,} queries · "
                  f"MultipleNegativesRanking · batch {batch_size} · seq {max_seq_length}")

    console.print("\n[bold]baseline[/bold] on the independent eval")
    base = load_model(model_name, device=device, max_seq_length=max_seq_length)
    before = EV.run(base, conn, n=n_eval)
    console.print(EV.format_report(before))
    del base
    ldev.empty_cache(device)

    train, val = KF.inner_split(triples, rec.inner_val_frac, seed=1)
    console.print(f"\n[bold]training[/bold] on {len(train):,} triples "
                  f"({len(val):,} held back for early stopping)")

    def reporter():
        while True:
            st = yield
            if st.get("early_stop"):
                console.print(f"  [yellow]early stop[/yellow] step {st['step']} "
                              f"best val {st['best_val']:.4f}")
                continue
            v = st.get("val_loss")
            console.print(f"  step {st['step']:>4}/{st['steps']}  loss {st['loss']:7.4f}  "
                          f"lr {st['lr']:.2e}" + (f"  val {v:7.4f}" if v is not None else "")
                          + f"  {st['elapsed']/60:.0f}m")

    pr = reporter(); next(pr)
    model = KF.train_on_mnrl(list(train), model_name, device, rec,
                             progress=pr, val_triples=val)
    console.print(f"[green]trained[/green] {getattr(model, 'steps_trained', 0)} steps"
                  + (" (early stopped)" if getattr(model, "stopped_early", False) else ""))

    console.print("\n[bold]after training[/bold]")
    after = EV.run(model, conn, n=n_eval)
    console.print(EV.format_report(before, after))

    # The same guard `lara finetune` uses, and for the same reason: re-embedding 29.5 M
    # chunks costs hours and a worse encoder degrades every search until it is undone.
    won = after["citation"]["mrr"] > before["citation"]["mrr"]
    kept = after["paraphrase"]["mrr"] > before["paraphrase"]["mrr"] - 0.02
    verdict = ("citation MRR "
               f"{before['citation']['mrr']:.4f} -> {after['citation']['mrr']:.4f}, "
               f"paraphrase MRR {before['paraphrase']['mrr']:.4f} -> "
               f"{after['paraphrase']['mrr']:.4f}")
    if not (won and kept) and not force:
        console.print(f"\n[yellow]not saved[/yellow] — {verdict}\n"
                      "Citation MRR must improve and paraphrase MRR must not fall more "
                      "than 0.02. Re-embedding the corpus with a worse encoder costs "
                      "hours and degrades every search until it is undone.\n"
                      "[dim]--force overrides, if you want the weights to inspect.[/dim]")
        conn.close()
        raise typer.Exit(1)

    dest = _P(out or (cfg.get_path("disk.root") / "models" / "embeddinggemma-pairs"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(dest))
    console.print(f"\n[green]saved[/green] {dest} — {verdict}")
    console.print(
        "\nAdopting it is a separate decision and not a cheap one:\n"
        "  1. point [bold]embedding.model[/bold] at this path in config.local.yaml\n"
        "  2. [bold]lara embed --restart[/bold] — re-embeds all 28.7 M chunks (~8 GPU-hours)\n"
        "  3. refit the whitener, whose statistics belong to the old encoder\n"
        "[dim]Until step 2 finishes the index and the encoder disagree, and search is wrong.[/dim]"
    )
    conn.close()


@app.command("lr-sweep")
def lr_sweep(
    config: str = typer.Option(None, help="Path to config.yaml"),
    lrs: str = typer.Option("1e-5,3e-5,1e-4,3e-4,1e-3", help="Muon LRs, comma-separated"),
    batch_size: int = typer.Option(512, help="Optimiser batch (reached by accumulation)"),
    micro_batch: int = typer.Option(64, help="Sequences per forward pass; 0 = no accumulation"),
    epochs: int = typer.Option(4),
    max_per_query: int = typer.Option(32, help="Triples per query when building the set"),
    patience: int = typer.Option(3, help="Evals without improvement before stopping (0=off)"),
    eval_every: int = typer.Option(5, help="Steps between validation passes"),
    val_frac: float = typer.Option(0.25, help="Held-out query fraction for scoring"),
    device: str = typer.Option(None, help="Override device; default auto-detects"),
    out: str = typer.Option(None, help="Write the sweep table as JSON"),
) -> None:
    """Sweep the Muon learning rate at a fixed recipe, scored on held-out queries.

    Every run starts from the same pretrained checkpoint on the same query-split, so the
    only variable is the learning rate. Early stopping runs against an inner split of the
    training half, never against the slice the results are reported on.
    """
    import json as _json

    from lara.finetune import kfold as KF
    from lara.store import db

    cfg = config_mod.load(config)
    conn = db.connect(cfg.get_path("paths.metadata_db"))
    triples = KF.make_triples(conn, max_per_query=max_per_query)
    n_q = len({t.query_hash for t in triples})
    grid = [float(x) for x in lrs.split(",") if x.strip()]

    rec = KF.Recipe(batch_size=batch_size, micro_batch=micro_batch, epochs=epochs,
                    patience=patience, eval_every=eval_every)
    per_epoch = int(len(triples) * (1 - val_frac) * (1 - rec.inner_val_frac)) // batch_size
    console.print(
        f"[bold]{len(triples):,}[/bold] triples from {n_q:,} queries · "
        f"batch {batch_size} (micro {micro_batch or batch_size}) · "
        f"~{per_epoch} steps/epoch x {epochs} epochs · {len(grid)} learning rates"
    )
    if per_epoch < 4:
        console.print(
            "[yellow]few steps per epoch[/yellow] — at this batch size the sweep is mostly "
            "measuring noise. Raise --max-per-query or lower --batch-size."
        )

    def reporter():
        while True:
            s = yield
            if s.get("early_stop"):
                console.print(f"      [yellow]early stop[/yellow] step {s['step']}  "
                              f"best val {s['best_val']:.4f}")
                continue
            v = s.get("val_loss")
            console.print(f"      step {s['step']:>4}/{s['steps']}  loss {s['loss']:8.4f}  "
                          f"lr {s['lr']:.2e}" + (f"  val {v:7.4f}" if v is not None else ""))

    p = reporter(); next(p)
    rows = []
    for lr in grid:
        console.print(f"\n[bold]muon lr {lr:.1e}[/bold]")
        rows += KF.sweep(triples, cfg.get_in("embedding.model"), device, rec, [lr],
                         val_frac=val_frac, progress=p)
        console.print("  " + _json.dumps({k: rows[-1][k] for k in
                                          ("steps", "early_stopped", "d_pair_acc",
                                           "d_spearman", "d_margin_mae")}))

    console.print("\n" + KF.format_sweep(rows))
    if out:
        _P = __import__("pathlib").Path(out)
        _P.parent.mkdir(parents=True, exist_ok=True)
        _P.write_text(_json.dumps(rows, indent=1))
        console.print(f"\nwrote {out}")
    conn.close()


@app.command("fit-check")
def fit_check(
    config: str = typer.Option(None, help="Path to config.yaml"),
    mode: str = typer.Option("overfit", help="overfit | kfold"),
    k: int = typer.Option(5, help="Folds, for kfold mode"),
    n: int = typer.Option(400, help="Triples to use in overfit mode"),
    epochs: int = typer.Option(3),
    batch_size: int = typer.Option(128),
    lr_muon: float = typer.Option(5e-5),
    patience: int = typer.Option(3, help="Evals without improvement before stopping (0=off)"),
    eval_every: int = typer.Option(5, help="Steps between validation passes"),
    device: str = typer.Option(None, help="Override device; default auto-detects"),
) -> None:
    """Can the embedder fit the harvested judgements at all?

    `overfit` trains and evaluates on the same triples: a working setup drives pairwise
    accuracy toward 1.0. Failing that, the recipe is broken and more data cannot help.
    `kfold` splits by query to measure whether it generalises within-distribution.
    """
    from lara.finetune import kfold as KF
    from lara.store import db

    cfg = config_mod.load(config)
    conn = db.connect(cfg.get_path("paths.metadata_db"))
    triples = KF.make_triples(conn)
    console.print(f"[bold]{len(triples):,}[/bold] triples from "
                  f"{len({t.query_hash for t in triples}):,} queries")
    if len(triples) < batch_size * 2:
        console.print("[red]not enough triples[/red] — run `lara explore` first")
        raise typer.Exit(1)

    rec = KF.Recipe(lr_muon=lr_muon, batch_size=batch_size, epochs=epochs,
                    patience=patience, eval_every=eval_every)
    console.print(f"recipe: muon lr {rec.lr_muon:.1e} · adam lr {rec.lr_adam:.1e} · "
                  f"batch {rec.batch_size} · max {rec.epochs} epochs · "
                  + (f"early stop patience {rec.patience} every {rec.eval_every} steps"
                     if rec.patience else "no early stopping"))
    model_name = cfg.get_in("embedding.model")

    def reporter():
        while True:
            s = yield
            if s.get("early_stop"):
                console.print(f"    [yellow]early stop[/yellow] at step {s['step']}  "
                              f"best val loss {s['best_val']:.4f}")
                continue
            v = s.get("val_loss")
            console.print(f"    step {s['step']:>4}/{s['steps']}  loss {s['loss']:8.4f}  "
                          f"lr {s['lr']:.2e}"
                          + (f"  val {v:7.4f}" if v is not None else ""))

    if mode == "overfit":
        sub = triples[:n]
        console.print(f"\n[bold]overfit check[/bold] on {len(sub)} triples "
                      "(train == eval; this measures capacity, not generalisation)")
        from lara.index.embed import load_model
        base = load_model(model_name, device=device, max_seq_length=rec.max_seq_length)
        before = KF.evaluate(base, sub, device)
        console.print(f"  before: pair_acc {before['pair_acc']:.3f}  "
                      f"margin_mae {before['margin_mae']:.3f}  spearman {before['spearman']}")
        del base
        ldev.empty_cache(device)

        p = reporter(); next(p)
        model = KF.train_on(list(sub), model_name, device,
                            KF.replace(rec, patience=0), progress=p)
        after = KF.evaluate(model, sub, device)
        console.print(f"  after : pair_acc {after['pair_acc']:.3f}  "
                      f"margin_mae {after['margin_mae']:.3f}  spearman {after['spearman']}")
        gain = after["pair_acc"] - before["pair_acc"]
        if after["pair_acc"] > 0.9:
            console.print("[green]  the setup can fit its data[/green] — recipe is sane; "
                          "generalisation is the next question")
        elif gain > 0.05:
            console.print("[yellow]  learning, but not fitting[/yellow] — try more epochs "
                          "or a higher LR before adding data")
        else:
            console.print("[red]  cannot fit even the training set[/red] — the loss, "
                          "optimiser or LR is wrong; more data will not help")
        conn.close()
        return

    folds = KF.split_by_query(triples, k, seed=rec.seed)
    console.print(f"\n[bold]{k}-fold CV[/bold], split by query "
                  f"(sizes {[len(f) for f in folds]})")
    results = []
    for i in range(k):
        val_idx = set(folds[i])
        train = [t for j, t in enumerate(triples) if j not in val_idx]
        val = [triples[j] for j in folds[i]]
        console.print(f"\n  fold {i+1}/{k}: train {len(train):,} · val {len(val):,}")
        from lara.index.embed import load_model
        base = load_model(model_name, device=device, max_seq_length=rec.max_seq_length)
        before = KF.evaluate(base, val, device)
        del base
        ldev.empty_cache(device)
        # Early stopping selects on an INNER split of the training folds. Selecting on
        # `val` would leak: the checkpoint would be chosen using the very data the fold
        # metrics are computed from, making them optimistic by an unknown amount.
        inner_train, inner_val = (KF.inner_split(train, rec.inner_val_frac, rec.seed)
                                  if rec.patience else (train, []))
        if inner_val:
            console.print(f"    inner split: {len(inner_train):,} train · "
                          f"{len(inner_val):,} early-stop val (by query)")
        p = reporter(); next(p)
        model = KF.train_on(list(inner_train), model_name, device, rec, progress=p,
                            val_triples=inner_val)
        after = KF.evaluate(model, val, device)
        note = ""
        if getattr(model, "stopped_early", False):
            note = f"   [dim](stopped at step {model.steps_trained})[/dim]"
        del model
        ldev.empty_cache(device)
        console.print(f"    pair_acc {before['pair_acc']:.3f} -> {after['pair_acc']:.3f}   "
                      f"spearman {before['spearman']} -> {after['spearman']}{note}")
        results.append((before, after))

    import statistics as st
    for key in ("pair_acc", "spearman", "margin_mae"):
        b = [r[0][key] for r in results]; a = [r[1][key] for r in results]
        console.print(f"  {key:12} {st.mean(b):.4f} ± {st.pstdev(b):.4f}  ->  "
                      f"{st.mean(a):.4f} ± {st.pstdev(a):.4f}")
    conn.close()


@app.command()
def finetune(
    config: str = typer.Option(None, help="Path to config.yaml"),
    out: str = typer.Option(None, help="Where to save the tuned model"),
    epochs: int = typer.Option(1),
    batch_size: int = typer.Option(16, help="Citation edges (bags) per step"),
    chunks_per_paper: int = typer.Option(4, help="Chunks sampled from each side of a bag"),
    max_edges: int = typer.Option(200000, help="Cap citation edges used (0 = all)"),
    device: str = typer.Option(None, help="Override device; default auto-detects"),
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
    ldev.empty_cache(device)
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

    devices = ldev.resolve_all(device if device else ecfg.get("devices"))
    model = emb.load_model(
        ecfg["model"], device=devices[0], max_seq_length=int(ecfg["max_seq_len"]),
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


def _probe_generators(timeout: float = 0.6) -> list[tuple[str, str]]:
    """Look for an already-running OpenAI-compatible server.

    Many people already have Ollama or LM Studio going, and asking them to re-download a
    model they have is the fastest way to lose them. Probed in the order a local setup is
    most likely to be using.
    """
    import httpx

    found = []
    for name, url in (("vLLM", "http://127.0.0.1:8000/v1"),
                      ("Ollama", "http://127.0.0.1:11434/v1"),
                      ("LM Studio", "http://127.0.0.1:1234/v1"),
                      ("llama.cpp", "http://127.0.0.1:8080/v1")):
        try:
            r = httpx.get(f"{url}/models", timeout=timeout)
            if r.status_code == 200:
                ids = [m.get("id", "?") for m in (r.json().get("data") or [])]
                found.append((f"{name} ({', '.join(ids[:2]) or 'no models listed'})",
                              url, ids[0] if ids else None))
        except Exception:
            continue
    return found


@app.command()
def setup(
    config: str = typer.Option(None, help="Path to config.yaml"),
    non_interactive: bool = typer.Option(False, "--non-interactive",
                                         help="Accept every recommendation"),
    prefer: str = typer.Option("balanced", help="balanced | speed | memory"),
    show: bool = typer.Option(False, "--show", help="Print the plan without writing"),
) -> None:
    """Configure this machine and write config.local.yaml.

    Detects the hardware, plans a tier-1 index that fits it, lets you pick a generator,
    and saves the result. The server reads that file on every start.
    """
    from lara import setup as SU
    from lara.index.vectors import VectorStore
    from lara.serve import devices as DV

    cfg = config_mod.load(config)
    # --show writes nothing, so let it report on a machine that has no access yet.
    if not show:
        _require_hf(cfg)
    device = DV.detect()
    ecfg = cfg.get_in("embedding")

    # ── 1. what is this machine ────────────────────────────────────────────────
    console.print("\n[bold]1. Hardware[/bold]")
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_row("platform", f"{device.system} / {device.machine}")
    t.add_row("accelerator", f"{device.accelerator}"
                             + (f" — {', '.join(g['name'] for g in device.gpus)}"
                                if device.gpus else ""))
    t.add_row("memory", f"{device.total_ram_gb:.0f} GB RAM"
                        + (f", {device.total_vram_gb:.0f} GB VRAM" if device.total_vram_gb
                           else " (unified)" if device.unified_memory else ""))
    t.add_row("index budget", f"{device.single_device_gb:.0f} GB"
                          + (f" (largest of {len(device.gpus)} GPUs; the index is not sharded)"
                             if len(device.gpus) > 1 else ""))
    t.add_row("generator budget", f"{device.budget_gb:.0f} GB"
                              + (" across all GPUs" if len(device.gpus) > 1 else ""))
    t.add_row("generator backend", f"{device.backend} — {device.backend_reason}")
    console.print(t)
    for n in device.notes:
        console.print(f"  [dim]{n}[/dim]")

    # ── 2. corpus ──────────────────────────────────────────────────────────────
    store = VectorStore(
        cfg.get_path("paths.vectors_fp16"), cfg.get_path("paths.vectors_int8"),
        dim_full=int(ecfg["dim_full"]), dim_trunc=int(ecfg["dim_truncated"]),
    )
    have = store.rows()
    n_chunks = have or SU.REFERENCE_CHUNKS
    console.print("\n[bold]2. Corpus[/bold]")
    if have:
        console.print(f"  {have:,} vectors present")
    else:
        console.print(f"  [yellow]no vectors yet[/yellow] — planning against the published "
                      f"corpus ({SU.REFERENCE_CHUNKS:,} chunks). "
                      f"Fetch it with `lara dataset fetch`.")

    # ── 3. retrieval ───────────────────────────────────────────────────────────
    plan = SU.plan_index(
        device, n_chunks, int(ecfg["dim_truncated"]),
        hot_tier_bytes=int(cfg.get_in("hot_tier.max_bytes", 2_000_000_000)),
        cross_encoder=bool(cfg.get_in("index.rerank.cross_encoder.enabled", True)),
        prefer=prefer,
    )
    console.print("\n[bold]3. Retrieval backend[/bold]")
    opts = [o for o, _, _ in plan.alternatives]
    totals = {o.key: total for o, total, _ in plan.alternatives}
    keys = [o.key for o in opts]
    recommended = plan.option.key

    # Every option is selectable. The memory columns are reported so you can judge the
    # trade yourself; nothing is struck out for being large, because "it does not fit
    # today" depends on the corpus you scope to and what else the machine is doing.
    def backend_table(cursor: int | None) -> Table:
        t = Table(show_header=True, header_style="bold")
        for c, j in (("", "left"), ("search engine", "left"), ("index RAM", "right"),
                     ("total RAM", "right"), ("room for AI model", "right"),
                     ("speed", "right"), ("accuracy", "right"), ("trade-off", "left")):
            t.add_column(c, justify=j, overflow="fold")
        for i, opt in enumerate(opts):
            on_cursor = cursor is not None and i == cursor
            mark = "❯" if on_cursor else ("→" if opt.key == recommended else "")
            style = ("bold cyan" if on_cursor
                     else "green" if opt.key == recommended else "")
            cell = f"[{style}]{opt.label}[/{style}]" if style else opt.label
            # Every column is quoted at the size this machine will actually build. Mixing
            # a full-corpus index with leftover-memory computed after scoping made two
            # columns that could not both be true at once.
            idx = opt.index_gb(int(n_chunks * gen_keep), plan.dim)
            full = opt.index_gb(n_chunks, plan.dim)
            idx_cell = (f"{idx:.1f} GB" if gen_keep >= 1
                        else f"{idx:.1f} GB\n[dim]of {full:.1f} full[/dim]")
            resident = idx + plan.overhead_gb
            head = SU.generator_headroom_gb(device.budget_gb, resident)
            params = SU.generator_params_at_4bit(head)
            gen = (f"{head:.1f} GB\n~{SU.format_params(params)} @4bit" if params >= 5e8
                   else f"{head:.1f} GB\n[red]no room[/red]")
            t.add_row(mark, cell, idx_cell, f"{resident:.1f} GB", gen,
                      f"{opt.p50_ms:.1f}ms", f"{opt.recall:.3f}", opt.note)
        return t

    # Scoping shrinks only the index, and "unnecessary" leaves scope_keep at whatever the
    # solver last computed — which is not 1.0 — so it has to be read as 1.0 here or a
    # machine that needs no scoping is shown a shrunken index it will never build.
    #
    # Backend and keep fraction are one decision, not two: the whole reason to shrink the
    # corpus is to afford a backend and still have room to generate, and that trade is
    # unreadable if you pick the backend on one screen and the fraction on the next.
    # `gen_keep` is mutable state driven by the left/right arrows below.
    KEEP_STEPS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33,
                  0.50, 0.66, 0.75, 0.90, 1.00]
    default_keep = 1.0 if plan.scope == "unnecessary" else plan.scope_keep
    keep_idx = min(range(len(KEEP_STEPS)),
                   key=lambda i: abs(KEEP_STEPS[i] - default_keep))
    gen_keep = KEEP_STEPS[keep_idx]

    def nudge_keep(delta: int) -> bool:
        nonlocal keep_idx, gen_keep
        new = min(max(keep_idx + delta, 0), len(KEEP_STEPS) - 1)
        if new == keep_idx:
            return False
        keep_idx, gen_keep = new, KEEP_STEPS[new]
        return True

    def slider() -> str:
        filled = round(gen_keep * 24)
        bar = "█" * filled + "░" * (24 - filled)
        chunks = int(n_chunks * gen_keep)
        head = "[green]◀ ▶[/green]" if prompt_mod.interactive() else "   "
        return (f"  {head} corpus kept resident  [cyan]{bar}[/cyan]  "
                f"[bold]{gen_keep:>4.0%}[/bold]  ({chunks / 1e6:.1f}M of "
                f"{n_chunks / 1e6:.1f}M chunks)")

    # "+models" hid the fact that a third of the fixed cost is cache, not a model, and
    # named no figure you could check. Spell the addends out instead.
    fixed = [f"embedder {plan.embedder_gb:.1f}"]
    if plan.reranker_gb:
        fixed.append(f"cross-encoder reranker {plan.reranker_gb:.1f}")
    fixed.append(f"tier-0 hot cache {plan.hot_tier_bytes / 1e9:.1f}")
    width = "fp32" if device.accelerator == "cpu" else "half precision"
    pool = ("This machine shares one pool of memory between the search index and the AI "
            "model, so every GB one takes is a GB the other cannot have."
            if device.unified_memory else
            "The index sits on a single card; a generator can be sharded across all of "
            "them, so this is the cautious figure.")
    basis = "at the share of the corpus the slider keeps resident"

    def legend_table() -> Table:
        g = Table(show_header=False, box=None, padding=(0, 2))
        g.add_column(style="bold", no_wrap=True)
        g.add_column(overflow="fold")
        g.add_row("search engine", "which engine stores and searches the vectors")
        g.add_row("index RAM", f"memory the search index alone needs, {basis}")
        g.add_row("total RAM", f"index + {' + '.join(fixed)} — the {plan.overhead_gb:.1f} GB "
                               f"of models and cache is always loaded, whatever you pick")
        g.add_row("room for AI model",
                  f"what is left of this machine's {plan.budget_gb:.0f} GB for the model "
                  f"that writes answers, and the biggest one that fits if it is "
                  f"4-bit quantised (KV cache and activations take {SU.KV_OVERHEAD:.2f}x "
                  f"the weights on top)")
        g.add_row("speed", "typical time for one search — lower is better")
        g.add_row("accuracy", "share of the truly-best results returned; 1.000 is exact, "
                              "0.979 means ~2 in 100 are missed")
        return g

    def print_legend() -> None:
        console.print("  [dim]→ marks the recommendation. Any row can be chosen.[/dim]")
        console.print(legend_table())
        console.print(f"  [dim]The embedder and reranker load in {width} on "
                      f"{device.accelerator} and are not further compressed, so those two "
                      f"figures double on a CPU-only machine. {pool}\n"
                      f"  Speed and accuracy are measured — reproduce with "
                      f"`lara bench-index`.[/dim]")

    def screen(cursor: int | None):
        from rich.console import Group
        return Group(backend_table(cursor), slider())

    chosen = plan.option
    if non_interactive or show:
        console.print(screen(None))
        print_legend()
    elif prompt_mod.interactive():
        print_legend()
        console.print("  [dim]↑/↓ choose a search engine · ◀/▶ move the slider · "
                      "enter to confirm · esc for the recommendation.[/dim]")
        start = keys.index(recommended)
        picked = prompt_mod.select(len(opts), screen, console=console,
                                   initial=start, horizontal=nudge_keep)
        if picked is None:
            gen_keep = default_keep      # esc restores the recommendation wholesale
        chosen = opts[picked] if picked is not None else plan.option
        plan.option = chosen
        console.print(f"  search engine: [bold]{chosen.key}[/bold]   "
                      f"corpus kept: [bold]{gen_keep:.0%}[/bold]")
    else:
        # Not a terminal — piped input, CI, TERM=dumb. No slider, so both halves of the
        # decision get a typed prompt.
        console.print(screen(None))
        print_legend()
        while True:
            pick = typer.prompt(f"\n  backend [{'/'.join(keys)}]", default=recommended)
            pick = pick.strip()
            if pick in keys:
                break
            # Silently falling back to the default here meant a typo picked a backend you
            # did not ask for, and nothing said so.
            console.print(f"  [red]{pick!r} is not one of[/red] {', '.join(keys)}")
        chosen = SU.OPTIONS_BY_KEY[pick]
        plan.option = chosen
        while True:
            raw = typer.prompt("  fraction of the corpus to keep resident (0-1)",
                               default=f"{gen_keep:.2f}")
            try:
                val = float(raw)
            except ValueError:
                console.print(f"  [red]{raw!r} is not a number[/red]")
                continue
            if 0 < val <= 1:
                gen_keep = val
                break
            console.print("  [red]must be greater than 0 and at most 1[/red]")

    # The slider decided *how much* to keep; this decides *which* papers. Fold the
    # chosen fraction back into the plan so the config, the header and the saved
    # generator sizing all describe the machine you just configured.
    plan.scope_keep = gen_keep
    plan.scope = "unnecessary" if gen_keep >= 1.0 else "required"

    # ── 4. what to keep ────────────────────────────────────────────────────────
    console.print("\n[bold]4. Your interests[/bold]")
    # Recomputed for whatever is selected now. plan.reasons was written for the
    # *recommended* option during planning, so after you change the selection it
    # cheerfully reports the memory cost of a backend you did not choose.
    if gen_keep < 1.0:
        console.print(f"  [dim]{chosen.label} over the whole corpus would need "
                      f"{plan.index_gb + plan.overhead_gb:.1f} GB against a "
                      f"{plan.budget_gb:.0f} GB budget.[/dim]")
    topics: list[str] = []
    if gen_keep >= 1.0:
        console.print("  Keeping the whole corpus resident — no topics needed.")
    else:
        console.print(
            f"  Keeping [bold]{gen_keep:.0%}[/bold] of papers means keeping the "
            f"{gen_keep:.0%} most relevant to *you*: "
            f"[bold]{plan.planned_index_gb:.1f} GB[/bold] resident instead of "
            f"{plan.index_gb:.1f} GB ({int(n_chunks * gen_keep):,} chunks). Name the "
            f"topics that matter and they are what survives the cut.")
        console.print("  [dim]Nothing is deleted: dropped papers stay searchable through "
                      "BM25 and open normally. Only their dense vectors leave RAM.[/dim]")
        if not non_interactive and not show:
            while True:
                topic = typer.prompt("  topic to keep (blank when done)", default="",
                                     show_default=False)
                if not topic.strip():
                    break
                topics.append(topic.strip())
            if not topics:
                # Without topics there is nothing to score against, so a keep fraction
                # cannot be honoured. Saying so beats writing a config that silently
                # keeps everything.
                console.print("  [yellow]No topics given[/yellow] — keeping the whole "
                              "corpus resident instead.")
                plan.scope_keep, plan.scope = 1.0, "unnecessary"

    # ── 5. generator ───────────────────────────────────────────────────────────
    console.print("\n[bold]5. Generator[/bold]")
    running = _probe_generators()
    for label, url, _ in running:
        console.print(f"  [green]found running[/green] {label} at {url}")

    model = quant = base_url = None
    if running and not non_interactive and not show:
        if typer.confirm(f"  use {running[0][0]}?", default=True):
            base_url, model = running[0][1], running[0][2]
    elif running:
        # --show previews what --non-interactive would do, so it must make the same choice.
        base_url, model = running[0][1], running[0][2]
        console.print(f"  [dim]would use {base_url}"
                      + (f" serving {model}" if model else "") + "[/dim]")

    if not base_url:
        cached = [m for m in models_mod.scan(cfg.get_path("huggingface.home"),
                                             backend=device.backend) if m.servable]
        fitting = [m for m in cached if DV.fits(m.size_gb, device)["fits"]]
        if not cached:
            console.print("  [yellow]no servable model in the HF cache[/yellow] — "
                          "download one from the reader UI, or run Ollama / LM Studio "
                          "and re-run setup.")
        else:
            t = Table(show_header=True, header_style="bold")
            t.add_column("GB", justify="right"); t.add_column("repo")
            t.add_column("fits?", justify="left")
            for m in sorted(cached, key=lambda m: m.size_gb)[:12]:
                f = DV.fits(m.size_gb, device)
                t.add_row(f"{m.size_gb:.1f}", m.repo,
                          f"[green]yes[/green] ({f['margin_gb']:.0f} GB spare)" if f["fits"]
                          else f"[red]no[/red] (needs {f['needed_gb']:.0f} of "
                               f"{f['budget_gb']:.0f} GB {f['where']})")
            console.print(t)
            if fitting and not non_interactive and not show:
                model = typer.prompt("  model repo (blank to skip)",
                                     default=fitting[-1].repo, show_default=True) or None
            elif fitting and non_interactive:
                model = fitting[-1].repo
            if model:
                m = next((x for x in cached if x.repo == model), None)
                if m and m.runtime_quant_options():
                    quant = m.runtime_quant_options()[0]

    # ── 6. write ───────────────────────────────────────────────────────────────
    overrides = SU.overrides_for(
        plan, model=model, quantization=quant, base_url=base_url,
        disk_root=str(cfg.get_path("disk.root")),
        devices=[int(g) for g in range(len(device.gpus))] if device.gpus else "auto",
        topics=topics,
    )
    # Pin the filesystem the corpus actually lives on. Detectable, and the check that
    # catches a symlink quietly redirecting 30 GB onto the wrong disk.
    backing = preflight_mod._device_for(cfg.get_path("disk.root"))
    if backing and backing != "unknown":
        overrides.setdefault("disk", {})["required_device"] = backing

    # Carry over anything the wizard does not manage — notably hand-written disk safety
    # pins, which it cannot derive and must not silently drop.
    import yaml as _yaml
    existing = {}
    if config_mod.LOCAL_CONFIG.exists():
        existing = _yaml.safe_load(config_mod.LOCAL_CONFIG.read_text()) or {}

    if show:
        console.print("\n[bold]config.local.yaml would be:[/bold]\n")
        console.print(SU.render(config_mod.deep_merge(existing, overrides), device, plan))
        kept = SU.carry_forward(existing, overrides)
        if kept:
            console.print(f"[dim]({', '.join(kept)} carried over from the existing "
                          f"file)[/dim]")
        return

    path, backup, kept = SU.write_local(config_mod.LOCAL_CONFIG, overrides, device, plan,
                                        existing=existing)
    console.print(f"\n[green]wrote[/green] {path}")
    if backup:
        console.print(f"  [dim]previous version saved as {backup.name}[/dim]")
    if kept:
        console.print(f"  [dim]kept your existing {', '.join(kept)}[/dim]")

    checks, ok = preflight_mod.run(config_mod.load())
    bad = [c for c in checks if not c.ok]
    console.print(f"\npreflight: [{'green' if ok else 'red'}]"
                  f"{'passed' if ok else 'FAILED'}[/{'green' if ok else 'red'}]")
    for c in bad:
        console.print(f"  [red]{c.name}[/red]: {c.detail}")

    console.print("\n[bold]next[/bold]")
    if not have:
        console.print("  lara dataset fetch <source>")
    console.print("  lara serve")
    if topics:
        # The keep-set is built from the config on first start, so there is no command
        # to run here. Say what will happen rather than handing over a chore.
        console.print(f"  [dim]…which builds the {plan.effective_keep:.0%} keep-set from "
                      f"{len(topics)} topic(s) the first time, then reuses it. Inspect or "
                      f"override it with `lara corpus scope --preview`.[/dim]")
    elif plan.scope != "unnecessary":
        console.print("  [dim]…no topics were given, so the whole corpus stays "
                      "resident.[/dim]")


@app.command("bench-index")
def bench_index(
    config: str = typer.Option(None, help="Path to config.yaml"),
    n: int = typer.Option(2_000_000, help="Vectors to benchmark over (0 = whole corpus)"),
    queries: int = typer.Option(20, help="Queries to time"),
    k: int = typer.Option(200, help="Candidates per query, as tier 1 is actually used"),
    only: str = typer.Option(None, help="Comma-separated subset, e.g. 'faiss sq8,torch int8'"),
) -> None:
    """Measure every tier-1 backend on THIS machine, so the choice is not a guess.

    Recall is reported against exact fp16, which is the definition of correct here — not
    against ground truth, which tier 1 was never trying to produce on its own.
    """
    import time

    import numpy as np

    from lara.index import backends as BK
    from lara.index.vectors import VectorStore

    cfg = config_mod.load(config)
    ecfg = cfg.get_in("embedding")
    store = VectorStore(
        cfg.get_path("paths.vectors_fp16"), cfg.get_path("paths.vectors_int8"),
        dim_full=int(ecfg["dim_full"]), dim_trunc=int(ecfg["dim_truncated"]),
    )
    total = store.rows()
    if total == 0:
        console.print("[red]no vectors[/red] — build or fetch the corpus first")
        raise typer.Exit(1)

    use = total if not n else min(n, total)
    mat = store.load_int8(mmap=True)[:use]
    console.print(f"benchmarking [bold]{use:,}[/bold] of {total:,} vectors, "
                  f"dim {mat.shape[1]}, k={k}, {queries} queries\n")

    rng = np.random.default_rng(0)
    picks = rng.choice(use, queries, replace=False)
    qs = [(lambda v: (v / np.linalg.norm(v)).astype(np.float16))(mat[i].astype(np.float32))
          for i in picks]

    candidates = [
        ("torch fp16", lambda: BK.TorchBackend(mat, precision="fp16")),
        ("torch int8", lambda: BK.TorchBackend(mat, precision="int8")),
        ("torch fp16 cpu", lambda: BK.TorchBackend(mat, device="cpu", precision="fp16")),
        ("faiss flat", lambda: BK.FaissBackend(mat, cfg=BK.FaissConfig(kind="flat"))),
        ("faiss sq8", lambda: BK.FaissBackend(mat, cfg=BK.FaissConfig(kind="sq8"))),
        ("faiss hnsw", lambda: BK.FaissBackend(mat, cfg=BK.FaissConfig(kind="hnsw"))),
    ]
    if only:
        want = {s.strip().lower() for s in only.split(",")}
        candidates = [c for c in candidates if c[0].lower() in want]
    if not BK.faiss_available():
        candidates = [c for c in candidates if not c[0].startswith("faiss")]
        console.print("[yellow]faiss not installed[/yellow] — skipping those "
                      "(pip install 'lara[cpu]')\n")

    gold: list[set[int]] = []
    table = Table(show_header=True, header_style="bold")
    for col, just in (("backend", "left"), ("memory", "right"), ("build", "right"),
                      ("p50", "right"), ("p95", "right"), ("recall@k", "right")):
        table.add_column(col, justify=just)

    for name, make in candidates:
        try:
            t0 = time.time()
            ix = make()
            build_s = time.time() - t0
            lat = []
            got = []
            for q in qs:
                t0 = time.time()
                rows, _ = ix.search(q, k=k)
                lat.append((time.time() - t0) * 1000)
                got.append(set(rows.tolist()))
            if not gold:
                gold = got                       # first backend is exact fp16 = reference
            rec = sum(len(a & b) for a, b in zip(got, gold)) / max(1, sum(len(g) for g in gold))
            lat.sort()
            table.add_row(
                name, f"{ix.memory_bytes() / 1e9:.2f} GB", f"{build_s:.1f}s",
                f"{lat[len(lat) // 2]:.1f}ms", f"{lat[int(len(lat) * 0.95) - 1]:.1f}ms",
                f"{rec:.3f}",
            )
            del ix
            ldev.empty_cache()
        except Exception as exc:                  # a backend that cannot build is a result
            table.add_row(name, "-", "-", "-", "-", f"[red]{type(exc).__name__}[/red]")

    console.print(table)
    scale = total / use if use else 1
    console.print(f"\n[dim]Memory scales linearly: multiply by {scale:.1f}x for the full "
                  f"{total:,}-vector corpus. Recall is measured against exact fp16.[/dim]")
    console.print("[dim]Write the winner into config.local.yaml as index.backend / "
                  "index.precision / index.faiss.kind.[/dim]")


corpus_app = typer.Typer(help="Corpus residency — what stays in RAM")
app.add_typer(corpus_app, name="corpus")


def _scope_inputs(cfg, topics: list[str], device: str | None):
    """Embed the topics and load the paper-level index they are scored against.

    The work lives in ``scope.inputs`` so the server's automatic build and this command
    score against the same vectors; this wrapper only turns its error into CLI output.
    """
    from lara.index import scope as SC

    try:
        return SC.inputs(cfg, topics, device)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e


@corpus_app.command("scope")
def corpus_scope(
    topic: list[str] = typer.Option(
        None, "--topic", "-t",
        help="Topic to keep, repeatable. Papers score by their NEAREST topic."),
    config: str = typer.Option(None, help="Path to config.yaml"),
    keep: float = typer.Option(0.1, help="Fraction of papers to keep resident (0-1)"),
    expand: int = typer.Option(
        3, help="Also keep papers cited by >= N kept papers (0 disables)"),
    preview: bool = typer.Option(False, "--preview", help="Show the cut without applying"),
    apply: bool = typer.Option(False, "--apply", help="Write the keep-set"),
    device: str = typer.Option(None, help="Override device; default auto-detects"),
) -> None:
    """Keep only the topically relevant slice of the corpus resident in RAM (D22)."""
    from lara.index import scope as SC

    if not topic:
        console.print("[red]at least one --topic is required[/red]")
        raise typer.Exit(1)
    if not 0 < keep <= 1:
        console.print("[red]--keep must be in (0, 1][/red]")
        raise typer.Exit(1)
    if not (preview or apply):
        preview = True      # default to the safe half

    cfg = config_mod.load(config)
    conn, vecs, paper_int8, row_to_id, dim = _scope_inputs(cfg, list(topic), device)

    if preview and not apply:
        cut = SC.cut_preview(conn, vecs, paper_int8, row_to_id, keep)
        console.print(f"\n[bold]{cut['n_keep']:,}[/bold] of {cut['n_total']:,} papers "
                      f"at keep={keep}\n")

        def show(title, rows, style):
            if not rows:
                return
            t = Table(show_header=True, header_style="bold", title=title, title_justify="left")
            t.add_column("rank", justify="right")
            t.add_column("score", justify="right")
            t.add_column("arxiv")
            t.add_column("title", overflow="fold")
            for r in rows:
                t.add_row(f"{r['rank']:,}", f"{r['score']:.3f}", r["arxiv_id"],
                          f"[{style}]{r['title'][:80]}[/{style}]")
            console.print(t)

        show("best matches", cut["top"], "green")
        show("last kept — the cut line is below this", cut["kept_edge"], "green")
        show("first dropped", cut["dropped_edge"], "yellow")
        console.print(
            "\n[dim]A similarity cut is not a clean topical boundary. Dropped papers stay "
            "searchable via BM25 and open normally; only their dense vectors leave RAM.[/dim]"
        )
        console.print("[dim]Re-run with --apply to write this keep-set.[/dim]")
        conn.close()
        return

    console.print(f"scoring {len(row_to_id):,} papers against {len(topic)} topic(s)…")
    sc = SC.build(conn, vecs, list(topic), paper_int8, row_to_id, keep,
                  expand_min_citations=expand, dim_truncated=dim)
    where = sc.save(cfg.get_path("disk.root"))
    conn.close()

    via = sc.by_via()
    full = sc.corpus_chunks * dim * 2
    console.print(
        f"\n[green]scope written[/green] to {where}\n"
        f"  papers    {sc.n_papers:,}  "
        f"({via.get('topic', 0):,} by topic + {via.get('citation', 0):,} by citation)\n"
        f"  chunks    {sc.n_rows:,} of {sc.corpus_chunks:,} "
        f"({100 * sc.n_rows / max(1, sc.corpus_chunks):.1f}%)\n"
        f"  tier-1    {sc.resident_bytes() / 1e9:.2f} GB resident "
        f"(was {full / 1e9:.2f} GB)\n"
    )
    console.print("[dim]Restart the reader, or POST /api/reload, to apply it.[/dim]")


@corpus_app.command("scope-status")
def corpus_scope_status(config: str = typer.Option(None, help="Path to config.yaml")) -> None:
    """Show the active keep-set, if any."""
    from lara.index import scope as SC

    cfg = config_mod.load(config)
    sc = SC.Scope.load(cfg.get_path("disk.root"))
    if sc is None:
        console.print("no scope set — the whole corpus is resident")
        return
    via = sc.by_via()
    console.print(
        f"[bold]topics[/bold]   {', '.join(sc.topics)}\n"
        f"[bold]keep[/bold]     {sc.fraction}  (expand >= {sc.expand_min_citations} citations)\n"
        f"[bold]papers[/bold]   {sc.n_papers:,} "
        f"({via.get('topic', 0):,} topic + {via.get('citation', 0):,} citation)\n"
        f"[bold]chunks[/bold]   {sc.n_rows:,} of {sc.corpus_chunks:,} at build time\n"
        f"[bold]tier-1[/bold]   {sc.resident_bytes() / 1e9:.2f} GB\n"
        f"[bold]created[/bold]  {sc.created_utc}"
    )


@corpus_app.command("unscope")
def corpus_unscope(config: str = typer.Option(None, help="Path to config.yaml")) -> None:
    """Drop the keep-set and go back to a fully resident corpus."""
    from lara.index import scope as SC

    cfg = config_mod.load(config)
    if SC.Scope.clear(cfg.get_path("disk.root")):
        console.print("[green]scope cleared[/green] — restart the reader to apply")
    else:
        console.print("no scope was set")


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


@dataset_app.command("pull")
def dataset_pull(
    repo: str = typer.Option(None, help="Hugging Face dataset repo (default: the published corpus)"),
    config: str = typer.Option(None, help="Path to config.yaml"),
    tiers: str = typer.Option("core", help="core | full | archive (comma-separated)"),
    revision: str = typer.Option(None, help="Branch, tag or commit to pin"),
    list_only: bool = typer.Option(False, "--list", help="Show what is in the repo and exit"),
    workers: int = typer.Option(8, help="Parallel download workers"),
    extract: bool = typer.Option(True, help="Unpack raw/*.tar after pulling the archive tier"),
) -> None:
    """Download the corpus from Hugging Face. No account needed; resumes if interrupted.

    Tiers: `core` gives working search and answers, `full` adds the fp16 vectors for exact
    rescoring, `archive` adds the raw crawled HTML (only useful if you intend to re-parse).
    """
    from huggingface_hub import list_repo_files, snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    from lara.serve import dataset as DS

    cfg = config_mod.load(config)
    root = cfg.get_path("disk.root")
    repo = repo or DS.HF_REPO
    want = tuple(t.strip() for t in tiers.split(",") if t.strip())
    unknown = [t for t in want if t not in DS.HF_TIER_PREFIXES]
    if unknown:
        console.print(f"[red]unknown tier(s)[/red] {', '.join(unknown)}; "
                      f"choose from {', '.join(DS.HF_TIER_PREFIXES)}")
        raise typer.Exit(1)

    # The corpus itself is ungated, but a corpus with no encoder cannot be searched.
    # Checking before ~50 GB moves is the whole point of checking at all.
    if not list_only:
        _require_hf(cfg)

    try:
        files = list_repo_files(repo, repo_type="dataset", revision=revision)
    except RepositoryNotFoundError:
        console.print(f"[red]{repo} not found[/red] as a dataset repo. Check the name, and "
                      f"note the Hub returns the same error for private repos as for "
                      f"missing ones.")
        raise typer.Exit(1) from None
    except GatedRepoError:
        console.print(f"[red]{repo} is gated[/red] — accept its terms on the Hub, then "
                      f"`huggingface-cli login`.")
        raise typer.Exit(1) from None

    patterns = DS.hf_patterns(want)
    import fnmatch
    chosen = [f for f in files if any(fnmatch.fnmatch(f, p) for p in patterns)]

    if list_only or not chosen:
        table = Table(show_header=True, header_style="bold")
        table.add_column("file"); table.add_column("tier")
        for f in sorted(files):
            tier = next((t for t, ps in DS.HF_TIER_PREFIXES.items()
                         if any(f.startswith(p) for p in ps)), "-")
            table.add_row(f, tier)
        console.print(table)
        console.print(f"\n{len(files)} files in [bold]{repo}[/bold]")
        if not chosen and not list_only:
            console.print(f"[yellow]nothing matches tier(s) {', '.join(want)}[/yellow] — "
                          f"the upload may still be in progress.")
        return

    console.print(f"pulling [bold]{len(chosen)}[/bold] file(s) from {repo} "
                  f"(tiers: {', '.join(want)}) into {root}")
    console.print("[dim]Resumable: re-run after an interruption and it continues. "
                  "Hub downloads are checksummed on arrival.[/dim]\n")
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo, repo_type="dataset", revision=revision,
        local_dir=str(root), allow_patterns=patterns, max_workers=int(workers),
    )
    # The archive tier arrives as one tar per year; nothing else in the codebase reads a
    # tar, so without this step it is 40 GB the parser cannot see.
    if "archive" in want and extract:
        console.print("\nunpacking the raw archive…")
        done, skipped = DS.extract_archives(root)
        if done:
            console.print(f"  extracted {len(done)} tar(s) into {root / 'raw'}")
        if skipped:
            console.print(f"  [dim]{len(skipped)} already extracted, left alone[/dim]")
        tars = sorted((root / "raw").glob("raw-*.tar"))
        if tars:
            freeable = sum(t.stat().st_size for t in tars) / 1e9
            console.print(
                f"  [dim]the .tar files are kept so re-running `pull` stays a no-op; "
                f"delete them to reclaim {freeable:.0f} GB once you are happy:[/dim]\n"
                f"  [dim]  rm {root / 'raw'}/raw-*.tar[/dim]"
            )
    elif "archive" in want:
        console.print(
            "\n[yellow]--no-extract given[/yellow]: raw/*.tar are on disk but the parser "
            "reads raw/{YYMM}/*.zst. Unpack them from the corpus root, not from inside "
            f"raw/:\n  tar xf {root / 'raw'}/raw-2015.tar -C {root}"
        )

    console.print(f"\n[green]done[/green] — {root}")
    console.print("next: [bold]lara setup[/bold], then [bold]lara serve[/bold]")


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
    no_llm: bool = typer.Option(False, "--no-llm",
                                help="Do not start a generator; retrieval only"),
) -> None:
    """Run the reader server, starting the generation backend alongside it.

    The generator is stopped when the reader exits, unless it was already running, in
    which case it is adopted and left alone.
    """
    import os

    import uvicorn

    cfg = config_mod.load(config)
    if config:
        os.environ["LARA_CONFIG"] = config
    host = host or cfg.get_in("serving.host", "127.0.0.1")
    port = port or int(cfg.get_in("serving.port", 8080))

    # Fail closed. Binding off loopback without a token used to expose model downloads
    # (using the operator's HF_TOKEN), free GPU inference, the arXiv crawler and every
    # delete endpoint to anything that could route here. Refusing to start is the only
    # version of this that survives a hurried evening.
    from lara.serve import auth as AUTH

    token = AUTH.resolve_token(cfg)
    if AUTH.require_token_for(host, cfg) and not token:
        fresh = AUTH.generate_token()
        console.print(
            f"[red]refusing to serve on {host} without authentication.[/red]\n\n"
            f"Every endpoint is reachable by anyone who can route here, including model "
            f"downloads that use your HF_TOKEN, generation on your GPU, and the endpoints "
            f"that delete your library.\n\nSet a token and try again:\n\n"
            f"  [bold]export LARA_TOKEN={fresh}[/bold]\n\n"
            f"or put it in config.local.yaml under serving.auth.token, then open\n"
            f"  http://{host}:{port}/?token=$LARA_TOKEN\n\n"
            f"[dim]To bind loopback-only instead, drop --host. To disable this check "
            f"deliberately, set serving.auth.mode: off[/dim]"
        )
        raise typer.Exit(1)
    if token:
        os.environ["LARA_TOKEN"] = token          # the app builds its middleware from this
        console.print(f"[green]authentication on[/green] — open "
                      f"http://{host}:{port}/?token=<your token> once per browser")
    elif not AUTH.is_loopback(host):
        console.print("[yellow]serving without authentication[/yellow] (auth.mode: off)")

    gen = None
    if not no_llm and cfg.get_in("serving.generator.autostart", True):
        from lara.serve import devices as DV
        from lara.serve import generator as GEN

        gen = GEN.from_config(cfg, DV.detect().accelerator)
        if gen is None:
            console.print("[yellow]no generator configured[/yellow] — retrieval will work, "
                          "answers will not. Run `lara setup` to pick one.")
        else:
            gen.start()
            if gen.adopted:
                console.print(f"[dim]using the {gen.backend.label} already serving at "
                              f"{gen.base_url}[/dim]")
            elif not gen.backend.available():
                console.print(f"[yellow]{gen.backend.label} is not installed[/yellow] — "
                              f"{gen.backend.install_hint}. Retrieval works; answers will "
                              f"not until a generator is running.")
            elif gen.proc is None:
                console.print(f"[yellow]{gen.backend.label} is configured but nothing is "
                              f"running at {gen.base_url}[/yellow]")
            else:
                budget = int(cfg.get_in("serving.generator.startup_timeout_sec", 300))
                console.print(f"starting {gen.backend.label} with {gen.model} "
                              f"[dim](up to {budget}s; log: {gen.log_path})[/dim]")
                if gen.wait_ready(timeout=budget):
                    console.print(f"[green]{gen.backend.label} ready[/green]")
                else:
                    console.print(f"[red]{gen.backend.label} did not come up[/red] — "
                                  f"see {gen.log_path}. Retrieval still works.")

    console.print(f"[bold]http://{host}:{port}[/bold]  (warming models — first start is slow)")
    try:
        # One worker: the GPU index, embedder and reranker are process-local and would be
        # duplicated per worker, tripling VRAM for no throughput gain on a single-user tool.
        uvicorn.run("lara.serve.app:app", host=host, port=port, workers=1, log_level="info")
    finally:
        # Only stops what we started. A server that was already running is left alone --
        # it may be shared with something else, and killing it is not this process's call.
        if gen is not None and gen.proc is not None:
            console.print(f"stopping {gen.backend.label}…")
            gen.stop()


@app.command("serve-llm")
def serve_llm(
    config: str = typer.Option(None, help="Path to config.yaml"),
    model: str = typer.Option(None, help="Override the model"),
    backend: str = typer.Option(None, help="vllm | llamacpp | mlx | external"),
    show: bool = typer.Option(False, "--show", help="Print the command instead of running"),
) -> None:
    """Start the generation server on its own.

    `lara serve` does this for you; this command exists for running the generator on a
    different machine, or keeping it up across reader restarts (model loads are slow).
    """
    from lara.serve import devices as DV
    from lara.serve import generator as GEN

    cfg = config_mod.load(config)
    accel = DV.detect().accelerator
    chosen = GEN.choose(accel, backend or cfg.get_in("serving.generator.backend"))
    serving = cfg.get_in("serving") or {}
    merged = {**(serving.get("generator") or {}), "vllm": serving.get("vllm") or {}}
    repo = model or GEN.model_for(chosen.name, merged)

    if chosen.name == "external":
        console.print("backend is [bold]external[/bold] — nothing to start. "
                      f"lara will use {serving.get('vllm', {}).get('base_url')}")
        return
    if not repo:
        console.print(f"[red]no model configured for {chosen.label}[/red] "
                      f"({chosen.model_format}). Set serving.generator.{chosen.name}.model "
                      f"or run `lara setup`.")
        raise typer.Exit(1)
    if not chosen.available():
        console.print(f"[red]{chosen.label} is not installed[/red] — {chosen.install_hint}")
        raise typer.Exit(1)

    port = GEN.port_of((serving.get("vllm") or {}).get("base_url", "http://127.0.0.1:8000/v1"))
    cmd = chosen.command(repo, port, merged)
    env = chosen.env(merged)
    if "CUDA_VISIBLE_DEVICES" in env and chosen.name == "vllm":
        console.print(f"[dim]CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}[/dim]")
    console.print(" ".join(cmd))
    if show:
        return
    import subprocess
    subprocess.run(cmd, env=env, check=False)


@app.command("backends")
def list_backends(config: str = typer.Option(None, help="Path to config.yaml")) -> None:
    """Show the generation backends, which are installed, and which one would be used."""
    from lara.serve import devices as DV
    from lara.serve import generator as GEN

    cfg = config_mod.load(config)
    dev_info = DV.detect()
    serving = cfg.get_in("serving") or {}
    merged = {**(serving.get("generator") or {}), "vllm": serving.get("vllm") or {}}
    active = GEN.choose(dev_info.accelerator, cfg.get_in("serving.generator.backend"))

    table = Table(show_header=True, header_style="bold")
    for c in ("", "backend", "format", "installed", "model", "notes"):
        table.add_column(c, overflow="fold")
    for b in GEN.BACKENDS.values():
        fit = dev_info.accelerator in b.platforms or b.name == "external"
        table.add_row(
            "→" if b.name == active.name else "",
            b.label if fit else f"[dim]{b.label}[/dim]",
            b.model_format,
            "[green]yes[/green]" if b.available() else f"[dim]no — {b.install_hint}[/dim]",
            str(GEN.model_for(b.name, merged) or "-"),
            b.notes.split(". ")[0] + ".",      # full text lives in the module docstring
        )
    console.print(table)
    console.print(f"\ndetected [bold]{dev_info.accelerator}[/bold]; "
                  f"→ marks what `lara serve` would start")
    base = (serving.get("vllm") or {}).get("base_url", "http://127.0.0.1:8000/v1")
    ids = GEN.probe(base)
    console.print(f"already running at {base}: "
                  + (f"[green]yes[/green] — {', '.join(ids) or 'no models listed'}"
                     if ids is not None else "[dim]no[/dim]"))


@app.command("bench-generate")
def bench_generate(
    config: str = typer.Option(None, help="Path to config.yaml"),
    prompt: str = typer.Option("Explain the role of the KV cache in transformer inference.",
                               help="Prompt to time"),
    max_tokens: int = typer.Option(256),
    runs: int = typer.Option(3, help="Timed runs after one warm-up"),
) -> None:
    """Measure time-to-first-token and throughput against the running generator.

    Backend-agnostic on purpose: it speaks the same OpenAI-compatible API the reader uses,
    so llama.cpp and MLX can be compared on one machine by pointing this at each in turn.
    """
    import json
    import time

    import httpx

    from lara.serve import generator as GEN

    cfg = config_mod.load(config)
    base = (cfg.get_in("serving.vllm.base_url") or "http://127.0.0.1:8000/v1").rstrip("/")
    ids = GEN.probe(base)
    if ids is None:
        console.print(f"[red]nothing answering at {base}[/red] — start one with "
                      f"`lara serve-llm`")
        raise typer.Exit(1)
    model = ids[0]
    console.print(f"benchmarking [bold]{model}[/bold] at {base}\n")

    def once() -> tuple[float, float, int]:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0.0, "stream": True}
        t0 = time.time()
        ttft = None
        n = 0
        with httpx.stream("POST", f"{base}/chat/completions", json=body, timeout=300) as r:
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                except Exception:
                    continue
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    n += 1
        return (ttft or 0.0), time.time() - t0, n

    once()                                    # warm the prefix cache and the kernels
    rows = [once() for _ in range(runs)]
    table = Table(show_header=True, header_style="bold")
    for c in ("run", "TTFT", "total", "tokens", "tok/s"):
        table.add_column(c, justify="right")
    for i, (ttft, total, n) in enumerate(rows, 1):
        rate = (n - 1) / (total - ttft) if n > 1 and total > ttft else 0.0
        table.add_row(str(i), f"{ttft * 1000:.0f}ms", f"{total:.2f}s", str(n), f"{rate:.1f}")
    console.print(table)
    med = sorted(r[0] for r in rows)[len(rows) // 2]
    rates = sorted((n - 1) / (t - f) for f, t, n in rows if n > 1 and t > f)
    console.print(f"\nmedian TTFT [bold]{med * 1000:.0f}ms[/bold]"
                  + (f", median [bold]{rates[len(rates) // 2]:.1f} tok/s[/bold]" if rates else ""))


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
