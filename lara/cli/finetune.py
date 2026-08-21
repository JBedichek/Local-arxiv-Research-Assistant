"""Fine-tuning the embedder: mining training pairs, sweeping, checking, training."""

from __future__ import annotations

import typer

from lara import config as config_mod
from lara import device as ldev
from lara.cli._base import app, console


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
    concurrency: int = typer.Option(
        32, help="Question generations in flight at once; vLLM serves max_num_seqs=64"),
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
                                      topics=topic_list or None,
                                      gen_concurrency=concurrency, progress=p))
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
    contextual: bool = typer.Option(
        True, help="Render training documents as the corpus is embedded "
                   "(title > section | text). --no-contextual is the ablation."),
    seed: int = typer.Option(0, help="Training seed; vary it to get error bars"),
    sam_rho: float = typer.Option(
        0.0, help="Sharpness-Aware Minimisation radius; 0 off. Doubles step cost."),
    ema_decay: float = typer.Option(
        0.0, help="EMA of weights, e.g. 0.999; 0 off. Nearly free."),
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

    triples = KF.make_triples(conn, max_per_query=max_per_query, contextual=contextual)
    n_q = len({t.query_hash for t in triples})
    if len(triples) < batch_size * 4:
        console.print(f"[red]only {len(triples):,} triples[/red] — run `lara explore` first")
        raise typer.Exit(1)
    rec = KF.Recipe(lr_muon=lr_muon, lr_adam=lr_muon / 5, batch_size=batch_size,
                    micro_batch=micro_batch, epochs=epochs, max_seq_length=max_seq_length,
                    patience=patience, eval_every=eval_every,
                    sam_rho=sam_rho, ema_decay=ema_decay, seed=seed,
                    compile_mode=None if compile_mode.lower() == "none" else compile_mode)
    console.print(f"[bold]{len(triples):,}[/bold] triples from {n_q:,} queries · "
                  f"MultipleNegativesRanking · batch {batch_size} · seq {max_seq_length} · "
                  f"docs {'contextual' if contextual else 'bare (ablation)'}"
                  + (f" · SAM rho={sam_rho}" if sam_rho else "")
                  + (f" · EMA {ema_decay}" if ema_decay else ""))

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
        seen_epoch = 0
        while True:
            st = yield
            if st.get("early_stop"):
                console.print(
                    f"  [yellow]early stop[/yellow] at step {st['step']}/{st['steps']} — "
                    f"epoch {st.get('epoch','?')}/{st.get('epochs','?')}, "
                    f"step {st.get('epoch_step','?')}/{st.get('steps_per_epoch','?')} "
                    f"within it = [bold]{st.get('epoch_frac', 0):.2f} epochs[/bold] · "
                    f"best val {st['best_val']:.4f}")
                continue
            # A line whenever the epoch rolls over, so the trace shows where the
            # boundaries are without reading step arithmetic.
            ep = st.get("epoch")
            if ep and ep != seen_epoch:
                seen_epoch = ep
                console.print(f"  [dim]--- epoch {ep}/{st.get('epochs')} "
                              f"({st.get('steps_per_epoch')} steps) ---[/dim]")
            v = st.get("val_loss")
            frac = st.get("epoch_frac")
            if frac is None and st.get("steps_per_epoch"):
                frac = st["step"] / st["steps_per_epoch"]
            console.print(f"  step {st['step']:>4}/{st['steps']}"
                          + (f"  ep {frac:5.2f}" if frac is not None else "")
                          + f"  loss {st['loss']:7.4f}  lr {st['lr']:.2e}"
                          + (f"  val {v:7.4f}" if v is not None else "")
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
