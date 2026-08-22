"""Querying the built corpus from the terminal, and benchmarking the index."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from lara import config as config_mod
from lara.cli._base import app, console


@app.command()
def search(
    query: str = typer.Argument(..., help="Question to retrieve for"),
    config: str = typer.Option(None, help="Path to config.yaml"),
    k: int = typer.Option(8, help="Results to show"),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip the cross-encoder"),
    no_bm25: bool = typer.Option(False, "--no-bm25", help="Dense only"),
    paper: str = typer.Option(None, help="Restrict to one arXiv id"),
    corpus: str = typer.Option(None, help="Search a `lara kb` corpus instead of arXiv"),
    device: str = typer.Option(None, help="Override device; default auto-detects"),
) -> None:
    """Retrieve chunks for a query and show anchored citations with timings."""
    from lara import device as ldev
    from lara.index import embed as emb
    from lara.index import retrieve as R
    from lara.index.vectors import VectorStore
    from lara.store import db

    cfg = config_mod.load(config)
    ecfg, icfg = cfg.get_in("embedding"), cfg.get_in("index")

    # A built corpus exposes `papers` and `sections` as views over its own tables, so the
    # retriever below does not know or care which kind it was handed. See the schema note
    # in lara/corpus/build.py.
    if corpus:
        from lara.corpus import build as C
        from lara.corpus.store import Registry

        c = Registry(Path(cfg.get_in("paths")["corpora"])).get(corpus)
        if not c.built:
            console.print(f"[red]{corpus} is not built[/red] — "
                          f"run [bold]lara kb build {corpus}[/bold]")
            raise typer.Exit(1)
        conn = C.connect(c.db_path)
        db_paths = (c.fp16_path, c.int8_path)
    else:
        conn = db.connect(cfg.get_path("paths.metadata_db"))
        db_paths = (cfg.get_path("paths.vectors_fp16"),
                    cfg.get_path("paths.vectors_int8"))

    store = VectorStore(
        *db_paths,
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
    console.print(f"index memory: {retr.dense.memory_bytes()/1e9:.2f} GB\n")

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

    from lara import device as ldev
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

    # Recall is meaningless without a fixed reference. Taking "whatever ran first" meant
    # `--only "faiss hnsw"` made HNSW its own gold and reported 1.000 — the one number the
    # flag exists to interrogate, silently inverted. Exact fp16 is always the reference,
    # built separately when it is not among the selected rows (it is also the cheapest to
    # build, so this costs little).
    gold: list[set[int]] = []
    if not any(name == "torch fp16" for name, _ in candidates):
        console.print("[dim]building exact fp16 as the recall reference "
                      "(not selected for timing)…[/dim]")
        ref = BK.TorchBackend(mat, precision="fp16")
        gold = [set(ref.search(q, k=k)[0].tolist()) for q in qs]
        del ref
        ldev.empty_cache()
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
