"""``lara corpus`` -- which slice of the corpus stays resident in memory (D22)."""

from __future__ import annotations

import typer
from rich.table import Table

from lara import config as config_mod
from lara.cli._base import app, console

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
