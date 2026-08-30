"""``lara kb`` — build and manage corpora that are not arXiv.

The interactive flow is the point of this module:

    lara kb new flight-manuals

    Goal? the systems and emergency procedures of the Cessna 172S
      -> the model proposes eight queries; you edit them
      -> each is searched, each result fetched, read and scored
      -> you see title, size, licence and a preview, and accept or reject
      -> build: chunk, embed, index

**Nothing is downloaded without being counted, and nothing is built without being shown.**
Both halves matter. A builder that silently pulled four gigabytes would be a liability on a
laptop, and one that silently embedded a marketing page would quietly poison every search
afterwards. So the budget is displayed as it is spent and adjustable in place, and every
document is a decision the reader makes with the licence in front of them.

The commands are thin: everything real is in :mod:`lara.corpus.builder`, so the same flow
can be driven from the web UI later without reimplementing any of it.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer
from rich.table import Table

from lara.cli._base import app, console
from lara.corpus.store import human

kb = typer.Typer(add_completion=False, help="Build and manage your own corpora")
app.add_typer(kb, name="kb")

SIZE = re.compile(r"^\s*([\d.]+)\s*([kmgt]?)b?\s*$", re.I)
UNITS = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}


def parse_size(text: str) -> int | None:
    """'500MB', '2g', '1024' -> bytes. None if it is not a size."""
    m = SIZE.match(text or "")
    if not m:
        return None
    return int(float(m.group(1)) * UNITS[m.group(2).lower()])


def _registry(cfg):
    from lara.corpus.store import Registry
    return Registry(Path(cfg.get_in("paths")["corpora"]))


def _config(path: str | None):
    from lara import config as config_mod
    return config_mod.load(path)


def _licence_style(verdict: str) -> str:
    return {"public-domain": "green", "permissive": "green", "restricted": "yellow",
            "copyrighted": "red", "unknown": "dim"}.get(verdict, "dim")


# ── listing and inspection ──────────────────────────────────────────────────────────

@kb.command("list")
def list_corpora(config: str = typer.Option(None, help="Path to config.yaml")) -> None:
    """Show every corpus and whether it is ready to search."""
    cfg = _config(config)
    rows = _registry(cfg).list()
    if not rows:
        console.print("[dim]No corpora yet. Create one with[/dim] "
                      "[bold]lara kb new <name>[/bold]")
        return
    t = Table(box=None, pad_edge=False)
    for col in ("name", "sources", "chunks", "on disk", "built", "goal"):
        t.add_column(col, justify="right" if col in ("sources", "chunks", "on disk") else "left")
    for c, r in rows:
        t.add_row(c.root.name, str(len(r.accepted())), f"{r.chunks:,}",
                  human(c.size_on_disk()),
                  "[green]yes[/green]" if c.built else "[yellow]no[/yellow]",
                  (r.goal or "")[:44])
    console.print(t)


@kb.command()
def show(name: str, config: str = typer.Option(None, help="Path to config.yaml")) -> None:
    """Every source in one corpus, with its licence and verdict."""
    from lara.corpus import builder as B

    cfg = _config(config)
    c = _registry(cfg).get(name)
    if not c.recipe_path.exists():
        console.print(f"[red]No corpus named[/red] {name}")
        raise typer.Exit(1)
    r = c.load()
    console.print(f"[bold]{c.root.name}[/bold]  {r.goal or '[dim]no goal set[/dim]'}")
    console.print(f"budget {human(r.text_budget)}   on disk {human(c.size_on_disk())}   "
                  f"{'built ' + r.built_utc if r.built_utc else '[yellow]not built[/yellow]'}")
    t = Table(box=None, pad_edge=False)
    for col in ("", "rel", "size", "licence", "title"):
        t.add_column(col)
    for s in r.sources:
        mark = {"accepted": "[green]+[/green]", "rejected": "[red]-[/red]"}.get(s.decided, "[yellow]?[/yellow]")
        t.add_row(mark, f"{s.relevance:.2f}" if s.relevance is not None else " — ",
                  human(s.chars),
                  f"[{_licence_style(s.licence)}]{s.licence}[/]", (s.title or s.url)[:52])
    console.print(t)
    console.print("\n" + B.summarise(r))


@kb.command()
def delete(name: str, config: str = typer.Option(None, help="Path to config.yaml"),
           yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation")) -> None:
    """Delete a corpus and everything in it."""
    cfg = _config(config)
    c = _registry(cfg).get(name)
    if not c.recipe_path.exists():
        console.print(f"[red]No corpus named[/red] {name}")
        raise typer.Exit(1)
    size = c.size_on_disk()
    if not yes and not typer.confirm(f"Delete {c.root.name} and its {human(size)}?"):
        raise typer.Exit()
    c.delete()
    console.print(f"Deleted {c.root.name}, freed {human(size)}.")


@kb.command()
def budget(name: str, set_to: str = typer.Argument(..., metavar="SIZE",
                                                   help="e.g. 2GB, 500MB"),
           config: str = typer.Option(None, help="Path to config.yaml")) -> None:
    """Change how much raw text a corpus may accumulate."""
    cfg = _config(config)
    c = _registry(cfg).get(name)
    r = c.load()
    n = parse_size(set_to)
    if n is None:
        console.print(f"[red]Not a size:[/red] {set_to}   try 2GB, 500MB, 1500000")
        raise typer.Exit(1)
    r.text_budget = n
    c.save(r)
    console.print(f"{c.root.name}: budget {human(n)} "
                  f"({human(r.text_bytes())} of it already used)")


# ── adding documents by hand ────────────────────────────────────────────────────────

@kb.command("add")
def add_files(name: str, files: list[Path] = typer.Argument(..., help="Files to include"),
              config: str = typer.Option(None, help="Path to config.yaml")) -> None:
    """Add local files to a corpus. Creates it if it does not exist."""
    from lara.corpus import builder as B

    cfg = _config(config)
    reg = _registry(cfg)
    c = reg.create(name)
    r = c.load()
    added = 0
    for f in files:
        for path in (sorted(f.rglob("*")) if f.is_dir() else [f]):
            if not path.is_file():
                continue
            src = B.add_file(c, r, path)
            if src is None:
                console.print(f"  [dim]skipped[/dim] {path.name} — no extractable text")
                continue
            added += 1
            console.print(f"  [green]+[/green] {path.name}  {human(src.chars)}  "
                          f"[{_licence_style(src.licence)}]{src.licence}[/]")
    console.print(f"\n{added} file(s) added to {c.root.name}. "
                  f"Build with [bold]lara kb build {c.root.name}[/bold]")


@kb.command("build")
def build_cmd(name: str, config: str = typer.Option(None, help="Path to config.yaml"),
              prune: bool = typer.Option(True, help="Delete downloads no longer needed")) -> None:
    """Chunk, embed and index every accepted source."""
    from lara.corpus import builder as B

    cfg = _config(config)
    c = _registry(cfg).get(name)
    if not c.recipe_path.exists():
        console.print(f"[red]No corpus named[/red] {name}")
        raise typer.Exit(1)
    r = c.load()
    _build(cfg, c, r, prune=prune)


def _build(cfg, c, r, *, prune: bool = True) -> None:
    """The build step, shared by `lara kb build` and the end of `lara kb new`."""
    from lara.cli._base import _require_hf
    from lara.corpus import builder as B
    from lara.index.embed import load_model
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

    if not r.accepted():
        console.print("[yellow]Nothing accepted — nothing to build.[/yellow]")
        return
    _require_hf(cfg)
    ecfg = cfg.get_in("embedding")
    console.print(f"\nLoading {ecfg['model']}…")
    embedder = load_model(ecfg["model"], max_seq_length=ecfg.get("max_seq_len", 512))

    with Progress(TextColumn("  [progress.description]{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"), TimeElapsedColumn(),
                  console=console) as prog:
        task = prog.add_task("embedding", total=1)

        def on_event(e):
            if e["kind"] == "document":
                prog.console.print(f"  [green]+[/green] {e['title'][:60]}  "
                                   f"{e['chunks']} chunks")
            elif e["kind"] == "missing":
                prog.console.print(f"  [yellow]![/yellow] {e['title'][:60]} — {e['detail']}")
            elif e["kind"] == "embedded":
                prog.update(task, completed=e["done"], total=max(e["total"], e["done"]))

        stats = B.build(c, r, embedder,
                        dim_full=ecfg.get("dim_full", 768),
                        dim_trunc=ecfg.get("dim_truncated", 256), on_event=on_event,
                        # M20: the config's chunking.* must reach add_document.
                        chunking=cfg.get_in("chunking") or {})

    freed = B.prune_raw(c, r) if prune else 0
    console.print(f"\n[bold green]Built {c.root.name}[/bold green] — "
                  f"{stats.documents} document(s), {stats.chunks:,} chunks, "
                  f"{stats.embedded:,} embedded in {stats.seconds:.0f}s"
                  + (f", {human(freed)} of downloads pruned" if freed else ""))
    console.print(f"On disk: {human(c.size_on_disk())}. "
                  f"Search it with [bold]lara search --corpus {c.root.name} '…'[/bold]")


# ── the interactive builder ─────────────────────────────────────────────────────────

def _review(c, r, candidates) -> int:
    """Show the candidates and take the reader's decisions. Returns how many accepted.

    Everything already fetched is shown, including what the judges rejected: a scored
    rejection is a suggestion, and the reader is allowed to disagree with it. The default
    answer accepts exactly what passed, so agreeing is one keypress and overriding is
    possible rather than obligatory.
    """
    from lara.corpus import builder as B

    pending = [x for x in candidates if x.source.decided == "pending"]
    dropped = [x for x in candidates if x.source.decided == "rejected"]
    if not pending and not dropped:
        console.print("[yellow]Nothing new found.[/yellow]")
        return 0

    # Content-hash dedup catches the same bytes from two mirrors, but not the same book in
    # three editions — which is what a search for any manual actually returns. Those have
    # different hashes and are genuinely different documents, so they are shown rather than
    # dropped; flagging them is what stops a reader embedding one manual three times.
    dup_of: dict[int, int] = {}
    first_seen: dict[str, int] = {}
    for i, x in enumerate(pending, 1):
        key = re.sub(r"[^a-z0-9]+", " ", (x.source.title or "").lower()).strip()
        if not key:
            continue
        if key in first_seen:
            dup_of[i] = first_seen[key]
        else:
            first_seen[key] = i

    t = Table(box=None, pad_edge=False, title=None)
    for col, just in (("#", "right"), ("rel", "right"), ("size", "right"),
                      ("licence", "left"), ("title", "left")):
        t.add_column(col, justify=just)
    for i, x in enumerate(pending, 1):
        s = x.source
        note = "  [yellow](thin)[/yellow]" if x.thin else ""
        if i in dup_of:
            note += f"  [yellow](same title as #{dup_of[i]})[/yellow]"
        t.add_row(str(i), f"{s.relevance:.2f}" if s.relevance is not None else " — ",
                  human(s.chars), f"[{_licence_style(s.licence)}]{s.licence}[/]",
                  (s.title or s.url)[:56] + note)
    console.print("\n[bold]Candidates[/bold]")
    console.print(t)

    if dropped:
        console.print(f"[dim]{len(dropped)} rejected by the judges "
                      f"(see them with `lara kb show {c.root.name}`)[/dim]")
    if not pending:
        return 0

    # Licences are called out separately rather than left in a column, because this is the
    # part with legal consequences and a coloured word in a table is easy to slide past.
    risky = [x for x in pending if not x.source.redistributable]
    if risky:
        from lara.corpus import licence as LIC
        console.print()
        for verdict in sorted({x.source.licence for x in risky}):
            n = sum(1 for x in risky if x.source.licence == verdict)
            console.print(f"  [{_licence_style(verdict)}]{verdict}[/] × {n}: "
                          f"{LIC.PLAIN[verdict]}")

    if dup_of:
        console.print(f"[yellow]{len(dup_of)} candidate(s) share a title with another — "
                      f"likely the same document twice. Accepting both embeds it "
                      f"twice.[/yellow]")
    console.print("\n[bold]Accept which?[/bold] [dim]all / none / 1,3,5 / "
                  "one — to step through them[/dim]")
    default = "all" if not dup_of else ",".join(
        str(i) for i in range(1, len(pending) + 1) if i not in dup_of)
    answer = (typer.prompt("accept", default=default) or "").strip().lower()

    chosen: list
    if answer in ("all", "a", ""):
        chosen = pending
    elif answer in ("none", "n"):
        chosen = []
    elif answer in ("one", "o", "1by1"):
        chosen = []
        for x in pending:
            s = x.source
            console.print(f"\n[bold]{s.title[:70]}[/bold]\n  {s.url}")
            console.print(f"  {human(s.chars)}   relevance "
                          f"{s.relevance if s.relevance is not None else '—'}   "
                          f"[{_licence_style(s.licence)}]{s.licence_label or s.licence}[/]")
            if x.preview:
                console.print(f"  [dim]{x.preview[:300]}…[/dim]")
            if typer.confirm("  include", default=True):
                chosen.append(x)
    else:
        want = {int(n) for n in re.findall(r"\d+", answer)}
        chosen = [x for i, x in enumerate(pending, 1) if i in want]

    keep = {x.source.url for x in chosen}
    for x in pending:
        B.decide(c, r, x.source.url, "accepted" if x.source.url in keep else "rejected",
                 "" if x.source.url in keep else "declined by the reader")
    console.print(f"[green]{len(chosen)}[/green] accepted, "
                  f"{len(pending) - len(chosen)} declined.")
    return len(chosen)


@kb.command("new")
def new(name: str,
        goal: str = typer.Option(None, "--goal", "-g", help="What the corpus is for"),
        queries: int = typer.Option(8, help="How many search queries to propose"),
        per_query: int = typer.Option(6, help="Results to consider per query"),
        text_budget: str = typer.Option(None, "--budget", help="Cap on raw text, e.g. 2GB"),
        model: str = typer.Option(None, help="Generator model for queries and judging"),
        build_now: bool = typer.Option(True, "--build/--no-build",
                                       help="Build once sources are chosen"),
        config: str = typer.Option(None, help="Path to config.yaml")) -> None:
    """Search, review and build a corpus, one goal at a time."""
    import asyncio

    from lara.cli._base import _require_hf
    from lara.corpus import builder as B
    from lara.index.embed import load_model

    cfg = _config(config)
    reg = _registry(cfg)
    goal = goal or typer.prompt("What is this corpus for?")
    limit = parse_size(text_budget) if text_budget else None
    if text_budget and limit is None:
        console.print(f"[red]Not a size:[/red] {text_budget}")
        raise typer.Exit(1)

    c = reg.create(name, goal=goal, **({"text_budget": limit} if limit else {}))
    r = c.load()
    if not r.goal:
        r.goal = goal
    if limit:
        r.text_budget = limit
    c.save(r)

    free = c.disk_free()
    console.print(f"[bold]{c.root.name}[/bold] — {goal}")
    console.print(f"[dim]budget {human(r.text_budget)} of text · {human(free)} free on "
                  f"disk · change with `lara kb budget {c.root.name} <size>`[/dim]")

    _require_hf(cfg)
    ecfg = cfg.get_in("embedding")
    embedder = load_model(ecfg["model"], max_seq_length=ecfg.get("max_seq_len", 512))

    console.print("\nAsking the model for search queries…")
    qs = asyncio.run(B.propose_queries(cfg, goal, n=queries, model=model))
    for i, q in enumerate(qs, 1):
        console.print(f"  {i}. {q}")
    if qs == [goal]:
        console.print("[dim](no generator reachable — searching your own words)[/dim]")

    edited = typer.prompt("\nEdit queries? blank to keep, or type your own, ; separated",
                          default="", show_default=False).strip()
    if edited:
        qs = [q.strip() for q in edited.split(";") if q.strip()]

    from lara.corpus.search import MIN_INTERVAL_SEC
    console.print(f"\nSearching [dim]({len(qs)} queries, one every "
                  f"{MIN_INTERVAL_SEC:.0f}s to stay unblocked)[/dim]")

    def on_event(e):
        k = e["kind"]
        if k == "searched":
            console.print(f"  {e['results']} result(s) from {e['queries']} queries"
                          + (f", {e['cached']} cached" if e["cached"] else "")
                          + (f", [yellow]{e['blocked']} blocked[/yellow]" if e["blocked"] else ""))
            for err in e["errors"][:3]:
                console.print(f"  [dim]{err}[/dim]")
        elif k == "candidate":
            mark = "[green]+[/green]" if e["relevant"] else "[red]-[/red]"
            console.print(f"  {mark} {(e['title'] or e['url'])[:54]:56s} "
                          f"{human(e['chars']):>8s}  "
                          f"[{_licence_style(e['licence'])}]{e['licence']}[/]"
                          + ("" if e["relevant"] else f"  [dim]{e['reason'][:40]}[/dim]"))
        elif k == "skipped":
            console.print(f"  [dim]· {e['url'][:60]} — {e['why']}[/dim]")
        elif k in ("warning", "blocked", "budget"):
            console.print(f"  [yellow]{e['detail']}[/yellow]")

    candidates = asyncio.run(B.discover(cfg, embedder, c, r, queries=qs,
                                        per_query=per_query, model=model,
                                        on_event=on_event))
    r = c.load()                       # discovery wrote as it went; take its version
    n = _review(c, r, candidates)
    r = c.load()

    console.print("\n" + B.summarise(r))
    if n and build_now and typer.confirm("\nBuild the knowledge base now?", default=True):
        _build(cfg, c, r)
    elif n:
        console.print(f"Build later with [bold]lara kb build {c.root.name}[/bold]")
