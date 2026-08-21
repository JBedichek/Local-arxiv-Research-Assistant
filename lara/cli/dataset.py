"""``lara dataset`` -- publish the built corpus over the LAN, or fetch one."""

from __future__ import annotations

import typer
from rich.table import Table

from lara import config as config_mod
from lara.cli._base import _require_hf, app, console

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
