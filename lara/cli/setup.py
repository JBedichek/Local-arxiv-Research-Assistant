"""Getting a machine ready: the preflight check and the setup wizard."""

from __future__ import annotations

import typer
from rich.table import Table

from lara import config as config_mod
from lara import models as models_mod
from lara import preflight as preflight_mod
from lara import prompt as prompt_mod
from lara.cli._base import _require_hf, app, console


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
        # Zero, not the configured value: hot_tier.max_bytes is read by this planner
        # and by nothing that serves, so budgeting for it shrank every machine by
        # 2 GB in favour of a cache that is never allocated.
        hot_tier_bytes=0,
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
                      f"{SU.p50_for(opt, device.accelerator):.1f}ms",
                      f"{opt.recall:.3f}", opt.note)
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
    # Whole-corpus regardless of scoping, which is exactly why it is worth naming: it is
    # the one fixed cost the slider cannot move.
    row_map = n_chunks * SU.ROW_MAP_BYTES_PER_ROW / 1e9
    if row_map >= 0.05:
        fixed.append(f"row map {row_map:.1f}")
    if plan.hot_tier_bytes:
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
    from lara.serve import generator as GEN

    # Not device.backend: that is advisory prose and says "llama.cpp" on any Mac, while
    # the runtime actually chosen prefers MLX when mlx-lm is installed. They name
    # different weight formats, so the list of servable models has to follow the real one.
    gen_backend = GEN.resolve_backend(cfg, device.accelerator,
                                      cfg.get_path("huggingface.home"))
    console.print(f"  [dim]backend [bold]{gen_backend}[/bold], which loads "
                  f"{models_mod.wants_format(gen_backend)} weights[/dim]")
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
                                             backend=gen_backend) if m.servable]
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
                t.add_row(f"{m.size_gb:.1f}", m.spec,
                          f"[green]yes[/green] ({f['margin_gb']:.0f} GB spare)" if f["fits"]
                          else f"[red]no[/red] (needs {f['needed_gb']:.0f} of "
                               f"{f['budget_gb']:.0f} GB {f['where']})")
            console.print(t)
            if fitting and not non_interactive and not show:
                model = typer.prompt("  model (blank to skip)",
                                     default=fitting[-1].spec, show_default=True) or None
            elif fitting and (non_interactive or show):
                # --show previews what --non-interactive would do, as it already does for
                # an already-running generator. Without `show` here it previewed a config
                # with no model at all, which is not what running it would produce.
                model = fitting[-1].spec
            if model:
                m = next((x for x in cached if model in (x.repo, x.spec)), None)
                if m and m.runtime_quant_options():
                    quant = m.runtime_quant_options()[0]

    # ── 5b. context length ─────────────────────────────────────────────────────
    # The KV cache is per-token and at 32k outweighs the weights it serves, so this is
    # the largest generator-side memory decision and nothing used to surface it.
    SLOT_CHOICES = (1, 2, 4)
    ctx_size = int(cfg.get_in("serving.generator.llamacpp.ctx_size", 32768) or 32768)
    kv_quant = bool(cfg.get_in("serving.generator.llamacpp.cache_type_k"))
    slots = int(cfg.get_in("serving.generator.llamacpp.parallel", 2) or 2)
    if model and gen_backend in ("llamacpp", "ollama"):
        model_gb = next((m.size_gb for m in (cached if not base_url else [])
                         if model in (m.repo, m.spec)), 0.0)
        spare = SU.generator_headroom_gb(device.budget_gb, plan.overhead_gb
                                         + plan.planned_index_gb)

        def ctx_screen(cursor: int | None):
            from rich.console import Group

            t = Table(show_header=True, header_style="bold")
            for c, j in (("", "left"), ("context", "right"), ("per request", "right"),
                         ("KV cache", "right"), ("+ weights", "right"), ("verdict", "left")):
                t.add_column(c, justify=j, overflow="fold")
            for i, n in enumerate(SU.CONTEXT_CHOICES):
                kv = SU.kv_cache_gb(n, kv_quant)
                total = kv + model_gb
                on = cursor is not None and i == cursor
                fits = total <= spare
                t.add_row("❯" if on else ("→" if n == ctx_size else ""),
                          f"[bold cyan]{n:,}[/bold cyan]" if on else f"{n:,}",
                          f"{n // max(1, slots):,}",
                          f"{kv:.2f} GB", f"{total:.2f} GB",
                          "[green]fits[/green]" if fits
                          else f"[red]needs {total - spare:.1f} GB more[/red]")
            arrows = "[green]◀ ▶[/green]" if prompt_mod.interactive() else "   "
            kv_label = ("[bold]q8_0[/bold] — half the cache, slight quality cost"
                        if kv_quant else "[bold]fp16[/bold] — full precision, full size")
            return Group(t, f"  {arrows} KV cache precision: {kv_label}")

        def toggle_kv(_delta: int) -> bool:
            nonlocal kv_quant
            kv_quant = not kv_quant
            return True

        def slots_table(cursor: int | None) -> Table:
            t = Table(show_header=True, header_style="bold")
            for c, j in (("", "left"), ("slots", "right"), ("per request", "right"),
                         ("what it buys", "left")):
                t.add_column(c, justify=j, overflow="fold")
            for i, n in enumerate(SLOT_CHOICES):
                on = cursor is not None and i == cursor
                t.add_row("❯" if on else ("→" if n == slots else ""),
                          f"[bold cyan]{n}[/bold cyan]" if on else str(n),
                          f"{ctx_size // n:,}",
                          "one request at a time, whole window each"
                          if n == 1 else f"{n} concurrent requests, window split {n} ways")
            return t

        console.print("\n[bold]5b. Context length[/bold]")
        console.print(f"  [dim]How much text the generator can hold at once — the question, "
                      f"the excerpts it cites, and the answer. Its KV cache costs "
                      f"{SU.KV_BYTES_PER_TOKEN_FP16 / 1024:.0f} KB per token at fp16, so "
                      f"this is usually the largest generator-side memory decision.\n"
                      f"  Estimated for an 8B-class model; llama.cpp prints the exact "
                      f"figure when it loads. {spare:.1f} GB is free after retrieval.[/dim]")
        start = (list(SU.CONTEXT_CHOICES).index(ctx_size)
                 if ctx_size in SU.CONTEXT_CHOICES else 1)
        if non_interactive or show:
            console.print(ctx_screen(None))
        elif prompt_mod.interactive():
            console.print("  [dim]↑/↓ context · ◀/▶ KV precision · enter to confirm[/dim]")
            picked = prompt_mod.select(len(SU.CONTEXT_CHOICES), ctx_screen,
                                       console=console, initial=start,
                                       horizontal=toggle_kv)
            if picked is not None:
                ctx_size = SU.CONTEXT_CHOICES[picked]
            console.print(f"  context: [bold]{ctx_size:,}[/bold]   KV: "
                          f"[bold]{'q8_0' if kv_quant else 'fp16'}[/bold]")
        else:
            console.print(ctx_screen(None))
            raw = typer.prompt("  context length", default=str(ctx_size))
            ctx_size = int(raw) if raw.strip().isdigit() else ctx_size
            kv_quant = typer.confirm("  quantise the KV cache to q8_0 (halves it)",
                                     default=kv_quant)

        # Slots do not change total KV -- `-c` is the whole cache, split between them --
        # so this is a throughput/window trade rather than a memory one, and belongs
        # after the memory decision rather than tangled into it.
        console.print("\n[bold]5c. Concurrent requests[/bold]")
        console.print(f"  [dim]llama.cpp splits the {ctx_size:,}-token context evenly "
                      f"between slots, so each extra slot halves what one request can "
                      f"read. Total memory is unchanged.[/dim]")
        s_start = SLOT_CHOICES.index(slots) if slots in SLOT_CHOICES else 0
        if non_interactive or show:
            console.print(slots_table(None))
        elif prompt_mod.interactive():
            got = prompt_mod.select(len(SLOT_CHOICES), slots_table, console=console,
                                    initial=s_start)
            if got is not None:
                slots = SLOT_CHOICES[got]
            console.print(f"  slots: [bold]{slots}[/bold] "
                          f"({ctx_size // slots:,} tokens per request)")
        else:
            console.print(slots_table(None))
            raw = typer.prompt("  concurrent slots", default=str(slots))
            slots = int(raw) if raw.strip().isdigit() and int(raw) > 0 else slots

    # ── 6. write ───────────────────────────────────────────────────────────────
    overrides = SU.overrides_for(
        plan, model=model, quantization=quant, base_url=base_url,
        disk_root=str(cfg.get_path("disk.root")),
        devices=[int(g) for g in range(len(device.gpus))] if device.gpus else "auto",
        topics=topics, backend=gen_backend, ctx_size=ctx_size,
        kv_quant=kv_quant, slots=slots,
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
