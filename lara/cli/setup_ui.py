"""The interactive screen `lara setup` uses to choose a search engine.

Lifted out of ``setup()``, where it was six nested closures sharing two mutable locals.
That works, but it reads badly in both directions: ``backend_table`` was defined above
the ``gen_keep`` it renders, and ``setup()`` itself was 500 lines of which 170 were this
one screen. Six functions over shared mutable state is a class, so it is one here.

**Backend and keep fraction are one decision, not two.** The whole reason to shrink the
resident corpus is to afford a better search engine and still leave room to generate, and
that trade is unreadable if you pick the engine on one screen and the fraction on the
next. So the table and the slider live on the same screen and redraw together.
"""

from __future__ import annotations

import typer
from rich.console import Group
from rich.table import Table

from lara import tui
from lara.cli._base import console

#: The fractions the slider stops at. Coarse at the top, fine at the bottom, because the
#: difference between 1% and 2% of the corpus is a real decision and the difference
#: between 90% and 91% is not.
KEEP_STEPS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33,
              0.50, 0.66, 0.75, 0.90, 1.00]


class BackendChooser:
    """Section 3 of the wizard: pick a tier-1 backend and how much corpus stays resident.

    Construct it with the planner's output, then call :meth:`run`. It reports the choice
    and also writes ``plan.option`` back, because later sections size the generator from
    the plan rather than from the return value.
    """

    def __init__(self, plan, device, n_chunks: int) -> None:
        from lara import setup as SU

        self.SU = SU
        self.plan = plan
        self.device = device
        self.n_chunks = n_chunks

        self.opts = [o for o, _, _ in plan.alternatives]
        self.keys = [o.key for o in self.opts]
        self.recommended = plan.option.key

        # Scoping shrinks only the index, and "unnecessary" leaves scope_keep at whatever
        # the solver last computed — which is not 1.0 — so it has to be read as 1.0 here,
        # or a machine that needs no scoping is shown a shrunken index it will never
        # build.
        self.default_keep = 1.0 if plan.scope == "unnecessary" else plan.scope_keep
        self.keep_idx = min(range(len(KEEP_STEPS)),
                            key=lambda i: abs(KEEP_STEPS[i] - self.default_keep))
        self.keep = KEEP_STEPS[self.keep_idx]

        # "+models" hid the fact that a third of the fixed cost is cache, not a model, and
        # named no figure you could check. Spell the addends out instead.
        self.fixed = [f"embedder {plan.embedder_gb:.1f}"]
        if plan.reranker_gb:
            self.fixed.append(f"cross-encoder reranker {plan.reranker_gb:.1f}")
        # Whole-corpus regardless of scoping, which is exactly why it is worth naming: it
        # is the one fixed cost the slider cannot move.
        row_map = n_chunks * SU.ROW_MAP_BYTES_PER_ROW / 1e9
        if row_map >= 0.05:
            self.fixed.append(f"row map {row_map:.1f}")
        if plan.hot_tier_bytes:
            self.fixed.append(f"tier-0 hot cache {plan.hot_tier_bytes / 1e9:.1f}")

        self.width = "fp32" if device.accelerator == "cpu" else "half precision"
        self.pool = ("This machine shares one pool of memory between the search index and "
                     "the AI model, so every GB one takes is a GB the other cannot have."
                     if device.unified_memory else
                     "The index sits on a single card; a generator can be sharded across "
                     "all of them, so this is the cautious figure.")
        self.basis = "at the share of the corpus the slider keeps resident"

    # ── rendering ────────────────────────────────────────────────────────────────

    # Every option is selectable. The memory columns are reported so you can judge the
    # trade yourself; nothing is struck out for being large, because "it does not fit
    # today" depends on the corpus you scope to and what else the machine is doing.
    def backend_table(self, cursor: int | None) -> Table:
        SU, plan, device = self.SU, self.plan, self.device
        t = Table(show_header=True, header_style="bold")
        for c, j in (("", "left"), ("search engine", "left"), ("index RAM", "right"),
                     ("total RAM", "right"), ("room for AI model", "right"),
                     ("speed", "right"), ("accuracy", "right"), ("trade-off", "left")):
            t.add_column(c, justify=j, overflow="fold")
        for i, opt in enumerate(self.opts):
            on_cursor = cursor is not None and i == cursor
            mark = "❯" if on_cursor else ("→" if opt.key == self.recommended else "")
            style = ("bold cyan" if on_cursor
                     else "green" if opt.key == self.recommended else "")
            cell = f"[{style}]{opt.label}[/{style}]" if style else opt.label
            # Every column is quoted at the size this machine will actually build. Mixing
            # a full-corpus index with leftover-memory computed after scoping made two
            # columns that could not both be true at once.
            idx = opt.index_gb(int(self.n_chunks * self.keep), plan.dim)
            full = opt.index_gb(self.n_chunks, plan.dim)
            idx_cell = (f"{idx:.1f} GB" if self.keep >= 1
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

    def slider(self) -> str:
        filled = round(self.keep * 24)
        bar = "█" * filled + "░" * (24 - filled)
        chunks = int(self.n_chunks * self.keep)
        head = "[green]◀ ▶[/green]" if tui.interactive() else "   "
        return (f"  {head} corpus kept resident  [cyan]{bar}[/cyan]  "
                f"[bold]{self.keep:>4.0%}[/bold]  ({chunks / 1e6:.1f}M of "
                f"{self.n_chunks / 1e6:.1f}M chunks)")

    def legend_table(self) -> Table:
        SU, plan = self.SU, self.plan
        g = Table(show_header=False, box=None, padding=(0, 2))
        g.add_column(style="bold", no_wrap=True)
        g.add_column(overflow="fold")
        g.add_row("search engine", "which engine stores and searches the vectors")
        g.add_row("index RAM", f"memory the search index alone needs, {self.basis}")
        g.add_row("total RAM", f"index + {' + '.join(self.fixed)} — the "
                               f"{plan.overhead_gb:.1f} GB "
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

    def print_legend(self) -> None:
        console.print("  [dim]→ marks the recommendation. Any row can be chosen.[/dim]")
        console.print(self.legend_table())
        console.print(f"  [dim]The embedder and reranker load in {self.width} on "
                      f"{self.device.accelerator} and are not further compressed, so those "
                      f"two figures double on a CPU-only machine. {self.pool}\n"
                      f"  Speed and accuracy are measured — reproduce with "
                      f"`lara bench-index`.[/dim]")

    def screen(self, cursor: int | None) -> Group:
        return Group(self.backend_table(cursor), self.slider())

    # ── the decision ─────────────────────────────────────────────────────────────

    def nudge_keep(self, delta: int) -> bool:
        """Move the slider one stop. Returns whether anything changed, so the arrow keys
        do not force a redraw at either end."""
        new = min(max(self.keep_idx + delta, 0), len(KEEP_STEPS) - 1)
        if new == self.keep_idx:
            return False
        self.keep_idx, self.keep = new, KEEP_STEPS[new]
        return True

    def run(self, *, allow_interactive: bool):
        """Show the screen, take the decision, and return ``(option, keep_fraction)``."""
        if not allow_interactive:
            console.print(self.screen(None))
            self.print_legend()
            return self.plan.option, self.keep

        if tui.interactive():
            return self._arrow_keys()
        return self._typed()

    def _arrow_keys(self):
        self.print_legend()
        console.print("  [dim]↑/↓ choose a search engine · ◀/▶ move the slider · "
                      "enter to confirm · esc for the recommendation.[/dim]")
        start = self.keys.index(self.recommended)
        picked = tui.select(len(self.opts), self.screen, console=console,
                                   initial=start, horizontal=self.nudge_keep)
        if picked is None:
            self.keep = self.default_keep     # esc restores the recommendation wholesale
        chosen = self.opts[picked] if picked is not None else self.plan.option
        self.plan.option = chosen
        console.print(f"  search engine: [bold]{chosen.key}[/bold]   "
                      f"corpus kept: [bold]{self.keep:.0%}[/bold]")
        return chosen, self.keep

    def _typed(self):
        # Not a terminal — piped input, CI, TERM=dumb. No slider, so both halves of the
        # decision get a typed prompt.
        console.print(self.screen(None))
        self.print_legend()
        while True:
            pick = typer.prompt(f"\n  backend [{'/'.join(self.keys)}]",
                                default=self.recommended)
            pick = pick.strip()
            if pick in self.keys:
                break
            # Silently falling back to the default here meant a typo picked a backend you
            # did not ask for, and nothing said so.
            console.print(f"  [red]{pick!r} is not one of[/red] {', '.join(self.keys)}")
        chosen = self.SU.OPTIONS_BY_KEY[pick]
        self.plan.option = chosen
        while True:
            raw = typer.prompt("  fraction of the corpus to keep resident (0-1)",
                               default=f"{self.keep:.2f}")
            try:
                val = float(raw)
            except ValueError:
                console.print(f"  [red]{raw!r} is not a number[/red]")
                continue
            if 0 < val <= 1:
                self.keep = val
                break
            console.print("  [red]must be greater than 0 and at most 1[/red]")
        return chosen, self.keep
