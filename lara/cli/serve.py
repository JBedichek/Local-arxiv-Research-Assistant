"""Running the reader and the generator, and inspecting what can serve what."""

from __future__ import annotations

import typer
from rich.table import Table

from lara import config as config_mod
from lara import models as models_mod
from lara.cli._base import app, console


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

def _announce_when_ready(host: str, port: int, token: str | None,
                         timeout: float = 900.0) -> None:
    """Print the reader's URL once it will actually serve a page.

    Model loading takes tens of seconds, and during it every endpoint answers 503. A URL
    printed before that is an invitation to click something broken, so this waits for the
    server to say it is ready and announces it then.

    Daemon thread: the announcement is a convenience and must never hold up shutdown or
    outlive the server it is describing.
    """
    import threading
    import time

    # 0.0.0.0 is a bind address, not somewhere to point a browser.
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    url = f"http://{shown}:{port}"

    def watch() -> None:
        import httpx

        params = {"token": token} if token else None
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(1.0)
            try:
                r = httpx.get(f"{url}/api/health", params=params, timeout=2.0)
                if r.status_code == 200 and r.json().get("ready"):
                    break
            except Exception:
                continue        # not listening yet, or still warming
        else:
            console.print(f"[yellow]still warming after {timeout:.0f}s[/yellow] — {url}")
            return
        suffix = "/?token=<your token>" if token else ""
        console.print(f"\n[bold green]ready[/bold green]  [bold]{url}{suffix}[/bold]")

    threading.Thread(target=watch, daemon=True, name="ready-announcer").start()

def _reader_log_path(cfg):
    """Where the reader's own log goes, or None if the logs directory is unusable.

    Never fatal: a reader that refuses to start because it could not open a log file
    would be a worse failure than the missing log this exists to fix.
    """
    try:
        from pathlib import Path

        logs = Path(cfg.get_path("paths.logs"))
        logs.mkdir(parents=True, exist_ok=True)
        return logs / "reader.log"
    except Exception:
        return None


def _log_config(path):
    """uvicorn's logging config, with a file handler alongside the console one.

    uvicorn installs its own config and would otherwise discard a handler added to the
    root logger, which is why this replaces the config rather than appending a handler.

    Which loggers get the file handler is not uniform, and cannot be. uvicorn sets
    `propagate: False` on `uvicorn` and `uvicorn.access` but *not* on `uvicorn.error`,
    which therefore reaches the file through its parent. Adding the handler to all three
    wrote every error twice.
    """
    import copy

    from uvicorn.config import LOGGING_CONFIG

    cfg = copy.deepcopy(LOGGING_CONFIG)
    if path is None:
        return cfg
    cfg["formatters"]["file"] = {
        "()": "logging.Formatter",
        "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s",
    }
    cfg["handlers"]["file"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "formatter": "file",
        "filename": str(path),
        # A long research run is chatty and this is a file nobody prunes by hand.
        "maxBytes": 10_000_000,
        "backupCount": 3,
    }
    for name in ("uvicorn", "uvicorn.access"):
        if name in cfg["loggers"]:
            cfg["loggers"][name]["handlers"] = [
                *cfg["loggers"][name].get("handlers", []), "file",
            ]
    # Anything the application logs -- and any traceback uvicorn re-raises -- reaches the
    # file too, which is the point: the interesting failures are not uvicorn's own.
    cfg["root"] = {"handlers": ["file"], "level": "INFO"}
    return cfg


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

    tokens = AUTH.resolve_tokens(cfg)
    token = next(iter(tokens.values()), None)
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
    # Only turn auth ON where this bind actually demands it. Keying off "a token exists"
    # alone meant a token left in config.local.yaml — one `lara setup` writes — forced a
    # login prompt on plain loopback, where the mode says none is needed. The token is a
    # credential to use when required, not a switch that requires it.
    if token and AUTH.require_token_for(host, cfg):
        import json as _json
        # The whole set, so each person can hold their own secret and be revoked alone.
        os.environ["LARA_TOKENS"] = _json.dumps(tokens)
        os.environ.pop("LARA_TOKEN", None)
        console.print(
            f"[green]authentication on[/green] — {len(tokens)} secret(s): "
            f"{', '.join(sorted(tokens))}\n"
            f"each person opens http://{host}:{port}/?token=<their own> once per browser")
    else:
        # Stale env vars must not re-enable it, nor leak one person's secret into a run
        # that is meant to be open.
        os.environ.pop("LARA_TOKEN", None)
        os.environ.pop("LARA_TOKENS", None)
        if not AUTH.is_loopback(host):
            console.print("[yellow]serving without authentication[/yellow] (auth.mode: off)")

    gen = None
    if not no_llm and cfg.get_in("serving.generator.autostart", True):
        from lara.serve import devices as DV
        from lara.serve import generator as GEN

        accel = DV.detect().accelerator
        gen = GEN.from_config(cfg, accel)
        if gen is None:
            # Nothing configured, but a usable model may well be sitting in the cache --
            # it is on a fresh install that downloaded one from the reader. Adopting it
            # beats printing "run lara setup" at someone who just downloaded a model and
            # reasonably expects to be able to ask a question.
            backend = GEN.resolve_backend(cfg, accel, cfg.get_path("huggingface.home"))
            found = models_mod.servable(cfg.get_path("huggingface.home"), backend=backend)
            fitting = [m for m in found if DV.fits(m.size_gb, DV.detect())["fits"]]
            pick = (fitting or found or [None])[-1]
            if pick is not None:
                gen = GEN.from_config(cfg, accel, model_override=pick.spec,
                                      backend_override=backend)
                console.print(f"[yellow]no generator configured[/yellow] — using cached "
                              f"[bold]{pick.spec}[/bold] ({pick.size_gb:.1f} GB) on "
                              f"{backend} for this run. `lara setup` makes it permanent.")
        if gen is None:
            console.print("[yellow]no generator configured[/yellow] and none cached — "
                          "retrieval will work, answers will not. Download one from the "
                          "reader, or run `lara setup`.")
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

    # The URL used to be printed here, before uvicorn had even bound the port, so it was
    # clickable for the ~30s of model loading during which every request 503s. Announce it
    # from a watcher that waits for the server to report itself ready.
    _announce_when_ready(host, port, token)
    console.print("[dim]starting — loading models, the first start is the slow one[/dim]")
    # The generator has had a log file since it was first spawned; the reader never did,
    # so its warnings and tracebacks existed only in whichever terminal happened to run it.
    # "Check the server log" was answerable for half the system, and the half that reports
    # a failed research run was the missing half.
    reader_log = _reader_log_path(cfg)
    if reader_log:
        console.print(f"[dim]logging to {reader_log}[/dim]")
    try:
        # One worker: the GPU index, embedder and reranker are process-local and would be
        # duplicated per worker, tripling VRAM for no throughput gain on a single-user tool.
        uvicorn.run("lara.serve.app:app", host=host, port=port, workers=1,
                    log_level="info", log_config=_log_config(reader_log))
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
    backend: str = typer.Option(None, help="vllm | llamacpp | mlx | ollama | external"),
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
    merged = GEN.generator_cfg(cfg)
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
    merged = GEN.generator_cfg(cfg)
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
