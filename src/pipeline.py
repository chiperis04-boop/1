"""Football Highlight Studio — command-line interface.

Thin wrapper over `src.runner`; the WebUI uses the same runner.

    python -m src.pipeline run input/match.mp4 --profile tiktok
    python -m src.pipeline detect input/match.mp4
    python -m src.pipeline list-profiles
    python -m src.pipeline webui
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress as RichProgress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from .detect.types import Moment
from .ingest import ingest
from .runner import run_pipeline
from .utils.io import ensure_dir, get_logger, load_config

app = typer.Typer(add_completion=False,
                  help="Local football highlight + telestration studio")
console = Console()
log = get_logger()


@app.command()
def run(
    match: str = typer.Argument(..., help="Path to the full match video"),
    profile: str = typer.Option("tiktok", help="Output profile (see list-profiles)"),
    config: str = typer.Option("config/config.yaml"),
    branding: str = typer.Option("config/branding.yaml"),
    out: str = typer.Option("output", help="Output root directory"),
    limit: int = typer.Option(0, help="Only render the top-N moments (0 = all)"),
    no_vision: bool = typer.Option(False, "--no-vision",
                                   help="Disable GPU telestration/reframe"),
):
    """Full pipeline: match video -> finished branded vertical clips."""
    overrides = {"vision": {"enabled": False}} if no_vision else None
    with RichProgress(SpinnerColumn(), TextColumn("[bold]{task.fields[stage]}"),
                      BarColumn(), TextColumn("{task.description}"),
                      console=console) as prog:
        task = prog.add_task("starting", total=100, stage="init")

        def cb(p):
            prog.update(task, completed=p.percent, stage=p.stage,
                        description=p.message)

        result = run_pipeline(match, profile=profile, config=config,
                              branding=branding, out_root=out, limit=limit,
                              on_progress=cb, overrides=overrides)

    if result.status == "failed" and not result.clips:
        console.print(f"[red]Failed:[/red] {result.error}")
        raise typer.Exit(1)

    _result_table(result)


@app.command()
def studio(
    match: str = typer.Argument(..., help="Path to the full match video"),
    profile: str = typer.Option("tiktok", help="Output profile (see list-profiles)"),
    config: str = typer.Option("config/config.yaml"),
    branding: str = typer.Option("config/branding.yaml"),
    out: str = typer.Option("output", help="Output root directory"),
    limit: int = typer.Option(0, help="Only render the top-N events (0 = all)"),
):
    """v2 pipeline: Scout -> Director -> Cameraman (BoT-SORT+CMC) -> Composer."""
    from .studio_pipeline import run_studio
    with RichProgress(SpinnerColumn(), TextColumn("[bold]{task.fields[stage]}"),
                      BarColumn(), TextColumn("{task.description}"),
                      console=console) as prog:
        task = prog.add_task("starting", total=100, stage="init")

        def cb(stage, pct, msg):
            prog.update(task, completed=pct, stage=stage, description=msg)

        result = run_studio(match, profile=profile, config=config,
                            branding=branding, out_root=out, limit=limit,
                            on_progress=cb)

    if result.status == "failed" and not result.clips:
        console.print(f"[red]Failed:[/red] {result.error}")
        raise typer.Exit(1)
    table = Table(title=f"Studio result — {result.status}")
    table.add_column("#"); table.add_column("kind"); table.add_column("status")
    table.add_column("hook / err")
    for c in result.clips:
        info = ((c.manifest or {}).get("video_hook_text", "")
                if c.status == "ok" else f"{c.stage_failed}: {c.error}")
        table.add_row(str(c.index), c.kind, c.status, str(info))
    console.print(table)
    console.print(f"[green]Output:[/green] {result.out_dir}  "
                  f"(events={result.windows}, goals={result.goals})")


@app.command()
def detect(match: str, config: str = typer.Option("config/config.yaml")):
    """Only run detection and print the ranked moments (no rendering)."""
    from .runner import _detect_and_fuse
    cfg = load_config(config)
    name = Path(match).stem
    workdir = ensure_dir(Path("output") / name / "work")
    info = ingest(match, workdir, cfg)
    moments = _detect_and_fuse(info, cfg, workdir, lambda *a, **k: None)
    _print_moments(moments)


@app.command("list-profiles")
def list_profiles(config: str = typer.Option("config/config.yaml")):
    cfg = load_config(config)
    table = Table(title="Output profiles")
    table.add_column("name"); table.add_column("WxH"); table.add_column("fps")
    for k, v in cfg["render"]["profiles"].items():
        table.add_row(k, f"{v['width']}x{v['height']}", str(v["fps"]))
    console.print(table)


@app.command()
def webui(host: str = typer.Option("0.0.0.0"), port: int = typer.Option(7860),
          share: bool = typer.Option(False, help="Create a public Gradio link")):
    """Launch the browser UI for upload + management."""
    from app.webui import launch
    launch(host=host, port=port, share=share)


# --------------------------------------------------------------------------- #
def _print_moments(moments: list[Moment]):
    table = Table(title="Detected moments")
    for c in ("#", "time", "kind", "conf", "min", "sources"):
        table.add_column(c)
    for i, m in enumerate(moments):
        table.add_row(str(i), f"{m.t:.1f}s", m.kind, f"{m.confidence:.2f}",
                      str(m.minute or "-"), ",".join(m.sources))
    console.print(table)


def _result_table(result):
    table = Table(title=f"Result — {result.status}")
    table.add_column("#"); table.add_column("kind"); table.add_column("status")
    table.add_column("file/err")
    for c in result.clips:
        info = c.path if c.status == "ok" else f"{c.stage_failed}: {c.error}"
        table.add_row(str(c.index), c.kind, c.status, str(info))
    console.print(table)
    console.print(f"[green]Output:[/green] {result.out_dir}  "
                  f"(moments={result.moments}, goals={result.goals})")


if __name__ == "__main__":
    app()
