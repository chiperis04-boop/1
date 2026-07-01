"""Dry-run moment preview + event-feed override plumbing (WebUI-agnostic).

The WebUI lets an operator validate WHICH moments will be clipped before paying
for a full render (ClipMaker-style). This module holds the pure, gradio-free
logic so it is unit-testable:

  * `event_feed_overrides()` turns the UI moment-source choice (detectors /
    pasted text-CSV report / ESPN fixture) + per-half kick-off offsets into the
    `detect.event_feed.*` config overrides.
  * `preview_windows()` runs ONLY the Scout (no render) and returns a table of
    the windows it would clip: minute, type, confidence, video timecodes,
    score-verified flag and sources.
"""
from __future__ import annotations

from pathlib import Path

from ..edit import ff
from ..utils.io import get_logger, load_config
from .scout import scout_events

log = get_logger()


def fmt_timecode(seconds) -> str:
    """Seconds -> mm:ss timecode of the VIDEO file."""
    try:
        s = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "?"
    return f"{s // 60:d}:{s % 60:02d}"


def _kickoffs(ko1, ko2) -> dict:
    kickoffs: dict = {}
    try:
        if ko1 is not None and float(ko1) >= 0:
            kickoffs[1] = float(ko1)
    except (TypeError, ValueError):
        pass
    try:
        if ko2 is not None and float(ko2) > 0:
            kickoffs[2] = float(ko2)
    except (TypeError, ValueError):
        pass
    return kickoffs


def event_feed_overrides(event_source, feed_text, espn_fixture, espn_slug,
                         ko1, ko2, match_stem, feed_dir: str = "input") -> dict:
    """Build detect.event_feed.* overrides from the UI moment-source choice.

    Returns {} for the detector path. For the text/CSV path the pasted report is
    written to <feed_dir>/<match_stem>_feed.txt and used as the source; for ESPN
    the keyless fixture is wired. Kick-offs map the match clock to video time.
    """
    src = str(event_source or "").strip().lower()
    kickoffs = _kickoffs(ko1, ko2)

    if src.startswith("espn"):
        if not str(espn_fixture or "").strip():
            return {}
        return {"enabled": True, "kickoffs": kickoffs or {1: 0.0},
                "espn": {"enabled": True,
                         "slug": str(espn_slug or "esp.1").strip(),
                         "fixture_id": str(espn_fixture).strip()}}

    if src.startswith(("текст", "csv", "text")):
        text = (feed_text or "").strip()
        if not text:
            return {}
        Path(feed_dir).mkdir(parents=True, exist_ok=True)
        feed_path = Path(feed_dir) / f"{match_stem}_feed.txt"
        feed_path.write_text(text, encoding="utf-8")
        return {"enabled": True, "source": str(feed_path),
                "kickoffs": kickoffs or {1: 0.0}}

    return {}                                    # detectors (auto) — no feed


def preview_windows(video_path: str, ef_overrides: dict | None = None,
                    limit: int = 0, config: str = "config/config.yaml"):
    """Run ONLY the Scout on `video_path` and return (rows, summary) without
    rendering. `rows`: [minute, kind, conf, start, anchor, end, verified,
    sources]. Never raises — returns ([], error_string) on failure."""
    try:
        ff.ensure_tools()
        cfg = load_config(config)
        if ef_overrides:
            cur = cfg.setdefault("detect", {}).get("event_feed", {})
            cfg["detect"]["event_feed"] = {**cur, **ef_overrides}
        duration = ff.duration(video_path)
        windows = scout_events(video_path, video_path, cfg, duration)
    except Exception as exc:  # noqa: BLE001
        return [], f"preview failed: {exc}"

    if not windows:
        return [], "no moments found"
    if limit and int(limit) > 0:
        windows = sorted(windows, key=lambda w: w.confidence, reverse=True)[:int(limit)]
        windows.sort(key=lambda w: w.anchor_t)

    rows = [[
        w.minute if w.minute is not None else "—",
        w.kind, f"{w.confidence:.2f}",
        fmt_timecode(w.start), fmt_timecode(w.anchor_t), fmt_timecode(w.end),
        "✓" if w.verified else "", ", ".join(w.sources),
    ] for w in windows]
    goals = sum(w.kind == "goal" for w in windows)
    return rows, f"{len(windows)} moment(s), {goals} goal(s)"
