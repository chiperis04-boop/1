"""Dry-run moment preview + event-feed override plumbing (WebUI-agnostic).

The WebUI lets an operator validate WHICH moments will be clipped before paying
for a full render (ClipMaker-style). This module holds the pure, gradio-free
logic so it is unit-testable:

  * `event_feed_overrides()` turns the UI moment-source choice into the
    `detect.event_feed.*` config overrides. Priority, so MANUAL data always
    wins over API guessing:
        uploaded StatsBomb/SoccerNet JSON  >  pasted descriptive log / captions
        >  ESPN fixture (optional)  >  detectors (auto).
  * `preview_markdown()` returns a scannable **markdown** table of the selected
    event windows (match clock, video timecode, type, verified, description)
    WITHOUT rendering. The manual-log path is parsed with
    `event_feed.load_descriptive_events` and works even before a video is chosen.
  * `preview_windows()` (rows form) is kept for the detector/ESPN scout path and
    for unit tests.
"""
from __future__ import annotations

import shutil
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
    if s < 0:
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


def event_feed_overrides(event_source, feed_text, feed_file, espn_fixture,
                         espn_slug, ko1, ko2, match_stem,
                         feed_dir: str = "input") -> dict:
    """Build detect.event_feed.* overrides from the UI inputs.

    MANUAL data takes precedence (that is the whole point — supply the log
    yourself instead of guessing):
      1. `feed_file`  — an uploaded StatsBomb/SoccerNet JSON; copied onto the
         input dir (so it persists for a detached render) and used as the source.
      2. `feed_text`  — a pasted descriptive log / captions; written to a feed
         file (.json if it looks like JSON, else .txt) and used as the source.
      3. ESPN fixture — optional secondary source when no manual data is given.
      4. detectors    — fallback (returns {}).
    Kick-offs map the match clock to video time.
    """
    kickoffs = _kickoffs(ko1, ko2)

    # 1) uploaded StatsBomb/SoccerNet JSON
    if feed_file:
        try:
            srcp = Path(str(feed_file))
            if srcp.exists() and srcp.stat().st_size > 0:
                Path(feed_dir).mkdir(parents=True, exist_ok=True)
                dst = Path(feed_dir) / f"{match_stem}_feed.json"
                if srcp.resolve() != dst.resolve():
                    shutil.copy(srcp, dst)
                return {"enabled": True, "source": str(dst),
                        "kickoffs": kickoffs or {1: 0.0}}
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[preview] could not use uploaded feed file: {exc}")

    # 2) pasted descriptive log / captions
    text = (feed_text or "").strip()
    if text:
        Path(feed_dir).mkdir(parents=True, exist_ok=True)
        ext = ".json" if text[:1] in "[{" else ".txt"
        feed_path = Path(feed_dir) / f"{match_stem}_feed{ext}"
        feed_path.write_text(text, encoding="utf-8")
        return {"enabled": True, "source": str(feed_path),
                "kickoffs": kickoffs or {1: 0.0}}

    # 3) optional ESPN fixture
    if str(event_source or "").strip().lower().startswith("espn"):
        if str(espn_fixture or "").strip():
            return {"enabled": True, "kickoffs": kickoffs or {1: 0.0},
                    "espn": {"enabled": True,
                             "slug": str(espn_slug or "esp.1").strip(),
                             "fixture_id": str(espn_fixture).strip()}}

    # 4) detectors (auto) — no feed
    return {}


def _desc(w) -> str:
    """Human-readable description for a window (label — player (team))."""
    meta = w.meta or {}
    label = str(meta.get("label") or "").strip()
    player = str(meta.get("player") or "").strip()
    team = str(meta.get("team") or "").strip()
    head = " — ".join(p for p in (label, player) if p) or ", ".join(w.sources)
    return f"{head} ({team})" if team else head


def _windows_to_rows(windows, limit: int = 0):
    if limit and int(limit) > 0:
        windows = sorted(windows, key=lambda w: w.confidence, reverse=True)[:int(limit)]
        windows.sort(key=lambda w: w.anchor_t)
    rows = [[
        f"{w.minute}'" if w.minute is not None else "—",
        fmt_timecode(w.anchor_t), w.kind, f"{w.confidence:.2f}",
        "✓" if w.verified else "", _desc(w),
    ] for w in windows]
    return rows, windows


def _rows_to_markdown(rows, summary: str) -> str:
    if not rows:
        return f"### 👁 Dry-run\n\n_{summary}_"
    head = ("| # | Clock | ⏱ Video | Type | ✓ | Description |\n"
            "|--:|:-----:|:-------:|:----:|:-:|:------------|\n")
    body = "\n".join(
        f"| {i} | {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |"
        for i, r in enumerate(rows, 1))
    return f"### 👁 Dry-run — {summary}\n\n{head}{body}"


def preview_windows(video_path: str, ef_overrides: dict | None = None,
                    limit: int = 0, config: str = "config/config.yaml"):
    """Run ONLY the Scout on `video_path` (detector/ESPN path) and return
    (rows, summary). rows: [clock, video_tc, kind, conf, verified, desc].
    Never raises — returns ([], error_string) on failure."""
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
    rows, kept = _windows_to_rows(windows, limit)
    goals = sum(w.kind == "goal" for w in kept)
    return rows, f"{len(kept)} moment(s), {goals} goal(s)"


def preview_markdown(video_path, event_source, feed_text, feed_file, espn_fixture,
                     espn_slug, ko1, ko2, limit, feed_dir: str = "input",
                     config: str = "config/config.yaml"):
    """DRY-RUN → markdown table of the windows that WOULD be clipped, without
    rendering. Manual log (uploaded JSON / pasted text) is parsed with
    `load_descriptive_events` and shown even if no video is chosen yet; the
    detector/ESPN path uses the Scout and needs a video. Returns
    (status_line, markdown)."""
    try:
        cfg = load_config(config)
    except Exception as exc:  # noqa: BLE001
        return f"❌ config error: {exc}", _rows_to_markdown([], f"config error: {exc}")

    stem = Path(video_path).stem if video_path else "manual"
    ef = event_feed_overrides(event_source, feed_text, feed_file, espn_fixture,
                              espn_slug, ko1, ko2, stem, feed_dir)
    kickoffs = _kickoffs(ko1, ko2) or {1: 0.0}

    duration = None
    if video_path and Path(video_path).exists():
        try:
            ff.ensure_tools()
            duration = ff.duration(video_path)
        except Exception:  # noqa: BLE001
            duration = None

    # ---- manual descriptive log (JSON upload or pasted text) ----
    if ef.get("source"):
        try:
            from .event_feed import events_to_windows, load_descriptive_events
            cur = cfg.setdefault("detect", {}).get("event_feed", {})
            cfg["detect"]["event_feed"] = {**cur, **ef}
            events = load_descriptive_events(ef["source"], cfg)
            if not events:
                msg = ("no events parsed — check the log format (text lines like "
                       "\"67' GOAL — Messi\", CSV, or StatsBomb/SoccerNet JSON)")
                return f"👁 {msg}", _rows_to_markdown([], msg)
            windows = events_to_windows(events, kickoffs, cfg, duration)
        except Exception as exc:  # noqa: BLE001
            return f"❌ preview failed: {exc}", _rows_to_markdown([], f"failed: {exc}")
        if not windows:
            msg = "events parsed but none placed — set the kick-off (seconds)"
            return f"👁 {msg}", _rows_to_markdown([], msg)
        rows, kept = _windows_to_rows(windows, int(limit) if limit else 0)
        goals = sum(w.kind == "goal" for w in kept)
        note = "" if duration else " (video timecodes are from kick-off; pick a video to bound them)"
        summary = f"{len(kept)} moment(s), {goals} goal(s){note}"
        return (f"👁 Dry-run: {summary} — проверьте отбор, затем «Render highlights».",
                _rows_to_markdown(rows, summary))

    # ---- detector / ESPN path (needs a video) ----
    if not (video_path and Path(video_path).exists()):
        msg = ("paste a descriptive log / upload StatsBomb-SoccerNet JSON, or "
               "pick a video and a source (ESPN/detectors)")
        return f"👁 {msg}", _rows_to_markdown([], msg)
    espn_ef = ef if ef.get("espn") else None
    rows, summary = preview_windows(video_path, espn_ef, int(limit) if limit else 0,
                                    config)
    return f"👁 Dry-run: {summary}", _rows_to_markdown(rows, summary)
