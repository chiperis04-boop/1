"""Dry-run preview + event-feed override tests (Block E, CPU-only, no gradio).

Guards the WebUI's moment-source plumbing and the "show the selected moments
before rendering" dry-run, via the gradio-free src.detection.preview module.

Run directly:  python tests/test_preview.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detection.preview import (event_feed_overrides, fmt_timecode,   # noqa: E402
                                   preview_windows)


def test_fmt_timecode():
    assert fmt_timecode(0) == "0:00"
    assert fmt_timecode(75) == "1:15"
    assert fmt_timecode(2740) == "45:40"
    assert fmt_timecode(None) == "?"
    print("  \u2713 timecode: seconds -> mm:ss (video time)")


def test_overrides_detectors_is_empty():
    assert event_feed_overrides("Детекторы (авто)", "", "", "esp.1", 0, 0, "m") == {}
    print("  \u2713 overrides: detector source -> no event-feed override")


def test_overrides_espn():
    ov = event_feed_overrides("ESPN (fixture)", "", "704321", "fifa.world",
                              0, 2740, "m")
    assert ov["enabled"] and ov["espn"]["fixture_id"] == "704321"
    assert ov["espn"]["slug"] == "fifa.world"
    assert ov["kickoffs"] == {1: 0.0, 2: 2740.0}
    # empty fixture -> no override
    assert event_feed_overrides("ESPN (fixture)", "", "", "esp.1", 0, 0, "m") == {}
    print("  \u2713 overrides: ESPN fixture + kick-offs wired; empty fixture -> {}")


def test_overrides_text_writes_feed_file():
    tmp = Path(tempfile.mkdtemp(prefix="fhs_feed_"))
    ov = event_feed_overrides("Текст/CSV отчёт",
                              "67' GOAL — Messi (Argentina)\n73' yellow card, Rodri",
                              "", "esp.1", 0, 2740, "match1", feed_dir=str(tmp))
    assert ov["enabled"] and ov["kickoffs"] == {1: 0.0, 2: 2740.0}
    feed = Path(ov["source"])
    assert feed.exists() and "Messi" in feed.read_text(encoding="utf-8")
    # empty text -> no override
    assert event_feed_overrides("Текст/CSV отчёт", "  ", "", "esp.1", 0, 0, "m",
                                feed_dir=str(tmp)) == {}
    print("  \u2713 overrides: pasted report written to feed file + used as source")


def test_preview_windows_from_text_feed():
    """End-to-end dry-run: a tiny synthetic video + a pasted report -> the Scout
    places the right windows (no render), with correct video timecodes."""
    from src.edit import ff
    ff.ensure_tools()
    tmp = Path(tempfile.mkdtemp(prefix="fhs_preview_"))
    vid = str(tmp / "match.mp4")
    # 200s silent black video is enough for the scout to place feed windows
    ff.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:d=200:r=5",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", vid], desc="preview source")
    # goal at 1' (period 1, kick-off at 0s) -> anchor ~60s; card at 2'
    ov = event_feed_overrides("Текст/CSV отчёт",
                              "1' GOAL — Tester (Reds)\n2' yellow card, Foe (Blues)",
                              "", "esp.1", 0, 0, "match", feed_dir=str(tmp))
    rows, summary = preview_windows(vid, ov, limit=0)
    assert rows, summary
    kinds = [r[1] for r in rows]
    assert "goal" in kinds, kinds
    goal_row = next(r for r in rows if r[1] == "goal")
    assert goal_row[6] == "✓", goal_row            # goals are feed-verified
    assert "event_feed" in goal_row[7], goal_row    # source tag
    assert goal_row[4] == "1:00", goal_row          # anchor timecode = 1'
    print(f"  \u2713 dry-run: text feed -> {summary}; goal anchored at 1:00, verified")


def main() -> int:
    print("dry-run preview + event-feed override tests (Block E)")
    for t in (test_fmt_timecode,
              test_overrides_detectors_is_empty,
              test_overrides_espn,
              test_overrides_text_writes_feed_file,
              test_preview_windows_from_text_feed):
        t()
    print("\nALL PREVIEW TESTS PASSED \u2705")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
