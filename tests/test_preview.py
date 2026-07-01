"""Dry-run preview + event-feed override tests (Block E, CPU-only, no gradio).

Guards the WebUI's MANUAL moment-source plumbing (uploaded StatsBomb/SoccerNet
JSON or a pasted descriptive log takes precedence over API guessing) and the
markdown dry-run that shows the selected windows BEFORE rendering, via the
gradio-free src.detection.preview module.

Run directly:  python tests/test_preview.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detection.preview import (event_feed_overrides, fmt_timecode,   # noqa: E402
                                   preview_markdown)


def test_fmt_timecode():
    assert fmt_timecode(0) == "0:00"
    assert fmt_timecode(75) == "1:15"
    assert fmt_timecode(2740) == "45:40"
    assert fmt_timecode(None) == "?"
    print("  \u2713 timecode: seconds -> mm:ss (video time)")


def test_overrides_detectors_is_empty():
    assert event_feed_overrides("Детекторы (авто)", "", None, "", "esp.1",
                                0, 0, "m") == {}
    print("  \u2713 overrides: detector source, no manual data -> {}")


def test_overrides_espn_only_without_manual():
    ov = event_feed_overrides("ESPN (fixture)", "", None, "704321", "fifa.world",
                              0, 2740, "m")
    assert ov["enabled"] and ov["espn"]["fixture_id"] == "704321"
    assert ov["espn"]["slug"] == "fifa.world"
    assert ov["kickoffs"] == {1: 0.0, 2: 2740.0}
    assert event_feed_overrides("ESPN (fixture)", "", None, "", "esp.1",
                                0, 0, "m") == {}
    print("  \u2713 overrides: ESPN fixture + kick-offs wired; empty fixture -> {}")


def test_overrides_text_writes_feed_file():
    tmp = Path(tempfile.mkdtemp(prefix="fhs_feed_"))
    ov = event_feed_overrides("Текст/CSV отчёт",
                              "67' GOAL — Messi (Argentina)\n73' yellow card, Rodri",
                              None, "", "esp.1", 0, 2740, "match1", feed_dir=str(tmp))
    assert ov["enabled"] and ov["kickoffs"] == {1: 0.0, 2: 2740.0}
    feed = Path(ov["source"])
    assert feed.suffix == ".txt" and feed.exists()
    assert "Messi" in feed.read_text(encoding="utf-8")
    print("  \u2713 overrides: pasted text log written to .txt feed + used as source")


def test_overrides_pasted_json_saved_as_json():
    tmp = Path(tempfile.mkdtemp(prefix="fhs_feed_"))
    payload = '[{"type": {"name": "Shot"}, "minute": 5, "shot": {"outcome": {"name": "Goal"}}}]'
    ov = event_feed_overrides("Текст/CSV отчёт", payload, None, "", "esp.1",
                              0, 0, "m", feed_dir=str(tmp))
    assert Path(ov["source"]).suffix == ".json", ov["source"]
    print("  \u2713 overrides: pasted JSON saved as .json (parser detects StatsBomb)")


def test_overrides_uploaded_file_takes_precedence():
    tmp = Path(tempfile.mkdtemp(prefix="fhs_feed_"))
    up = tmp / "statsbomb.json"
    up.write_text('{"annotations": []}', encoding="utf-8")
    # a file upload wins even if text + ESPN are also provided
    ov = event_feed_overrides("ESPN (fixture)", "some text", str(up), "999",
                              "esp.1", 0, 0, "match2", feed_dir=str(tmp))
    assert ov["enabled"] and Path(ov["source"]).name == "match2_feed.json"
    assert "espn" not in ov, "uploaded JSON must take precedence over ESPN"
    print("  \u2713 overrides: uploaded JSON beats pasted text and ESPN")


def test_preview_markdown_from_pasted_log_no_video():
    """DRY-RUN before a video is chosen: a pasted descriptive log is parsed via
    load_descriptive_events and rendered as a scannable markdown table."""
    tmp = Path(tempfile.mkdtemp(prefix="fhs_prev_"))
    status, md = preview_markdown(
        None, "Текст/CSV отчёт",
        "1' GOAL — Tester (Reds)\n2' yellow card, Foe (Blues)",
        None, "", "esp.1", 0, 0, 0, feed_dir=str(tmp))
    assert "moment(s)" in status, status
    assert "| goal |" in md and "Tester" in md, md
    assert md.strip().startswith("###"), "must be markdown"
    assert "| card |" in md, md
    print("  \u2713 dry-run markdown: pasted log -> scannable table (no video needed)")


def test_preview_markdown_from_statsbomb_upload():
    """DRY-RUN from an uploaded StatsBomb JSON -> markdown with the goal."""
    tmp = Path(tempfile.mkdtemp(prefix="fhs_prev_"))
    sb = tmp / "sb.json"
    sb.write_text(json.dumps([
        {"type": {"name": "Shot"}, "minute": 12, "second": 5, "period": 1,
         "team": {"name": "Reds"}, "player": {"name": "Ann"},
         "shot": {"outcome": {"name": "Goal"}}},
        {"type": {"name": "Pass"}, "minute": 3, "period": 1},
    ]), encoding="utf-8")
    status, md = preview_markdown(None, "Текст/CSV отчёт", "", str(sb), "",
                                  "esp.1", 0, 0, 0, feed_dir=str(tmp))
    assert "1 goal" in status or "goal(s)" in status, status
    assert "| goal |" in md and "Ann" in md, md
    print("  \u2713 dry-run markdown: uploaded StatsBomb JSON -> goal window listed")


def test_preview_markdown_empty_input_is_graceful():
    status, md = preview_markdown(None, "Детекторы (авто)", "", None, "", "esp.1",
                                  0, 0, 0)
    assert "👁" in status and md.strip().startswith("###")
    print("  \u2713 dry-run markdown: empty input -> friendly guidance, no crash")


def main() -> int:
    print("dry-run preview + event-feed override tests (Block E)")
    for t in (test_fmt_timecode,
              test_overrides_detectors_is_empty,
              test_overrides_espn_only_without_manual,
              test_overrides_text_writes_feed_file,
              test_overrides_pasted_json_saved_as_json,
              test_overrides_uploaded_file_takes_precedence,
              test_preview_markdown_from_pasted_log_no_video,
              test_preview_markdown_from_statsbomb_upload,
              test_preview_markdown_empty_input_is_graceful):
        t()
    print("\nALL PREVIEW TESTS PASSED \u2705")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
