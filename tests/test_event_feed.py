"""Event-feed Scout tests (CPU-only): parsing + match-clock -> video-time mapping
+ OCR alignment. No model/GPU/network needed."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.event_feed import (MatchEvent, align_to_ocr, events_to_windows,
                                       load_events)

CFG = {"detect": {"scout": {"goal_pre_seconds": 20.0, "goal_post_seconds": 10.0,
                            "pre_seconds": 7.0, "post_seconds": 5.0,
                            "merge_gap_seconds": 15.0},
                  "event_feed": {"ocr_align_radius_seconds": 40.0}}}


def test_text_parsing():
    feed = ("3' GOAL — Messi (Argentina)\n"
            "23: yellow card, Rodri (Spain)\n"
            "60' substitution, Pedri off\n"        # must be skipped
            "67' GREAT SAVE by Martinez (Argentina)\n")
    # load_events expects a path or list/dict; use the text parser via a temp file
    import tempfile
    p = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    p.write(feed); p.close()
    events = load_events(p.name, CFG)
    kinds = [e.kind for e in events]
    assert "goal" in kinds and "card" in kinds and "save" in kinds, kinds
    assert "sub" not in kinds and len(events) == 3, kinds
    goal = next(e for e in events if e.kind == "goal")
    assert goal.minute == 3 and goal.period == 1 and goal.team == "Argentina", goal
    save = next(e for e in events if e.kind == "save")
    assert save.period == 2, "67' must be 2nd half"
    print("  \u2713 text feed parsed; sub skipped; period inferred from minute")


def test_csv_and_importance():
    rows = [
        {"minute": "12", "period": "1", "team": "A", "player": "X", "type": "goal"},
        {"minute": "70", "period": "2", "team": "B", "type": "shot", "importance": "0.3"},
        {"minute": "80", "period": "2", "type": "corner"},      # skipped kind
    ]
    events = load_events(rows, CFG)
    assert len(events) == 2, [e.kind for e in events]
    assert events[0].kind == "goal" and events[0].importance == 1.0
    assert events[1].kind == "chance" and abs(events[1].importance - 0.3) < 1e-6
    print("  \u2713 CSV-style rows parsed; corner skipped; importance honoured")


def test_clock_to_video_mapping():
    events = [MatchEvent(minute=3, kind="goal", period=1),
              MatchEvent(minute=67, kind="goal", period=2)]
    # 2nd-half kick-off is at 45:40 (2740s) of the video file
    kickoffs = {1: 0, 2: 2740}
    ws = events_to_windows(events, kickoffs, CFG, duration=6000)
    assert len(ws) == 2, len(ws)
    # 1st-half 3' -> anchor 180s
    assert abs(ws[0].anchor_t - 180.0) < 1e-6, ws[0].anchor_t
    # 2nd-half 67' -> 2740 + (67-45)*60 = 2740 + 1320 = 4060s
    assert abs(ws[1].anchor_t - 4060.0) < 1e-6, ws[1].anchor_t
    # goal window padding: -20/+10 around the anchor
    assert abs(ws[0].start - 160.0) < 1e-6 and abs(ws[0].end - 190.0) < 1e-6
    assert all(w.verified for w in ws), "goals are feed-verified"
    print("  \u2713 match-clock -> video-time via per-period kick-off offsets")


def test_missing_kickoff_skips():
    events = [MatchEvent(minute=67, kind="goal", period=2)]
    ws = events_to_windows(events, {}, CFG, duration=6000)   # no kick-offs
    assert ws == [], "no kick-off mapping -> nothing placed"
    print("  \u2713 no kick-off mapping yields no windows (honest, not guessed)")


def test_ocr_alignment_corrects_drift():
    # feed says the goal is at 180s, but the OCR score-change (frame-accurate)
    # is at 188s -> the window should snap by +8s.
    ws = events_to_windows([MatchEvent(minute=3, kind="goal", period=1)],
                           {1: 0}, CFG, duration=6000)
    ocr = [SimpleNamespace(t=188.0, meta={"prev": "0-0", "score": "1-0"})]
    aligned = align_to_ocr(ws, ocr, CFG)
    assert abs(aligned[0].anchor_t - 188.0) < 1e-6, aligned[0].anchor_t
    assert aligned[0].score_after == "1-0"
    assert "scoreboard_ocr" in aligned[0].sources
    print("  \u2713 OCR clock snaps feed goal window, corrects kick-off drift")


if __name__ == "__main__":
    test_text_parsing()
    test_csv_and_importance()
    test_clock_to_video_mapping()
    test_missing_kickoff_skips()
    test_ocr_alignment_corrects_drift()
    print("\nALL EVENT-FEED TESTS PASSED \u2705")
