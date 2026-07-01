"""Reel-survey tests (Block: whole-video VLM moment selection, CPU-only).

Verifies the pure logic (moment merging + window building) and the end-to-end
survey with a MOCK vision-LLM (no network): filler/replay picks are dropped,
highlight picks become EventWindows on the moment boundaries.

Run directly:  python tests/test_reel_survey.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detection import reel_survey                      # noqa: E402


def _shot(start, end):
    return NS(start=start, end=end, start_frame=int(start * 25),
              end_frame=int(end * 25))


def test_merge_moments_absorbs_tiny_and_caps():
    shots = [_shot(0, 1), _shot(1, 2), _shot(2, 12),      # two tiny + one long
             _shot(12, 30), _shot(30, 33), _shot(33, 60)]
    moments = reel_survey._merge_moments(shots, min_seconds=6.0, max_moments=40,
                                         duration=60.0)
    # tiny leading scenes absorbed; every moment (except maybe last) >= ~6s
    assert all(e > s for s, e in moments)
    assert len(moments) <= 6 and moments[0][0] == 0.0
    # capping: force a small max
    capped = reel_survey._merge_moments(shots, 6.0, 2, 60.0)
    assert len(capped) <= 2, capped
    print(f"  \u2713 merge_moments: {len(moments)} moments; cap-to-2 -> {len(capped)}")


def test_to_windows_keeps_highlights_drops_filler():
    moments = [(0, 10), (10, 25), (25, 40), (40, 55)]
    picks = {
        0: {"i": 0, "type": "goal", "keep": True, "hook": "WHAT A GOAL"},
        1: {"i": 1, "type": "pass", "keep": True},          # unknown kind -> drop
        2: {"i": 2, "type": "skill", "keep": True, "hook": "NUTMEG"},
        3: {"i": 3, "type": "celebration", "keep": True},    # -> drop
    }
    ws = reel_survey._to_windows(moments, picks, {})
    kinds = [w.kind for w in ws]
    assert kinds == ["goal", "skill"], kinds
    assert ws[0].start == 0 and ws[0].end == 10 and ws[0].verified is False
    assert ws[0].meta["hook"] == "WHAT A GOAL"
    print("  \u2713 to_windows: goal+skill kept on scene bounds; pass/celebration dropped")


def test_survey_reel_end_to_end_mocked(monkeypatch=None):
    """survey_reel with a mock VLM + mock scene/montage: picks -> windows."""
    class _Client:
        def is_configured(self):
            return True
        def chat_json(self, system, text, images):
            return {"moments": [
                {"i": 0, "type": "goal", "keep": True, "hook": "GOLAZO"},
                {"i": 1, "type": "pass", "keep": True},        # dropped (kind)
                {"i": 2, "type": "save", "keep": True, "hook": "HUGE SAVE"},
            ]}

    shots = [_shot(0, 12), _shot(12, 28), _shot(28, 45)]
    reel_survey.segment_shots = lambda v, c: shots            # not used (imported inside)
    # patch the module-level lookups used inside survey_reel
    import src.perception.shots as shots_mod
    shots_mod.segment_shots = lambda v, c: shots
    reel_survey._labeled_montage = lambda v, m, b, c: b"jpegbytes"

    cfg = {"detect": {"reel_survey": {"enabled": True, "moment_min_seconds": 6.0,
                                      "max_moments": 40, "cells_per_sheet": 20}}}
    ws = reel_survey.survey_reel("fake.mp4", cfg, duration=45.0, client=_Client())
    kinds = [w.kind for w in ws]
    assert kinds == ["goal", "save"], kinds
    assert ws[0].meta["hook"] == "GOLAZO" and ws[1].meta["hook"] == "HUGE SAVE"
    assert ["reel_survey"] == ws[0].sources
    print("  \u2713 survey_reel: mock VLM -> goal+save windows, pass dropped")


def test_survey_reel_graceful_without_vlm():
    class _NoClient:
        def is_configured(self):
            return False
    ws = reel_survey.survey_reel("fake.mp4", {"detect": {"reel_survey": {"enabled": True}}},
                                 client=_NoClient())
    assert ws == []
    print("  \u2713 survey_reel: no VLM -> [] (Scout falls back to detectors)")


def main() -> int:
    print("reel-survey tests (whole-video VLM moment selection)")
    for t in (test_merge_moments_absorbs_tiny_and_caps,
              test_to_windows_keeps_highlights_drops_filler,
              test_survey_reel_end_to_end_mocked,
              test_survey_reel_graceful_without_vlm):
        t()
    print("\nALL REEL-SURVEY TESTS PASSED \u2705")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
