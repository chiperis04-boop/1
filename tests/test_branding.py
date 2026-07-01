"""Branding overlay-selection tests (CPU-only, no ffmpeg).

Guards the Block-A fix: in the v2 Studio the Composer already burns the
event-driven hook (from the Director's EditPlan) and the reaction-gated stat
cards, so `apply_branding` must NOT add a second static hook or the big
persistent stats block — that overlap was the "two hooks + POSSESSION 100% over
the player" UI overload. The v1 (classic) path keeps the full branding set.

Run directly (no pytest needed):

    python tests/test_branding.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.branding.overlays import _overlay_specs        # noqa: E402
from src.detect.types import Moment                     # noqa: E402


def _moment():
    return Moment(t=100.0, start=90.0, end=110.0, confidence=0.9, kind="goal",
                  minute=67)


def _branding():
    return {
        "channel": {"font": ""},
        "hook": {"enabled": True,
                 "templates": {"goal": ["Rate this finish 1-10"]}},
        "lower_third": {"enabled": True, "show_minute": True},
        "stats_overlay": {"enabled": True, "show_shot_distance": True,
                          "show_players_beaten": True},
        "watermark": {"enabled": True, "text": "@yourchannel", "opacity": 0.6},
    }


def _texts(specs):
    return [t for t, _ in specs]


def test_v2_minimal_drops_duplicate_hook_and_stats():
    """composer_typography=True -> no static hook, no big stats block."""
    stats = {"shot_distance_m": 18, "players_beaten": 3}
    specs = _overlay_specs(_moment(), stats, _branding(), composer_typography=True)
    texts = _texts(specs)
    # the static branding hook must be gone (Composer owns the hook)
    assert "Rate this finish 1-10" not in texts, texts
    # the big stats block must be gone (Composer owns reaction stat cards)
    assert not any("Shot:" in t or "Beaten:" in t for t in texts), texts
    # a compact lower-third + watermark remain -> ONE clean overlay set
    assert any(t.startswith("GOAL") for t in texts), texts
    assert "@yourchannel" in texts, texts
    print("  \u2713 v2 minimal: no duplicate hook, no big stats block; "
          "compact lower-third + watermark kept")


def test_v2_lower_third_is_edge_safe_and_smaller():
    """The v2 lower-third sits inside the horizontal safe margin (x=w*0.06, not
    a hard x=60 that clipped on some frames) and is smaller than v1."""
    specs = _overlay_specs(_moment(), {}, _branding(), composer_typography=True)
    lt = next(opts for txt, opts in specs if txt.startswith("GOAL"))
    assert "x=w*0.06" in lt, lt                 # margin as a fraction, not clipped
    assert "fontsize=38" in lt, lt              # smaller reference-look size
    assert "y=h*0.80" in lt, lt                 # above the bottom UI band
    print("  \u2713 v2 lower-third: edge-safe x=w*0.06, compact fontsize, safe y")


def test_v1_keeps_full_branding_set():
    """Classic v1 path (composer_typography=False) keeps hook + stats + more."""
    stats = {"shot_distance_m": 18, "players_beaten": 3}
    specs = _overlay_specs(_moment(), stats, _branding(), composer_typography=False)
    texts = _texts(specs)
    assert "Rate this finish 1-10" in texts, texts          # static hook present
    assert any("Shot:" in t for t in texts), texts          # stats present
    assert any("Beaten:" in t for t in texts), texts
    assert any(t.startswith("GOAL") for t in texts), texts   # lower-third present
    assert "@yourchannel" in texts, texts                    # watermark present
    print("  \u2713 v1 classic: full branding set (hook + stats + lower-third + wm)")


def test_watermark_respects_disabled_flag():
    b = _branding()
    b["watermark"]["enabled"] = False
    specs = _overlay_specs(_moment(), {}, b, composer_typography=True)
    assert "@yourchannel" not in _texts(specs)
    print("  \u2713 watermark honours its enabled=false flag")


def main() -> int:
    print("branding overlay-selection tests (Block A)")
    for t in (test_v2_minimal_drops_duplicate_hook_and_stats,
              test_v2_lower_third_is_edge_safe_and_smaller,
              test_v1_keeps_full_branding_set,
              test_watermark_respects_disabled_flag):
        t()
    print("\nALL BRANDING TESTS PASSED \u2705")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
