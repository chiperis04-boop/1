"""Under-player occlusion tests (CPU-only, no GPU/YOLO needed).

Block B: tactical graphics must be composited BENEATH players. These tests
target the pure compositing function `composite_under_players` with a SYNTHETIC
player mask (so no seg model is required) and the honest graceful-fallback
behaviour of `Occluder` when no seg model can load.

Run directly:  python tests/test_occlusion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.graphics.homography import composite_under_players   # noqa: E402
from src.graphics.occlusion import Occluder                   # noqa: E402


def _scene():
    """A 40x40 grey frame, a full-frame red BGRA graphics layer, and a player
    mask covering the left half of the frame."""
    frame = np.full((40, 40, 3), 100, dtype=np.uint8)          # grey background
    layer = np.zeros((40, 40, 4), dtype=np.uint8)
    layer[:, :, 2] = 255                                       # red (BGR)
    layer[:, :, 3] = 255                                       # fully opaque graphics
    mask = np.zeros((40, 40), dtype=bool)
    mask[:, :20] = True                                        # player = left half
    return frame, layer, mask


def test_graphics_kept_under_player_mask():
    """Where the player mask is set, the ORIGINAL frame pixels survive (graphics
    are hidden BEHIND the player); elsewhere the graphics show through."""
    frame, layer, mask = _scene()
    out = composite_under_players(frame, layer, mask, alpha=1.0)
    # left half (player): pixels identical to the original grey frame
    assert np.array_equal(out[:, :20], frame[:, :20]), "graphics leaked over player"
    # right half (no player): graphics are visible (red channel maxed, changed)
    assert out[:, 20:, 2].min() == 255, "graphics not drawn outside the player"
    assert not np.array_equal(out[:, 20:], frame[:, 20:]), "graphics missing"
    print("  \u2713 composite: player pixels preserved, graphics only OUTSIDE the mask")


def test_alpha_blend_outside_mask():
    """Outside the mask the graphics blend by alpha over the frame (not a hard
    overwrite), so the halo reads as translucent."""
    frame, layer, mask = _scene()
    out = composite_under_players(frame, layer, mask, alpha=0.5)
    # right-half red channel = 0.5*255 + 0.5*100 = 177 (blended, not 255)
    val = int(out[0, 30, 2])
    assert 170 <= val <= 184, val
    print(f"  \u2713 composite: alpha blend outside mask (red={val}, translucent)")


def test_no_mask_draws_on_top():
    """With no mask (seg unavailable) the graphics blend on top everywhere —
    the honest fallback that never loses the overlay or crashes."""
    frame, layer, _ = _scene()
    out = composite_under_players(frame, layer, None, alpha=1.0)
    assert out[:, :, 2].min() == 255, "graphics should cover the whole frame"
    print("  \u2713 composite: None mask -> graphics on top everywhere (no crash)")


def test_occluder_graceful_without_model():
    """Occlusion enabled but the seg model can't load (no ultralytics / bad
    path) -> available() is False and player_mask() is None, so the caller falls
    back to the grounded-arc halo. Occlusion is never faked."""
    occ = Occluder({"telestration": {"occlusion": True,
                                     "occlusion_model": "/nonexistent-seg-model.pt"},
                    "vision": {"device": "cpu"}})
    assert occ.available() is False
    assert occ.player_mask(np.zeros((16, 16, 3), dtype=np.uint8)) is None
    print("  \u2713 occluder: honest fallback when seg model can't load")


def test_occluder_disabled_is_inert():
    occ = Occluder({"telestration": {"occlusion": False}})
    assert occ.available() is False
    assert occ.player_mask(np.zeros((8, 8, 3), dtype=np.uint8)) is None
    print("  \u2713 occluder: disabled -> inert (no model load attempted)")


def test_composer_occluded_annotator_routes_through_composite():
    """The Composer's world annotator, given an Occluder, must composite the
    halo/trail UNDER the player mask. With a full-frame mask every graphic is
    re-pasted behind the player, so the annotated frame equals the input;
    with no mask the graphics appear (frame changes)."""
    from src.render.composer import Composer
    from src.tracking.cameraman import FrameTrack

    cfg = {"telestration": {"occlusion": True, "spotlight_scorer": True,
                            "team_halos": False, "ball_trail": True,
                            "halo_grounded": False},
           "render": {"encoder": "libx264", "fps": 30}}
    comp = Composer(cfg, {})
    frame = np.full((80, 60, 3), 90, dtype=np.uint8)
    ft = FrameTrack(idx=0, players=[{"id": 1, "cls": 0,
                                     "xyxy": [10, 10, 40, 70],
                                     "center": [25, 40]}])
    ft.ball = {"xyxy": [20, 20, 26, 26], "center": [23, 23]}
    tele = cfg["telestration"]

    class _FullMask:
        def player_mask(self, f):
            return np.ones(f.shape[:2], dtype=bool)

    class _NoMask:
        def player_mask(self, f):
            return None

    out_full = comp._annotate_frame(frame.copy(), ft, 1, [], tele,
                                    analytics=None, hero_state={}, occluder=_FullMask())
    assert np.array_equal(out_full, frame), "full mask must hide all graphics"

    out_none = comp._annotate_frame(frame.copy(), ft, 1, [], tele,
                                    analytics=None, hero_state={}, occluder=_NoMask())
    assert not np.array_equal(out_none, frame), "graphics should show with no mask"
    print("  \u2713 composer: occluded annotator composites halo/trail under the mask")


def main() -> int:
    print("under-player occlusion tests (Block B, CPU)")
    for t in (test_graphics_kept_under_player_mask,
              test_alpha_blend_outside_mask,
              test_no_mask_draws_on_top,
              test_occluder_graceful_without_model,
              test_occluder_disabled_is_inert,
              test_composer_occluded_annotator_routes_through_composite):
        t()
    print("\nALL OCCLUSION TESTS PASSED \u2705")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
