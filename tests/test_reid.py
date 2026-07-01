"""Cross-shot hero Re-ID tests (pure numpy; no GPU/ffmpeg).

  * histogram_embedding distinguishes kit colours
  * best_match / cross_shot_hero_map link the hero across shots by appearance
  * per_frame_hero + cameraman._focus_points follow the right player per shot

Run directly:  python tests/test_reid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.perception.shots import Shot                       # noqa: E402
from src.tracking.cameraman import FrameTrack, _focus_points  # noqa: E402
from src.vision.reid import (best_match, cosine,            # noqa: E402
                             cross_shot_hero_map, histogram_embedding,
                             per_frame_hero)


def _solid(color_bgr, w=20, h=40):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return img


def test_histogram_embedding_separates_colours():
    red = histogram_embedding(_solid((0, 0, 200)))
    red2 = histogram_embedding(_solid((0, 0, 180)))
    blue = histogram_embedding(_solid((200, 0, 0)))
    assert cosine(red, red2) > cosine(red, blue)
    assert cosine(red, blue) < 0.5
    print(f"  ✓ histogram embedding: sim(red,red)={cosine(red, red2):.2f} "
          f"> sim(red,blue)={cosine(red, blue):.2f}")


def test_best_match_and_cross_shot_map():
    red = histogram_embedding(_solid((0, 0, 200)))
    blue = histogram_embedding(_solid((200, 0, 0)))
    # shot 0: hero (id 5) is red. shot 1: tracker renumbered -> red is id 22,
    # blue is id 23. Re-ID must link the hero to 22 in shot 1.
    shot_embs = {0: {5: red, 6: blue},
                 1: {22: red, 23: blue}}
    mid, sim = best_match(red, shot_embs[1])
    assert mid == 22 and sim > 0.9
    hero_map = cross_shot_hero_map(hero_id=5, shot_track_embs=shot_embs,
                                   hero_shot=0, min_sim=0.5)
    assert hero_map[0] == 5 and hero_map[1] == 22, hero_map
    print(f"  ✓ cross-shot map: hero 5 (red) -> id {hero_map[1]} in next shot")


def _player(pid, cx, cy):
    return {"id": pid, "cls": 0, "xyxy": [cx - 10, cy - 20, cx + 10, cy + 20],
            "center": [float(cx), float(cy)]}


def test_per_frame_hero_follows_across_shots():
    # shot 0 (frames 0-9): hero is id 5 at x=100; shot 1 (10-19): hero id 22 at x=900
    frames = []
    for i in range(10):
        frames.append(FrameTrack(idx=i, players=[_player(5, 100, 360),
                                                 _player(6, 700, 360)]))
    for i in range(10, 20):
        frames.append(FrameTrack(idx=i, players=[_player(22, 900, 360),
                                                 _player(23, 300, 360)]))
    shots = [Shot(0, 0, 1, 0, 10), Shot(1, 1, 2, 10, 20)]
    hero_map = {0: 5, 1: 22}
    ids = per_frame_hero(len(frames), shots, hero_map, default_hero=5)
    assert ids[0] == 5 and ids[15] == 22
    focus = _focus_points(frames, ids, 1280, 720)
    assert focus[2][0] < 300, focus[2]        # shot 0 -> following x=100
    assert focus[15][0] > 700, focus[15]      # shot 1 -> following x=900 (relinked)
    print(f"  ✓ per-frame hero: focus x {focus[2][0]:.0f} (shot0) -> "
          f"{focus[15][0]:.0f} (shot1, relinked id)")


def main() -> int:
    print("cross-shot Re-ID tests (pure)")
    test_histogram_embedding_separates_colours()
    test_best_match_and_cross_shot_map()
    test_per_frame_hero_follows_across_shots()
    print("\nALL RE-ID TESTS PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
