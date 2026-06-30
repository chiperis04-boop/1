"""Shot-segmentation tests (need ffmpeg + scenedetect + cv2).

Builds a synthetic clip of three visually distinct takes (the third is a near
duplicate of the first) and checks:
  * segment_shots finds multiple shots and tiles the whole clip contiguously
  * frame_segments exactly covers [0, n_frames)
  * mark_duplicate_shots flags the repeated take (literal-duplicate heuristic)
  * graceful single-shot fallback when segmentation is disabled

Run directly:  python tests/test_shots.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.edit import ff                                   # noqa: E402
from src.perception.shots import (frame_segments,         # noqa: E402
                                   mark_duplicate_shots, segment_shots)


def _make_three_shot_clip(path, fps=30):
    """Concatenate three 2s takes: red(moving) | green(moving) | red(moving).

    Hard cuts between solid, very different colours give PySceneDetect crisp
    boundaries; the 1st and 3rd takes share a colour so the duplicate heuristic
    has something to catch.
    """
    def take(color, motion, out):
        # a moving box over a solid colour = motion + a stable dominant hue
        ff.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c={color}:s=320x240:d=2:r={fps}",
            "-f", "lavfi", "-i", f"color=c=white:s=40x40:d=2:r={fps}",
            "-filter_complex",
            "[0:v][1:v]overlay=x='mod(t*120,280)':y=100,format=yuv420p[v]",
            "-map", "[v]", *ff.venc_args("libx264"), out,
        ], desc="take")

    d = Path(path).parent
    a, b, c = d / "a.mp4", d / "b.mp4", d / "c.mp4"
    take("red", 1, str(a))
    take("green", 1, str(b))
    take("red", 1, str(c))      # near-duplicate of take A
    lst = d / "list.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in (a, b, c)), encoding="utf-8")
    ff.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
            "-c", "copy", str(path)], desc="concat takes")


def test_segment_and_tile(tmp):
    clip = tmp / "three.mp4"
    _make_three_shot_clip(clip)
    shots = segment_shots(str(clip), {"shots": {"enabled": True,
                                                 "threshold": 27.0,
                                                 "min_seconds": 0.4}})
    assert len(shots) >= 3, f"expected >=3 shots, got {len(shots)}"
    # contiguous, starts at 0
    assert shots[0].start_frame == 0
    for i in range(1, len(shots)):
        assert shots[i].start_frame == shots[i - 1].end_frame, "shots not contiguous"
    n = shots[-1].end_frame
    segs = frame_segments(shots, n)
    # segments exactly tile [0, n)
    assert segs[0][0] == 0 and segs[-1][1] == n
    for i in range(1, len(segs)):
        assert segs[i][0] == segs[i - 1][1]
    print(f"  ✓ segment_shots: {len(shots)} shots, contiguous + tiling [0,{n})")


def test_duplicate_replay_flag(tmp):
    clip = tmp / "three.mp4"
    if not clip.exists():
        _make_three_shot_clip(clip)
    shots = segment_shots(str(clip), {"shots": {"enabled": True, "min_seconds": 0.4}})
    shots = mark_duplicate_shots(str(clip), shots,
                                 {"shots": {"detect_replays": True,
                                            "replay_correlation": 0.9}})
    # the last red take should be flagged as a duplicate of the first red take
    assert any(s.is_replay for s in shots[1:]), \
        "expected a later take to be flagged as a duplicate"
    # and the very first shot is never a replay
    assert shots[0].is_replay is False
    print("  ✓ mark_duplicate_shots: repeated take flagged (literal-duplicate)")


def test_disabled_single_shot(tmp):
    clip = tmp / "three.mp4"
    if not clip.exists():
        _make_three_shot_clip(clip)
    shots = segment_shots(str(clip), {"shots": {"enabled": False}})
    assert len(shots) == 1 and shots[0].start_frame == 0
    print("  ✓ disabled -> graceful single shot spanning the clip")


def main() -> int:
    ff.ensure_tools()
    print("shot-segmentation tests")
    tmp = Path(tempfile.mkdtemp(prefix="fhs_shots_"))
    try:
        test_segment_and_tile(tmp)
        test_duplicate_replay_flag(tmp)
        test_disabled_single_shot(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL SHOT TESTS PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
