"""Music beat-sync tests (need ffmpeg; numpy only otherwise).

  * detect_beats recovers ~120 BPM from a synthetic click track
  * snap_to_beats snaps within tolerance, leaves far times alone
  * beat_align_plan snaps a slow-mo onset onto a beat

Run directly:  python tests/test_music.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.edit import ff                                   # noqa: E402
from src.edit.music import beat_align_plan, detect_beats, snap_to_beats  # noqa: E402


def _click_track(path, bpm=120, secs=8):
    period = 60.0 / bpm
    # 1 kHz tone for the first 25 ms of each beat period => crisp onsets
    expr = f"lt(mod(t\\,{period})\\,0.025)*0.8*sin(2*PI*1000*t)"
    ff.run(["ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=gray:s=64x64:r=10:d={secs}",
            "-f", "lavfi", "-i",
            f"aevalsrc={expr}:s=16000:d={secs}",
            "-map", "0:v", "-map", "1:a",
            *ff.venc_args("libx264"), "-c:a", "aac", "-ar", "16000",
            "-shortest", str(path)], desc="click track")


def test_detect_beats_bpm(tmp):
    p = tmp / "click.mp4"
    _click_track(p, bpm=120, secs=8)
    grid = detect_beats(str(p))
    assert grid, "no beats detected"
    assert 100 <= grid.bpm <= 140, f"bpm {grid.bpm} off ~120"
    diffs = np.diff(grid.beats)
    assert abs(float(np.median(diffs)) - 0.5) < 0.1, np.median(diffs)
    print(f"  ✓ detect_beats: {grid.bpm:.0f} BPM, {len(grid.beats)} beats "
          f"(median gap {np.median(diffs):.2f}s)")


def test_snap_to_beats():
    beats = [0.0, 0.5, 1.0, 1.5, 2.0]
    snapped = snap_to_beats([0.52, 1.46, 3.0], beats, max_shift=0.12)
    assert snapped[0] == 0.5 and snapped[1] == 1.5    # snapped
    assert snapped[2] == 3.0                          # too far -> unchanged
    print("  ✓ snap_to_beats: near times snap, far times unchanged")


def test_beat_align_plan(tmp):
    from src.agents.editplan import EditPlan, SlowmoBeat
    p = tmp / "click.mp4"
    if not p.exists():
        _click_track(p, bpm=120, secs=8)
    plan = EditPlan(cut_in=0.0, slowmo_beats=[SlowmoBeat(2.46, 5.46, 0.4)])
    plan2, changed = beat_align_plan(plan, str(p))
    assert changed
    # 2.46 should snap to the nearest 0.5s-grid beat (2.5) and keep its duration
    assert abs(plan2.slowmo_beats[0].start - 2.5) < 0.12
    assert abs((plan2.slowmo_beats[0].end - plan2.slowmo_beats[0].start) - 3.0) < 1e-3
    print(f"  ✓ beat_align_plan: slow-mo onset 2.46 -> "
          f"{plan2.slowmo_beats[0].start:.2f}s (on beat)")


def test_strong_beats_selects_accented():
    """strong_beats() keeps the accented beats (bass drops) — top-frac by onset
    strength — and falls back to all beats when strengths are absent."""
    from src.edit.music import BeatGrid, strong_beats
    g = BeatGrid(bpm=120.0, beats=[0.0, 0.5, 1.0, 1.5],
                 strengths=[0.1, 0.9, 0.2, 0.95])
    drops = strong_beats(g, frac=0.5)
    assert 0.5 in drops and 1.5 in drops, drops
    assert 0.0 not in drops and 1.0 not in drops, drops
    assert strong_beats(BeatGrid(beats=[0.0, 1.0])) == [0.0, 1.0]   # no strengths
    print("  \u2713 strong_beats: keeps accented beats (bass drops), all when unknown")


def test_beat_align_lands_on_bass_drop():
    """AI slow-mo onset snaps to the nearest ACCENTED beat (bass drop) within the
    wider drop tolerance, not just any beat."""
    import src.edit.music as m
    from src.agents.editplan import EditPlan, SlowmoBeat
    grid = m.BeatGrid(bpm=120.0, beats=[0.0, 0.5, 1.0, 1.5, 2.0],
                      strengths=[0.1, 0.2, 0.1, 0.95, 0.2])   # drop at 1.5s
    orig = m.detect_beats
    m.detect_beats = lambda path, cfg=None: grid
    try:
        plan = EditPlan(slowmo_beats=[SlowmoBeat(1.62, 2.62, 0.4)])  # 0.12s from drop
        cfg = {"edit": {"audio": {"beat_snap_max_shift": 0.05,
                                  "beat_drop_max_shift": 0.35, "beat_drop_frac": 0.5}}}
        plan, changed = m.beat_align_plan(plan, "x.wav", cfg)
        assert changed and abs(plan.slowmo_beats[0].start - 1.5) < 1e-6, \
            plan.slowmo_beats[0].start
        assert abs((plan.slowmo_beats[0].end - plan.slowmo_beats[0].start) - 1.0) < 1e-6
    finally:
        m.detect_beats = orig
    print("  \u2713 beat_align: AI slow-mo lands on the bass drop (1.62 -> 1.50s)")


def main() -> int:
    ff.ensure_tools()
    print("music beat-sync tests")
    tmp = Path(tempfile.mkdtemp(prefix="fhs_music_"))
    try:
        test_detect_beats_bpm(tmp)
        test_snap_to_beats()
        test_beat_align_plan(tmp)
        test_strong_beats_selects_accented()
        test_beat_align_lands_on_bass_drop()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL MUSIC TESTS PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
