"""Audio-event detection tests (need ffmpeg; numpy only otherwise).

Synthesise audio with a quiet base and a loud sustained 'crowd roar' burst, and
check analyze_audio() builds an excitement curve and flags a roar overlapping
the burst. A quiet clip yields no roar.

Run directly:  python tests/test_audio_events.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.edit import ff                                   # noqa: E402
from src.perception.audio_events import analyze_audio     # noqa: E402


def _make_audio(path, with_burst=True, secs=10):
    if with_burst:
        # quiet 200Hz tone for the whole clip + a loud noise burst at 4-6s
        fc = (f"sine=frequency=200:duration={secs}:sample_rate=16000,volume=0.05[base];"
              f"anoisesrc=duration=2:sample_rate=16000:amplitude=0.8,adelay=4000[nz];"
              f"[base][nz]amix=inputs=2:duration=first:normalize=0[a]")
    else:
        fc = (f"sine=frequency=200:duration={secs}:sample_rate=16000,volume=0.05[a]")
    ff.run(["ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=gray:s=64x64:r=10:d={secs}",
            "-filter_complex", fc, "-map", "0:v", "-map", "[a]",
            *ff.venc_args("libx264"), "-c:a", "aac", "-ar", "16000",
            "-shortest", str(path)], desc="make audio clip")


def test_roar_detected(tmp):
    p = tmp / "roar.mp4"
    _make_audio(p, with_burst=True)
    res = analyze_audio(str(p))
    assert res.curve and len(res.curve) == len(res.times)
    roars = [e for e in res.events if e.label == "roar"]
    assert roars, f"expected a roar event, got {[e.to_dict() for e in res.events]}"
    # at least one roar overlaps the 4-6s burst
    assert any(e.start <= 6.0 and e.end >= 4.0 for e in roars), \
        [e.to_dict() for e in roars]
    # excitement is higher during the burst than at the start
    assert res.curve_at(5.0) > res.curve_at(1.0)
    print(f"  ✓ roar detected at ~{roars[0].t:.1f}s; excitement "
          f"{res.curve_at(1.0):.2f}->{res.curve_at(5.0):.2f}")


def test_quiet_no_roar(tmp):
    p = tmp / "quiet.mp4"
    _make_audio(p, with_burst=False)
    res = analyze_audio(str(p))
    assert not [e for e in res.events if e.label == "roar"], \
        [e.to_dict() for e in res.events]
    print("  ✓ steady quiet clip -> no roar event")


def main() -> int:
    ff.ensure_tools()
    print("audio-event tests")
    tmp = Path(tempfile.mkdtemp(prefix="fhs_audio_"))
    try:
        test_roar_detected(tmp)
        test_quiet_no_roar(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL AUDIO-EVENT TESTS PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
