"""Deterministic QA tests (need ffmpeg + cv2).

Builds synthetic outputs that exhibit specific defects and checks qa_report
flags them, and that a clean clip passes:
  * clean colourful clip with audio -> passes
  * pillarboxed clip (black side bars) -> 'letterbox'
  * all-black clip -> 'black_frames'
  * duration outside the expected window -> 'too_long'

Run directly:  python tests/test_qa.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.edit import ff                       # noqa: E402
from src.qa.checks import qa_report           # noqa: E402


def _clean(path, w=216, h=384, secs=3, fps=30):
    ff.run(["ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate={fps}:duration={secs}",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={secs}",
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
            *ff.venc_args("libx264"), "-c:a", "aac", "-ar", "48000", "-ac", "2",
            "-shortest", str(path)], desc="clean")


def _pillarbox(path, w=216, h=384, secs=3, fps=30):
    # a narrow centred picture on a black canvas => big black side bars
    ff.run(["ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc=size=80x{h}:rate={fps}:duration={secs}",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={secs}",
            "-vf", f"pad={w}:{h}:(ow-iw)/2:0:color=black,format=yuv420p",
            *ff.venc_args("libx264"), "-c:a", "aac", "-ar", "48000", "-ac", "2",
            "-shortest", str(path)], desc="pillarbox")


def _black(path, w=216, h=384, secs=3, fps=30):
    ff.run(["ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:rate={fps}:duration={secs}",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={secs}",
            *ff.venc_args("libx264"), "-c:a", "aac", "-ar", "48000", "-ac", "2",
            "-shortest", str(path)], desc="black")


def test_clean_passes(tmp):
    p = tmp / "clean.mp4"
    _clean(p)
    r = qa_report(str(p), cfg={"edit": {"audio": {"loudness_target_lufs": -14}}},
                  expected={"width": 216, "height": 384,
                            "min_seconds": 1, "max_seconds": 10})
    assert r.passed, (r.score, r.issues, r.checks)
    assert "letterbox" not in r.issues and "black_frames" not in r.issues
    print(f"  ✓ clean clip passes QA (score {r.score}, no issues)")


def test_pillarbox_flagged(tmp):
    p = tmp / "pillar.mp4"
    _pillarbox(p)
    r = qa_report(str(p))
    assert "letterbox" in r.issues and not r.passed, (r.issues, r.checks)
    print("  ✓ pillarboxed clip -> 'letterbox' flagged, QA fails")


def test_black_flagged(tmp):
    p = tmp / "black.mp4"
    _black(p)
    r = qa_report(str(p))
    assert "black_frames" in r.issues and not r.passed, (r.issues, r.checks)
    print("  ✓ all-black clip -> 'black_frames' flagged, QA fails")


def test_too_long_flagged(tmp):
    p = tmp / "clean.mp4"
    if not p.exists():
        _clean(p, secs=3)
    r = qa_report(str(p), expected={"max_seconds": 1.0})
    assert "too_long" in r.issues, r.issues
    print("  ✓ over-length clip -> 'too_long' flagged")


def main() -> int:
    ff.ensure_tools()
    print("QA checks tests")
    tmp = Path(tempfile.mkdtemp(prefix="fhs_qa_"))
    try:
        test_clean_passes(tmp)
        test_pillarbox_flagged(tmp)
        test_black_flagged(tmp)
        test_too_long_flagged(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL QA TESTS PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
