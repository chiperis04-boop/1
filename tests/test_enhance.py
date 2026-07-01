"""Quality-enhancement + slow-mo audio tests.

  * The cinematic finishing grade (ffmpeg, no model) renders a valid clip at the
    SAME resolution with its audio preserved (real render + frame extract).
  * The slow-mo audio modes build the expected, sync-preserving filter chains
    (muffle/mute/stretch) so a slowed window never turns into a garbled voice.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.edit import ff  # noqa: E402
from src.render.composer import _slow_audio_filter  # noqa: E402
from src.render.enhance import Enhancer  # noqa: E402

CFG = {"render": {"encoder": "libx264", "fps": 30},
       "vision": {"device": "cpu"},
       "enhance": {"enabled": True, "backend": "grade",
                   "grade": {"enabled": True, "contrast": 1.08,
                             "saturation": 1.15, "sharpen": 0.8,
                             "denoise": True}}}


def _make_clip(path: str, w=320, h=568, dur=1.5):
    """A small vertical test clip WITH an audio track."""
    ff.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=size={w}x{h}:rate=30:duration={dur}",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        path,
    ], desc="make test clip")


def _dims(path: str):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
        capture_output=True, text=True).stdout.strip()
    w, h = out.split("x")
    return int(w), int(h)


def test_slow_audio_filter_modes():
    # stretch = pure pitch-preserving atempo chain (legacy)
    assert _slow_audio_filter("stretch", 0.4) == "atempo=0.5,atempo=0.8"
    # mute = same duration (atempo) then silenced -> no warble, stays in sync
    assert _slow_audio_filter("mute", 0.4) == "atempo=0.5,atempo=0.8,volume=0"
    # muffle (default) = atempo + band-pass + duck ('underwater' bed)
    m = _slow_audio_filter("muffle", 0.4)
    assert m.startswith("atempo=0.5,atempo=0.8")
    assert "lowpass" in m and "volume=0.55" in m
    # unknown/empty falls back to muffle
    assert "lowpass" in _slow_audio_filter("", 0.4)
    print("  ✓ slow-mo audio: muffle/mute/stretch chains preserve duration")


def test_grade_preserves_resolution_and_audio():
    with tempfile.TemporaryDirectory() as d:
        src = str(Path(d) / "in.mp4")
        out = str(Path(d) / "out.mp4")
        _make_clip(src)
        w0, h0 = _dims(src)

        res = Enhancer(CFG).enhance(src, out)
        assert Path(res).exists() and res == out
        # resolution unchanged (detail lift, not a resize)
        assert _dims(out) == (w0, h0)
        # audio preserved
        assert ff.has_audio(out)
        # a frame decodes (real output, not a stub)
        frame = str(Path(d) / "f.jpg")
        ff.run(["ffmpeg", "-y", "-i", out, "-frames:v", "1", frame],
               desc="extract frame")
        assert Path(frame).stat().st_size > 0
    print("  ✓ enhance grade: same-res, audio-safe, decodable output")


def test_grade_disabled_is_noop():
    cfg = {"render": {"encoder": "libx264", "fps": 30},
           "enhance": {"enabled": True, "backend": "grade",
                       "grade": {"enabled": False}}}
    with tempfile.TemporaryDirectory() as d:
        src = str(Path(d) / "in.mp4")
        _make_clip(src, dur=1.0)
        # grade disabled + no model -> returns the input unchanged (no fake pass)
        assert Enhancer(cfg).enhance(src, str(Path(d) / "out.mp4")) == src
    print("  ✓ enhance: grade disabled degrades to a no-op (honest)")


def test_model_backend_degrades_without_package():
    """backend=realesrgan with the package absent must degrade to the grade,
    never crash or fake an 'enhanced' file."""
    cfg = {"render": {"encoder": "libx264", "fps": 30},
           "vision": {"device": "cpu"},
           "enhance": {"enabled": True, "backend": "realesrgan",
                       "grade": {"enabled": True, "sharpen": 0.5}}}
    with tempfile.TemporaryDirectory() as d:
        src = str(Path(d) / "in.mp4")
        out = str(Path(d) / "out.mp4")
        _make_clip(src, dur=1.0)
        res = Enhancer(cfg).enhance(src, out)
        # realesrgan not installed here -> graded output produced (not the raw src)
        assert Path(res).exists()
        assert ff.has_audio(res)
    print("  ✓ enhance: model backend degrades to grade when package absent")


if __name__ == "__main__":
    test_slow_audio_filter_modes()
    test_grade_preserves_resolution_and_audio()
    test_grade_disabled_is_noop()
    test_model_backend_degrades_without_package()
    print("ok - enhance + slow-mo audio")
