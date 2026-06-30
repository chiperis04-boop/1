"""Quality-polish tests (need ffmpeg; cv2/numpy for the frame sink).

Covers the rendering/quality improvements that are verifiable on CPU:

  * ff.RawFrameSink         : piping raw BGR frames straight into one H.264
                              encode (no lossy mp4v intermediate) yields a valid
                              1-video + 1-audio mp4 at the requested geometry.
  * audio loudnorm + duck   : compose_clip with a music bed builds a VALID
                              ffmpeg filtergraph (sidechaincompress + loudnorm)
                              and produces 1 video + 1 stereo-48k audio.
  * caption safe-zone       : drawtext uses the configured safe_y, not 0.72.
  * interpolated slow-mo    : the slowed window actually contains more unique
                              frames than a plain frame-stretch (motion interp).

Run directly:  python tests/test_polish.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.edit import ff  # noqa: E402
from src.edit.compose import compose_clip, _caption_drawtext  # noqa: E402


def _probe_streams(path):
    data = ff.probe(path)
    v = [s for s in data["streams"] if s.get("codec_type") == "video"]
    a = [s for s in data["streams"] if s.get("codec_type") == "audio"]
    return v, a


def _make_clip(path, w=320, h=240, secs=3, fps=30, with_audio=True):
    """Synthesise a tiny moving-pattern clip (+ a sine 'commentary' track)."""
    cmd = ["ffmpeg", "-y",
           "-f", "lavfi", "-i",
           f"testsrc=size={w}x{h}:rate={fps}:duration={secs}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=220:duration={secs}"]
    cmd += [*ff.venc_args("libx264")]
    if with_audio:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2"]
    cmd += ["-shortest", str(path)]
    ff.run(cmd, desc="make test clip")


def _make_music(path, secs=5):
    ff.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
            f"sine=frequency=440:duration={secs}",
            "-c:a", "aac", "-b:a", "128k", str(path)], desc="make music")


# --------------------------------------------------------------------------- #
def test_raw_frame_sink(tmp):
    out = str(tmp / "sink.mp4")
    w, h, n, fps = 160, 288, 24, 30
    sink = ff.RawFrameSink(out, w, h, fps, "libx264", audio_src=None)
    for i in range(n):
        frame = np.full((h, w, 3), 0, dtype=np.uint8)
        frame[:, :, 0] = (i * 10) % 256          # moving colour ramp
        sink.write(frame)
    sink.close()
    v, a = _probe_streams(out)
    assert len(v) == 1 and len(a) == 1, (len(v), len(a))
    assert (int(v[0]["width"]), int(v[0]["height"])) == (w, h)
    assert v[0]["codec_name"] == "h264", v[0]["codec_name"]
    print("  ✓ RawFrameSink: raw BGR frames -> single H.264 encode, 1V+1A, no mp4v")


def test_raw_frame_sink_surfaces_errors(tmp):
    # an unwritable output path must surface an ffmpeg error on close, not be
    # silently swallowed (errors-are-visible principle).
    out = str(tmp / "does_not_exist_dir" / "x.mp4")   # parent dir missing
    sink = ff.RawFrameSink(out, 100, 100, 30, "libx264")
    raised = False
    try:
        for _ in range(5):
            sink.write(np.zeros((100, 100, 3), dtype=np.uint8))
        sink.close()
    except ff.FFmpegError:
        raised = True
    assert raised, "failed encode should raise FFmpegError (errors not swallowed)"
    print("  ✓ RawFrameSink: failed encode surfaces FFmpegError (no silent fail)")


def test_compose_audio_ducking_and_loudnorm(tmp):
    clip = tmp / "clip.mp4"
    music = tmp / "music.m4a"
    _make_clip(clip, secs=3, with_audio=True)
    music_dir = tmp / "music"
    music_dir.mkdir()
    _make_music(music_dir / "track.m4a", secs=6)

    cfg = {
        "render": {"encoder": "libx264", "fps": 30},
        "edit": {
            "captions": {"enabled": False, "font": "", "fontsize": 40},
            "audio": {"music_dir": str(music_dir), "music_volume": 0.35,
                      "duck_under_commentary": True, "loudnorm": True,
                      "loudness_target_lufs": -14},
        },
        "_active_profile": {"width": 320, "height": 240},
    }
    out = str(tmp / "composed.mp4")
    compose_clip(str(clip), out, cfg, {}, captions=None, add_music=True)
    v, a = _probe_streams(out)
    assert len(v) == 1 and len(a) == 1, (len(v), len(a))
    assert int(a[0]["sample_rate"]) == 48000 and int(a[0]["channels"]) == 2
    print("  ✓ compose: sidechain-duck + loudnorm filtergraph valid -> 1V + 48k stereo")


def test_caption_safe_zone():
    cfg = {"edit": {"captions": {"enabled": True, "font": "",
                                 "fontsize": 58, "safe_y": 0.62}}}
    caps = [{"text": "GOLAZO", "start": 0.2, "end": 1.5}]
    filt, files = _caption_drawtext(caps, cfg, "/tmp/x.mp4")
    for f in files:
        Path(f).unlink(missing_ok=True)
    assert "y=h*0.620" in filt, filt
    assert "h*0.72" not in filt, "captions must not sit in the bottom UI band"
    print("  ✓ captions: drawtext honours safe_y (0.62), not the old 0.72")


def _unique_frame_count(path, w=64, h=64):
    """Decode to raw and count frames that differ from their predecessor."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={w}:{h}",
         "-pix_fmt", "gray", "-f", "rawvideo", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buf = proc.stdout
    fsize = w * h
    frames = [buf[i:i + fsize] for i in range(0, len(buf) - fsize + 1, fsize)]
    uniq = 1
    for i in range(1, len(frames)):
        if frames[i] != frames[i - 1]:
            uniq += 1
    return uniq, len(frames)


def test_slowmo_interpolation_adds_frames(tmp):
    from src.edit.effects import apply_slowmo
    clip = tmp / "clip.mp4"
    _make_clip(clip, w=320, h=240, secs=6, fps=30, with_audio=True)
    base_cfg = {
        "render": {"encoder": "libx264", "fps": 30},
        "_active_profile": {"width": 320, "height": 240},
        "edit": {"effects": {"slowmo_on_key": True, "slowmo_factor": 0.4,
                             "slowmo_window": 2.5}},
    }
    # plain stretch (no interpolation)
    plain = str(tmp / "plain.mp4")
    base_cfg["edit"]["effects"]["slowmo_interpolate"] = False
    apply_slowmo(str(clip), 3.0, plain, base_cfg)
    # motion-interpolated
    interp = str(tmp / "interp.mp4")
    base_cfg["edit"]["effects"]["slowmo_interpolate"] = True
    apply_slowmo(str(clip), 3.0, interp, base_cfg)

    u_plain, _ = _unique_frame_count(plain)
    u_interp, _ = _unique_frame_count(interp)
    assert u_interp > u_plain, (u_interp, u_plain)
    print(f"  ✓ slow-mo interpolation: unique frames {u_plain} -> {u_interp} "
          "(smoother motion, less judder)")


# --------------------------------------------------------------------------- #
def main() -> int:
    ff.ensure_tools()
    print("quality-polish tests (ffmpeg)")
    tmp = Path(tempfile.mkdtemp(prefix="fhs_polish_"))
    try:
        test_raw_frame_sink(tmp)
        test_raw_frame_sink_surfaces_errors(tmp)
        test_compose_audio_ducking_and_loudnorm(tmp)
        test_caption_safe_zone()
        test_slowmo_interpolation_adds_frames(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL POLISH TESTS PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
