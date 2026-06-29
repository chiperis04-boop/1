"""Audit edge-case tests (no GPU): encoder fallback, audio synthesis, and an
end-to-end runner run (per_clip + compilation) with ingest/detectors mocked.

Run:  python tests/test_audit.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.edit import ff                       # noqa: E402
from src.detect.types import Signal           # noqa: E402
from src.ingest import MediaInfo              # noqa: E402
import src.runner as runner                   # noqa: E402


def _mk_source(path: str, dur: int = 40, with_audio: bool = True):
    cmd = ["ffmpeg", "-y",
           "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=30:duration={dur}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [path]
    ff.run(cmd, desc="mk source")


# --------------------------------------------------------------------------- #
def test_encoder_fallback():
    # no GPU here, so an nvenc request must fall back to libx264
    enc = ff.pick_encoder("h264_nvenc")
    assert enc == "libx264", f"expected libx264 fallback, got {enc}"
    assert ff.pick_encoder("libx264") == "libx264"
    print("  ✓ encoder fallback: h264_nvenc -> libx264 (no GPU)")


def test_standardize_video_only():
    tmp = Path(tempfile.mkdtemp(prefix="aud_av_"))
    vid = str(tmp / "noaudio.mp4")
    _mk_source(vid, dur=3, with_audio=False)
    assert not ff.has_audio(vid), "fixture should have no audio"
    out = str(tmp / "std.mp4")
    ff.standardize(vid, out, 1080, 1920, 30, "libx264")
    assert ff.has_audio(out), "standardize must synthesise a silent track"
    print("  ✓ standardize: video-only input gains a silent audio track")


def _run_with_mocks(out_root: str, mode: str, target: float = 30.0):
    """Run the real runner with ingest + detectors mocked so we exercise
    fusion -> clipper -> per-clip/compilation edit chain end-to-end (CPU only,
    vision disabled), without needing librosa/whisper/easyocr/models."""
    tmp = Path(tempfile.mkdtemp(prefix="aud_src_"))
    src = str(tmp / "match.mp4")
    _mk_source(src, dur=40)

    def fake_ingest(s, workdir, cfg):
        return MediaInfo(src_path=src, duration=40.0, fps=30.0, width=1280,
                         height=720, has_audio=True, audio_path=None,
                         proxy_path=src)

    def fake_audio(audio_path, cfg):
        return [Signal(t=t, source="audio", strength=1.0) for t in (10, 28)]

    none = lambda *a, **k: []
    runner.ingest = fake_ingest
    runner.detect_audio = fake_audio
    runner.detect_scenes = none
    runner.detect_scoreboard = none
    runner.detect_commentary = none

    overrides = {
        "vision": {"enabled": False},
        # orchestration test: effects are covered by test_montage; disable the
        # heavy ones here to keep CPU runtime sane while still exercising
        # fusion -> clipper -> reframe -> compose -> branding -> compilation.
        "edit": {"effects": {"slowmo_on_key": False, "freeze_zoom": False}},
        "clip": {"pre_seconds": 3.0, "post_seconds": 2.0,
                 "goal_pre_seconds": 3.0, "goal_post_seconds": 2.0},
        "render": {"output_mode": mode,
                   "compilation": {"target_seconds": target,
                                   "min_seconds": 8, "max_seconds": 60}},
    }
    return runner.run_pipeline(src, profile="tiktok", out_root=out_root,
                              overrides=overrides)


def test_runner_per_clip():
    out = tempfile.mkdtemp(prefix="aud_out_")
    res = _run_with_mocks(out, "per_clip")
    assert res.status == "ok", f"status={res.status} err={res.error}"
    assert res.moments >= 1, "no moments fused from mocked signals"
    ok = [c for c in res.clips if c.status == "ok"]
    assert ok, "no clips rendered"
    for c in ok:
        assert Path(c.path).exists(), f"missing clip file {c.path}"
        d = ff.probe(c.path)
        v = [s for s in d["streams"] if s["codec_type"] == "video"]
        a = [s for s in d["streams"] if s["codec_type"] == "audio"]
        assert len(v) == 1 and len(a) == 1
        assert (v[0]["width"], v[0]["height"]) == (1080, 1920)
    assert (Path(res.out_dir) / "result.json").exists()
    print(f"  ✓ runner per_clip: {len(ok)} clips @1080x1920, result.json written")


def test_runner_compilation():
    out = tempfile.mkdtemp(prefix="aud_out2_")
    res = _run_with_mocks(out, "compilation", target=10.0)
    assert res.status == "ok", f"status={res.status} err={res.error}"
    assert res.reel_path and Path(res.reel_path).exists(), "reel not produced"
    d = ff.probe(res.reel_path)
    v = [s for s in d["streams"] if s["codec_type"] == "video"]
    a = [s for s in d["streams"] if s["codec_type"] == "audio"]
    assert len(v) == 1 and len(a) == 1
    assert (v[0]["width"], v[0]["height"]) == (1080, 1920)
    dur = ff.duration(res.reel_path)
    assert 6 <= dur <= 70, f"reel duration {dur:.1f}s out of expected window"
    assert Path(res.reel_path).with_suffix(".txt").exists(), "no reel caption"
    print(f"  ✓ runner compilation: reel {dur:.1f}s @1080x1920 + caption")


def main() -> int:
    ff.ensure_tools()
    print("audit edge-case + integration tests")
    test_encoder_fallback()
    test_standardize_video_only()
    test_runner_per_clip()
    test_runner_compilation()
    print("\nAUDIT TESTS PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
