"""Synthetic end-to-end test of the montage chain (no GPU, no models).

Builds a wide synthetic clip with ffmpeg `lavfi` (moving test pattern + tone),
then runs the non-vision montage stages exactly as the runner would:

    letterbox reframe -> slow-mo -> freeze-zoom -> compose -> branding(intro/outro)

and asserts every output is a valid MP4 with exactly one video + one audio
stream, and that the final clip matches the chosen profile geometry.

Run directly (no pytest required):

    python tests/test_montage.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detect.types import Moment            # noqa: E402
from src.edit import ff                         # noqa: E402
from src.edit.compose import compose_clip       # noqa: E402
from src.edit.effects import apply_slowmo, freeze_zoom_intro  # noqa: E402
from src.edit.reframe import _letterbox         # noqa: E402
from src.branding.overlays import apply_branding  # noqa: E402


PROFILE = {"width": 1080, "height": 1920, "fps": 30}

CFG = {
    "render": {"encoder": "libx264", "fps": 30},
    "_active_profile": PROFILE,
    "edit": {
        "reframe": {"target_aspect": "9:16", "mode": "letterbox", "smoothing": 0.85},
        "effects": {"slowmo_on_key": True, "slowmo_factor": 0.4,
                    "slowmo_window": 2.0, "freeze_zoom": True,
                    "freeze_zoom_scale": 1.4},
        "audio": {"music_dir": "assets/music", "crowd_sfx": "assets/sfx/none.wav",
                  "music_volume": 0.35},
        "captions": {"enabled": True, "font": "", "fontsize": 58,
                     "max_words_per_line": 4},
    },
}

BRAND = {
    "channel": {"name": "TEST", "font": ""},
    "hook": {"enabled": True, "templates": {"goal": ["Spot the mistake 👀, [test]: 100%"]}},
    "lower_third": {"enabled": True, "show_minute": True},
    "stats_overlay": {"enabled": True, "show_shot_distance": True,
                      "show_sprint_distance": True, "show_players_beaten": True},
    "watermark": {"enabled": True, "text": "@test", "opacity": 0.6},
    "intro": {"enabled": True, "fallback_text": "TEST CHANNEL"},
    "outro": {"enabled": True, "cta_text": "Follow!"},
}


def _assert_av(path: str, label: str, w=None, h=None):
    data = ff.probe(path)
    streams = data.get("streams", [])
    v = [s for s in streams if s.get("codec_type") == "video"]
    a = [s for s in streams if s.get("codec_type") == "audio"]
    assert len(v) == 1, f"{label}: expected 1 video stream, got {len(v)}"
    assert len(a) == 1, f"{label}: expected 1 audio stream, got {len(a)}"
    if w and h:
        assert (v[0]["width"], v[0]["height"]) == (w, h), \
            f"{label}: geometry {v[0]['width']}x{v[0]['height']} != {w}x{h}"
    print(f"  ✓ {label}: 1 video + 1 audio"
          f"{f' @ {w}x{h}' if w else ''}  ({Path(path).name})")


def main() -> int:
    ff.ensure_tools()
    tmp = Path(tempfile.mkdtemp(prefix="fhs_test_"))
    print(f"workdir: {tmp}")

    # 1) synthetic wide source clip (6s, 1280x720, pattern + tone)
    src = str(tmp / "src.mp4")
    ff.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", src,
    ], desc="make source")
    _assert_av(src, "source", 1280, 720)

    key_t = 3.0
    w, h = PROFILE["width"], PROFILE["height"]

    # 2) reframe (letterbox path; action-track needs the vision TrackResult)
    r = _letterbox(src, str(tmp / "reframed.mp4"), CFG)
    _assert_av(r, "reframe", w, h)

    # 3) slow-mo on the key beat
    s = apply_slowmo(r, key_t, str(tmp / "slowmo.mp4"), CFG)
    _assert_av(s, "slowmo", w, h)

    # 4) freeze-zoom call-out intro
    f = freeze_zoom_intro(s, key_t, str(tmp / "freeze.mp4"), CFG)
    _assert_av(f, "freeze-zoom", w, h)

    # 5) compose: grade + captions (+ music if any present)
    caps = [{"text": "GOLAZO! 100%: top corner", "start": 0.2, "end": 1.5},
            {"text": "watch [this] run", "start": 1.6, "end": 2.6}]
    c = compose_clip(f, str(tmp / "composed.mp4"), CFG, BRAND, caps)
    _assert_av(c, "compose", w, h)

    # 6) branding overlays + intro/outro concat
    m = Moment(t=key_t, start=0.0, end=6.0, confidence=0.9, kind="goal",
               minute=67, sources=["audio", "scoreboard_ocr"])
    final = apply_branding(c, m, {"shot_distance_m": 24, "sprint_distance_m": 40,
                                  "players_beaten": 2},
                           str(tmp / "final.mp4"), CFG, BRAND)
    _assert_av(final, "final (with intro/outro)", w, h)

    # final must be longer than the body (intro+outro added)
    assert ff.duration(final) > ff.duration(c), "intro/outro did not extend clip"
    print(f"  ✓ final duration {ff.duration(final):.2f}s > body {ff.duration(c):.2f}s")

    # 7) compilation: stitch two segments into one reel
    from src.edit.compilation import build_compilation, select_for_duration
    seg_a = compose_clip(r, str(tmp / "seg_a.mp4"), CFG, BRAND, caps, add_music=False)
    seg_b = compose_clip(s, str(tmp / "seg_b.mp4"), CFG, BRAND, None, add_music=False)
    items = [
        {"path": seg_a, "duration": ff.duration(seg_a), "confidence": 0.9, "order": 0},
        {"path": seg_b, "duration": ff.duration(seg_b), "confidence": 0.8, "order": 1},
    ]
    chosen = select_for_duration(items, target=20, dur_max=60, per_moment_max=0)
    assert chosen, "selection returned no segments"
    reel = build_compilation([x["path"] for x in chosen], str(tmp / "reel.mp4"),
                             CFG, BRAND)
    _assert_av(reel, "compilation reel", w, h)
    seg_sum = sum(ff.duration(x["path"]) for x in chosen)
    assert ff.duration(reel) > seg_sum, "reel shorter than its segments"
    print(f"  ✓ reel duration {ff.duration(reel):.2f}s from {len(chosen)} "
          f"segments (sum {seg_sum:.2f}s) + intro/outro")

    # 8) duration targeting: select_for_duration lands a 30-60s reel window
    synth = [{"path": f"s{i}", "duration": 9.0, "confidence": 0.9 - i * 0.01,
              "order": i} for i in range(12)]
    picked = select_for_duration(synth, target=45, dur_max=60, per_moment_max=12)
    picked_dur = sum(min(p["duration"], 12) for p in picked)
    assert 30.0 <= picked_dur <= 60.0, f"reel target out of 30-60s: {picked_dur}"
    # chronological order is restored after the confidence-first pick
    assert [p["order"] for p in picked] == sorted(p["order"] for p in picked)
    print(f"  ✓ duration target: picked {len(picked)} moments ~{picked_dur:.0f}s "
          "(inside 30-60s), chronological")

    # 9) beat-synced compilation: cut points snap to the music beat grid
    _test_beat_synced_cuts(tmp, [x["path"] for x in chosen])

    print("\nALL MONTAGE CHECKS PASSED ✅")
    return 0


def _test_beat_synced_cuts(tmp: Path, segments: list[str]):
    """Generate a 120-BPM click track, then assert the compilation cut plan
    trims each segment to a whole number of beat periods (transitions on beat),
    and that a beat-synced reel still renders to a valid 1V+1A mp4."""
    from src.edit.compilation import _beat_cut_plan, build_compilation
    from src.edit.music import detect_beats

    music_dir = tmp / "music"
    music_dir.mkdir(exist_ok=True)
    click = str(music_dir / "click.wav")
    # 1 kHz tone gated ON for 30 ms every 0.5 s -> clear onsets at 120 BPM
    ff.run(["ffmpeg", "-y", "-f", "lavfi",
            "-i", "aevalsrc=0.6*sin(2*PI*1000*t)*lt(mod(t\\,0.5)\\,0.03):d=8",
            "-ac", "1", click], desc="click track")

    beat_cfg = {
        "_active_profile": PROFILE, "render": {"encoder": "libx264", "fps": 30},
        "audio_events": {"sample_rate": 16000},
        "edit": {"audio": {"music_dir": str(music_dir), "beat_sync": True,
                           "music_volume": 0.35, "beat_min_beats_per_segment": 2}},
    }
    grid = detect_beats(click, beat_cfg)
    assert grid and grid.beats, "click track produced no beats"
    import numpy as np
    period = float(np.median(np.diff(grid.beats)))

    trims = _beat_cut_plan(segments, beat_cfg)
    assert trims and len(trims) == len(segments), trims
    for seg, t in zip(segments, trims):
        beats = t / period
        assert abs(beats - round(beats)) < 0.15, f"cut {t:.2f}s not on beat ({beats})"
        assert t <= ff.duration(seg) + 1e-3, "trim exceeds segment length"
    print(f"  ✓ beat-sync: {grid.bpm:.0f} BPM, cut points on beat "
          f"(period {period:.2f}s) -> trims {['%.2f' % t for t in trims]}")

    reel = build_compilation(segments, str(tmp / "reel_beat.mp4"), beat_cfg, BRAND)
    _assert_av(reel, "beat-synced reel", PROFILE["width"], PROFILE["height"])


if __name__ == "__main__":
    raise SystemExit(main())
