"""CPU tests for the refactored CV/HUD layers (no GPU, no YOLO models).

Covers the pieces the GPU end-to-end run can't exercise on a CPU box:
  * tracker-coordinate smoothing actually reduces jitter
  * the virtual-camera planner holds in the dead-zone and never overshoots
  * telestration renders a valid clip AND layers graphics *under* players
  * the Pillow HUD engine composites time-bounded overlays into a valid clip
  * the scoreboard minute parser reads the clock, not the scoreline

Run:  python tests/test_telestration.py   (needs ffmpeg + opencv + pillow)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.edit import ff                                    # noqa: E402
from src.edit import overlay_render as ovr                 # noqa: E402
from src.edit.reframe import plan_camera                   # noqa: E402
from src.vision.smoothing import smooth_series, smooth_track  # noqa: E402
from src.vision.detect_track import TrackResult, FrameDets    # noqa: E402


def test_smoothing_reduces_jitter():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 150)
    clean = 40 * t
    noisy = clean + rng.normal(0, 6, t.shape)
    sg = smooth_series(noisy, "savgol", window=11, poly=2)
    raw_rmse = float(np.sqrt(np.mean((noisy - clean) ** 2)))
    sg_rmse = float(np.sqrt(np.mean((sg - clean) ** 2)))
    assert sg_rmse < raw_rmse * 0.7, (sg_rmse, raw_rmse)
    print(f"  ✓ smoothing: jitter RMSE {raw_rmse:.2f} -> {sg_rmse:.2f}")


def test_camera_deadzone_and_no_overshoot():
    cfg = {"deadzone": 0.12, "smooth_time": 0.55, "max_pan_frac": 0.7,
           "presmooth_window": 11}
    src_w, crop_w, fps = 1920.0, 1080.0, 30.0
    rng = np.random.default_rng(1)
    # jitter around centre -> camera must stay essentially still
    foc = np.full(90, 960.0) + rng.normal(0, 8, 90)
    cx = plan_camera(foc, src_w, crop_w, fps, cfg)
    assert cx.max() - cx.min() < 5.0, cx.ptp()
    # step pan -> no overshoot past the target, velocity capped
    foc = np.concatenate([np.full(30, 760.0), np.full(70, 1160.0)])
    cx = plan_camera(foc, src_w, crop_w, fps, cfg)
    assert cx.max() <= 1160.0 + 1.0, cx.max()
    vel = np.abs(np.diff(cx)) * fps
    assert vel.max() <= 0.7 * src_w + 1.0, vel.max()
    print(f"  ✓ camera: dead-zone hold (ptp<5px), no overshoot, "
          f"vmax {vel.max():.0f}px/s")


def _synthetic_play(tmp: Path):
    import cv2
    W, H, N = 1280, 720, 50
    src = str(tmp / "play.mp4")
    vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    frames, ball_path = [], []
    for i in range(N):
        img = np.full((H, W, 3), (40, 120, 40), np.uint8)
        fd = FrameDets(idx=i)
        px, py = 300 + i * 9, 380
        cv2.rectangle(img, (px - 20, py - 60), (px + 20, py + 60),
                      (235, 235, 235), -1)
        fd.players.append({"id": 1, "xyxy": [px - 20, py - 60, px + 20, py + 60],
                           "center": [px, py]})
        bx, by = px + 55, py - 8
        cv2.circle(img, (bx, by), 8, (255, 255, 255), -1)
        fd.ball = {"xyxy": [bx - 8, by - 8, bx + 8, by + 8], "center": [bx, by]}
        ball_path.append((i, float(bx), float(by)))
        frames.append(fd)
        vw.write(img)
    vw.release()
    tr = TrackResult(width=W, height=H, fps=30.0, frames=frames,
                     ball_path=ball_path, key_track_id=1)
    return src, tr


def test_telestration_layers_under_players():
    import cv2
    from src.vision import telestration as tel
    tmp = Path(tempfile.mkdtemp(prefix="fhs_tel_"))
    src, tr = _synthetic_play(tmp)
    smooth_track(tr, "savgol", 9, 2)
    cfg = {"telestration": {"enabled": True, "spotlight_scorer": True,
                            "motion_arrows": True, "ball_trail": True,
                            "highlight_zone": True, "pass_line": True,
                            "arrow_color": [0, 200, 255],
                            "spotlight_color": [0, 220, 255],
                            "trail_color": [255, 255, 255], "line_thickness": 4},
           "render": {"encoder": "libx264"}}
    out = str(tmp / "tele.mp4")
    tel.render_telestration(src, tr, out, cfg, calib=None)
    data = ff.probe(out)
    v = [s for s in data["streams"] if s["codec_type"] == "video"]
    assert len(v) == 1 and (v[0]["width"], v[0]["height"]) == (1280, 720)

    # verify a player's interior pixels are (close to) the original kit colour,
    # i.e. graphics did NOT paint over the player body.
    frame = str(tmp / "f.png")
    ff.run(["ffmpeg", "-y", "-ss", "1.0", "-i", out, "-frames:v", "1", frame],
           desc="grab")
    img = cv2.imread(frame)
    fd = tr.frames[30]
    p = fd.players[0]
    cx, cy = int(p["center"][0]), int(p["center"][1])
    patch = img[cy - 10:cy + 10, cx - 5:cx + 5].reshape(-1, 3).mean(axis=0)
    # white kit -> all channels high; a yellow/colour graphic would drop a channel
    assert patch.min() > 170, f"graphics bled onto the player body: {patch}"
    print(f"  ✓ telestration: valid clip + players above graphics "
          f"(kit pixel {patch.astype(int).tolist()})")


def test_overlay_engine_composites():
    tmp = Path(tempfile.mkdtemp(prefix="fhs_ovr_"))
    src = str(tmp / "s.mp4")
    ff.run(["ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=c=0x2e7d32:size=1080x1920:rate=30:duration=3",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", src], desc="src")
    z = ovr.SafeZones()
    els = [
        ovr.hook_overlay("Spot the defender's mistake 👀", 1080, 1920, "",
                         str(tmp / "h.png"), z, start=0.0, end=1.6),
        ovr.lower_third_overlay("GOAL   67'", 1080, 1920, "", str(tmp / "l.png"),
                                z, start=0.3),
        ovr.stats_card_overlay(["Shot: 24 m", "Beaten: 2"], 1080, 1920, "",
                               str(tmp / "st.png"), z, start=0.5),
        ovr.caption_overlay("GOLAZO top corner", 1080, 1920, "",
                            str(tmp / "c.png"), z, 0.2, 1.8),
    ]
    out = str(tmp / "out.mp4")
    ovr.composite(src, els, out, "libx264")
    data = ff.probe(out)
    v = [s for s in data["streams"] if s["codec_type"] == "video"]
    a = [s for s in data["streams"] if s["codec_type"] == "audio"]
    assert len(v) == 1 and len(a) == 1
    assert (v[0]["width"], v[0]["height"]) == (1080, 1920)
    print("  ✓ overlay engine: 4 timed HUD overlays composited (1 video+1 audio)")


def test_minute_reads_clock_not_score():
    from src.detect.scoreboard_ocr import _parse_minute, _SCORE_RE
    cases = {"2 - 1 67'": 67, "ARS 2 - 1 CHE 67'": 67, "0-0 45+2'": 45,
             "FT 3 - 2 90'": 90}
    for txt, want in cases.items():
        got = _parse_minute(txt, _SCORE_RE.search(txt))
        assert got == want, f"{txt!r} -> {got} (want {want})"
    print("  ✓ minute sync: clock parsed, not the scoreline")


def main() -> int:
    ff.ensure_tools()
    print("telestration / camera / HUD tests")
    test_smoothing_reduces_jitter()
    test_camera_deadzone_and_no_overshoot()
    test_minute_reads_clock_not_score()
    test_overlay_engine_composites()
    test_telestration_layers_under_players()
    print("\nTELESTRATION CHECKS PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
