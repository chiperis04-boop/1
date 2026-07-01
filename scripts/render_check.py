"""CPU render-check — reproducible visual proof of the overlay/camera/render
chain WITHOUT a GPU.

Drives the REAL Cameraman -> Composer -> branding chain on a synthetic clip with
known player/ball positions, so you can extract frames and confirm Block A (one
clean overlay set: single event hook, reaction-gated stat card, compact edge-safe
lower-third), Block C (camera leads the ball, no letterbox) and the grounded
halos / neon ball trail render correctly. GPU vision (YOLO detect/track/seg,
vLLM Director) is intentionally NOT exercised here — that is "verify on GPU".

Usage:
    python scripts/render_check.py [out_dir]
Then read out_dir/chk_*.jpg as images and compare with the reference look.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.editplan import EditPlan, SlowmoBeat        # noqa: E402
from src.branding.overlays import apply_branding            # noqa: E402
from src.detect.types import Moment                         # noqa: E402
from src.edit import ff                                     # noqa: E402
from src.perception.shots import Shot                       # noqa: E402
from src.render.composer import Composer                    # noqa: E402
from src.tracking.cameraman import Cameraman, FrameTrack    # noqa: E402
from src.utils.io import load_branding, load_config         # noqa: E402

W, H, N, FPS = 1280, 720, 90, 30


class _Poss:
    def run_at(self, idx):
        return None


class _Jer:
    number_of = {1: 10}


class _Analytics:
    """Minimal analytics stand-in (team colours + hero number + possession)."""
    hero_id = 1
    hero_number = 10
    hero_source = "geometric"
    jerseys = _Jer()
    possession = _Poss()

    def color_for_track(self, tid, default=(0, 220, 255)):
        return (60, 60, 255) if tid == 1 else (40, 230, 255)

    def possession_share_pct(self):
        return {0: 60, 1: 40}


def _box(cx, cy, w=46, h=120):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def main(out_dir: str = "output/_render_check") -> int:
    ff.ensure_tools()
    out = Path(out_dir)
    frames_dir = out / "src_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i in range(N):
        t = i / (N - 1)
        img = np.full((H, W, 3), (40, 120, 40), dtype=np.uint8)
        cv2.line(img, (W // 2, 0), (W // 2, H), (60, 150, 60), 2)
        hx, hy = int(200 + t * 760), int(380 + 40 * np.sin(t * 3))
        rx, ry = int(700 - t * 200), 300
        bx, by = hx + 40, hy + 60
        cv2.rectangle(img, (hx - 23, hy - 60), (hx + 23, hy + 60), (30, 30, 200), -1)
        cv2.rectangle(img, (rx - 23, ry - 60), (rx + 23, ry + 60), (30, 210, 210), -1)
        cv2.circle(img, (bx, by), 9, (255, 255, 255), -1)
        cv2.imwrite(str(frames_dir / f"f{i:04d}.png"), img)
        ft = FrameTrack(idx=i, players=[
            {"id": 1, "cls": 0, "xyxy": _box(hx, hy), "center": [float(hx), float(hy)]},
            {"id": 2, "cls": 0, "xyxy": _box(rx, ry), "center": [float(rx), float(ry)]},
        ])
        ft.ball = {"xyxy": [bx - 9, by - 9, bx + 9, by + 9],
                   "center": [float(bx), float(by)]}
        frames.append(ft)

    src = str(out / "src.mp4")
    ff.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", str(frames_dir / "f%04d.png"),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            "-t", str(N / FPS), src], desc="synthetic source")

    cfg = load_config("config/config.yaml")
    brand = load_branding("config/branding.yaml")
    cfg["_active_profile"] = cfg["render"]["profiles"]["tiktok"]
    cfg["render"]["encoder"] = ff.pick_encoder(cfg["render"]["encoder"])
    cfg["telestration"]["occlusion"] = False

    cam = Cameraman(cfg)
    composer = Composer(cfg, brand)
    analytics = _Analytics()
    meta = {"w": W, "h": H, "fps": float(FPS)}
    shots = [Shot(0, 0.0, N / FPS, 0, N)]
    plan = cam.build_plan(frames, meta, hero_id=1, shots=shots)

    editplan = EditPlan(event="goal", hook_text="WHAT A GOAL!", lower_third="",
                        slowmo_beats=[SlowmoBeat(1.5, 2.3, 0.4)])
    manifest = editplan.to_manifest()
    world, screen = composer.make_annotators(plan, manifest, analytics)
    reframed = str(out / "reframed.mp4")
    cam.render(src, plan, reframed, annotate_world=world, annotate_screen=screen,
               intermediate=True)
    final = str(out / "final.mp4")
    composer.finish(reframed, final, manifest=manifest,
                    stats={"possession_pct": [60, 40], "top_speed_kmh": 31},
                    beats=editplan.slowmo_beats, reaction=[(0.0, 0.8)])
    moment = Moment(t=1.5, start=0.0, end=N / FPS, confidence=0.9, kind="goal",
                    minute=67)
    branded = str(out / "branded.mp4")
    apply_branding(final, moment, {}, branded, cfg, brand,
                   with_intro_outro=False, composer_typography=True)

    for tsec in (0.3, 1.0, 2.0, 2.6):
        ff.run(["ffmpeg", "-y", "-ss", str(tsec), "-i", branded, "-frames:v", "1",
                str(out / f"chk_{int(tsec * 10):02d}.jpg")], desc=f"frame {tsec}")
    print(f"render-check done -> {branded} ({ff.duration(branded):.1f}s); "
          f"frames chk_*.jpg in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "output/_render_check"))
