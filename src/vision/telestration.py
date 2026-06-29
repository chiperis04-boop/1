"""Telestration renderer.

Takes a clip + its TrackResult and renders an annotated clip with the channel's
analyst-style graphics:

  * spotlight   — soft glowing ellipse under the key player that follows them
  * motion_arrow— curved arrow tracing the key player's path so far
  * ball_trail  — fading polyline behind the ball
  * highlight_zone — shaded area ahead of the key player (free space / danger)

All drawing is done with OpenCV onto each frame, then re-encoded with FFmpeg
(preserving the clip's audio). Colors come from config (BGR).
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from ..utils.io import get_logger
from ..edit import ff
from .detect_track import TrackResult

log = get_logger()


def render_telestration(clip_path: str, track: TrackResult, out_path: str,
                        cfg: dict) -> str:
    t = cfg["telestration"]
    if not t.get("enabled", True):
        return clip_path

    cap = cv2.VideoCapture(clip_path)
    fps = track.fps
    w, h = track.width, track.height

    tmp = out_path.replace(".mp4", "_silent.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp, fourcc, fps, (w, h))

    arrow_color = tuple(t["arrow_color"])
    spot_color = tuple(t["spotlight_color"])
    trail_color = tuple(t["trail_color"])
    thick = t["line_thickness"]

    # precompute key player path
    key_id = track.key_track_id
    key_path = _key_player_path(track, key_id)
    ball_lookup = {i: (x, y) for i, x, y in track.ball_path}
    trail = deque(maxlen=18)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        overlay = frame.copy()

        # --- highlight zone (drawn first, underneath) ---
        if t["highlight_zone"] and idx in key_path:
            _draw_zone(overlay, key_path[idx], w, h, spot_color)

        # --- spotlight under key player ---
        if t["spotlight_scorer"] and idx in key_path:
            _draw_spotlight(overlay, key_path[idx], spot_color)

        # blend the soft overlays
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        # --- ball trail (crisp) ---
        if t["ball_trail"] and idx in ball_lookup:
            trail.append(ball_lookup[idx])
            _draw_trail(frame, list(trail), trail_color)

        # --- motion arrow along key player path (crisp) ---
        if t["motion_arrows"] and key_id is not None:
            pts = [key_path[i] for i in range(max(0, idx - 30), idx + 1) if i in key_path]
            if len(pts) >= 5:
                _draw_motion_arrow(frame, pts, arrow_color, thick)

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()

    # remux original audio back in
    encoder = ff.pick_encoder(cfg["render"]["encoder"])
    ff.mux_audio(tmp, clip_path, out_path, encoder)
    log.info(f"[telestration] rendered -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
def _key_player_path(track: TrackResult, key_id) -> dict[int, tuple[float, float]]:
    path: dict[int, tuple[float, float]] = {}
    if key_id is None:
        return path
    for fd in track.frames:
        for p in fd.players:
            if p["id"] == key_id:
                path[fd.idx] = (p["center"][0], p["center"][1])
    return path


def _draw_spotlight(img, center, color):
    cx, cy = int(center[0]), int(center[1])
    axes = (46, 18)
    cv2.ellipse(img, (cx, cy + 30), axes, 0, 0, 360, color, -1)


def _draw_zone(img, center, w, h, color):
    cx, cy = int(center[0]), int(center[1])
    pts = np.array([[cx, cy], [cx + 220, cy - 70], [cx + 220, cy + 70]], np.int32)
    cv2.fillPoly(img, [pts], color)


def _draw_trail(img, pts, color):
    n = len(pts)
    for i in range(1, n):
        alpha = i / n
        p0 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        p1 = (int(pts[i][0]), int(pts[i][1]))
        cv2.line(img, p0, p1, color, max(1, int(4 * alpha)), cv2.LINE_AA)


def _draw_motion_arrow(img, pts, color, thick):
    # smooth the path then draw a tapered arrow
    arr = np.array(pts, dtype=np.float32)
    k = max(1, len(arr) // 6)
    sm = cv2.blur(arr.reshape(-1, 1, 2), (1, k)).reshape(-1, 2)
    for i in range(1, len(sm)):
        cv2.line(img, tuple(sm[i - 1].astype(int)), tuple(sm[i].astype(int)),
                 color, thick, cv2.LINE_AA)
    cv2.arrowedLine(img, tuple(sm[-2].astype(int)), tuple(sm[-1].astype(int)),
                    color, thick + 1, cv2.LINE_AA, tipLength=0.5)
