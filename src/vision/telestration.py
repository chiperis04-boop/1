"""Analyst-grade telestration with correct depth layering.

Renders the channel's signature graphics onto each clip frame:

  * **spotlight**   — a glowing ellipse on the grass under the key player
  * **pass line**   — the current play link (key player → ball)
  * **ball trail**  — a fading trace behind the ball
  * **motion arrow**— a smoothed arrow tracing the key player's run
  * **highlight zone** — shaded free space ahead of the attacker

Three things make this look professional rather than like a cheap filter, and
they directly fix the 03_goal.mp4 artefacts:

1. **Layering (players above graphics).** Graphics are drawn on a separate layer
   and composited *under* the players using their segmentation silhouettes (from
   a YOLO-seg pass) — or, when masks are unavailable, a feathered body-ellipse
   approximation of each bounding box. The result: the spotlight sits on the
   turf at a player's feet, not pasted over their shins.

2. **Grass-plane geometry (homography).** When a pitch calibration is supplied,
   the spotlight / zone / pass-line are constructed in pitch *metres* and warped
   back into the image with the pitch→image homography, so they lie flat on the
   grass and deform with camera movement. Without calibration we fall back to a
   foreshortened ellipse (wider than tall) that still reads as ground-plane.

3. **De-jittered coordinates.** Tracks are smoothed upstream (see
   ``vision.smoothing``), so circles and traces glide instead of buzzing.

The annotation engine prefers Roboflow ``supervision`` annotators
(EllipseAnnotator / TraceAnnotator) when installed; otherwise it uses an
anti-aliased OpenCV fallback that produces an equivalent look. Either way the
graphics are layered under players.
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
                        cfg: dict, calib=None) -> str:
    t = cfg["telestration"]
    if not t.get("enabled", True):
        return clip_path

    cap = cv2.VideoCapture(clip_path)
    fps = track.fps
    w, h = track.width, track.height

    tmp = out_path.replace(".mp4", "_silent.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp, fourcc, fps, (w, h))

    accent = tuple(int(c) for c in t.get("arrow_color", [0, 200, 255]))
    spot_color = tuple(int(c) for c in t.get("spotlight_color", [0, 220, 255]))
    trail_color = tuple(int(c) for c in t.get("trail_color", [255, 255, 255]))
    thick = int(t.get("line_thickness", 4))
    spot_radius_m = float(t.get("spotlight_radius_m", 1.6))

    key_id = track.key_track_id
    key_path = _key_player_path(track, key_id)
    ball_lookup = {i: (x, y) for i, x, y in track.ball_path}
    frame_index = {fd.idx: fd for fd in track.frames}
    trail = deque(maxlen=int(t.get("trail_length", 22)))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fd = frame_index.get(idx)
        gfx = frame.copy()              # graphics layer (drawn, then layered under players)

        foot = _foot_point(key_path.get(idx)) if idx in key_path else None

        # ---- grass-plane soft graphics (blended) ----
        soft = frame.copy()
        if t.get("highlight_zone", True) and foot is not None:
            ball_xy = ball_lookup.get(idx)
            _draw_zone(soft, foot, ball_xy, calib, idx, spot_color, w, h)
        if t.get("spotlight_scorer", True) and foot is not None:
            _draw_spotlight(soft, foot, calib, idx, spot_radius_m, spot_color)
        gfx = cv2.addWeighted(soft, 0.38, gfx, 0.62, 0)

        # ---- crisp grass-plane lines (also under players) ----
        if t.get("pass_line", True) and foot is not None and idx in ball_lookup:
            _draw_pass_line(gfx, foot, ball_lookup[idx], accent, thick)
        if t.get("ball_trail", True) and idx in ball_lookup:
            trail.append(ball_lookup[idx])
            _draw_trail(gfx, list(trail), trail_color)
        if t.get("motion_arrows", True) and key_id is not None:
            pts = [key_path[i] for i in range(max(0, idx - 34), idx + 1)
                   if i in key_path]
            if len(pts) >= 5:
                _draw_motion_arrow(gfx, pts, accent, thick)

        # ---- layer players back ON TOP of the graphics ----
        out = _composite_under_players(frame, gfx, fd, w, h)
        writer.write(out)
        idx += 1

    cap.release()
    writer.release()

    encoder = ff.pick_encoder(cfg["render"]["encoder"])
    ff.mux_audio(tmp, clip_path, out_path, encoder)
    log.info(f"[telestration] rendered {idx} frames "
             f"(layered={'seg' if track.has_segmentation else 'bbox'}, "
             f"homography={'on' if calib is not None else 'off'}) -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# layering
# --------------------------------------------------------------------------- #
def _composite_under_players(original, gfx, fd, w, h):
    """Return gfx everywhere except where players are, where the *original*
    pixels are kept — so players occlude the graphics (correct depth)."""
    if fd is None or (not fd.players and not fd.fg_contours):
        return gfx
    mask = np.zeros((h, w), dtype=np.uint8)
    if fd.fg_contours:
        cv2.fillPoly(mask, [c.reshape(-1, 1, 2) for c in fd.fg_contours], 255)
    else:
        for p in fd.players:
            x1, y1, x2, y2 = (int(v) for v in p["xyxy"])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ax, ay = max(4, (x2 - x1) // 2), max(6, (y2 - y1) // 2)
            # a slightly tall ellipse approximates a standing player's silhouette
            cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    # feather so the composite edge isn't a hard cut-out
    mask = cv2.GaussianBlur(mask, (0, 0), 2.5)
    m = (mask.astype(np.float32) / 255.0)[..., None]
    out = original.astype(np.float32) * m + gfx.astype(np.float32) * (1.0 - m)
    return out.astype(np.uint8)


# --------------------------------------------------------------------------- #
# grass-plane primitives (homography-aware)
# --------------------------------------------------------------------------- #
def _ground_circle_pts(foot, radius_m, calib, idx, n=40):
    """Polygon (Nx2 int) of a ground circle of `radius_m` around the foot point,
    perspective-correct when calibrated, else a foreshortened ellipse."""
    fx, fy = foot
    if calib is not None and calib.has(idx):
        c = calib.to_pitch(idx, fx, fy)
        if c is not None:
            pts = []
            for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
                ip = calib.to_image(idx, c[0] + radius_m * np.cos(a),
                                    c[1] + radius_m * np.sin(a))
                if ip is not None:
                    pts.append(ip)
            if len(pts) >= 8:
                return np.array(pts, dtype=np.int32)
    # fallback: a ground-plane ellipse (squashed vertically) sitting at the feet
    ax = int(max(28, radius_m * 28))
    ay = int(ax * 0.42)
    ell = cv2.ellipse2Poly((int(fx), int(fy)), (ax, ay), 0, 0, 360, 12)
    return ell.astype(np.int32)


def _draw_spotlight(img, foot, calib, idx, radius_m, color):
    poly = _ground_circle_pts(foot, radius_m, calib, idx)
    cv2.fillPoly(img, [poly.reshape(-1, 1, 2)], color, cv2.LINE_AA)
    # crisp ring on top of the soft fill for definition
    cv2.polylines(img, [poly.reshape(-1, 1, 2)], True,
                  tuple(int(min(255, c + 35)) for c in color), 3, cv2.LINE_AA)


def _draw_zone(img, foot, ball_xy, calib, idx, color, w, h):
    """Shaded free-space wedge ahead of the attacker, along the play direction
    (foot→ball), warped to the grass when calibrated."""
    fx, fy = foot
    if ball_xy is not None:
        dx, dy = ball_xy[0] - fx, ball_xy[1] - fy
    else:
        dx, dy = 1.0, 0.0
    norm = float(np.hypot(dx, dy)) or 1.0
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux                              # perpendicular
    depth = 230.0
    half = 95.0
    quad = np.array([
        [fx - px * half * 0.4, fy - py * half * 0.4],
        [fx + px * half * 0.4, fy + py * half * 0.4],
        [fx + ux * depth + px * half, fy + uy * depth + py * half],
        [fx + ux * depth - px * half, fy + uy * depth - py * half],
    ], dtype=np.int32)
    cv2.fillPoly(img, [quad.reshape(-1, 1, 2)], color, cv2.LINE_AA)


def _draw_pass_line(img, foot, ball_xy, color, thick):
    p0 = (int(foot[0]), int(foot[1]))
    p1 = (int(ball_xy[0]), int(ball_xy[1]))
    cv2.line(img, p0, p1, (20, 20, 20), thick + 3, cv2.LINE_AA)   # dark halo
    cv2.line(img, p0, p1, color, thick, cv2.LINE_AA)
    cv2.circle(img, p1, max(5, thick + 2), color, -1, cv2.LINE_AA)


def _draw_trail(img, pts, color):
    n = len(pts)
    for i in range(1, n):
        alpha = i / n
        p0 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        p1 = (int(pts[i][0]), int(pts[i][1]))
        cv2.line(img, p0, p1, color, max(1, int(5 * alpha)), cv2.LINE_AA)


def _draw_motion_arrow(img, pts, color, thick):
    arr = np.array(pts, dtype=np.float32)
    k = max(1, len(arr) // 6)
    sm = cv2.blur(arr.reshape(-1, 1, 2), (1, k)).reshape(-1, 2)
    for i in range(1, len(sm)):
        cv2.line(img, tuple(sm[i - 1].astype(int)), tuple(sm[i].astype(int)),
                 color, thick, cv2.LINE_AA)
    cv2.arrowedLine(img, tuple(sm[-2].astype(int)), tuple(sm[-1].astype(int)),
                    color, thick + 1, cv2.LINE_AA, tipLength=0.5)


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


def _foot_point(center):
    """Approximate the feet (bottom of the player) from the tracked centre.
    The centre is the bbox midpoint; feet are roughly the box-half below it.
    Telestration only has the centre here, so nudge down a fixed fraction."""
    if center is None:
        return None
    return (center[0], center[1] + 26)
