"""Action-aware vertical reframing (cinematic virtual camera).

Converts a wide broadcast clip to 9:16 (or 1:1) by computing a per-frame crop
window that follows the *play* — not the crowd, not the referee on the touchline.

The previous version followed the centroid of *all* detected players, so the
crop lurched toward whoever was on screen (referee, coach, far-side players) and
jittered frame-to-frame. This rewrite fixes the camerawork seen in 03_goal.mp4:

  * **Object prioritisation** — the camera targets a weighted blend of the
    **ball** and the **active player** (the tracked protagonist) only. Every
    other player, and anything off the ball, is ignored, so the framing stays on
    the actual attack.
  * **Dead-zone (comfort zone)** — while the target sits within a central band
    the camera holds perfectly still; it only begins to pan once the action
    nears the edge of frame. This removes the constant micro-panning.
  * **Cinematic interpolation** — motion uses a critically-damped *SmoothDamp*
    (no overshoot) with a velocity cap, so pans accelerate and settle smoothly
    instead of snapping. The focus signal is also pre-smoothed (Savitzky-Golay).

The crop path is computed from the already-smoothed `TrackResult`, so no second
detection pass is needed.
"""
from __future__ import annotations

import numpy as np

from ..utils.io import get_logger
from ..vision.detect_track import TrackResult
from . import ff

log = get_logger()


def reframe_clip(clip_path: str, track: TrackResult, out_path: str, cfg: dict) -> str:
    import cv2  # lazy: only the action-track path needs OpenCV
    r = cfg["edit"]["reframe"]
    if r["mode"] == "letterbox":
        return _letterbox(clip_path, out_path, cfg)

    aspect_w, aspect_h = (int(x) for x in r["target_aspect"].split(":"))
    src_w, src_h = track.width, track.height

    crop_h = src_h
    crop_w = int(round(crop_h * aspect_w / aspect_h))
    crop_w = min(crop_w, src_w)

    if r["mode"] == "center":
        focus_x = np.full(max(1, len(track.frames)), src_w / 2.0, np.float32)
    else:
        focus_x = _focus_track(track, src_w)

    cx = plan_camera(focus_x, src_w=src_w, crop_w=crop_w, fps=track.fps or 30.0,
                     cfg=r)

    cap = cv2.VideoCapture(clip_path)
    fps = track.fps
    tmp = out_path.replace(".mp4", "_silent.mp4")
    out_w = cfg_profile_w(cfg)
    out_h = cfg_profile_h(cfg)
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

    half = crop_w / 2
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        c = float(cx[min(idx, len(cx) - 1)])
        x0 = int(round(np.clip(c - half, 0, src_w - crop_w)))
        crop = frame[0:crop_h, x0:x0 + crop_w]
        crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(crop)
        idx += 1

    cap.release()
    writer.release()
    encoder = ff.pick_encoder(cfg["render"]["encoder"])
    ff.mux_audio(tmp, clip_path, out_path, encoder)
    log.info(f"[reframe] {r['target_aspect']} action-track (ball+player, "
             f"dead-zone) -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
# focus signal: ball + active player only
# --------------------------------------------------------------------------- #
def _focus_track(track: TrackResult, src_w: float) -> np.ndarray:
    """Per-frame horizontal focus target from the ball and the active player.

    Weighting keeps the *ball carrier* in frame: mostly the ball, biased toward
    the protagonist. Referees / coaches / off-ball players are never considered.
    Missing frames hold the previous value (no snap to screen centre).
    """
    ball = {i: x for i, x, _ in track.ball_path}
    key_id = track.key_track_id
    key_x: dict[int, float] = {}
    if key_id is not None:
        for fd in track.frames:
            for p in fd.players:
                if p["id"] == key_id:
                    key_x[fd.idx] = p["center"][0]

    xs: list[float] = []
    last = src_w / 2.0
    for fd in track.frames:
        b = ball.get(fd.idx)
        k = key_x.get(fd.idx)
        if b is not None and k is not None:
            val = 0.65 * b + 0.35 * k
        elif b is not None:
            val = b
        elif k is not None:
            val = k
        else:
            val = last
        last = val
        xs.append(val)
    if not xs:
        return np.full(1, src_w / 2.0, np.float32)
    return np.asarray(xs, dtype=np.float32)


# --------------------------------------------------------------------------- #
# camera planner: pre-smooth -> dead-zone -> SmoothDamp (pure / testable)
# --------------------------------------------------------------------------- #
def plan_camera(focus_x: np.ndarray, src_w: float, crop_w: float, fps: float,
                cfg: dict) -> np.ndarray:
    """Turn a noisy per-frame focus signal into a smooth, clamped camera path."""
    n = len(focus_x)
    if n == 0:
        return np.array([src_w / 2.0])
    half = crop_w / 2.0

    # 1) pre-smooth the raw focus to remove residual detection jitter
    from ..vision.smoothing import smooth_series
    pre_win = int(cfg.get("presmooth_window", 11))
    focus = smooth_series(np.asarray(focus_x, dtype=np.float64), "savgol",
                          window=pre_win, poly=2)

    # 2) parameters
    deadzone = float(cfg.get("deadzone", 0.12)) * crop_w
    smooth_time = float(cfg.get("smooth_time", 0.55))      # seconds to settle
    max_speed = float(cfg.get("max_pan_frac", 0.7)) * src_w  # px / second
    dt = 1.0 / max(1.0, fps)

    # 3) integrate the critically-damped follower with a central dead-zone
    cx = float(np.clip(focus[0], half, src_w - half))
    vel = 0.0
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        target = float(focus[i])
        err = target - cx
        if abs(err) <= deadzone:
            eff_target = cx                       # inside comfort zone: hold
        else:
            eff_target = target - np.sign(err) * deadzone
        eff_target = float(np.clip(eff_target, half, src_w - half))
        cx, vel = _smooth_damp(cx, eff_target, vel, smooth_time, max_speed, dt)
        cx = float(np.clip(cx, half, src_w - half))
        out[i] = cx
    return out


def _smooth_damp(current: float, target: float, vel: float, smooth_time: float,
                 max_speed: float, dt: float):
    """Critically-damped follower (Game-Programming-Gems / Unity SmoothDamp).

    Produces overshoot-free, velocity-limited motion — the key to a cinematic
    pan that eases in and settles without snapping or oscillating.
    """
    smooth_time = max(1e-4, smooth_time)
    omega = 2.0 / smooth_time
    x = omega * dt
    exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = current - target
    max_change = max_speed * smooth_time
    change = max(-max_change, min(change, max_change))
    tgt = current - change
    temp = (vel + omega * change) * dt
    vel = (vel - omega * temp) * exp
    out = tgt + (change + temp) * exp
    # prevent overshooting the target
    if (target - current > 0.0) == (out > target):
        out = target
        vel = (out - target) / dt
    return out, vel


# --------------------------------------------------------------------------- #
def cfg_profile_w(cfg: dict) -> int:
    return cfg["_active_profile"]["width"]


def cfg_profile_h(cfg: dict) -> int:
    return cfg["_active_profile"]["height"]


def _letterbox(clip_path: str, out_path: str, cfg: dict) -> str:
    w, h = cfg_profile_w(cfg), cfg_profile_h(cfg)
    encoder = ff.pick_encoder(cfg["render"]["encoder"])
    return ff.standardize(clip_path, out_path, w, h, cfg["render"]["fps"], encoder)
