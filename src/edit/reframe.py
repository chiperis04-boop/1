"""Action-aware vertical reframing.

Converts a wide broadcast clip to 9:16 (or 1:1) by computing a per-frame crop
window that follows the action — preferring the ball, falling back to the key
player, then the centroid of all players. The crop path is heavily smoothed so
the virtual camera glides instead of jittering (this is what makes auto-reframe
look professional rather than robotic).

Uses the already-computed TrackResult so no second detection pass is needed.
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

    # crop window height = full height; width set by target aspect
    crop_h = src_h
    crop_w = int(round(crop_h * aspect_w / aspect_h))
    crop_w = min(crop_w, src_w)

    # focus x per frame
    focus_x = _focus_track(track, src_w)
    if r["mode"] == "center" or focus_x is None:
        focus_x = np.full(len(track.frames), src_w / 2.0)

    # smooth the camera path (exponential + clamp velocity)
    cx = _smooth(focus_x, r["smoothing"])
    half = crop_w / 2
    cx = np.clip(cx, half, src_w - half)

    cap = cv2.VideoCapture(clip_path)
    fps = track.fps
    out_w = cfg_profile_w(cfg)
    out_h = cfg_profile_h(cfg)
    encoder = ff.pick_encoder(cfg["render"]["encoder"])
    # crop (~607px wide for 9:16 from 1080p) is upscaled to the profile width,
    # so use a high-quality interpolation instead of INTER_AREA (down-only).
    sink = ff.RawFrameSink(out_path, out_w, out_h, fps, encoder,
                           audio_src=clip_path)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        c = cx[min(idx, len(cx) - 1)]
        x0 = int(round(c - half))
        x0 = max(0, min(x0, src_w - crop_w))
        crop = frame[0:crop_h, x0:x0 + crop_w]
        crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        sink.write(crop)
        idx += 1

    cap.release()
    sink.close()
    log.info(f"[reframe] {r['target_aspect']} action-track -> {out_path}")
    return out_path


# --------------------------------------------------------------------------- #
def _focus_track(track: TrackResult, src_w: float):
    ball = {i: x for i, x, _ in track.ball_path}
    xs = []
    for fd in track.frames:
        if fd.idx in ball:
            xs.append(ball[fd.idx])
        elif fd.players:
            xs.append(float(np.mean([p["center"][0] for p in fd.players])))
        else:
            xs.append(xs[-1] if xs else src_w / 2.0)
    return np.array(xs, dtype=np.float32) if xs else None


def _smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    """Zero-phase camera smoothing (forward + backward EMA).

    A plain causal EMA lags behind the action, so the crop trails the ball on
    fast plays. Because the whole clip is known up-front we can filter forwards
    AND backwards and average the two — this removes the phase lag, so the
    virtual camera anticipates motion instead of chasing it.
    """
    fwd = _ema_pass(x, alpha)
    bwd = _ema_pass(x[::-1], alpha)[::-1]
    return 0.5 * (fwd + bwd)


def _ema_pass(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * x[i]
    return out


def cfg_profile_w(cfg: dict) -> int:
    return cfg["_active_profile"]["width"]


def cfg_profile_h(cfg: dict) -> int:
    return cfg["_active_profile"]["height"]


def _letterbox(clip_path: str, out_path: str, cfg: dict) -> str:
    w, h = cfg_profile_w(cfg), cfg_profile_h(cfg)
    encoder = ff.pick_encoder(cfg["render"]["encoder"])
    return ff.standardize(clip_path, out_path, w, h, cfg["render"]["fps"], encoder)
