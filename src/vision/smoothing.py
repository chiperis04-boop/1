"""Temporal smoothing for tracker coordinates.

Raw per-frame detections from YOLO + ByteTrack/BoT-SORT jitter by a few pixels
every frame. Drawn directly, that jitter makes player circles, traces and the
virtual camera shake — the exact artefact seen in 03_goal.mp4. We therefore
smooth every coordinate *path* before it is used for telestration or reframing.

Three filters are provided, in order of preference:

  * **Savitzky-Golay** (`scipy.signal.savgol_filter`) — fits a low-order
    polynomial over a sliding window; excellent at removing jitter while
    preserving the shape/curvature of a run (so an arrow still bends naturally).
  * **One-Euro** — an adaptive low-pass filter: heavy smoothing when a player is
    near-stationary (kills micro-jitter) but low lag when they sprint. Great for
    the live spotlight/circle.
  * **EMA** — a dependency-free exponential fallback.

Gaps (missing frames for a track id, or undetected ball frames) are linearly
interpolated first so a filter never smooths across a hole.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..utils.io import get_logger
from .detect_track import TrackResult

log = get_logger()


# --------------------------------------------------------------------------- #
# 1-D smoothers
# --------------------------------------------------------------------------- #
def smooth_series(values: np.ndarray, method: str = "savgol",
                  window: int = 9, poly: int = 2,
                  alpha: float = 0.4) -> np.ndarray:
    """Smooth a 1-D array of samples. Falls back gracefully on short inputs."""
    n = len(values)
    if n < 3:
        return values
    if method == "savgol":
        try:
            from scipy.signal import savgol_filter
            win = min(window, n if n % 2 == 1 else n - 1)
            if win < 3:
                win = 3
            if win % 2 == 0:
                win -= 1
            p = min(poly, win - 1)
            return savgol_filter(values, win, p)
        except Exception:  # noqa: BLE001  (scipy missing / degenerate window)
            method = "ema"
    if method == "ema":
        return _ema(values, alpha)
    return values


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


class OneEuroFilter:
    """Adaptive low-pass filter (Casiez et al. 2012).

    Smooths hard when slow, tracks fast when quick — ideal for a circle/spotlight
    that must sit rock-steady on a standing player yet keep up with a sprint.
    """

    def __init__(self, freq: float = 30.0, min_cutoff: float = 1.0,
                 beta: float = 0.02, d_cutoff: float = 1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


# --------------------------------------------------------------------------- #
# gap-filling
# --------------------------------------------------------------------------- #
def interpolate_path(frames: list[int], xs: list[float], ys: list[float],
                     max_gap: int = 12):
    """Return dense (idx, x, y) over [min..max] frame, linearly filling gaps
    up to ``max_gap`` frames; longer gaps are left as breaks."""
    if not frames:
        return []
    out: list[tuple[int, float, float]] = []
    for k in range(len(frames) - 1):
        i0, i1 = frames[k], frames[k + 1]
        out.append((i0, xs[k], ys[k]))
        gap = i1 - i0
        if 1 < gap <= max_gap:
            for j in range(1, gap):
                a = j / gap
                out.append((i0 + j, xs[k] + a * (xs[k + 1] - xs[k]),
                            ys[k] + a * (ys[k + 1] - ys[k])))
    out.append((frames[-1], xs[-1], ys[-1]))
    return out


# --------------------------------------------------------------------------- #
# high level: smooth an entire TrackResult in place
# --------------------------------------------------------------------------- #
def smooth_track(track: TrackResult, method: str = "savgol",
                 window: int = 9, poly: int = 2) -> TrackResult:
    """Smooth every player track *and* the ball path on a TrackResult in place.

    Player centres are rewritten frame-by-frame with the smoothed value; bbox
    sizes are left intact (only the centre, which drives circles/traces/camera,
    is de-jittered). The ball path is interpolated then smoothed.
    """
    # ---- players: gather per-id centre series keyed by frame idx ----
    by_id_x: dict[int, list[float]] = defaultdict(list)
    by_id_y: dict[int, list[float]] = defaultdict(list)
    by_id_f: dict[int, list[int]] = defaultdict(list)
    for fd in track.frames:
        for p in fd.players:
            pid = p["id"]
            by_id_f[pid].append(fd.idx)
            by_id_x[pid].append(p["center"][0])
            by_id_y[pid].append(p["center"][1])

    smoothed: dict[int, dict[int, tuple[float, float]]] = {}
    for pid, fr in by_id_f.items():
        if len(fr) < 3:
            smoothed[pid] = {f: (x, y)
                             for f, x, y in zip(fr, by_id_x[pid], by_id_y[pid])}
            continue
        sx = smooth_series(np.asarray(by_id_x[pid], dtype=np.float64),
                           method, window, poly)
        sy = smooth_series(np.asarray(by_id_y[pid], dtype=np.float64),
                           method, window, poly)
        smoothed[pid] = {f: (float(sx[k]), float(sy[k]))
                         for k, f in enumerate(fr)}

    for fd in track.frames:
        for p in fd.players:
            pos = smoothed.get(p["id"], {}).get(fd.idx)
            if pos is not None:
                dx = pos[0] - p["center"][0]
                dy = pos[1] - p["center"][1]
                p["center"] = [pos[0], pos[1]]
                x1, y1, x2, y2 = p["xyxy"]
                p["xyxy"] = [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    # ---- ball: interpolate gaps then smooth ----
    if track.ball_path:
        fr = [i for i, _, _ in track.ball_path]
        xs = [x for _, x, _ in track.ball_path]
        ys = [y for _, _, y in track.ball_path]
        dense = interpolate_path(fr, xs, ys)
        if len(dense) >= 3:
            dfr = [i for i, _, _ in dense]
            dx = smooth_series(np.asarray([x for _, x, _ in dense]),
                               method, window, poly)
            dy = smooth_series(np.asarray([y for _, _, y in dense]),
                               method, window, poly)
            track.ball_path = [(dfr[k], float(dx[k]), float(dy[k]))
                               for k in range(len(dfr))]

    log.info(f"[smoothing] de-jittered {len(smoothed)} player tracks + ball "
             f"(method={method}, window={window})")
    return track
