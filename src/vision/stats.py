"""Derive presentable stats from tracking, metric when calibrated.

If a `PitchCalibration` (from `pitch.PitchEstimator`) is supplied, distances are
measured in real metres and speed in km/h by mapping image points to pitch
coordinates. Without calibration we fall back to a rough pixels-per-metre
estimate and flag the result as approximate (`metric=False`), so the overlay can
show "~" and we never present a guess as exact truth.
"""
from __future__ import annotations

import numpy as np

from .detect_track import TrackResult

try:  # avoid a hard import cycle at module load
    from .pitch import PitchCalibration
except Exception:  # pragma: no cover
    PitchCalibration = None  # type: ignore


def compute_stats(track: TrackResult, calib=None, px_per_m: float | None = None) -> dict:
    metric = calib is not None and getattr(calib, "coverage", 0.0) > 0.5
    if metric:
        return _metric_stats(track, calib)
    return _pixel_stats(track, px_per_m)


# --------------------------------------------------------------------------- #
# metric (homography-based)
# --------------------------------------------------------------------------- #
def _metric_stats(track: TrackResult, calib) -> dict:
    stats: dict = {"metric": True}
    fps = track.fps or 30.0

    key = track.key_track_id
    if key is not None:
        pts = []  # (idx, x_m, y_m)
        for fd in track.frames:
            for p in fd.players:
                if p["id"] == key:
                    m = calib.to_pitch(fd.idx, p["center"][0], p["center"][1])
                    if m is not None:
                        pts.append((fd.idx, m[0], m[1]))
        dist, top_speed = _metric_path(pts, fps)
        stats["sprint_distance_m"] = dist
        stats["top_speed_kmh"] = top_speed

    if track.ball_path:
        tail = track.ball_path[-max(2, int(fps // 2)):]
        bpts = []
        for idx, x, y in tail:
            m = calib.to_pitch(idx, x, y)
            if m is not None:
                bpts.append((idx, m[0], m[1]))
        d, _ = _metric_path(bpts, fps)
        if d:
            stats["shot_distance_m"] = d

    stats["players_beaten"] = _players_beaten(track)
    return stats


def _metric_path(pts, fps):
    """Return (total_distance_m, top_speed_kmh) from (idx, x_m, y_m) samples,
    rejecting unrealistic jumps (>12 m/frame ~ homography glitches)."""
    if len(pts) < 2:
        return 0.0, 0.0
    total = 0.0
    top = 0.0
    for (i0, x0, y0), (i1, x1, y1) in zip(pts, pts[1:]):
        d = float(np.hypot(x1 - x0, y1 - y0))
        dt = max(1, i1 - i0) / fps
        if d > 12.0:                       # reject calibration spikes
            continue
        total += d
        speed_kmh = (d / dt) * 3.6
        if speed_kmh < 45.0:               # human cap, ignore glitches
            top = max(top, speed_kmh)
    return total, top


# --------------------------------------------------------------------------- #
# pixel fallback (approximate)
# --------------------------------------------------------------------------- #
def _pixel_stats(track: TrackResult, px_per_m: float | None) -> dict:
    if px_per_m is None:
        px_per_m = track.width / 60.0      # assume ~60 m spans the frame width
    stats: dict = {"metric": False}

    key = track.key_track_id
    if key is not None:
        pts = [(p["center"][0], p["center"][1])
               for fd in track.frames for p in fd.players if p["id"] == key]
        stats["sprint_distance_m"] = _path_len(pts) / px_per_m

    if track.ball_path:
        tail = track.ball_path[-int(track.fps // 2):] or track.ball_path[-2:]
        stats["shot_distance_m"] = _path_len([(x, y) for _, x, y in tail]) / px_per_m

    stats["players_beaten"] = _players_beaten(track)
    return stats


def _path_len(pts) -> float:
    if len(pts) < 2:
        return 0.0
    a = np.array(pts, dtype=np.float32)
    return float(np.sum(np.linalg.norm(np.diff(a, axis=0), axis=1)))


def _players_beaten(track: TrackResult) -> int:
    key = track.key_track_id
    if key is None:
        return 0
    beaten = set()
    prev_rel: dict[int, float] = {}
    for fd in track.frames:
        kp = next((p for p in fd.players if p["id"] == key), None)
        if not kp:
            continue
        kx = kp["center"][0]
        for p in fd.players:
            if p["id"] == key:
                continue
            rel = p["center"][0] - kx
            if p["id"] in prev_rel and prev_rel[p["id"]] < 0 <= rel:
                beaten.add(p["id"])
            prev_rel[p["id"]] = rel
    return min(len(beaten), 6)
