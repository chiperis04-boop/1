"""Real-time ball possession.

Each frame we find the player nearest the ball and measure the gap. When a
player keeps the ball inside `possession_radius_m` for at least `min_frames`
consecutive frames (short detection dropouts bridged), that becomes a confirmed
*possession run* — which the Composer turns into a "POSSESSION" plate and which
feeds a team possession-share stat.

Distance is measured in **metres** when a pitch `PitchCalibration` is available
(ball + player centres projected to pitch coords). Without calibration we fall
back to a pixel threshold derived from player bounding-box height (a standing
player ~1.8 m), and flag the result `metric=False`.

Works on both the v1 `TrackResult` and the v2 `CropPlan` (duck-typed: `.fps`,
`.frames[*].idx`, `.frames[*].players[*]`, `.frames[*].ball`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.io import get_logger

log = get_logger()


@dataclass
class PossessionRun:
    start_idx: int
    end_idx: int
    track_id: int
    team: int = -1

    def contains(self, idx: int) -> bool:
        return self.start_idx <= idx <= self.end_idx

    def duration_frames(self) -> int:
        return self.end_idx - self.start_idx + 1


@dataclass
class PossessionResult:
    holder_by_frame: dict[int, int] = field(default_factory=dict)   # idx -> track_id
    runs: list[PossessionRun] = field(default_factory=list)
    share: dict[int, float] = field(default_factory=dict)           # team -> fraction
    metric: bool = False
    radius_m: float = 1.5

    def holder_at(self, idx: int) -> int | None:
        for r in self.runs:
            if r.contains(idx):
                return r.track_id
        return None

    def run_at(self, idx: int) -> PossessionRun | None:
        for r in self.runs:
            if r.contains(idx):
                return r
        return None


def analyze_possession(track, calib=None, cfg: dict | None = None,
                       team_of: dict[int, int] | None = None) -> PossessionResult:
    cfg = cfg or {}
    p = cfg.get("possession", {})
    radius_m = float(p.get("radius_m", 1.5))      # blueprint: ~1 m; 1.5 is practical
    min_frames = int(p.get("min_frames", 3))      # blueprint: > 3 frames
    bridge = int(p.get("bridge_frames", 3))       # tolerate brief holder dropouts
    team_of = team_of or {}

    frames = getattr(track, "frames", []) or []
    metric = calib is not None and getattr(calib, "coverage", 0.0) > 0.5
    px_per_m = _estimate_px_per_m(frames)

    # 1) nearest player + gap per frame -> raw holder (or None if too far)
    raw: dict[int, int] = {}
    for fd in frames:
        if not getattr(fd, "ball", None) or not fd.players:
            continue
        holder, gap = _nearest(fd, calib if metric else None, px_per_m)
        if holder is None:
            continue
        thresh = radius_m if metric else radius_m * px_per_m
        if gap <= thresh:
            raw[fd.idx] = holder

    # 2) collapse into stable runs (bridge short gaps / single-frame flips)
    runs = _runs_from_holders(frames, raw, min_frames, bridge)
    for r in runs:
        r.team = team_of.get(r.track_id, -1)

    holder_by_frame = {idx: r.track_id for r in runs for idx in
                       range(r.start_idx, r.end_idx + 1)}
    share = _team_share(runs, frames)

    log.info(f"[possession] {len(runs)} runs, metric={metric}, "
             f"radius={radius_m}m, share={ {k: round(v,2) for k,v in share.items()} }")
    return PossessionResult(holder_by_frame=holder_by_frame, runs=runs,
                            share=share, metric=metric, radius_m=radius_m)


# --------------------------------------------------------------------------- #
def _nearest(fd, calib, px_per_m):
    """Return (track_id, gap) for the player nearest the ball this frame.
    gap is metres when `calib` is given, else pixels."""
    bx, by = fd.ball["center"]
    if calib is not None:
        bm = calib.to_pitch(fd.idx, bx, by)
    best_id, best_d = None, float("inf")
    for pl in fd.players:
        px, py = pl["center"]
        if calib is not None:
            pm = calib.to_pitch(fd.idx, px, py)
            if bm is None or pm is None:
                continue
            d = float(np.hypot(pm[0] - bm[0], pm[1] - bm[1]))
        else:
            d = float(np.hypot(px - bx, py - by))
        if d < best_d:
            best_id, best_d = pl["id"], d
    return best_id, best_d


def _runs_from_holders(frames, raw, min_frames, bridge):
    """Merge per-frame holders into runs, bridging gaps <= `bridge` frames where
    the holder is unchanged or briefly missing."""
    idxs = [fd.idx for fd in frames]
    runs: list[PossessionRun] = []
    cur_id, cur_start, cur_last = None, None, None
    for idx in idxs:
        h = raw.get(idx)
        if h is None:
            # keep the current run alive across a short gap
            if cur_id is not None and (idx - cur_last) <= bridge:
                continue
            _flush(runs, cur_id, cur_start, cur_last, min_frames)
            cur_id = None
            continue
        if h == cur_id and (idx - cur_last) <= bridge:
            cur_last = idx
        else:
            _flush(runs, cur_id, cur_start, cur_last, min_frames)
            cur_id, cur_start, cur_last = h, idx, idx
    _flush(runs, cur_id, cur_start, cur_last, min_frames)
    return runs


def _flush(runs, tid, start, last, min_frames):
    if tid is not None and start is not None and (last - start + 1) >= min_frames:
        runs.append(PossessionRun(start_idx=start, end_idx=last, track_id=tid))


def _team_share(runs, frames) -> dict[int, float]:
    total = sum(r.duration_frames() for r in runs)
    if total == 0:
        return {}
    by_team: dict[int, int] = {}
    for r in runs:
        by_team[r.team] = by_team.get(r.team, 0) + r.duration_frames()
    return {t: n / total for t, n in by_team.items() if t != -1}


def _estimate_px_per_m(frames) -> float:
    """Pixels-per-metre from the median player bounding-box height (~1.8 m)."""
    heights = [pl["xyxy"][3] - pl["xyxy"][1]
               for fd in frames for pl in getattr(fd, "players", [])]
    if not heights:
        return 30.0
    return max(8.0, float(np.median(heights)) / 1.8)
