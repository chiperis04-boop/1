"""GPU-free unit tests for the v2 "Studio" pipeline logic.

These exercise the pure-Python decision logic of the v2 chain — the parts that
do NOT need a GPU, YOLO, EasyOCR or real video — so the studio's "football
math" stays correct under refactors:

  * audio-safe slow-mo  : the atempo filter chain for factors < 0.5
  * Cameraman           : constant-velocity Kalman viewport smoothing + EMA
                          fallback, focus-point fusion, geometric hero vote
  * possession          : per-frame holder -> confirmed runs (min_frames +
                          gap bridging) + team share
  * jerseys             : parsing a shirt number out of a Director description,
                          and highest-confidence track lookup for a number
  * analytics           : hero resolution order (jersey > team_nearest >
                          geometric)
  * teams               : club-colour lookup per track
  * scout               : near-duplicate event-window de-duplication

Run directly (no pytest needed):

    python tests/test_studio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.render.composer import _atempo_chain                       # noqa: E402
from src.tracking.cameraman import (FrameTrack, _ema, _focus_points,  # noqa: E402
                                     _kalman_1d, _kalman_smooth, _pick_hero)
from src.vision.possession import analyze_possession                # noqa: E402
from src.vision.jerseys import JerseyResult, number_from_description  # noqa: E402
from src.vision.analytics import _resolve_hero                      # noqa: E402
from src.vision.teams import TeamAssignment, pick_key_player        # noqa: E402
from src.detection.scout import EventWindow, _dedupe                # noqa: E402


# --------------------------------------------------------------------------- #
# small duck-typed stand-ins (match the attributes the code reads)
# --------------------------------------------------------------------------- #
class _Track:
    """Minimal stand-in for a CropPlan/TrackResult: just `.frames` + `.fps`."""
    def __init__(self, frames, fps=30.0, hero_id=None):
        self.frames = frames
        self.fps = fps
        self.hero_id = hero_id
        self.key_track_id = None


class _Manifest:
    def __init__(self, desc):
        self.main_hero_description = desc


def _player(pid, cx, cy, h=36.0, w=20.0):
    return {"id": pid, "cls": 0,
            "xyxy": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            "center": [float(cx), float(cy)]}


def _frame(idx, players, ball_xy=None):
    ft = FrameTrack(idx=idx, players=players)
    if ball_xy is not None:
        ft.ball = {"xyxy": [ball_xy[0] - 3, ball_xy[1] - 3,
                            ball_xy[0] + 3, ball_xy[1] + 3],
                   "center": [float(ball_xy[0]), float(ball_xy[1])]}
    return ft


# --------------------------------------------------------------------------- #
# 1) audio-safe slow-mo: atempo chain for factors < 0.5
# --------------------------------------------------------------------------- #
def test_atempo_chain():
    def product(chain: str) -> float:
        vals = [float(p.split("=")[1]) for p in chain.split(",")]
        # ffmpeg requires each atempo factor in [0.5, 2.0]
        for v in vals:
            assert 0.5 - 1e-6 <= v <= 2.0 + 1e-6, f"atempo {v} out of [0.5,2.0]"
        out = 1.0
        for v in vals:
            out *= v
        return out

    # 0.4 needs chaining (single atempo can't go below 0.5)
    c04 = _atempo_chain(0.4)
    assert "," in c04, f"factor 0.4 must chain, got {c04!r}"
    assert abs(product(c04) - 0.4) < 1e-6, c04

    # extreme slow factor still resolves to a legal, correct chain
    c01 = _atempo_chain(0.1)
    assert abs(product(c01) - 0.1) < 1e-6, c01

    # >= 0.5 stays a single filter
    c06 = _atempo_chain(0.6)
    assert c06 == "atempo=0.6", c06
    c10 = _atempo_chain(1.0)
    assert c10 == "atempo=1.0", c10
    print("  ✓ atempo chain: <0.5 factors chained & products exact, "
          "each stage in [0.5,2.0]")


# --------------------------------------------------------------------------- #
# 2) Cameraman: Kalman smoothing + EMA fallback + focus fusion + hero vote
# --------------------------------------------------------------------------- #
def test_kalman_smoothing_reduces_jitter():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 120)
    truth = 500 + 200 * t                       # smooth ramp (camera pan)
    noisy = truth + rng.normal(0, 25, size=truth.shape)
    sm = _kalman_1d(noisy, q=2.0, r=120.0, fps=30.0)
    # smoothed signal is closer to the truth than the raw measurements
    err_raw = np.mean((noisy - truth) ** 2)
    err_sm = np.mean((sm - truth) ** 2)
    assert err_sm < err_raw, f"kalman did not reduce error ({err_sm} !< {err_raw})"
    # and is materially less jittery frame-to-frame
    assert np.mean(np.abs(np.diff(sm))) < np.mean(np.abs(np.diff(noisy)))
    print(f"  ✓ kalman: MSE {err_raw:.0f}->{err_sm:.0f}, jitter reduced")


def test_kalman_smooth_ema_fallback():
    xs = np.array([0.0, 10.0, 0.0, 10.0, 0.0, 10.0])
    ys = xs.copy()
    # a broken cfg value forces the except-branch -> EMA fallback (no crash)
    cx, cy = _kalman_smooth(xs, ys, fps=30.0,
                            cfg={"kalman_process_noise": "not-a-number",
                                 "smoothing": 0.8})
    assert len(cx) == len(xs) and len(cy) == len(ys)
    # EMA must damp the oscillation amplitude
    assert np.max(cx) - np.min(cx) < np.max(xs) - np.min(xs)
    # matches the dedicated EMA helper
    assert np.allclose(cx, _ema(xs, 0.8))
    print("  ✓ kalman_smooth: bad cfg degrades to EMA fallback (no crash)")


def test_focus_points_fallback_chain():
    w, h = 1280, 720
    frames = [
        _frame(0, [_player(1, 100, 100), _player(2, 200, 200)], ball_xy=(150, 150)),
        _frame(1, [_player(1, 110, 100)]),                       # hero only
        _frame(2, [], ball_xy=(600, 360)),                       # ball only
        _frame(3, []),                                           # nothing -> last
    ]
    pts = _focus_points(frames, hero_id=1, w=w, h=h)
    assert pts.shape == (4, 2)
    # frame 0: average of hero(100,100) and ball(150,150) -> (125,125)
    assert np.allclose(pts[0], [125, 125]), pts[0]
    # frame 2: ball only
    assert np.allclose(pts[2], [600, 360]), pts[2]
    # frame 3: no detections -> carries last point forward
    assert np.allclose(pts[3], pts[2]), pts[3]
    print("  ✓ focus points: hero+ball fusion, ball-only, last-known fallbacks")


def test_geometric_hero_vote():
    # player 7 is nearest the ball for most of the final 40% of the clip
    frames = []
    for i in range(10):
        frames.append(_frame(i, [_player(7, 100, 100), _player(9, 900, 500)],
                             ball_xy=(105, 100)))
    assert _pick_hero(frames) == 7
    print("  ✓ geometric hero: nearest-to-ball vote picks the right track")


def test_kalman_rts_is_zero_phase():
    """The RTS (forward+backward) smoother must not lag the action.

    On a symmetric bump a causal forward-only filter shifts the peak LATER in
    time (lag), which is exactly why a naive auto-reframe trails the ball. The
    RTS smoother should keep the peak essentially where it is.
    """
    n = 121
    t = np.arange(n)
    peak = 60
    z = 1000.0 * np.exp(-((t - peak) ** 2) / (2 * 8.0 ** 2))  # gaussian bump
    sm = _kalman_1d(z, q=2.0, r=120.0, fps=30.0)
    causal = _ema(z, 0.85)
    sm_peak = int(np.argmax(sm))
    causal_peak = int(np.argmax(causal))
    # RTS peak stays on top of the true peak ...
    assert abs(sm_peak - peak) <= 3, f"RTS peak lagged: {sm_peak} vs {peak}"
    # ... while the causal filter visibly lags later in time
    assert causal_peak > sm_peak + 3, (causal_peak, sm_peak)
    print(f"  ✓ kalman RTS zero-phase: peak@{sm_peak} (true {peak}), "
          f"causal lags to {causal_peak}")


def test_per_shot_smoothing_resets_at_cut():
    """With shot segments, the camera path must reset at the cut (sharp step)
    instead of gliding across it (the v2 lurch bug)."""
    n = 120
    xs = np.concatenate([np.full(60, 100.0), np.full(60, 900.0)])  # hard cut @60
    ys = np.full(n, 500.0)
    cfg = {"kalman_process_noise": 2.0, "kalman_measurement_noise": 120.0,
           "smoothing": 0.85}
    whole, _ = _kalman_smooth(xs, ys, 30.0, cfg)                    # one path
    seg, _ = _kalman_smooth(xs, ys, 30.0, cfg, segments=[(0, 60), (60, 120)])
    # segmented: each shot pinned near its own value -> big step at the boundary
    assert seg[59] < 300 and seg[60] > 700, (seg[59], seg[60])
    # whole-clip path bleeds the cut -> its boundary step is far smaller
    assert (seg[60] - seg[59]) > (whole[60] - whole[59]) * 3
    print(f"  ✓ per-shot smoothing: step@cut seg={seg[60]-seg[59]:.0f} "
          f"vs whole-clip {whole[60]-whole[59]:.0f} (no cross-cut glide)")


# --------------------------------------------------------------------------- #
# 3) possession: per-frame holders -> confirmed runs + bridging + share
# --------------------------------------------------------------------------- #
def test_possession_runs_and_bridge():
    # player 7 holds the ball; frame 3 has a one-frame detection dropout that
    # must be bridged into a single continuous run (min_frames=3, bridge=3).
    frames = []
    for i in range(6):
        if i == 3:
            frames.append(_frame(i, [_player(7, 100, 100), _player(9, 900, 500)]))
        else:
            frames.append(_frame(i, [_player(7, 100, 100), _player(9, 900, 500)],
                                 ball_xy=(104, 100)))
    res = analyze_possession(_Track(frames),
                             cfg={"possession": {"radius_m": 1.5, "min_frames": 3,
                                                 "bridge_frames": 3}},
                             team_of={7: 0, 9: 1})
    assert len(res.runs) == 1, f"expected 1 bridged run, got {len(res.runs)}"
    run = res.runs[0]
    assert run.track_id == 7 and run.team == 0
    assert run.start_idx == 0 and run.end_idx == 5, (run.start_idx, run.end_idx)
    assert res.holder_at(4) == 7
    # team share: all confirmed possession is team 0
    assert res.share.get(0, 0) == 1.0, res.share
    print("  ✓ possession: dropout bridged into one run, team share computed")


def test_possession_rejects_far_ball():
    # ball always far from every player -> no possession run
    frames = [_frame(i, [_player(7, 100, 100)], ball_xy=(1200, 700))
              for i in range(6)]
    res = analyze_possession(_Track(frames),
                             cfg={"possession": {"radius_m": 1.5, "min_frames": 3}})
    assert res.runs == [], "ball far from players must yield no runs"
    print("  ✓ possession: ball out of range yields no false run")


# --------------------------------------------------------------------------- #
# 4) jerseys: number parsing + highest-confidence track lookup
# --------------------------------------------------------------------------- #
def test_number_from_description():
    assert number_from_description("Player 18 in blue jersey") == 18
    assert number_from_description("follow #7") == 7
    assert number_from_description("no. 23 on the left") == 23
    assert number_from_description("number 9") == 9
    assert number_from_description("the player on the ball") is None
    assert number_from_description("") is None
    # explicit '#NN' marker takes the first 1-2 digits ('#100' -> 10)
    assert number_from_description("#100") == 10
    print("  ✓ jersey parse: #/no./number/'player N' all extracted; prose -> None")


def test_jersey_track_for_number_prefers_confidence():
    jr = JerseyResult(number_of={5: 10, 8: 10, 3: 7},
                      confidence_of={5: 0.4, 8: 0.9, 3: 0.8})
    # two tracks both read as "10" -> pick the higher-confidence one (8)
    assert jr.track_for_number(10) == 8
    assert jr.track_for_number(7) == 3
    assert jr.track_for_number(99) is None
    print("  ✓ jersey lookup: ambiguous number resolves to highest-confidence track")


# --------------------------------------------------------------------------- #
# 5) analytics: hero resolution order (jersey > team_nearest > geometric)
# --------------------------------------------------------------------------- #
def test_hero_resolution_priority():
    frames = [_frame(i, [_player(7, 100, 100), _player(9, 900, 500)],
                     ball_xy=(105, 100)) for i in range(10)]
    track = _Track(frames, hero_id=9)
    teams = TeamAssignment(team_of={7: 0, 9: 1}, colors={0: (255, 0, 0)})

    # (a) jersey wins when the Director names a readable number
    jr = JerseyResult(number_of={7: 10}, confidence_of={7: 0.9})
    tid, num, src = _resolve_hero(_Manifest("lock on #10"), jr, teams, track,
                                  geometric_hero=9)
    assert (tid, num, src) == (7, 10, "jersey"), (tid, num, src)

    # (b) no number in description -> team-aware nearest-to-ball pick
    jr2 = JerseyResult()
    tid, num, src = _resolve_hero(_Manifest("the striker"), jr2, teams, track,
                                  geometric_hero=9)
    assert src == "team_nearest" and tid == 7, (tid, src)

    # (c) no teams known -> plain geometric fallback (Cameraman's pick)
    tid, num, src = _resolve_hero(_Manifest("the striker"), jr2,
                                  TeamAssignment(), track, geometric_hero=9)
    assert (tid, src) == (9, "geometric"), (tid, src)
    print("  ✓ hero resolution: jersey > team_nearest > geometric, in order")


# --------------------------------------------------------------------------- #
# 6) teams: club-colour lookup per track
# --------------------------------------------------------------------------- #
def test_hero_halo_persistence():
    """The hero halo must persist across short detection gaps (no flicker) and
    then release after `halo_hold_frames`."""
    from src.render.composer import _smooth_hero
    tele = {"halo_hold_frames": 4, "halo_smooth": 0.0}
    state: dict = {}
    assert _smooth_hero(_frame(0, [_player(7, 100, 100)]), 7, state, tele) is not None
    persisted = [_smooth_hero(_frame(i, [_player(9, 500, 500)]), 7, state, tele)
                 for i in range(1, 6)]            # hero 7 absent for 5 frames
    assert all(p is not None for p in persisted[:4]), persisted
    assert persisted[4] is None                   # released after hold=4
    print("  ✓ hero halo persists across short gaps then releases (no flicker)")


def test_per_shot_zoom_sizes():
    """Director per-shot zoom must change the crop size per shot (punch-in)."""
    from src.agents.editplan import ShotEdit
    from src.perception.shots import Shot
    from src.tracking.cameraman import Cameraman, FrameTrack
    cfg = {"edit": {"reframe": {"target_aspect": "9:16"}}, "tracking": {},
           "_active_profile": {"width": 1080, "height": 1920}}
    cam = Cameraman(cfg)
    n = 60
    frames = [FrameTrack(idx=i, players=[{"id": 1, "cls": 0,
                                          "xyxy": [600, 300, 640, 400],
                                          "center": [620, 350]}])
              for i in range(n)]
    meta = {"w": 1280, "h": 720, "fps": 30.0}
    shots = [Shot(0, 0, 1, 0, 30), Shot(1, 1, 2, 30, 60)]
    edits = [ShotEdit(0, zoom=1.0), ShotEdit(1, zoom=2.0)]   # punch-in on shot 2
    plan = cam.build_plan(frames, meta, hero_id=1, shots=shots, shot_edits=edits)
    base_cw = plan.sizes[0][0]
    tight_cw = plan.sizes[45][0]
    assert tight_cw < base_cw and abs(tight_cw - base_cw / 2) <= 2, (base_cw, tight_cw)
    print(f"  ✓ per-shot zoom: crop {base_cw}px -> {tight_cw}px on the punch-in shot")


def test_team_colors():
    ta = TeamAssignment(team_of={7: 0, 9: 1}, colors={0: (255, 0, 0), 1: (0, 0, 255)})
    assert ta.color_for_track(7) == (255, 0, 0)
    assert ta.color_for_track(9) == (0, 0, 255)
    # unknown track -> caller default
    assert ta.color_for_track(999, default=(1, 2, 3)) == (1, 2, 3)
    print("  ✓ team colours: halo colour resolves per track, default on unknown")


def test_pick_key_player_restricts_to_attacking_team():
    # team 0 (players 7,11) is in possession during the final third
    frames = []
    for i in range(10):
        frames.append(_frame(i, [_player(7, 100, 100), _player(11, 120, 110),
                                  _player(9, 900, 500)], ball_xy=(102, 100)))
    pid = pick_key_player(_Track(frames), team_of={7: 0, 11: 0, 9: 1})
    assert pid in (7, 11), pid          # must be on the attacking team
    print("  ✓ pick_key_player: protagonist restricted to the attacking team")


# --------------------------------------------------------------------------- #
# 7) scout: near-duplicate event-window de-duplication
# --------------------------------------------------------------------------- #
def test_scout_dedupe_merges_cluster():
    # three detectors fire on one goal within a few seconds -> one window,
    # keeping the highest confidence and unioning the source tags.
    ws = [
        EventWindow(kind="chance", anchor_t=100.0, start=80, end=110,
                    confidence=0.5, sources=["action_spotting"]),
        EventWindow(kind="goal", anchor_t=103.0, start=83, end=113,
                    confidence=0.9, verified=True, sources=["scoreboard_ocr"],
                    score_after="2-1"),
        EventWindow(kind="goal", anchor_t=140.0, start=120, end=150,
                    confidence=0.7, sources=["action_spotting"]),
    ]
    merged = _dedupe(ws, gap=15.0)
    assert len(merged) == 2, f"expected 2 windows after merge, got {len(merged)}"
    first = merged[0]
    assert first.kind == "goal"                       # goal wins the cluster
    assert first.verified is True
    assert set(first.sources) == {"action_spotting", "scoreboard_ocr"}
    assert first.score_after == "2-1"
    print("  ✓ scout dedupe: clustered detections merge, goal+verified+sources kept")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("v2 studio logic tests (no GPU)")
    tests = [
        test_atempo_chain,
        test_kalman_smoothing_reduces_jitter,
        test_kalman_smooth_ema_fallback,
        test_kalman_rts_is_zero_phase,
        test_per_shot_smoothing_resets_at_cut,
        test_focus_points_fallback_chain,
        test_geometric_hero_vote,
        test_possession_runs_and_bridge,
        test_possession_rejects_far_ball,
        test_number_from_description,
        test_jersey_track_for_number_prefers_confidence,
        test_hero_resolution_priority,
        test_hero_halo_persistence,
        test_per_shot_zoom_sizes,
        test_team_colors,
        test_pick_key_player_restricts_to_attacking_team,
        test_scout_dedupe_merges_cluster,
    ]
    for t in tests:
        t()
    print("\nALL STUDIO (v2) LOGIC TESTS PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
