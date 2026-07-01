"""Camera dynamic-subject-follow tests.

Verifies the Cameraman's focus target commits to the BALL when it is flying
(a pass/shot) and to the HERO player when the ball is slow / with the player
(a dribble / possession / skill accent).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.cameraman import (FrameTrack, _dynamic_ball_weight,  # noqa: E402
                                    _focus_points)

W, H = 1920, 1080
FPS = 30.0
HERO_ID = 7


def _frame(idx, hero_xy, ball_xy):
    ft = FrameTrack(idx=idx)
    ft.players = [{"id": HERO_ID, "cls": 0,
                   "xyxy": [hero_xy[0] - 20, hero_xy[1] - 40,
                            hero_xy[0] + 20, hero_xy[1] + 40],
                   "center": [float(hero_xy[0]), float(hero_xy[1])]}]
    if ball_xy is not None:
        ft.ball = {"xyxy": [ball_xy[0] - 6, ball_xy[1] - 6,
                            ball_xy[0] + 6, ball_xy[1] + 6],
                   "center": [float(ball_xy[0]), float(ball_xy[1])]}
    return ft


def test_flying_ball_pulls_focus_to_ball():
    """A fast ball leaving a stationary hero -> focus tracks the ball, not the
    (static) player."""
    n = 40
    hero = (400, 540)                     # hero stands still on the left
    frames = []
    for i in range(n):
        bx = 400 + i * 35                 # ball rockets right at 35 px/frame
        frames.append(_frame(i, hero, (bx, 540)))
    focus = _focus_points(frames, HERO_ID, W, H, fps=FPS)

    # by the end the ball is far right of the hero; the focus must have followed
    # the ball well past the mid-point between hero and ball.
    ball_end_x = 400 + (n - 1) * 35
    midpoint = (hero[0] + ball_end_x) / 2.0
    assert focus[-1][0] > midpoint, (focus[-1][0], midpoint)
    # and it should be much closer to the ball than to the hero
    assert abs(focus[-1][0] - ball_end_x) < abs(focus[-1][0] - hero[0])


def test_slow_ball_with_player_holds_the_hero():
    """A slow ball glued to a moving hero (dribble) -> focus stays on the hero
    side of the hero/ball pair, not a neutral 50/50 blend."""
    n = 40
    frames = []
    for i in range(n):
        hx = 500 + i * 6                  # hero jogs right slowly
        bx = hx + 25                      # ball dribbled just ahead, same speed
        frames.append(_frame(i, (hx, 540), (bx, 540)))
    focus = _focus_points(frames, HERO_ID, W, H, fps=FPS)

    # steady state: focus should sit closer to the hero than a 50/50 blend would
    hx = 500 + (n - 1) * 6
    bx = hx + 25
    blend_5050 = (hx + bx) / 2.0
    assert focus[-1][0] < blend_5050, (focus[-1][0], blend_5050)


def test_weight_rises_with_ball_speed():
    """The raw follow weight is higher for a fast ball than a slow one."""
    n = 30
    rf = {}
    slow = np.array([[500.0 + i * 3, 540.0] for i in range(n)])   # 3 px/frame
    fast = np.array([[500.0 + i * 40, 540.0] for i in range(n)])  # 40 px/frame
    w_slow = _dynamic_ball_weight(slow, W, H, FPS, None, rf)
    w_fast = _dynamic_ball_weight(fast, W, H, FPS, None, rf)
    assert w_fast[-1] > w_slow[-1]
    assert w_slow[-1] <= 0.6            # slow -> favours hero
    assert w_fast[-1] >= 0.7           # fast -> favours ball


def test_static_mode_is_5050():
    """follow_dynamic=false restores the exact old 50/50 blend."""
    n = 10
    pts = np.array([[500.0 + i * 40, 540.0] for i in range(n)])
    w = _dynamic_ball_weight(pts, W, H, FPS, None, {"follow_dynamic": False})
    assert np.allclose(w, 0.5)


def test_camera_reset_per_shot_no_speed_bleed():
    """Ball speed is measured inside each shot, so a cut can't fabricate a huge
    velocity across the boundary."""
    # shot A: ball still on the left; shot B: ball still on the right (a cut)
    left = np.array([[300.0, 540.0]] * 15)
    right = np.array([[1600.0, 540.0]] * 15)
    pts = np.vstack([left, right])
    segments = [(0, 15), (15, 30)]
    w = _dynamic_ball_weight(pts, W, H, FPS, segments, {})
    # both shots have a stationary ball -> low weight everywhere (no cut spike)
    assert w.max() <= 0.6, w.max()


if __name__ == "__main__":
    test_flying_ball_pulls_focus_to_ball()
    test_slow_ball_with_player_holds_the_hero()
    test_weight_rises_with_ball_speed()
    test_static_mode_is_5050()
    test_camera_reset_per_shot_no_speed_bleed()
    print("ok - camera dynamic focus")
