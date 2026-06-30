"""Homography validation gate (CPU-only): a trustworthy image->pitch H is
accepted; degenerate/over-fit ones (the 'tactical lines into the sky' bug) are
rejected so no garbage graphics are ever drawn. No model/GPU needed."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision.pitch import PITCH_LENGTH, PITCH_WIDTH, valid_homography

IMG_W, IMG_H = 1920.0, 1080.0


def _good_H():
    """A sane H mapping the full 1920x1080 image onto the 105x68m pitch."""
    sx = PITCH_LENGTH / IMG_W
    sy = PITCH_WIDTH / IMG_H
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)


def test_accepts_plausible():
    assert valid_homography(_good_H(), IMG_W, IMG_H) is True
    print("  \u2713 plausible H (image -> full pitch) accepted")


def test_rejects_none_and_nonfinite():
    assert valid_homography(None, IMG_W, IMG_H) is False
    bad = _good_H(); bad[0, 0] = np.nan
    assert valid_homography(bad, IMG_W, IMG_H) is False
    print("  \u2713 None / non-finite H rejected")


def test_rejects_out_of_pitch():
    # huge scale -> corners land far outside the pitch (the 'into the sky' case)
    blowup = np.array([[50.0, 0, 0], [0, 50.0, 0], [0, 0, 1]], dtype=np.float64)
    assert valid_homography(blowup, IMG_W, IMG_H) is False
    print("  \u2713 projection outside pitch bounds rejected")


def test_rejects_degenerate_conditioning():
    # near-singular / vanishing-perspective H -> bad conditioning or w->0
    degen = np.array([[1, 0, 0], [0, 1, 0], [1e-3, 1e-3, 1e-9]], dtype=np.float64)
    assert valid_homography(degen, IMG_W, IMG_H) is False
    print("  \u2713 ill-conditioned / vanishing-perspective H rejected")


def test_margin_is_tolerant():
    # slightly off-pitch within the margin is still accepted (broadcast pan)
    sx = (PITCH_LENGTH + 20) / IMG_W
    sy = (PITCH_WIDTH + 10) / IMG_H
    H = np.array([[sx, 0, -5], [0, sy, -3], [0, 0, 1]], dtype=np.float64)
    assert valid_homography(H, IMG_W, IMG_H, margin=40.0) is True
    print("  \u2713 small overshoot within margin tolerated")


if __name__ == "__main__":
    test_accepts_plausible()
    test_rejects_none_and_nonfinite()
    test_rejects_out_of_pitch()
    test_rejects_degenerate_conditioning()
    test_margin_is_tolerant()
    print("\nALL HOMOGRAPHY-GATE TESTS PASSED \u2705")
