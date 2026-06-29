"""Pitch keypoint detection + homography (camera calibration).

Maps image coordinates to real pitch coordinates in metres so stats become
metric (distance in m, speed in km/h) and a top-down "radar" view becomes
possible. A keypoint model (e.g. the roboflow/sports football-field-detection
model) locates known pitch landmarks each frame; we solve a homography from the
*visible subset* of those landmarks to a fixed pitch template, with RANSAC and
temporal smoothing for robustness against broadcast cuts/zoom.

This module is optional. If no pitch model is configured or too few keypoints
are visible, callers fall back to pixel-space estimates.

References:
  * roboflow/sports — pitch keypoint detection + calibration
    https://github.com/roboflow/sports
  * Camera calibration in sports with keypoints
    https://blog.roboflow.com/camera-calibration-sports-computer-vision/
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..utils.io import get_logger

log = get_logger()

# Standard pitch dimensions in metres.
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

# A canonical set of pitch landmarks in metres (origin at one corner).
# Index order MUST match the keypoint model's output channel order; adjust
# `keypoint_order` in config if your model differs. This is the common
# roboflow/sports-style 32-point layout (corners, box corners, circle, spots).
PITCH_TEMPLATE: dict[str, tuple[float, float]] = {
    "tl_corner": (0.0, 0.0),
    "tr_corner": (PITCH_LENGTH, 0.0),
    "br_corner": (PITCH_LENGTH, PITCH_WIDTH),
    "bl_corner": (0.0, PITCH_WIDTH),
    "halfway_top": (PITCH_LENGTH / 2, 0.0),
    "halfway_bottom": (PITCH_LENGTH / 2, PITCH_WIDTH),
    "center": (PITCH_LENGTH / 2, PITCH_WIDTH / 2),
    "l_box_tl": (0.0, (PITCH_WIDTH - 40.32) / 2),
    "l_box_bl": (0.0, (PITCH_WIDTH + 40.32) / 2),
    "l_box_tr": (16.5, (PITCH_WIDTH - 40.32) / 2),
    "l_box_br": (16.5, (PITCH_WIDTH + 40.32) / 2),
    "r_box_tr": (PITCH_LENGTH, (PITCH_WIDTH - 40.32) / 2),
    "r_box_br": (PITCH_LENGTH, (PITCH_WIDTH + 40.32) / 2),
    "r_box_tl": (PITCH_LENGTH - 16.5, (PITCH_WIDTH - 40.32) / 2),
    "r_box_bl": (PITCH_LENGTH - 16.5, (PITCH_WIDTH + 40.32) / 2),
    "l_pen_spot": (11.0, PITCH_WIDTH / 2),
    "r_pen_spot": (PITCH_LENGTH - 11.0, PITCH_WIDTH / 2),
}
TEMPLATE_POINTS = np.array(list(PITCH_TEMPLATE.values()), dtype=np.float32)


@dataclass
class FrameHomography:
    idx: int
    H: np.ndarray | None = None        # 3x3 image->pitch, or None if unsolved
    n_points: int = 0


@dataclass
class PitchCalibration:
    frames: dict[int, FrameHomography] = field(default_factory=dict)

    def to_pitch(self, idx: int, x: float, y: float):
        """Map an image point to pitch metres; None if no homography for idx."""
        fh = self.frames.get(idx)
        if fh is None or fh.H is None:
            return None
        pt = np.array([x, y, 1.0], dtype=np.float64)
        out = fh.H @ pt
        if abs(out[2]) < 1e-9:
            return None
        return float(out[0] / out[2]), float(out[1] / out[2])

    @property
    def coverage(self) -> float:
        if not self.frames:
            return 0.0
        solved = sum(1 for f in self.frames.values() if f.H is not None)
        return solved / len(self.frames)


class PitchEstimator:
    """Loads a pitch-keypoint model and produces per-frame homographies."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        p = cfg.get("vision", {}).get("pitch", {})
        self.enabled = bool(p.get("enabled", False))
        self.model_path = p.get("model")
        self.min_points = int(p.get("min_points", 4))
        self.conf = float(p.get("conf", 0.5))
        self.every = int(p.get("sample_every_frames", 1))
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def calibrate(self, clip_path: str) -> PitchCalibration:
        calib = PitchCalibration()
        if not self.enabled or not self.model_path:
            return calib
        try:
            import cv2
            model = self._load()
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[pitch] disabled (load failed): {exc}")
            return calib

        cap = cv2.VideoCapture(clip_path)
        device = self.cfg["vision"].get("device", "cpu")
        idx = 0
        last_H = None
        smooth = float(self.cfg["vision"].get("pitch", {}).get("smoothing", 0.8))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            H = last_H
            if idx % self.every == 0:
                H = self._solve(model, frame, device, cv2) or last_H
                if H is not None and last_H is not None:
                    H = smooth * last_H + (1 - smooth) * H
                last_H = H
            calib.frames[idx] = FrameHomography(idx=idx, H=H,
                                                n_points=0 if H is None else 1)
            idx += 1
        cap.release()
        log.info(f"[pitch] calibrated {idx} frames, coverage {calib.coverage:.0%}")
        return calib

    def _solve(self, model, frame, device, cv2):
        res = model.predict(frame, conf=self.conf, device=device, verbose=False)[0]
        kpts = getattr(res, "keypoints", None)
        if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
            return None
        xy = kpts.xy[0].cpu().numpy()                 # (K, 2)
        conf = (kpts.conf[0].cpu().numpy() if kpts.conf is not None
                else np.ones(len(xy)))
        # match visible keypoints to template by channel index
        img_pts, tmpl_pts = [], []
        for i, (pt, c) in enumerate(zip(xy, conf)):
            if c >= self.conf and i < len(TEMPLATE_POINTS) and (pt[0] + pt[1]) > 0:
                img_pts.append(pt)
                tmpl_pts.append(TEMPLATE_POINTS[i])
        if len(img_pts) < self.min_points:
            return None
        img_pts = np.array(img_pts, dtype=np.float32)
        tmpl_pts = np.array(tmpl_pts, dtype=np.float32)
        H, _ = cv2.findHomography(img_pts, tmpl_pts, cv2.RANSAC, 5.0)
        return H
