"""Tracking + virtual-camera layer (blueprint Module 3).

`cameraman` — YOLOv11 + BoT-SORT (with camera motion compensation / GMC) to
              differentiate true player motion from broadcast pan/zoom, lock the
              viewport onto the hero player + ball, and emit a Kalman-smoothed
              9:16 crop path so the vertical reframe glides instead of jittering.
"""
from __future__ import annotations

from .cameraman import CropPlan, Cameraman, plan_vertical_crop

__all__ = ["CropPlan", "Cameraman", "plan_vertical_crop"]
