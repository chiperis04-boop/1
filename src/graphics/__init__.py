"""Pitch geometry layer (blueprint Module 4, "Grass-Anchor").

`homography` — detect pitch line/keypoints per frame and solve the image->pitch
               projection matrix H (kornia `find_homography_lines_dlt` when line
               correspondences are available, else cv2 RANSAC on point keypoints)
               so tactical graphics can be warped to sit flat on the grass.
"""
from __future__ import annotations

from .homography import GraphicsHomography, PitchProjector, warp_points

__all__ = ["GraphicsHomography", "PitchProjector", "warp_points"]
