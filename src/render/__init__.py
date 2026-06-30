"""Compositing + render layer (blueprint Module 4, FX & Composer).

`composer` — MoviePy-based compositor: draws tactical graphics on a background
             layer, pastes player segmentation masks on top (so graphics stay
             under the boots), renders premium typography via MoviePy TextClip
             (drop shadows + semi-transparent data plates) and supervision
             halo/trace annotators, and builds audio-safe slow-motion.
"""
from __future__ import annotations

from .composer import Composer, compose_highlight

__all__ = ["Composer", "compose_highlight"]
