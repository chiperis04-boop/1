"""Perception layer (v3) — turning raw footage into structured context the
editing agents can reason over.

P0 ships shot segmentation (broadcast camera cuts) so tracking, crop planning
and graphics run *per shot* instead of gliding across cuts. Later phases add the
full PerceptionBundle (ASR transcript, detection summary, audio events) feeding
the frame-aware Director agent — see docs/IMPLEMENTATION_PLAN_AI_DIRECTOR.md.
"""
from .shots import Shot, frame_segments, mark_duplicate_shots, segment_shots

__all__ = ["Shot", "segment_shots", "frame_segments", "mark_duplicate_shots"]
