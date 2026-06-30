"""Discovery layer (blueprint Module 1 + 2).

`scout`    — SoccerNet action spotting + EasyOCR scoreboard verification that
             turns a full match into a short list of verified event windows.
`director` — a vision-LLM that watches each event window and emits a strict JSON
             editing manifest (hero player, hook text, slow-mo timing, …).

Both reuse the existing, tested detectors in `src/detect/` instead of
reimplementing them, so this package is an orchestration/upgrade layer.
"""
from __future__ import annotations

from .director import EditingManifest, generate_manifest
from .scout import EventWindow, scout_events

__all__ = [
    "EventWindow",
    "scout_events",
    "EditingManifest",
    "generate_manifest",
]
