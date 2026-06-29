"""Shared detection datatypes."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Signal:
    """A single timestamped cue from one detector."""
    t: float                 # seconds into the match
    source: str              # 'audio' | 'scene' | 'scoreboard_ocr' | 'commentary'
    strength: float          # normalized 0..1
    meta: dict = field(default_factory=dict)


@dataclass
class Moment:
    """A fused highlight candidate."""
    t: float                 # peak time (seconds)
    start: float             # clip start after padding
    end: float               # clip end after padding
    confidence: float        # 0..1
    kind: str                # 'goal' | 'chance' | 'save' | 'skill' | 'card'
    minute: int | None = None
    sources: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
