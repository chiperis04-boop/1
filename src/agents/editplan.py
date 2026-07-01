"""EditPlan — the structured editorial decision the Director hands to the
deterministic renderer.

This replaces the 6-field EditingManifest as the source of truth. It is always
*validated and coerced* (a malformed LLM reply degrades to safe defaults), and
it can produce an `EditingManifest` so the existing Composer keeps working while
the pipeline migrates.

`heuristic_plan()` builds a sensible EditPlan with no model/network, so the
studio runs fully offline (clearly the "blind" fallback, not the quality path).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..detection.director import (EditingManifest, _default_hook,
                                   _describe_hero, _peak_ball_speed_time)
from ..utils.io import get_logger

log = get_logger()

_EVENTS = {"goal", "chance", "save", "skill", "card", "buildup"}
_FRAMINGS = {"punch_in", "crop_follow", "letterbox_wide"}
_SUBJECTS = {"hero", "ball", "wide"}
_PACING = {"punchy", "cinematic"}
_ENERGY = {"low", "build", "high"}


@dataclass
class SlowmoBeat:
    start: float
    end: float
    factor: float = 0.4

    @classmethod
    def coerce(cls, d: dict) -> "SlowmoBeat | None":
        try:
            start = max(0.0, float(d.get("start", 0.0)))
            end = float(d.get("end", start))
            factor = float(d.get("factor", 0.4))
        except (TypeError, ValueError):
            return None
        if end <= start:
            return None
        factor = min(max(factor, 0.1), 1.0)
        return cls(start=start, end=end, factor=factor)


@dataclass
class ShotEdit:
    shot_idx: int
    keep: bool = True
    framing: str = "crop_follow"      # punch_in | crop_follow | letterbox_wide
    zoom: float = 1.0                 # >1 = tighter crop; <1 = wider (more pitch)
    subject: str = "hero"             # hero | ball | wide

    @classmethod
    def coerce(cls, d: dict, idx_default: int = 0) -> "ShotEdit":
        try:
            idx = int(d.get("shot_idx", idx_default))
        except (TypeError, ValueError):
            idx = idx_default
        framing = str(d.get("framing", "crop_follow")).lower().strip()
        if framing not in _FRAMINGS:
            framing = "crop_follow"
        subject = str(d.get("subject", "hero")).lower().strip()
        if subject not in _SUBJECTS:
            subject = "hero"
        try:
            zoom = float(d.get("zoom", 1.0))
        except (TypeError, ValueError):
            zoom = 1.0
        zoom = min(max(zoom, 0.6), 2.5)
        return cls(shot_idx=idx, keep=bool(d.get("keep", True)),
                   framing=framing, zoom=zoom, subject=subject)


@dataclass
class EditPlan:
    keep_clip: bool = True
    importance: float = 0.5
    event: str = "goal"
    cut_in: float = 0.0
    cut_out: float = 0.0              # 0 => to end of clip
    shots: list[ShotEdit] = field(default_factory=list)
    hero_description: str = ""
    hero_jersey_shot: int | None = None
    slowmo_beats: list[SlowmoBeat] = field(default_factory=list)
    hook_text: str = ""
    lower_third: str = ""
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    pacing: str = "punchy"
    music_energy: str = "build"
    # review-loop caption-safety corrections (set by apply_corrections when the
    # Critic flags overlapping/unreadable text) — honoured by the renderer.
    caption_safe: bool = False          # force the hook below the scoreboard band
    hook_scale_mult: float = 1.0        # shrink the hook on a text-overlap revision
    source: str = "heuristic"

    def to_dict(self) -> dict:
        return asdict(self)

    # --- bridge so the existing Composer (manifest-based) keeps working ---
    def to_manifest(self) -> EditingManifest:
        beat = self.slowmo_beats[0] if self.slowmo_beats else None
        return EditingManifest(
            event_detected=self.event,
            main_hero_yolo_cls="player",
            main_hero_description=self.hero_description or "the player on the ball",
            video_hook_text=self.hook_text,
            slomo_trigger_timestamp=float(beat.start) if beat else 0.0,
            slomo_duration=float(beat.end - beat.start) if beat else 3.0,
            source=self.source,
        )

    def kept_shots(self) -> list[ShotEdit]:
        return [s for s in self.shots if s.keep]

    # ------------------------------------------------------------------ coerce
    @classmethod
    def coerce(cls, data: dict, source: str, duration: float = 0.0) -> "EditPlan":
        """Validate + coerce a raw (LLM) dict into a safe EditPlan."""
        p = cls(source=source)
        if not isinstance(data, dict):
            return p
        p.keep_clip = bool(data.get("keep_clip", True))
        p.event = _coerce_choice(data.get("event"), _EVENTS, "goal")
        p.pacing = _coerce_choice(data.get("pacing"), _PACING, "punchy")
        p.music_energy = _coerce_choice(data.get("music_energy"), _ENERGY, "build")
        p.importance = _coerce_float(data.get("importance"), 0.5, 0.0, 1.0)
        p.cut_in = max(0.0, _coerce_float(data.get("cut_in"), 0.0))
        p.cut_out = max(0.0, _coerce_float(data.get("cut_out"), 0.0))
        if duration > 0 and p.cut_out <= 0:
            p.cut_out = duration
        if duration > 0:
            p.cut_in = min(p.cut_in, max(0.0, duration - 0.2))
            p.cut_out = min(p.cut_out, duration) if p.cut_out else duration
        if p.cut_out and p.cut_out <= p.cut_in:
            p.cut_out = duration or (p.cut_in + 1.0)
        p.hero_description = str(data.get("hero_description", "") or "").strip()
        p.hook_text = str(data.get("hook_text", "") or "").strip()
        p.lower_third = str(data.get("lower_third", "") or "").strip()
        p.caption = str(data.get("caption", "") or "").strip()
        hjs = data.get("hero_jersey_shot")
        p.hero_jersey_shot = int(hjs) if isinstance(hjs, (int, float)) else None
        tags = data.get("hashtags") or []
        if isinstance(tags, list):
            p.hashtags = [str(t).strip() for t in tags if str(t).strip()][:12]
        shots = data.get("shots") or []
        if isinstance(shots, list):
            p.shots = [ShotEdit.coerce(s, i) for i, s in enumerate(shots)
                       if isinstance(s, dict)]
        beats = data.get("slowmo_beats") or []
        if isinstance(beats, list):
            coerced = [SlowmoBeat.coerce(b) for b in beats if isinstance(b, dict)]
            p.slowmo_beats = [b for b in coerced if b is not None]
        return p


def _coerce_choice(v, allowed: set[str], default: str) -> str:
    s = str(v or "").lower().strip()
    return s if s in allowed else default


def _coerce_float(v, default: float, lo: float | None = None,
                  hi: float | None = None) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if lo is not None:
        f = max(lo, f)
    if hi is not None:
        f = min(hi, f)
    return f


# --------------------------------------------------------------------------- #
# offline heuristic plan (no model / no network) — the "blind" fallback
# --------------------------------------------------------------------------- #
def heuristic_plan(bundle, window=None, cfg: dict | None = None,
                   track=None) -> EditPlan:
    """Build a sensible EditPlan from geometry + the EventWindow, no LLM.

    `bundle` is a PerceptionBundle (or None); `track` is an optional tracking
    object used to time the slow-mo beat from the ball's fastest moment.
    """
    cfg = cfg or {}
    kind = getattr(window, "kind", "goal") if window else "goal"
    if kind not in _EVENTS:
        kind = "goal"
    duration = float(getattr(bundle, "duration", 0.0) or 0.0)
    shots = list(getattr(bundle, "shots", []) or [])

    p = EditPlan(source="heuristic", event=kind)
    p.keep_clip = True
    p.importance = float(getattr(window, "confidence", 0.5) or 0.5)
    p.cut_in = 0.0
    p.cut_out = duration

    # per-shot edits: keep non-replay shots, follow the action, normal zoom
    p.shots = [ShotEdit(shot_idx=getattr(s, "idx", i),
                        keep=not getattr(s, "is_replay", False),
                        framing="crop_follow", zoom=1.0, subject="hero")
               for i, s in enumerate(shots)]

    # one slow-mo beat at the decisive instant (peak ball speed if we have it)
    trigger = None
    if track is not None:
        trigger = _peak_ball_speed_time(track)
        p.hero_description = _describe_hero(track)
    if trigger is None and window is not None:
        trigger = max(0.0, (getattr(window, "anchor_t", 0.0)
                            - getattr(window, "start", 0.0)) - 0.5)
    if trigger is None:
        trigger = duration * 0.6 if duration else 0.0
    beat_dur = float(cfg.get("edit", {}).get("effects", {})
                     .get("slowmo_window", 3.0))
    beat_end = trigger + beat_dur
    if duration:
        beat_end = min(beat_end, duration)
    if beat_end > trigger:
        factor = float(cfg.get("edit", {}).get("effects", {})
                       .get("slowmo_factor", 0.4))
        p.slowmo_beats = [SlowmoBeat(start=float(trigger), end=float(beat_end),
                                     factor=factor)]

    if not p.hero_description:
        p.hero_description = "the player on the ball"
    p.hook_text = _default_hook(kind, window, cfg)
    p.pacing = "punchy"
    p.music_energy = "high" if (kind in ("goal", "skill")
                                or p.importance >= 0.8) else "build"
    return p
