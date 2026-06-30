"""Critic agent — watches the RENDERED clip and judges whether it looks like a
pro vertical highlight, returning machine-applicable feedback.

This is the subjective half of the review loop (the deterministic half is
src/qa). It samples output keyframes and asks the vision-LLM for a rating +
issue tags + concrete suggestions (reduce zoom, widen, drop a shot, trim
slow-mo). When no model is configured it is a no-op (trusts the QA report), so
the studio still runs offline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..utils.io import get_logger
from .llm_client import VisionLLMClient

log = get_logger()

_KNOWN_ISSUES = {"subject_off_frame", "shaky", "crop_jump", "text_unreadable",
                 "text_in_ui", "too_long", "too_short", "boring", "letterbox",
                 "graphics_flicker", "wrong_hero"}

_SYSTEM = (
    "You are a brutally honest reviewer of vertical (9:16) football highlights. "
    "You are shown keyframes of a FINISHED clip. Judge it as a viewer would and "
    "return ONLY JSON:\n"
    '{\n'
    '  "score": 0..1,                     // overall quality\n'
    '  "issues": ["subject_off_frame|shaky|crop_jump|text_unreadable|text_in_ui|'
    'too_long|too_short|boring|letterbox|graphics_flicker|wrong_hero"],\n'
    '  "notes": "one short sentence",\n'
    '  "suggestions": {"reduce_zoom": bool, "widen": bool, '
    '"trim_slowmo": bool, "drop_shots": [int]}\n'
    '}\n'
    "Return strictly valid JSON, no markdown."
)


@dataclass
class CriticReport:
    ok: bool = True
    score: float = 1.0
    issues: list[str] = field(default_factory=list)
    notes: str = ""
    suggestions: dict = field(default_factory=dict)
    source: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


def critique(out_path: str, plan=None, cfg: dict | None = None,
             client: VisionLLMClient | None = None) -> CriticReport:
    """Return a CriticReport for a rendered clip (VLM if configured, else no-op)."""
    cfg = cfg or {}
    client = client or VisionLLMClient(cfg)
    if not client.is_configured():
        return CriticReport(ok=True, score=1.0, notes="no critic backend",
                            source="none")
    try:
        frames = _sample(out_path, cfg)
        text = ("Review this finished 9:16 highlight. "
                + (f"Intended event: {getattr(plan, 'event', '')}. "
                   if plan is not None else "")
                + "Return the review JSON.")
        raw = client.chat_json(_SYSTEM, text, frames)
        rep = _coerce(raw, source=client.backend, cfg=cfg)
        log.info(f"[critic] score={rep.score:.2f} ok={rep.ok} issues={rep.issues}")
        return rep
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[critic] backend failed ({exc}); treating as pass")
        return CriticReport(ok=True, score=1.0, notes=f"critic error: {exc}",
                            source="error")


# --------------------------------------------------------------------------- #
def _coerce(data: dict, source: str, cfg: dict) -> CriticReport:
    q = cfg.get("qa", {})
    thresh = float(q.get("critic_pass_score", 0.7))
    try:
        score = float(data.get("score", 1.0))
    except (TypeError, ValueError):
        score = 1.0
    score = min(max(score, 0.0), 1.0)
    issues = [str(i).strip() for i in (data.get("issues") or [])
              if str(i).strip() in _KNOWN_ISSUES]
    sug = data.get("suggestions") or {}
    sug = sug if isinstance(sug, dict) else {}
    clean_sug = {
        "reduce_zoom": bool(sug.get("reduce_zoom", False)),
        "widen": bool(sug.get("widen", False)),
        "trim_slowmo": bool(sug.get("trim_slowmo", False)),
        "drop_shots": [int(x) for x in (sug.get("drop_shots") or [])
                       if isinstance(x, (int, float))],
    }
    return CriticReport(ok=score >= thresh and not issues, score=score,
                        issues=issues, notes=str(data.get("notes", "") or "")[:200],
                        suggestions=clean_sug, source=source)


def _sample(out_path: str, cfg: dict) -> list[bytes]:
    from ..perception.bundle import _grab_jpegs, _keyframe_times, _probe
    _, duration = _probe(out_path)
    times = _keyframe_times(duration,
                            max_frames=int(cfg.get("qa", {}).get("critic_frames", 8)),
                            peak_t=None)
    return _grab_jpegs(out_path, times)
