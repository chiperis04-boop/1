"""Director agent — watches a PerceptionBundle and returns an EditPlan.

If a vision-LLM backend is configured (local vLLM serving Qwen2-VL / MiniCPM-V,
or Gemini), the agent sends the clip's keyframes + a compact textual context
(shots, detection summary, commentary transcript, scoreboard) and asks for the
full editorial plan as JSON. Anything missing/malformed degrades to the offline
heuristic, so the studio always produces a plan.

This is the component that makes the studio *understand the shot* instead of
acting on fixed presets — see docs/IMPLEMENTATION_PLAN_AI_DIRECTOR.md.
"""
from __future__ import annotations

from ..utils.io import get_logger
from .editplan import EditPlan, heuristic_plan
from .llm_client import VisionLLMClient

log = get_logger()

_SYSTEM = (
    "You are the creative director of a viral football highlights channel. "
    "You are shown keyframes (some sampled densely around the decisive beat) "
    "from ONE short clip, plus context (shots/cuts, detections, commentary, "
    "score). Decide the full edit and return ONLY a JSON object:\n"
    '{\n'
    '  "keep_clip": bool,                  // false if this is not a real highlight\n'
    '  "importance": 0..1,                 // how strong a highlight it is\n'
    '  "event": "goal|chance|save|skill|card|buildup",\n'
    '  "cut_in": seconds, "cut_out": seconds,  // trim to the meaningful play\n'
    '  "shots": [ {"shot_idx":int, "keep":bool, '
    '"framing":"punch_in|crop_follow|letterbox_wide", "zoom":0.6..2.5, '
    '"subject":"hero|ball|wide"} ],   // drop replays/irrelevant shots\n'
    '  "hero_description": "jersey colour + number if visible",\n'
    '  "hero_jersey_shot": int|null,       // shot index with the best close-up to read the number\n'
    '  "slowmo_beats": [ {"start":s,"end":s,"factor":0.1..1.0} ], // the decisive instants\n'
    '  "hook_text": "ALL-CAPS punchy hook, max 7 words",\n'
    '  "lower_third": "short on-screen line",\n'
    '  "caption": "social caption", "hashtags": ["#..."],\n'
    '  "pacing": "punchy|cinematic", "music_energy": "low|build|high"\n'
    '}\n'
    "Return strictly valid JSON, no markdown, no commentary."
)


def plan_edit(bundle, window=None, cfg: dict | None = None, track=None,
              client: VisionLLMClient | None = None) -> EditPlan:
    """Return an EditPlan for one clip (vision-LLM if configured, else heuristic)."""
    cfg = cfg or {}
    client = client or VisionLLMClient(cfg)

    if not client.is_configured():
        return heuristic_plan(bundle, window, cfg, track)

    try:
        text = _context(bundle, window)
        raw = client.chat_json(_SYSTEM, text, getattr(bundle, "keyframes", []))
        plan = EditPlan.coerce(raw, source=client.backend,
                               duration=float(getattr(bundle, "duration", 0.0) or 0.0))
        plan = _post(plan, bundle, window, cfg, track)
        log.info(f"[director] EditPlan via {client.backend}: event={plan.event} "
                 f"keep={plan.keep_clip} hook='{plan.hook_text}' "
                 f"beats={len(plan.slowmo_beats)} shots={len(plan.shots)}")
        return plan
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[director] vision backend failed ({exc}); heuristic plan")
        return heuristic_plan(bundle, window, cfg, track)


# --------------------------------------------------------------------------- #
def _context(bundle, window) -> str:
    bits = []
    if window is not None:
        bits.append(f"Detector guess: '{getattr(window, 'kind', 'goal')}'.")
        if getattr(window, "verified", False) and getattr(window, "score_after", None):
            bits.append(f"Scoreboard confirms {window.score_before} -> "
                        f"{window.score_after}.")
        if getattr(window, "minute", None):
            bits.append(f"Match minute ~{window.minute}'.")
    dur = float(getattr(bundle, "duration", 0.0) or 0.0)
    bits.append(f"Clip duration {dur:.1f}s.")
    summary = getattr(bundle, "detections_summary", "")
    if summary:
        bits.append(f"Detections: {summary}.")
    shots = getattr(bundle, "shots", []) or []
    if shots:
        sl = "; ".join(f"#{s.idx} {s.start:.1f}-{s.end:.1f}s"
                       + ("(replay-like)" if getattr(s, "is_replay", False) else "")
                       for s in shots)
        bits.append(f"Shots: {sl}.")
    tx = bundle.transcript_text() if hasattr(bundle, "transcript_text") else ""
    if tx:
        bits.append(f"Commentary: \"{tx[:400]}\".")
    bits.append("Return the editing plan JSON.")
    return " ".join(bits)


def _post(plan: EditPlan, bundle, window, cfg, track) -> EditPlan:
    """Fill gaps / enforce invariants on a model-produced plan."""
    fallback = heuristic_plan(bundle, window, cfg, track)
    # never trust the model on event type when OCR already verified a goal
    if window is not None and getattr(window, "verified", False):
        plan.event = "goal"
    if not plan.shots:
        plan.shots = fallback.shots
    if not plan.slowmo_beats:
        plan.slowmo_beats = fallback.slowmo_beats
    if not plan.hook_text:
        plan.hook_text = fallback.hook_text
    if not plan.hero_description:
        plan.hero_description = fallback.hero_description
    dur = float(getattr(bundle, "duration", 0.0) or 0.0)
    if plan.cut_out <= 0 and dur:
        plan.cut_out = dur
    return plan
