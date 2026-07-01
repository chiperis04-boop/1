"""Editing agents (v3) — the frame-aware Director and (later) Critic.

The Director watches a PerceptionBundle (shots + keyframes + ASR transcript +
detection summary) and emits an EditPlan: the full set of editorial decisions
(moment curation, cut in/out, per-shot framing/zoom, slow-mo beats, hero, hook,
pacing). Deterministic code (Cameraman/Composer) then executes that plan.

See docs/IMPLEMENTATION_PLAN_AI_DIRECTOR.md. Everything degrades to an offline
heuristic so the studio runs with no model/network.
"""
from .editplan import EditPlan, ShotEdit, SlowmoBeat, heuristic_plan

__all__ = ["EditPlan", "ShotEdit", "SlowmoBeat", "heuristic_plan"]
