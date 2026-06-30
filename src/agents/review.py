"""Review loop — render, judge (QA + Critic), correct the EditPlan, repeat.

`review_and_revise` is pure orchestration (no ffmpeg/model knowledge) so it is
fully unit-testable with fakes: it renders a plan, scores the output with the
deterministic QA and the optional vision Critic, and if the result is sub-par it
asks `apply_corrections` for a concrete plan tweak and re-renders — up to a
bounded number of revisions, keeping the best result.

`apply_corrections` maps machine-readable issues (from QA and the Critic) to
deterministic EditPlan edits: relax zoom on crop-jump/shaky, widen/letterbox->
crop, trim slow-mo when too long, drop shots the Critic rejects.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..utils.io import get_logger

log = get_logger()


@dataclass
class ReviewResult:
    path: str
    plan: object
    qa: object
    critic: object = None
    score: float = 0.0
    attempts: int = 1
    history: list = field(default_factory=list)


def combined_score(qa, critic) -> float:
    qs = float(getattr(qa, "score", 1.0)) if qa is not None else 1.0
    if critic is None:
        return qs
    cs = float(getattr(critic, "score", 1.0))
    return round(0.5 * qs + 0.5 * cs, 4)


def _good(qa, critic) -> bool:
    return (qa is None or getattr(qa, "passed", True)) and \
           (critic is None or getattr(critic, "ok", True))


def review_and_revise(plan, *, render_fn, qa_fn, cfg=None, critic_fn=None,
                      max_revisions: int = 1) -> ReviewResult:
    """Render -> judge -> correct -> re-render (<= max_revisions), keep the best.

    render_fn(plan) -> output_path
    qa_fn(path)     -> QAReport
    critic_fn(path, plan) -> CriticReport   (optional)
    """
    cfg = cfg or {}
    cur = plan
    best: ReviewResult | None = None
    history = []
    for attempt in range(max_revisions + 1):
        path = render_fn(cur)
        qa = qa_fn(path)
        critic = critic_fn(path, cur) if critic_fn else None
        score = combined_score(qa, critic)
        history.append({"attempt": attempt, "path": path, "score": score,
                        "qa_issues": list(getattr(qa, "issues", []) or []),
                        "critic_issues": list(getattr(critic, "issues", []) or [])})
        if best is None or score > best.score:
            best = ReviewResult(path=path, plan=cur, qa=qa, critic=critic,
                                score=score, attempts=attempt + 1)
        if _good(qa, critic):
            break
        if attempt < max_revisions:
            new_plan, changed = apply_corrections(cur, qa, critic, cfg)
            if not changed:
                log.info("[review] no further corrections available; stopping")
                break
            log.info(f"[review] attempt {attempt + 1} score {score:.2f}; revising plan")
            cur = new_plan
    best.history = history
    best.attempts = len(history)
    return best


# --------------------------------------------------------------------------- #
def apply_corrections(plan, qa, critic, cfg=None) -> tuple[object, bool]:
    """Return (possibly revised plan copy, changed?) from QA + Critic feedback."""
    cfg = cfg or {}
    q = cfg.get("qa", {})
    issues = set(getattr(qa, "issues", []) or []) | set(getattr(critic, "issues", []) or [])
    sug = getattr(critic, "suggestions", {}) or {}
    p = copy.deepcopy(plan)
    changed = False

    relax = float(q.get("zoom_relax", 0.8))

    # crop-jump / shaky / explicit reduce_zoom -> relax per-shot zoom toward 1.0
    if issues & {"crop_jump", "shaky", "subject_off_frame"} or sug.get("reduce_zoom"):
        for s in getattr(p, "shots", []):
            if getattr(s, "zoom", 1.0) > 1.0:
                s.zoom = max(1.0, round(s.zoom * relax, 3))
                changed = True

    # widen -> go below 1.0 (show more pitch)
    if sug.get("widen"):
        for s in getattr(p, "shots", []):
            new = round(min(getattr(s, "zoom", 1.0), 0.85), 3)
            if new != s.zoom:
                s.zoom = new
                changed = True

    # letterbox bars -> switch those shots to a following crop
    if "letterbox" in issues:
        for s in getattr(p, "shots", []):
            if getattr(s, "framing", "") == "letterbox_wide":
                s.framing = "crop_follow"
                changed = True

    # too long / trim_slowmo -> shorten + speed up the slow-mo beats
    if "too_long" in issues or sug.get("trim_slowmo"):
        new_beats = []
        for b in getattr(p, "slowmo_beats", []):
            dur = (b.end - b.start) * 0.6
            b.end = b.start + max(0.4, dur)
            b.factor = min(1.0, round(b.factor * 1.3, 3))
            new_beats.append(b)
        if new_beats:
            p.slowmo_beats = new_beats
            changed = True

    # Critic asked to drop specific shots
    for idx in sug.get("drop_shots", []) or []:
        for s in getattr(p, "shots", []):
            if getattr(s, "shot_idx", None) == idx and s.keep:
                s.keep = False
                changed = True

    if changed:
        p.source = f"{getattr(plan, 'source', 'plan')}+revised"
    return p, changed
