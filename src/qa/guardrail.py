"""Factual guardrail for on-screen text (SynapseQuill-style claim check).

A vision-LLM Director writes punchy hooks/lower-thirds. Left unchecked it will
happily invent a scoreline, a shirt number or a possession figure the footage
does not support — and burning a WRONG stat onto the video is worse than showing
none. This guardrail is the deterministic fact-check between the Director and the
renderer:

  * it extracts the concrete, verifiable claims from the text — SCORE ("3-0"),
    JERSEY number ("#7" / "number 7") and POSSESSION ("68%") —
  * checks each against the known facts (the score-verified EventWindow, the
    resolved hero number, the measured possession share), and
  * SANITISES the text: an unsupported/contradicted claim is stripped; if that
    empties the hook it is regenerated from a safe, event-based default.

Names are intentionally NOT auto-stripped (reference hooks like "TOP BINS!" are
all-caps common words, so name-matching would mangle good copy) — that limit is
documented rather than guessed. Everything is best-effort and never raises.

Enabled by `qa.use_guardrail`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..utils.io import get_logger

log = get_logger()

_SCORE_RE = re.compile(r"\b(\d{1,2})\s*[-:\u2013]\s*(\d{1,2})\b")
_JERSEY_RE = re.compile(r"(?:#|\bno\.?\s*|\bnumber\s+)(\d{1,2})\b", re.IGNORECASE)
_PCT_RE = re.compile(r"\b(\d{1,3})\s*%")


@dataclass
class GuardrailReport:
    ok: bool = True
    issues: list[str] = field(default_factory=list)     # machine tags
    removed: list[str] = field(default_factory=list)    # human-readable claims dropped
    changed: bool = False

    def to_dict(self) -> dict:
        return {"ok": self.ok, "issues": self.issues, "removed": self.removed,
                "changed": self.changed}


def _norm_score(a: str, b: str) -> tuple[int, int]:
    return int(a), int(b)


def _score_set(score_str) -> set[tuple[int, int]]:
    """All orientations of a known score string ('2-1' -> {(2,1),(1,2)}) so a
    hook may state it either way round."""
    if not score_str:
        return set()
    m = _SCORE_RE.search(str(score_str))
    if not m:
        return set()
    a, b = _norm_score(m.group(1), m.group(2))
    return {(a, b), (b, a)}


def verify_text(text: str, facts: dict) -> tuple[str, list[str], list[str]]:
    """Return (clean_text, issue_tags, removed_claims) for one line of copy.

    facts: {score_after, hero_number, possession (list/dict of ints %),
            possession_tol}
    """
    if not text:
        return text, [], []
    issues: list[str] = []
    removed: list[str] = []
    clean = text

    # ---- SCORE ----
    known_scores = _score_set(facts.get("score_after"))
    def _score_ok(m):
        pair = _norm_score(m.group(1), m.group(2))
        if known_scores and (pair in known_scores):
            return m.group(0)                       # supported -> keep verbatim
        removed.append(f"score '{m.group(0).strip()}'")
        issues.append("unverified_score")
        return ""
    clean = _SCORE_RE.sub(_score_ok, clean)

    # ---- JERSEY NUMBER ----
    hero_num = facts.get("hero_number")
    def _num_ok(m):
        n = int(m.group(1))
        if hero_num is not None and n == int(hero_num):
            return m.group(0)                       # supported -> keep
        removed.append(f"number '{m.group(0).strip()}'")
        issues.append("unverified_number")
        return ""
    clean = _JERSEY_RE.sub(_num_ok, clean)

    # ---- POSSESSION % ----
    poss = facts.get("possession")
    poss_vals = (list(poss.values()) if isinstance(poss, dict)
                 else list(poss) if poss else [])
    tol = float(facts.get("possession_tol", 8))
    def _pct_ok(m):
        v = int(m.group(1))
        if poss_vals and any(abs(v - p) <= tol for p in poss_vals):
            return m.group(0)                       # matches a measured share
        removed.append(f"possession '{m.group(0).strip()}'")
        issues.append("unverified_possession")
        return ""
    clean = _PCT_RE.sub(_pct_ok, clean)

    # tidy leftover double spaces / dangling punctuation from removals
    clean = re.sub(r"\s{2,}", " ", clean)
    clean = re.sub(r"\s+([!?.,:;])", r"\1", clean).strip(" -\u2013:").strip()
    return clean, issues, removed


def guardrail_plan(plan, facts: dict, cfg: dict | None = None):
    """Sanitise an EditPlan's on-screen text against the facts, regenerating the
    hook from a safe event default if the fact-check emptied it. Mutates and
    returns (plan, GuardrailReport). Never raises."""
    cfg = cfg or {}
    rep = GuardrailReport()
    try:
        hook0 = getattr(plan, "hook_text", "") or ""
        lt0 = getattr(plan, "lower_third", "") or ""

        hook, hi, hr = verify_text(hook0, facts)
        lt, li, lr = verify_text(lt0, facts)
        rep.issues = sorted(set(hi) | set(li))
        rep.removed = hr + lr

        # regenerate an emptied/gutted hook from a safe, event-based default
        if hook0 and not hook:
            hook = _safe_hook(plan, facts, cfg)
            rep.issues = sorted(set(rep.issues) | {"hook_regenerated"})

        if hook != hook0:
            plan.hook_text = hook
            rep.changed = True
        if lt != lt0:
            plan.lower_third = lt
            rep.changed = True
        rep.ok = not rep.issues
        if rep.changed:
            log.info(f"[guardrail] sanitised copy (dropped {rep.removed or 'none'}); "
                     f"hook='{plan.hook_text}'")
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[guardrail] skipped ({exc})")
    return plan, rep


def _safe_hook(plan, facts: dict, cfg: dict) -> str:
    """A safe default hook that states no unverifiable number. Prefers a
    configured brand hook for the event, else an event-based phrase."""
    event = getattr(plan, "event", "goal") or "goal"
    hooks = (cfg.get("director", {}).get("hooks", {}) or {})
    pool = hooks.get(event) or hooks.get("goal")
    if pool:
        return str(pool[0]).strip()
    return {"goal": "WHAT A GOAL!", "chance": "SO CLOSE!", "save": "WHAT A SAVE!",
            "skill": "WATCH THIS!", "card": "OFF!", "buildup": "WATCH THIS!"
            }.get(event, "WATCH THIS!")


def facts_from(window, analytics) -> dict:
    """Assemble the known facts for the guardrail from the EventWindow + analytics."""
    facts: dict = {}
    if window is not None:
        facts["score_after"] = getattr(window, "score_after", None)
        facts["verified"] = bool(getattr(window, "verified", False))
    if analytics is not None:
        facts["hero_number"] = getattr(analytics, "hero_number", None)
        try:
            share = analytics.possession_share_pct()
            if share:
                facts["possession"] = share
        except Exception:  # noqa: BLE001
            pass
    return facts
