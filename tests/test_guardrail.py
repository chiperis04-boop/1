"""Factual-guardrail tests (Block D, CPU-only).

The guardrail must never let the renderer burn a SCORE / JERSEY number /
POSSESSION figure the data doesn't support: an unsupported claim is stripped,
and a hook gutted by that check is regenerated from a safe event default.

Run directly:  python tests/test_guardrail.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.editplan import EditPlan                       # noqa: E402
from src.qa.guardrail import (facts_from, guardrail_plan,       # noqa: E402
                              verify_text)


def test_strips_unverified_score_keeps_verified():
    # real score is 2-1; a hook claiming 3-0 is a fabrication -> stripped
    clean, issues, removed = verify_text("GOLAZO 3-0!", {"score_after": "2-1"})
    assert "3-0" not in clean and "unverified_score" in issues, (clean, issues)
    assert removed and "score" in removed[0]
    # a hook stating the real score (either orientation) is kept
    clean2, issues2, _ = verify_text("NOW 1-2!", {"score_after": "2-1"})
    assert "1-2" in clean2 and not issues2, (clean2, issues2)
    print("  \u2713 score: fabricated scoreline stripped, real score kept")


def test_strips_unverified_jersey_number():
    clean, issues, _ = verify_text("LOOK AT #10", {"hero_number": 7})
    assert "#10" not in clean and "unverified_number" in issues, (clean, issues)
    clean2, issues2, _ = verify_text("MAGIC FROM #7", {"hero_number": 7})
    assert "#7" in clean2 and not issues2, (clean2, issues2)
    print("  \u2713 jersey: wrong number stripped, the resolved hero number kept")


def test_strips_unverified_possession():
    # no possession data at all -> any % claim is unsupported
    clean, issues, _ = verify_text("70% DOMINANCE", {})
    assert "70%" not in clean and "unverified_possession" in issues, (clean, issues)
    # a % within tolerance of a measured share is kept
    clean2, issues2, _ = verify_text("68% OF THE BALL",
                                     {"possession": {0: 65, 1: 35}})
    assert "68%" in clean2 and not issues2, (clean2, issues2)
    print("  \u2713 possession: unsupported % stripped, measured share kept")


def test_guardrail_plan_regenerates_gutted_hook():
    """A hook that is ENTIRELY an unverified claim gets regenerated from a safe
    event default (the SynapseQuill 'regenerate' path) — never left blank."""
    plan = EditPlan(event="goal", hook_text="3-0", lower_third="GOAL #9")
    facts = {"score_after": None, "hero_number": 7}       # nothing supports it
    plan, rep = guardrail_plan(plan, facts, cfg={})
    assert rep.changed and "hook_regenerated" in rep.issues, rep.to_dict()
    assert plan.hook_text and "3-0" not in plan.hook_text, plan.hook_text
    # the lower-third's wrong number is stripped too
    assert "#9" not in plan.lower_third, plan.lower_third
    print(f"  \u2713 plan: gutted hook regenerated -> '{plan.hook_text}', "
          "bad number dropped from lower-third")


def test_guardrail_plan_keeps_clean_copy_untouched():
    plan = EditPlan(event="goal", hook_text="WHAT A GOAL!", lower_third="GOAL")
    plan, rep = guardrail_plan(plan, {"score_after": "2-1", "hero_number": 7}, {})
    assert not rep.changed and rep.ok, rep.to_dict()
    assert plan.hook_text == "WHAT A GOAL!"
    print("  \u2713 plan: clean copy passes the guardrail untouched")


def test_facts_from_assembles_known_facts():
    class _W:
        score_after = "3-2"
        verified = True

    class _A:
        hero_number = 11
        def possession_share_pct(self):
            return {0: 60, 1: 40}

    facts = facts_from(_W(), _A())
    assert facts["score_after"] == "3-2" and facts["hero_number"] == 11
    assert facts["possession"] == {0: 60, 1: 40}
    print("  \u2713 facts_from: score + hero number + possession assembled")


def test_review_replans_on_guardrail_issue():
    """Guardrail flagging an invented fact drives a re-plan: feeding its issue
    through a regenerate step yields a materially changed plan (no fabrication
    survives to render)."""
    plan = EditPlan(event="goal", hook_text="UNREAL 4-0 #22")
    before = plan.hook_text
    plan, rep = guardrail_plan(plan, {"score_after": "1-0", "hero_number": 9}, {})
    assert rep.issues and plan.hook_text != before
    # a second pass is now stable (idempotent) — the re-plan converged
    plan2, rep2 = guardrail_plan(plan, {"score_after": "1-0", "hero_number": 9}, {})
    assert not rep2.changed, (plan2.hook_text, rep2.to_dict())
    print("  \u2713 review/re-plan: guardrail issue forces a new plan, then converges")


def main() -> int:
    print("factual-guardrail tests (Block D)")
    for t in (test_strips_unverified_score_keeps_verified,
              test_strips_unverified_jersey_number,
              test_strips_unverified_possession,
              test_guardrail_plan_regenerates_gutted_hook,
              test_guardrail_plan_keeps_clean_copy_untouched,
              test_facts_from_assembles_known_facts,
              test_review_replans_on_guardrail_issue):
        t()
    print("\nALL GUARDRAIL TESTS PASSED \u2705")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
