"""Quality-assurance layer (v3) — the deterministic half of the review loop.

`qa_report()` watches the RENDERED output and reports objective problems
(black bars, dead/frozen frames, loudness off-target, wrong duration/streams)
as a machine-readable QAReport. The Critic agent adds the subjective "does it
look good" judgement; together they drive bounded revisions of the EditPlan.
"""
from .checks import QAReport, qa_report

__all__ = ["QAReport", "qa_report"]
