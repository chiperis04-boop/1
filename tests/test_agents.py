"""Agent-layer tests (pure Python, no GPU, no network).

Covers the frame-aware Director plumbing:
  * EditPlan.coerce      : garbage -> safe defaults; valid -> parsed + clamped
  * heuristic_plan       : offline plan (shots kept/dropped, a slow-mo beat, hook)
  * EditPlan.to_manifest : bridge to the existing Composer manifest
  * llm_client.parse_json: tolerant JSON extraction (fences / prose / non-dict)
  * director_agent.plan_edit : heuristic when unconfigured; uses a (mock) vision
    client when configured; falls back if the client raises

Run directly:  python tests/test_agents.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.director_agent import plan_edit                 # noqa: E402
from src.agents.editplan import EditPlan, ShotEdit, heuristic_plan  # noqa: E402
from src.agents.llm_client import parse_json                    # noqa: E402
from src.perception.bundle import PerceptionBundle              # noqa: E402
from src.perception.shots import Shot                           # noqa: E402


# --------------------------------------------------------------------------- #
class _Window:
    def __init__(self, kind="goal", verified=False, anchor_t=8.0, start=2.0,
                 confidence=0.7, score_after=None, score_before=None, minute=None):
        self.kind = kind; self.verified = verified; self.anchor_t = anchor_t
        self.start = start; self.confidence = confidence
        self.score_after = score_after; self.score_before = score_before
        self.minute = minute


def _bundle(duration=12.0, n_shots=3, replay_last=True):
    shots = []
    seg = duration / n_shots
    for i in range(n_shots):
        shots.append(Shot(idx=i, start=i * seg, end=(i + 1) * seg,
                          start_frame=int(i * seg * 30),
                          end_frame=int((i + 1) * seg * 30),
                          is_replay=(replay_last and i == n_shots - 1)))
    return PerceptionBundle(clip_path="x.mp4", fps=30.0, duration=duration,
                            shots=shots, detections_summary="3 shots; ball 80%")


class _FakeClient:
    backend = "vllm"

    def __init__(self, reply=None, configured=True, raise_exc=False):
        self._reply = reply or {}
        self._configured = configured
        self._raise = raise_exc

    def is_configured(self):
        return self._configured

    def chat_json(self, system, text, images):
        if self._raise:
            raise RuntimeError("model exploded")
        return self._reply


# --------------------------------------------------------------------------- #
def test_editplan_coerce_clamps_and_defaults():
    raw = {
        "keep_clip": True, "importance": 5.0,          # clamp -> 1.0
        "event": "wondergoal",                         # invalid -> goal
        "cut_in": -3, "cut_out": 999,                  # clamp to [0,duration]
        "shots": [{"shot_idx": 0, "keep": True, "framing": "zoomzoom",
                   "zoom": 9, "subject": "alien"}],    # invalid -> defaults
        "slowmo_beats": [{"start": 2, "end": 1},       # end<=start -> dropped
                         {"start": 3, "end": 5, "factor": 0.05}],  # clamp 0.1
        "hook_text": "WHAT A HIT", "hashtags": ["#goal", "  ", "#fc"],
        "pacing": "zen", "music_energy": "ultra",
    }
    p = EditPlan.coerce(raw, source="vllm", duration=10.0)
    assert p.importance == 1.0
    assert p.event == "goal"
    assert p.cut_in == 0.0 and p.cut_out == 10.0
    assert p.shots[0].framing == "crop_follow" and p.shots[0].subject == "hero"
    assert p.shots[0].zoom == 2.5                      # clamped to max
    assert len(p.slowmo_beats) == 1 and p.slowmo_beats[0].factor == 0.1
    assert p.pacing == "punchy" and p.music_energy == "build"
    assert p.hashtags == ["#goal", "#fc"]
    print("  ✓ EditPlan.coerce: invalid values clamped/defaulted, beats validated")


def test_editplan_to_manifest_bridge():
    p = EditPlan(event="save", hero_description="GK in green",
                 hook_text="DENIED!",
                 slowmo_beats=[__import__("src.agents.editplan", fromlist=["SlowmoBeat"])
                               .SlowmoBeat(start=4.0, end=7.0, factor=0.4)],
                 source="vllm")
    m = p.to_manifest()
    assert m.event_detected == "save" and m.video_hook_text == "DENIED!"
    assert m.slomo_trigger_timestamp == 4.0 and m.slomo_duration == 3.0
    assert m.source == "vllm"
    print("  ✓ EditPlan.to_manifest: bridges to the Composer's EditingManifest")


def test_heuristic_plan_offline():
    b = _bundle(duration=12.0, n_shots=3, replay_last=True)
    w = _Window(kind="goal", anchor_t=8.0, start=2.0, confidence=0.8)
    p = heuristic_plan(b, w, cfg={"edit": {"effects": {"slowmo_window": 3.0,
                                                       "slowmo_factor": 0.4}}})
    assert p.source == "heuristic" and p.event == "goal"
    assert len(p.shots) == 3
    assert p.shots[-1].keep is False               # replay shot dropped
    assert p.kept_shots() and len(p.kept_shots()) == 2
    assert len(p.slowmo_beats) == 1                # one decisive beat
    assert p.hook_text                              # a hook was chosen
    print("  ✓ heuristic_plan: offline plan keeps real shots, drops replay, 1 beat")


def test_parse_json_tolerant():
    assert parse_json('{"a":1}')["a"] == 1
    assert parse_json('```json\n{"a":2}\n```')["a"] == 2
    assert parse_json('Sure! {"a":3} hope that helps')["a"] == 3
    for bad in ("", "[1,2,3]", "not json"):
        try:
            parse_json(bad)
            raise AssertionError(f"should have raised for {bad!r}")
        except (ValueError, Exception):
            pass
    print("  ✓ parse_json: handles fences/prose, rejects non-objects")


def test_plan_edit_uses_mock_vision_client():
    b = _bundle(duration=12.0)
    w = _Window(kind="chance", anchor_t=8.0, start=2.0)
    reply = {
        "keep_clip": True, "importance": 0.9, "event": "goal",
        "cut_in": 1.0, "cut_out": 11.0,
        "shots": [{"shot_idx": 0, "keep": True, "framing": "punch_in",
                   "zoom": 1.3, "subject": "hero"}],
        "slowmo_beats": [{"start": 7.0, "end": 10.0, "factor": 0.35}],
        "hook_text": "TOP BINS!", "hero_description": "blue #10",
        "pacing": "punchy", "music_energy": "high",
    }
    p = plan_edit(b, w, cfg={}, client=_FakeClient(reply=reply))
    assert p.source == "vllm" and p.event == "goal"
    assert p.shots[0].framing == "punch_in" and abs(p.shots[0].zoom - 1.3) < 1e-6
    assert p.slowmo_beats[0].start == 7.0 and p.hook_text == "TOP BINS!"
    print("  ✓ plan_edit: consumes a (mock) vision-LLM reply into an EditPlan")


def test_plan_edit_falls_back_on_error_and_when_unconfigured():
    b = _bundle()
    w = _Window(kind="goal")
    # client raises -> heuristic
    p1 = plan_edit(b, w, cfg={}, client=_FakeClient(raise_exc=True))
    assert p1.source == "heuristic"
    # client not configured -> heuristic (never even called)
    p2 = plan_edit(b, w, cfg={}, client=_FakeClient(configured=False))
    assert p2.source == "heuristic"
    # default (no client, cfg backend defaults to heuristic) -> heuristic
    p3 = plan_edit(b, w, cfg={})
    assert p3.source == "heuristic"
    print("  ✓ plan_edit: falls back to heuristic on error / unconfigured / default")


def test_critic_coerce_and_no_backend():
    from src.agents.critic import CriticReport, _coerce, critique
    # mock reply coercion: unknown issues filtered, suggestions cleaned, ok=False
    rep = _coerce({"score": 0.4, "issues": ["crop_jump", "aliens"],
                   "notes": "ball leaves frame",
                   "suggestions": {"reduce_zoom": True, "drop_shots": [1, "x"]}},
                  source="vllm", cfg={})
    assert rep.ok is False and rep.score == 0.4
    assert rep.issues == ["crop_jump"]                       # 'aliens' dropped
    assert rep.suggestions["reduce_zoom"] is True and rep.suggestions["drop_shots"] == [1]
    # no configured backend -> no-op pass (never touches the file)
    out = critique("nonexistent.mp4", plan=None, cfg={})
    assert out.ok is True and out.source == "none"
    print("  ✓ critic: reply coerced (unknown issues dropped); no backend -> no-op pass")


def test_apply_corrections_variants():
    from src.agents.critic import CriticReport
    from src.agents.editplan import EditPlan, ShotEdit, SlowmoBeat
    from src.agents.review import apply_corrections
    from src.qa.checks import QAReport
    plan = EditPlan(
        shots=[ShotEdit(0, zoom=1.0, framing="letterbox_wide"),
               ShotEdit(1, zoom=1.5)],
        slowmo_beats=[SlowmoBeat(2.0, 8.0, 0.4)])
    qa = QAReport(score=0.4, passed=False, issues=["letterbox", "too_long"])
    crit = CriticReport(ok=False, score=0.5, issues=["crop_jump"],
                        suggestions={"drop_shots": [1]})
    p2, changed = apply_corrections(plan, qa, crit, cfg={})
    assert changed
    assert p2.shots[0].framing == "crop_follow"      # letterbox -> crop
    assert p2.shots[1].zoom < 1.5                    # crop_jump relaxes zoom
    assert p2.shots[1].keep is False                 # critic dropped shot 1
    assert p2.slowmo_beats[0].end < 8.0              # too_long trims slow-mo
    print("  ✓ apply_corrections: letterbox/zoom/slowmo/drop-shot edits applied")


def test_review_loop_revises_until_pass():
    from src.agents.editplan import EditPlan, ShotEdit
    from src.agents.review import review_and_revise
    from src.qa.checks import QAReport
    plan = EditPlan(shots=[ShotEdit(0, zoom=2.0)], source="heuristic")
    rendered_zoom: list[float] = []

    def render_fn(p):
        rendered_zoom.append(max((s.zoom for s in p.shots), default=1.0))
        return f"out_{len(rendered_zoom)}.mp4"

    def qa_fn(path):
        z = rendered_zoom[-1]
        if z > 1.3:
            return QAReport(score=0.5, passed=False, issues=["crop_jump"])
        return QAReport(score=0.95, passed=True, issues=[])

    res = review_and_revise(plan, render_fn=render_fn, qa_fn=qa_fn,
                            cfg={"qa": {"zoom_relax": 0.6}}, max_revisions=3)
    assert res.qa.passed, (res.score, [s.zoom for s in res.plan.shots])
    assert max(s.zoom for s in res.plan.shots) <= 1.3
    assert res.attempts >= 2                          # at least one revision
    print(f"  ✓ review loop: revised zoom 2.0 -> "
          f"{max(s.zoom for s in res.plan.shots)} until QA passed "
          f"({res.attempts} attempts)")


def main() -> int:
    print("agent-layer tests (no GPU/network)")
    for t in (test_editplan_coerce_clamps_and_defaults,
              test_editplan_to_manifest_bridge,
              test_heuristic_plan_offline,
              test_parse_json_tolerant,
              test_plan_edit_uses_mock_vision_client,
              test_plan_edit_falls_back_on_error_and_when_unconfigured,
              test_critic_coerce_and_no_backend,
              test_apply_corrections_variants,
              test_review_loop_revises_until_pass):
        t()
    print("\nALL AGENT TESTS PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
