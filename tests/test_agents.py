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


def main() -> int:
    print("agent-layer tests (no GPU/network)")
    for t in (test_editplan_coerce_clamps_and_defaults,
              test_editplan_to_manifest_bridge,
              test_heuristic_plan_offline,
              test_parse_json_tolerant,
              test_plan_edit_uses_mock_vision_client,
              test_plan_edit_falls_back_on_error_and_when_unconfigured):
        t()
    print("\nALL AGENT TESTS PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
