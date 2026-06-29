"""Scene / replay detector.

After a goal, broadcasts cut to tightly-edited slow-motion replays. A burst of
short scenes is a strong secondary cue. We use PySceneDetect on the proxy video
and emit a signal at the *start* of any cluster of short cuts (a likely replay
sequence).
"""
from __future__ import annotations

from ..utils.io import get_logger
from .types import Signal

log = get_logger()


def detect_scenes(proxy_path: str, cfg: dict) -> list[Signal]:
    s = cfg["detect"]["scene"]
    if not s.get("enabled", True):
        return []

    from scenedetect import ContentDetector, AdaptiveDetector, SceneManager, open_video

    video = open_video(proxy_path)
    sm = SceneManager()
    if s["detector"] == "adaptive":
        sm.add_detector(AdaptiveDetector())
    else:
        sm.add_detector(ContentDetector(threshold=s["threshold"]))
    sm.detect_scenes(video, frame_skip=int(s.get("frame_skip", 0)),
                     show_progress=False)
    scenes = sm.get_scene_list()

    # find clusters of short scenes => replay packages
    replay_max = s["replay_max_seconds"]
    signals: list[Signal] = []
    run_start = None
    run_count = 0
    for start, end in scenes:
        dur = (end - start).get_seconds()
        if dur <= replay_max:
            if run_start is None:
                run_start = start.get_seconds()
            run_count += 1
        else:
            if run_count >= 3 and run_start is not None:
                signals.append(Signal(t=run_start, source="scene",
                                      strength=min(1.0, run_count / 6.0),
                                      meta={"short_scenes": run_count}))
            run_start, run_count = None, 0
    if run_count >= 3 and run_start is not None:
        signals.append(Signal(t=run_start, source="scene",
                              strength=min(1.0, run_count / 6.0),
                              meta={"short_scenes": run_count}))

    log.info(f"[scene] {len(scenes)} scenes, {len(signals)} replay clusters")
    return signals
