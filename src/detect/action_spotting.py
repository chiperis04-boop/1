"""Action-spotting detector (optional, highest-precision event signal).

Where the audio/scene/OCR/commentary detectors infer "something happened", a
learned action-spotting model localises *named* events (goal, shot, card,
corner, …) directly from the video. When configured, each spotted action becomes
a high-weight `Signal`, so fusion can say "model spotted GOAL at 41:12" instead
of relying on a loud crowd.

This is a thin adapter so you can plug in any of the SoccerNet-family tools
without touching the rest of the pipeline:

  * `oslactionspotting` — pip-installable, unifies several algorithms
    https://pypi.org/project/oslactionspotting/
  * `recokick/ball-action-spotting` — 1st place SoccerNet 2023
    https://github.com/recokick/ball-action-spotting
  * `arturxe2/ASTRA` — transformer spotter
    https://github.com/arturxe2/ASTRA
  * `SoccerNet/sn-spotting` — task, baselines, labels
    https://github.com/SoccerNet/sn-spotting

If no backend/model is configured it is a graceful no-op and the other detectors
carry on. Implement `_spot_with_<backend>` for the tool you choose; it must
return a list of (timestamp_seconds, label, confidence).
"""
from __future__ import annotations

from ..utils.io import get_logger
from .types import Signal

log = get_logger()

# map model label vocab -> our moment kinds
_DEFAULT_LABEL_MAP = {
    "goal": "goal",
    "shot": "chance",
    "shots on target": "chance",
    "shots off target": "chance",
    "save": "save",
    "penalty": "goal",
    "yellow card": "card",
    "red card": "card",
    "direct free-kick": "chance",
    "corner": "chance",
}


def detect_actions(video_path: str, cfg: dict) -> list[Signal]:
    a = cfg.get("detect", {}).get("action_spotting", {})
    if not a.get("enabled", False) or not video_path:
        return []

    backend = a.get("backend", "oslactionspotting")
    label_map = {**_DEFAULT_LABEL_MAP, **(a.get("labels_map") or {})}
    min_conf = float(a.get("min_conf", 0.5))

    try:
        spots = _dispatch(backend, video_path, cfg, a)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[action_spotting] backend '{backend}' unavailable: {exc}")
        return []

    signals: list[Signal] = []
    for t, label, conf in spots:
        if conf < min_conf:
            continue
        kind = label_map.get(label.lower())
        if kind is None:
            continue
        signals.append(Signal(
            t=float(t), source="action_spotting",
            strength=float(min(1.0, conf)),
            meta={"kind_hint": kind, "label": label},
        ))
    log.info(f"[action_spotting] {len(signals)} events via {backend}")
    return signals


# --------------------------------------------------------------------------- #
def _dispatch(backend: str, video_path: str, cfg: dict, a: dict):
    if backend == "oslactionspotting":
        return _spot_with_osl(video_path, cfg, a)
    if backend == "custom":
        return _spot_with_custom(video_path, cfg, a)
    raise NotImplementedError(
        f"action_spotting backend '{backend}' not implemented. "
        f"Add a _spot_with_<backend>() returning [(t, label, conf), ...]."
    )


def _spot_with_osl(video_path: str, cfg: dict, a: dict):
    """Adapter for the `oslactionspotting` library.

    The library expects pre-extracted features + a config; wiring that fully is
    the L4 milestone (see docs/ROADMAP_FOOTBALL_CV.md). We import it here so a
    missing install degrades gracefully rather than crashing the pipeline.
    """
    import oslactionspotting  # noqa: F401  (presence check)
    raise NotImplementedError(
        "oslactionspotting is installed but the feature/inference wiring is the "
        "L4 milestone. Configure backend: custom with your own model meanwhile."
    )


def _spot_with_custom(video_path: str, cfg: dict, a: dict):
    """Hook for a user-supplied spotter.

    Point `detect.action_spotting.entrypoint` at a 'module:function' that takes
    (video_path, cfg) and returns [(timestamp_s, label, confidence), ...].
    """
    import importlib
    ep = a.get("entrypoint")
    if not ep or ":" not in ep:
        raise ValueError("set detect.action_spotting.entrypoint = 'module:func'")
    mod_name, func_name = ep.split(":", 1)
    func = getattr(importlib.import_module(mod_name), func_name)
    return func(video_path, cfg)
