"""Fusion stage.

Merges signals from every detector that fall within a time window into a single
ranked Moment. Confidence is a weighted blend of the contributing signals; the
moment 'kind' is inferred from which detectors fired (a score change => goal, a
'saved' commentary hit => save, etc.). Finally we apply clip padding.
"""
from __future__ import annotations

from ..utils.io import get_logger
from .types import Moment, Signal

log = get_logger()


def fuse(signals: list[Signal], cfg: dict, duration: float) -> list[Moment]:
    f = cfg["fusion"]
    weights = f["weights"]
    window = 8.0  # seconds: signals within this of each other are one moment

    signals = sorted(signals, key=lambda s: s.t)
    clusters: list[list[Signal]] = []
    for s in signals:
        if clusters and (s.t - clusters[-1][-1].t) <= window:
            clusters[-1].append(s)
        else:
            clusters.append([s])

    moments: list[Moment] = []
    for cluster in clusters:
        # weighted confidence (cap each source's contribution once)
        by_source: dict[str, float] = {}
        for s in cluster:
            by_source[s.source] = max(by_source.get(s.source, 0.0), s.strength)
        conf = sum(weights.get(src, 0.0) * strength
                   for src, strength in by_source.items())
        conf = min(1.0, conf)

        if conf < f["min_confidence"]:
            continue

        kind = _classify(cluster, by_source)
        # anchor time: prefer scoreboard, else strongest audio, else first
        anchor = _anchor_time(cluster)
        minute = _minute(cluster)

        moments.append(Moment(
            t=anchor, start=anchor, end=anchor, confidence=round(conf, 3),
            kind=kind, minute=minute,
            sources=sorted(by_source.keys()),
            meta={"signal_count": len(cluster)},
        ))

    # rank, cap, then pad
    moments.sort(key=lambda m: m.confidence, reverse=True)
    moments = moments[: f["max_moments"]]
    moments = _pad(moments, cfg, duration)
    moments.sort(key=lambda m: m.t)

    log.info(f"[fusion] {len(moments)} moments kept "
             f"({sum(m.kind == 'goal' for m in moments)} goals)")
    return moments


# --------------------------------------------------------------------------- #
def _classify(cluster: list[Signal], by_source: dict) -> str:
    # an action-spotting model is the most reliable label when present
    spot_hints = [s.meta.get("kind_hint") for s in cluster
                  if s.source == "action_spotting"]
    if spot_hints:
        for pref in ("goal", "card", "save", "chance"):
            if pref in spot_hints:
                return pref
    if "scoreboard_ocr" in by_source:
        return "goal"
    hints = [s.meta.get("kind_hint") for s in cluster if s.source == "commentary"]
    if "goal" in hints:
        return "goal"
    if "card" in hints:
        return "card"
    if "chance" in hints:
        # distinguish a save from a miss is hard from text alone; default chance
        return "save" if any("saved" in (s.meta.get("phrase") or "")
                             for s in cluster) else "chance"
    # only audio/scene fired -> probably a chance or skill moment
    return "chance"


def _anchor_time(cluster: list[Signal]) -> float:
    for s in cluster:
        if s.source == "scoreboard_ocr":
            # the score graphic updates slightly *after* the goal; back up a bit
            return max(0.0, s.t - 6.0)
    aud = [s for s in cluster if s.source == "audio"]
    if aud:
        return max(aud, key=lambda s: s.strength).t
    return cluster[0].t


def _minute(cluster: list[Signal]):
    for s in cluster:
        m = s.meta.get("minute")
        if m is not None:
            return m
    return None


def _pad(moments: list[Moment], cfg: dict, duration: float) -> list[Moment]:
    c = cfg["clip"]
    for m in moments:
        pre = c["goal_pre_seconds"] if m.kind == "goal" else c["pre_seconds"]
        post = c["goal_post_seconds"] if m.kind == "goal" else c["post_seconds"]
        m.start = max(0.0, m.t - pre)
        m.end = min(duration, m.t + post)
    return moments
