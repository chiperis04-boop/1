"""Blueprint Module 1 — the Scout (event discovery + scoreboard verification).

A full match is ~90 minutes; running heavy per-frame vision on all of it is
wasteful. The Scout does a *cheap discovery pass* and returns a short list of
`EventWindow`s (a -pre/+post slice around each candidate) that the expensive
modules (Director, Cameraman, Composer) then process.

It does NOT reimplement detection — it composes the existing, tested detectors:

  * action spotting  (src/detect/action_spotting.py)  -> named events w/ conf
                        adapter for yahoo-inc/spivak-action-spotting-soccernet
                        and the oslactionspotting family.
  * scoreboard OCR   (src/detect/scoreboard_ocr.py)    -> exact score changes

Verification logic (the blueprint's "Scoreboard Verification"): an action-spot
"Goal" anchor is *confirmed* when an OCR score increment lands inside its
window. Confirmed goals get a confidence boost and carry the before/after score
(e.g. "1-2" -> "2-2") for the lower-third overlay. OCR score changes with no
matching action anchor are still emitted as goals (OCR is near-certain), and
action anchors with no OCR support survive as unverified candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..detect.action_spotting import detect_actions
from ..detect.scoreboard_ocr import detect_scoreboard
from ..detect.types import Signal
from ..utils.io import get_logger

log = get_logger()


@dataclass
class EventWindow:
    """A verified candidate slice of the match for downstream processing."""
    kind: str                       # goal | chance | save | card | skill
    anchor_t: float                 # the event instant (seconds into match)
    start: float                    # window start (clip in-point)
    end: float                      # window end (clip out-point)
    confidence: float               # 0..1 (boosted when score-verified)
    verified: bool = False          # OCR confirmed a score change in-window
    minute: int | None = None
    score_before: str | None = None
    score_after: str | None = None
    sources: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def scout_events(
    video_path: str,
    proxy_path: str | None = None,
    cfg: dict | None = None,
    duration: float | None = None,
) -> list[EventWindow]:
    """Discover + verify event windows for a full match.

    `video_path` is the source (used by action spotting / clipping); `proxy_path`
    is the downscaled analysis copy used for OCR (falls back to `video_path`).
    """
    cfg = cfg or {}
    sc = cfg.get("detect", {}).get("scout", {})
    proxy = proxy_path or video_path

    # 0) event feed (textual play-by-play) — the most reliable + cheapest source
    #    when available. Map match-clock events to video time via kick-off
    #    offsets (ClipMaker-style), optionally snapped to the OCR score clock.
    ef = cfg.get("detect", {}).get("event_feed", {})
    if ef.get("enabled"):
        try:
            from .event_feed import (align_to_ocr, events_to_windows,
                                     load_descriptive_events, load_from_espn)
            events = []
            espn = ef.get("espn", {}) or {}
            if espn.get("enabled") and espn.get("fixture_id"):
                events = load_from_espn(espn.get("fixture_id"),
                                        espn.get("slug", "esp.1"), cfg)
            elif ef.get("source"):
                # descriptive log: text/CSV report OR StatsBomb/SoccerNet JSON
                events = load_descriptive_events(ef["source"], cfg)
            fw = events_to_windows(events, ef.get("kickoffs", {}), cfg, duration) \
                if events else []
            if fw:
                if ef.get("align_to_ocr", True):
                    try:
                        fw = align_to_ocr(fw, detect_scoreboard(proxy, cfg), cfg)
                    except Exception as exc:  # noqa: BLE001
                        log.warning(f"[scout] OCR alignment skipped: {exc}")
                log.info(f"[scout] using {len(fw)} event-feed windows "
                         f"({sum(w.verified for w in fw)} goals)")
                return fw
            log.warning("[scout] event feed yielded no windows; "
                        "falling back to detector discovery")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[scout] event feed unavailable ({exc}); detector discovery")

    # 1) named events (goal/shot/card/...) from the action-spotting model
    action_sigs: list[Signal] = []
    try:
        action_sigs = detect_actions(video_path, cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[scout] action spotting unavailable: {exc}")

    # 2) exact score changes (near-certain goals) from scoreboard OCR
    ocr_sigs: list[Signal] = []
    try:
        ocr_sigs = detect_scoreboard(proxy, cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[scout] scoreboard OCR unavailable: {exc}")

    windows = _build_windows(action_sigs, ocr_sigs, cfg, sc, duration)
    log.info(f"[scout] {len(windows)} event windows "
             f"({sum(w.verified for w in windows)} score-verified goals)")
    return windows


# --------------------------------------------------------------------------- #
def _window_bounds(kind: str, anchor: float, cfg: dict, sc: dict,
                   duration: float | None):
    """Pre/post padding around an anchor. Goals get a longer build-up.

    Defaults follow the blueprint (-20s/+10s for goals) but read from
    config.clip when present so the Scout and the legacy clipper stay aligned.
    """
    clip = cfg.get("clip", {})
    if kind == "goal":
        pre = sc.get("goal_pre_seconds", clip.get("goal_pre_seconds", 20.0))
        post = sc.get("goal_post_seconds", clip.get("goal_post_seconds", 10.0))
    else:
        pre = sc.get("pre_seconds", clip.get("pre_seconds", 7.0))
        post = sc.get("post_seconds", clip.get("post_seconds", 5.0))
    start = max(0.0, anchor - pre)
    end = anchor + post
    if duration:
        end = min(end, duration)
    return start, end


def _build_windows(action_sigs, ocr_sigs, cfg, sc, duration) -> list[EventWindow]:
    label_map = cfg.get("detect", {}).get("action_spotting", {}).get("labels_map", {})
    verify_radius = float(sc.get("verify_radius_seconds", 25.0))
    goal_boost = float(sc.get("verified_goal_boost", 0.25))

    windows: list[EventWindow] = []
    used_ocr: set[int] = set()

    # --- action anchors, verified against OCR score increments ---------------
    for sig in action_sigs:
        kind = sig.meta.get("kind_hint") or label_map.get(
            str(sig.meta.get("label", "")).lower(), "chance")
        start, end = _window_bounds(kind, sig.t, cfg, sc, duration)
        w = EventWindow(
            kind=kind, anchor_t=sig.t, start=start, end=end,
            confidence=float(sig.strength), sources=["action_spotting"],
            meta={"label": sig.meta.get("label")},
        )
        if kind == "goal":
            match_idx = _nearest_ocr(sig.t, ocr_sigs, used_ocr, verify_radius)
            if match_idx is not None:
                osig = ocr_sigs[match_idx]
                used_ocr.add(match_idx)
                w.verified = True
                w.confidence = min(1.0, w.confidence + goal_boost)
                w.score_before = osig.meta.get("prev")
                w.score_after = osig.meta.get("score")
                w.minute = osig.meta.get("minute")
                w.sources.append("scoreboard_ocr")
                # snap the anchor to the (precise) score-change instant
                w.anchor_t = osig.t
                w.start, w.end = _window_bounds("goal", osig.t, cfg, sc, duration)
        windows.append(w)

    # --- OCR score changes with no matching action anchor -> standalone goals -
    for i, osig in enumerate(ocr_sigs):
        if i in used_ocr:
            continue
        start, end = _window_bounds("goal", osig.t, cfg, sc, duration)
        windows.append(EventWindow(
            kind="goal", anchor_t=osig.t, start=start, end=end,
            confidence=float(min(1.0, osig.strength)), verified=True,
            minute=osig.meta.get("minute"),
            score_before=osig.meta.get("prev"),
            score_after=osig.meta.get("score"),
            sources=["scoreboard_ocr"],
        ))

    windows.sort(key=lambda w: w.anchor_t)
    return _dedupe(windows, float(sc.get("merge_gap_seconds", 15.0)))


def _nearest_ocr(t: float, ocr_sigs, used: set[int], radius: float):
    best, best_dt = None, radius
    for i, s in enumerate(ocr_sigs):
        if i in used:
            continue
        dt = abs(s.t - t)
        if dt <= best_dt:
            best, best_dt = i, dt
    return best


def _dedupe(windows: list[EventWindow], gap: float) -> list[EventWindow]:
    """Collapse near-duplicate anchors (e.g. OCR + action spotting + replay all
    firing on one goal). Keeps the highest-confidence window in each cluster and
    unions their source tags."""
    merged: list[EventWindow] = []
    for w in windows:
        if merged and (w.anchor_t - merged[-1].anchor_t) <= gap:
            prev = merged[-1]
            keep, drop = (prev, w) if prev.confidence >= w.confidence else (w, prev)
            keep.sources = sorted(set(keep.sources) | set(drop.sources))
            keep.verified = keep.verified or drop.verified
            # prefer a goal classification if either says goal
            if "goal" in (prev.kind, w.kind):
                keep.kind = "goal"
            if drop.score_after and not keep.score_after:
                keep.score_before, keep.score_after = drop.score_before, drop.score_after
                keep.minute = keep.minute or drop.minute
            merged[-1] = keep
        else:
            merged.append(w)
    return merged
