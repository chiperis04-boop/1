"""Reel Survey — whole-video understanding for pre-made highlights reels.

The operator now feeds 15-20 min highlights reels (not full matches), so the
Director should look at the WHOLE reel and decide what to clip, instead of the
Scout hunting a persistent scoreboard (which on a reel just snags random passes).

How it works (precise + cheap):
  1. Scene-segment the reel (PySceneDetect) — an edited reel has clean cuts, so
     scene boundaries ARE the natural, frame-accurate moment boundaries.
  2. Merge scenes into moments and lay ONE representative keyframe per moment
     into labelled contact-sheets (each cell = "index  mm:ss").
  3. The vision-LLM (NVIDIA NIM) surveys the sheets and returns which moments are
     real highlights (goal / skill / save / big chance / assist) and which are
     filler (midfield passes, replays, celebration-only, crowd) to drop.
  4. Kept moments become EventWindows on the exact scene boundaries; the per-clip
     Director + action-span trim then tighten each cut.

Everything is best-effort: no VLM configured / no scenes / any error -> returns
[] and the Scout falls back to its detector discovery.
"""
from __future__ import annotations

from ..utils.io import get_logger

log = get_logger()

_SURVEY_SYS = (
    "You are a senior football highlights editor. You are shown a CONTACT SHEET: "
    "one keyframe per numbered segment of a highlights reel; each cell is labelled "
    "'<index> <mm:ss>'. Judge the WHOLE reel and pick ONLY the segments that are "
    "real, postable highlight moments. Return STRICT JSON:\n"
    '{"moments":[{"i":<int>,"type":"goal|skill|save|chance|assist",'
    '"keep":true,"hook":"<=6 word clickbait"}]}\n'
    "Keep goals, great skills/dribbles/nutmegs, saves, big chances, key assists. "
    "DROP midfield passes, build-up, replays/duplicates, celebration-only, crowd, "
    "studio/graphic cards. Prefer precision (few strong moments) over recall."
)
_SURVEY_USER = ("Which numbered segments are real highlights worth clipping? "
                "Return the JSON only.")

# reel-survey type -> our clip kind (None = drop)
_KIND = {"goal": "goal", "skill": "skill", "dribble": "skill", "nutmeg": "skill",
         "save": "save", "chance": "chance", "shot": "chance", "assist": "chance",
         "penalty": "goal"}


def survey_reel(video_path: str, cfg: dict | None = None,
                duration: float | None = None, client=None):
    cfg = cfg or {}
    rs = cfg.get("detect", {}).get("reel_survey", {})
    try:
        from ..agents.llm_client import VisionLLMClient
        client = client or VisionLLMClient(cfg)
        if not client.is_configured():
            log.info("[reel_survey] no vision-LLM configured; skipping")
            return []

        from ..perception.shots import segment_shots
        shots = segment_shots(video_path, cfg)
        if not shots:
            return []
        moments = _merge_moments(
            shots, float(rs.get("moment_min_seconds", 6.0)),
            int(rs.get("max_moments", 40)), duration)
        if not moments:
            return []

        cells = int(rs.get("cells_per_sheet", 20))
        picks: dict[int, dict] = {}
        for base in range(0, len(moments), cells):
            chunk = moments[base:base + cells]
            sheet = _labeled_montage(video_path, chunk, base, cfg)
            if sheet is None:
                continue
            try:
                out = client.chat_json(_SURVEY_SYS, _SURVEY_USER, [sheet])
            except Exception as exc:  # noqa: BLE001
                log.warning(f"[reel_survey] VLM sheet failed ({exc})")
                continue
            for m in (out.get("moments") or []):
                try:
                    i = int(m.get("i", -1))
                except (TypeError, ValueError):
                    continue
                if 0 <= i < len(moments) and m.get("keep", True):
                    picks[i] = m

        windows = _to_windows(moments, picks, cfg)
        log.info(f"[reel_survey] {len(windows)} highlight moment(s) picked from "
                 f"{len(moments)} segments")
        return windows
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[reel_survey] unavailable ({exc}); detector discovery")
        return []


def _to_windows(moments, picks, cfg):
    from .scout import EventWindow
    top_n = int(cfg.get("detect", {}).get("reel_survey", {}).get("top_n", 0))
    windows = []
    for i, m in sorted(picks.items()):
        kind = _KIND.get(str(m.get("type", "")).lower().strip())
        if kind is None:
            continue
        s, e = moments[i]
        windows.append(EventWindow(
            kind=kind, anchor_t=(s + e) / 2.0, start=s, end=e,
            confidence=0.85 if kind == "goal" else 0.75, verified=False,
            sources=["reel_survey"],
            meta={"hook": m.get("hook"), "reel_type": m.get("type")}))
    windows.sort(key=lambda w: w.anchor_t)
    if top_n > 0 and len(windows) > top_n:
        strongest = sorted(windows, key=lambda w: w.confidence, reverse=True)[:top_n]
        strongest.sort(key=lambda w: w.anchor_t)
        return strongest
    return windows


def _merge_moments(shots, min_seconds, max_moments, duration):
    """Merge scene shots into moments (each >= min_seconds), then, if there are
    still more than max_moments, merge the shortest neighbours until it fits."""
    moments = [[s.start, s.end] for s in shots]
    # first pass: absorb tiny scenes into the previous moment
    merged = []
    for st, en in moments:
        if merged and (merged[-1][1] - merged[-1][0]) < min_seconds:
            merged[-1][1] = en
        else:
            merged.append([st, en])
    # cap the count by repeatedly merging the shortest moment into a neighbour
    while len(merged) > max_moments:
        durs = [(m[1] - m[0], k) for k, m in enumerate(merged)]
        _, k = min(durs)
        if k + 1 < len(merged):
            merged[k][1] = merged[k + 1][1]
            del merged[k + 1]
        elif k > 0:
            merged[k - 1][1] = merged[k][1]
            del merged[k]
        else:
            break
    if duration:
        merged = [[max(0.0, s), min(duration, e)] for s, e in merged]
    return [(s, e) for s, e in merged if e > s]


def _labeled_montage(video_path, moments, base_index, cfg):
    """Grab one mid-moment keyframe per moment, label it '<index> <mm:ss>', tile
    into a single contact-sheet JPEG (<180KB for the NIM inline limit)."""
    try:
        import math

        import cv2
        import numpy as np
        cap = cv2.VideoCapture(video_path)
        cell = 256
        tiles = []
        for k, (s, e) in enumerate(moments):
            mid = (s + e) / 2.0
            cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = np.zeros((cell, cell, 3), np.uint8)
            h, w = frame.shape[:2]
            sc = cell / float(max(h, w))
            frame = cv2.resize(frame, (max(1, int(w * sc)), max(1, int(h * sc))),
                               interpolation=cv2.INTER_AREA)
            idx = base_index + k
            label = f"{idx} {int(mid) // 60:d}:{int(mid) % 60:02d}"
            cv2.rectangle(frame, (0, 0), (len(label) * 11 + 8, 20), (0, 0, 0), -1)
            cv2.putText(frame, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
            tiles.append(frame)
        cap.release()
        if not tiles:
            return None
        cols = min(len(tiles), 5)
        rows = int(math.ceil(len(tiles) / cols))
        ch = max(t.shape[0] for t in tiles)
        cw = max(t.shape[1] for t in tiles)
        canvas = np.zeros((rows * ch, cols * cw, 3), np.uint8)
        for k, t in enumerate(tiles):
            y, x = (k // cols) * ch, (k % cols) * cw
            canvas[y:y + t.shape[0], x:x + t.shape[1]] = t
        m = max(canvas.shape[:2])
        if m > 1280:
            sc = 1280 / float(m)
            canvas = cv2.resize(canvas, (int(canvas.shape[1] * sc),
                                         int(canvas.shape[0] * sc)),
                                interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 72])
        return buf.tobytes() if ok else None
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[reel_survey] montage failed ({exc})")
        return None
