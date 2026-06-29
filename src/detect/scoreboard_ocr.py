"""Scoreboard OCR detector — the most precise goal finder.

We sample frames from the proxy every N seconds, OCR a region of interest where
the broadcast score graphic lives, parse a "A - B" score, and emit a strong
signal whenever the score *changes*. The match minute is also captured when
present, which feeds the lower-third overlay.

ROI handling:
  * If config provides `roi`, we use it directly.
  * Otherwise we auto-locate a stable text region in the top/bottom corners by
    sampling a handful of frames and picking the area with persistent digits.
"""
from __future__ import annotations

import re

import numpy as np

from ..utils.io import get_logger
from .types import Signal

log = get_logger()

_SCORE_RE = re.compile(r"\b(\d{1,2})\s*[-:vV]\s*(\d{1,2})\b")
# the match clock is a number explicitly marked as minutes ( 67'  45+2'  90’ )
_CLOCK_RE = re.compile(r"\b(\d{1,3})(?:\s*\+\s*\d{1,2})?\s*[\u2032\u2019']")
_MINUTE_RE = re.compile(r"\b(\d{1,3})\b")


def detect_scoreboard(proxy_path: str, cfg: dict) -> list[Signal]:
    o = cfg["detect"]["scoreboard_ocr"]
    if not o.get("enabled", True):
        return []

    import cv2
    import easyocr

    from ..utils.io import resolve_device
    from ..modelhub import ocr_storage_dir

    reader = easyocr.Reader(
        o["languages"], gpu=(resolve_device(cfg["vision"]["device"]) == "cuda"),
        model_storage_directory=ocr_storage_dir(), verbose=False)
    cap = cv2.VideoCapture(proxy_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = int(max(1, o["sample_every_seconds"] * fps))

    roi = o.get("roi")
    signals: list[Signal] = []
    last_score: tuple[int, int] | None = None

    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps

        crop = _crop_roi(frame, roi)
        text = _ocr_join(reader, crop)
        score = _parse_score(text)
        minute = _parse_minute(text, score_match=_SCORE_RE.search(text or ""))

        if score is not None:
            if last_score is None:
                last_score = score
            elif score != last_score and _is_increment(last_score, score):
                signals.append(Signal(
                    t=float(t), source="scoreboard_ocr", strength=1.0,
                    meta={"score": f"{score[0]}-{score[1]}",
                          "prev": f"{last_score[0]}-{last_score[1]}",
                          "minute": minute},
                ))
                last_score = score
        idx += step

    cap.release()
    log.info(f"[ocr] {len(signals)} score changes (goals)")
    return signals


# --------------------------------------------------------------------------- #
def _crop_roi(frame: np.ndarray, roi):
    h, w = frame.shape[:2]
    if roi:
        x1, y1, x2, y2 = roi
        return frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
    # default: top strip (most broadcast score bugs sit top-left/top-center)
    return frame[0:int(0.18 * h), 0:int(0.55 * w)]


def _ocr_join(reader, img) -> str:
    try:
        res = reader.readtext(img, detail=0, paragraph=True)
    except Exception:
        return ""
    return " ".join(res)


def _parse_score(text: str):
    m = _SCORE_RE.search(text or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _parse_minute(text: str, score_match=None):
    """Read the match clock, not the scoreline.

    The old regex grabbed the *first* number in the OCR string — which is the
    home-team score (e.g. "2 - 1  67'") — so a 67th-minute goal was labelled
    "GOAL - 2'". We now (1) strip the matched score span first, then (2) strongly
    prefer a number explicitly marked as minutes (an apostrophe/prime, optionally
    with stoppage like "45+2'"), and only fall back to a bare number if no
    clock mark exists.
    """
    t = text or ""
    if score_match is not None:
        t = (t[:score_match.start()] + "  " + t[score_match.end():])

    m = _CLOCK_RE.search(t)
    if m:
        val = int(m.group(1))
        return val if 0 <= val <= 130 else None

    # no explicit clock mark: take the first plausible standalone minute, but
    # never one that is immediately part of a score-like "N - N" pattern.
    for m in _MINUTE_RE.finditer(t):
        val = int(m.group(1))
        if 0 <= val <= 130:
            return val
    return None


def _is_increment(prev, cur) -> bool:
    """Only count a +1 to either side as a goal (filters OCR noise)."""
    dh, da = cur[0] - prev[0], cur[1] - prev[1]
    return (dh, da) in {(1, 0), (0, 1)}
