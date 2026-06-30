"""Jersey number recognition + hero matching.

The Director may say "follow player 18 in the blue jersey". The tracker only
knows numeric track IDs, so we bridge the two: read the number printed on each
player's shirt with OCR, aggregate per track ID (confidence-weighted majority
vote across the clip), and expose `{track_id: jersey_number}`. The studio then
locks the camera onto the track whose jersey matches the Director's hero — far
more reliable than "nearest player to the ball".

Reality check (kept honest, not over-promised): on wide broadcast frames shirt
numbers are tiny and often unreadable; this works best on close/replay frames
and degrades gracefully — unmatched heroes fall back to the geometric pick. We
only sample every Nth frame and skip boxes below a minimum height to keep it
cheap, and require a minimum aggregate confidence before trusting a number.

Uses EasyOCR (already a project dependency) with a digit allow-list, caching its
weights under `modelhub.ocr_storage_dir()` (the Modal Volume).
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..modelhub import ocr_storage_dir
from ..utils.io import get_logger, resolve_device

log = get_logger()

_NUM_RE = re.compile(r"\d{1,2}")


@dataclass
class JerseyResult:
    number_of: dict[int, int] = field(default_factory=dict)        # track_id -> number
    confidence_of: dict[int, float] = field(default_factory=dict)  # track_id -> 0..1

    def track_for_number(self, number: int) -> int | None:
        """Track id wearing `number` (highest-confidence if a digit repeats)."""
        cands = [(tid, self.confidence_of.get(tid, 0.0))
                 for tid, n in self.number_of.items() if n == number]
        if not cands:
            return None
        return max(cands, key=lambda c: c[1])[0]


class JerseyReader:
    def __init__(self, cfg: dict):
        j = cfg.get("vision", {}).get("jerseys", {})
        self.enabled = bool(j.get("enabled", False))
        self.sample_every = int(j.get("sample_every_frames", 3))
        self.min_box_h = int(j.get("min_box_height_px", 60))
        self.min_conf = float(j.get("min_confidence", 0.35))
        self.min_reads = int(j.get("min_reads", 2))
        self.device = resolve_device(cfg.get("vision", {}).get("device", "cuda"))
        self._reader = None

    def _ocr(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(
                ["en"], gpu=(self.device == "cuda"),
                model_storage_directory=ocr_storage_dir(),
                download_enabled=True, verbose=False)
        return self._reader

    def read(self, clip_path: str, track) -> JerseyResult:
        if not self.enabled:
            return JerseyResult()
        try:
            import cv2
            reader = self._ocr()
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[jerseys] disabled (deps missing): {exc}")
            return JerseyResult()

        # per track id: {number -> summed confidence}, and read count
        votes: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        reads: dict[int, int] = defaultdict(int)
        frame_index = {fd.idx: fd for fd in track.frames}

        cap = cv2.VideoCapture(clip_path)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % self.sample_every == 0:
                fd = frame_index.get(idx)
                if fd:
                    for pl in fd.players:
                        num, conf = self._read_one(frame, pl["xyxy"], reader, cv2)
                        if num is not None:
                            votes[pl["id"]][num] += conf
                            reads[pl["id"]] += 1
            idx += 1
        cap.release()

        result = JerseyResult()
        for tid, num_conf in votes.items():
            if reads[tid] < self.min_reads:
                continue
            best_num, best_conf = max(num_conf.items(), key=lambda kv: kv[1])
            norm = best_conf / max(1, reads[tid])
            if norm >= self.min_conf:
                result.number_of[tid] = best_num
                result.confidence_of[tid] = round(min(1.0, norm), 3)
        log.info(f"[jerseys] resolved {len(result.number_of)} numbers "
                 f"-> {result.number_of}")
        return result

    def _read_one(self, frame, xyxy, reader, cv2):
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        h = y2 - y1
        if h < self.min_box_h:
            return None, 0.0
        # upper-back band where the number sits; widen a touch
        by1, by2 = y1 + int(0.15 * h), y1 + int(0.5 * h)
        pad = int((x2 - x1) * 0.1)
        crop = frame[max(0, by1):max(0, by2), max(0, x1 - pad):x2 + pad]
        if crop.size == 0:
            return None, 0.0
        # upscale small crops so OCR has pixels to work with
        scale = max(1, int(80 / max(1, crop.shape[0])))
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        try:
            out = reader.readtext(crop, allowlist="0123456789", detail=1,
                                  paragraph=False)
        except Exception:  # noqa: BLE001
            return None, 0.0
        best_num, best_conf = None, 0.0
        for _box, text, conf in out:
            m = _NUM_RE.search(text or "")
            if m and conf > best_conf:
                val = int(m.group())
                if 0 <= val <= 99:
                    best_num, best_conf = val, float(conf)
        return best_num, best_conf


# --------------------------------------------------------------------------- #
def number_from_description(description: str) -> int | None:
    """Pull a jersey number out of a Director hero description.

    Looks for an explicit '#18' / 'no. 18' / 'number 18' first, then any small
    standalone integer, ignoring decoy numbers in colour words etc.
    """
    if not description:
        return None
    m = re.search(r"(?:#|no\.?\s*|number\s*|player\s*)(\d{1,2})", description,
                  flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = [int(n) for n in _NUM_RE.findall(description) if int(n) <= 99]
    return nums[0] if nums else None
