"""Deterministic QA on a rendered highlight.

Cheap, objective checks that need no model — run on the finished mp4 to catch
the failures that make an auto-edit look broken:

  * streams/resolution : exactly 1 video + 1 audio at the expected WxH
  * letterbox/pillarbox: unexpected black bars (crop/pad gone wrong)
  * dead frames        : near-black or frozen (duplicated) frames
  * loudness           : integrated LUFS within target (ffmpeg ebur128)
  * duration           : within the expected window

Each check yields a 0..1 score and may push a machine-readable issue tag
(consumed by agents.review.apply_corrections). Everything degrades gracefully:
if cv2/ffmpeg can't inspect the file, the check is skipped rather than failing
the render.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass, field

from ..edit import ff
from ..utils.io import get_logger

log = get_logger()


@dataclass
class QAReport:
    score: float = 1.0
    passed: bool = True
    issues: list[str] = field(default_factory=list)     # machine tags
    checks: dict = field(default_factory=dict)           # name -> {score,detail}

    def to_dict(self) -> dict:
        return asdict(self)

    def _add(self, name: str, score: float, detail: str = "",
             issue: str | None = None):
        self.checks[name] = {"score": round(float(score), 3), "detail": detail}
        if issue:
            self.issues.append(issue)


def qa_report(out_path: str, cfg: dict | None = None,
              expected: dict | None = None) -> QAReport:
    """Inspect a rendered clip and return a QAReport.

    expected: optional {width,height,min_seconds,max_seconds}.
    """
    cfg = cfg or {}
    q = cfg.get("qa", {})
    expected = expected or {}
    r = QAReport()

    _check_streams(r, out_path, expected)
    _check_frames(r, out_path, q)
    _check_loudness(r, out_path, cfg, q)
    _check_duration(r, out_path, expected)

    scores = [c["score"] for c in r.checks.values()] or [1.0]
    r.score = round(sum(scores) / len(scores), 3)
    r.passed = r.score >= float(q.get("pass_score", 0.75)) and not _hard(r.issues)
    return r


def _hard(issues: list[str]) -> bool:
    """Issues serious enough to fail QA regardless of average score."""
    return any(i in issues for i in ("no_video", "no_audio", "letterbox",
                                     "black_frames"))


# --------------------------------------------------------------------------- #
def _check_streams(r: QAReport, path: str, expected: dict):
    try:
        data = ff.probe(path)
    except Exception as exc:  # noqa: BLE001
        r._add("streams", 0.0, f"probe failed: {exc}", issue="probe_failed")
        return
    vids = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
    auds = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    if not vids:
        r._add("streams", 0.0, "no video stream", issue="no_video")
        return
    if not auds:
        r._add("streams", 0.2, "no audio stream", issue="no_audio")
    w, h = int(vids[0].get("width", 0)), int(vids[0].get("height", 0))
    detail = f"{len(vids)}V+{len(auds)}A {w}x{h}"
    if expected.get("width") and (w, h) != (int(expected["width"]), int(expected["height"])):
        r._add("streams", 0.4, detail + " (resolution mismatch)",
               issue="resolution_mismatch")
    else:
        r._add("streams", 1.0, detail)


def _check_frames(r: QAReport, path: str, q: dict):
    try:
        import cv2
        import numpy as np
    except Exception:  # noqa: BLE001
        return
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        cap.release()
        return
    sample_idx = [int(n * f) for f in (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)]
    grays, border_hits, dark_hits = [], 0, 0
    for fi in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grays.append(g)
        if float(g.mean()) < 8.0:
            dark_hits += 1
        if _has_black_bars(g, np):
            border_hits += 1
    cap.release()
    sampled = max(1, len(grays))

    # letterbox / pillarbox
    if border_hits >= max(2, sampled // 2):
        r._add("letterbox", 0.0, f"black bars in {border_hits}/{sampled} frames",
               issue="letterbox")
    else:
        r._add("letterbox", 1.0, "no unexpected black bars")

    # black frames
    if dark_hits >= max(2, sampled // 2):
        r._add("black_frames", 0.0, f"{dark_hits}/{sampled} near-black",
               issue="black_frames")
    else:
        r._add("black_frames", 1.0, "no dead-black frames")

    # frozen frames (consecutive samples ~identical)
    frozen = 0
    for a, b in zip(grays, grays[1:]):
        if a.shape == b.shape and float(np.mean(np.abs(a.astype("int16")
                                                       - b.astype("int16")))) < 0.5:
            frozen += 1
    if grays and frozen >= len(grays) - 1 and len(grays) > 2:
        r._add("motion", 0.2, "frames appear frozen", issue="frozen")
    else:
        r._add("motion", 1.0, "has motion")


def _has_black_bars(gray, np) -> bool:
    h, w = gray.shape
    m = max(2, int(min(h, w) * 0.06))
    top, bot = gray[:m].mean(), gray[-m:].mean()
    left, right = gray[:, :m].mean(), gray[:, -m:].mean()
    center = gray[h // 3:2 * h // 3, w // 3:2 * w // 3].mean()
    if center < 15:                      # whole frame dark -> not a bar issue
        return False
    bars_v = top < 6 and bot < 6
    bars_h = left < 6 and right < 6
    return bool(bars_v or bars_h)


def _check_loudness(r: QAReport, path: str, cfg: dict, q: dict):
    if not ff.has_audio(path):
        return
    target = float(cfg.get("edit", {}).get("audio", {})
                   .get("loudness_target_lufs", -14))
    tol = float(q.get("loudness_tolerance", 3.0))
    lufs = _measure_lufs(path)
    if lufs is None:
        return
    off = abs(lufs - target)
    if off <= tol:
        r._add("loudness", 1.0, f"{lufs:.1f} LUFS (target {target})")
    else:
        score = max(0.0, 1.0 - (off - tol) / 10.0)
        r._add("loudness", score, f"{lufs:.1f} LUFS off target {target}",
               issue="loudness_low" if lufs < target else "loudness_high")


def _measure_lufs(path: str) -> float | None:
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", path,
             "-af", "ebur128", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        # take the LAST "I:  -xx.x LUFS" (the summary integrated loudness)
        vals = re.findall(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", proc.stderr)
        return float(vals[-1]) if vals else None
    except Exception:  # noqa: BLE001
        return None


def _check_duration(r: QAReport, path: str, expected: dict):
    dur = ff.duration(path)
    lo = float(expected.get("min_seconds", 0) or 0)
    hi = float(expected.get("max_seconds", 0) or 0)
    if hi and dur > hi:
        r._add("duration", 0.3, f"{dur:.1f}s > max {hi}", issue="too_long")
    elif lo and dur < lo:
        r._add("duration", 0.3, f"{dur:.1f}s < min {lo}", issue="too_short")
    else:
        r._add("duration", 1.0, f"{dur:.1f}s")
