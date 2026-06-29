"""Audio-energy detector.

Goals and big chances produce a sharp, sustained rise in crowd noise and
commentator volume. We compute a short-time loudness envelope, smooth it, then
flag windows whose loudness is a configurable number of standard deviations
above a rolling baseline. This is the cheapest and one of the most reliable
signals, and it needs no GPU.
"""
from __future__ import annotations

import numpy as np

from ..utils.io import get_logger
from .types import Signal

log = get_logger()


def detect_audio(audio_path: str, cfg: dict) -> list[Signal]:
    import librosa

    a = cfg["detect"]["audio"]
    if not a.get("enabled", True) or not audio_path:
        return []

    sr = cfg["ingest"]["audio_sample_rate"]
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    hop = int(a["hop_seconds"] * sr)
    frame = int(a["frame_seconds"] * sr)

    # RMS loudness envelope in dB
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    times = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=hop)

    # smooth
    win = max(1, int(a["smoothing_seconds"] / a["hop_seconds"]))
    smooth = _moving_avg(db, win)

    # rolling baseline (median + std over a long window)
    base_win = max(win * 8, 60)
    med = _rolling(smooth, base_win, np.median)
    std = _rolling(smooth, base_win, np.std) + 1e-6
    z = (smooth - med) / std

    thr = a["zscore_threshold"]
    above = z > thr

    signals: list[Signal] = []
    for t, peak in _peaks(times, z, above, a["min_gap_seconds"]):
        strength = float(np.clip((peak - thr) / (thr + 1.0), 0.0, 1.0))
        signals.append(Signal(t=float(t), source="audio", strength=strength,
                              meta={"zscore": float(peak)}))

    log.info(f"[audio] {len(signals)} loudness spikes")
    return signals


# --------------------------------------------------------------------------- #
def _moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def _rolling(x: np.ndarray, w: int, fn) -> np.ndarray:
    out = np.empty_like(x)
    half = w // 2
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        out[i] = fn(x[lo:hi])
    return out


def _peaks(times, z, above, min_gap):
    """Yield (time, peak_z) for each contiguous above-threshold run, keeping the
    max, and enforcing a minimum gap between peaks."""
    peaks = []
    i = 0
    n = len(z)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            seg = z[i:j]
            k = i + int(np.argmax(seg))
            peaks.append((times[k], z[k]))
            i = j
        else:
            i += 1

    # enforce gap
    merged = []
    for t, p in sorted(peaks):
        if merged and (t - merged[-1][0]) < min_gap:
            if p > merged[-1][1]:
                merged[-1] = (t, p)
        else:
            merged.append((t, p))
    return merged
