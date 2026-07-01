"""Music beat detection + beat-snapping.

Cutting / launching slow-mo ON the musical beat is a big part of why a montage
feels professional. This module estimates beat times from a music track and
snaps editorial timestamps (slow-mo onsets, cut points) to the nearest beat.

Backends:
  * deterministic (default): decode with ffmpeg, build an onset-strength
    envelope with numpy and pick onset peaks -> beat times + BPM. No deps.
  * librosa (optional): if installed, `librosa.beat.beat_track` is used for a
    more robust estimate; we degrade to the numpy detector otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..perception.audio_events import decode_pcm
from ..utils.io import get_logger

log = get_logger()


@dataclass
class BeatGrid:
    bpm: float = 0.0
    beats: list[float] = field(default_factory=list)     # beat times (s)
    strengths: list[float] = field(default_factory=list)  # 0..1 accent per beat

    def __bool__(self) -> bool:
        return len(self.beats) >= 2


def detect_beats(path: str, cfg: dict | None = None) -> BeatGrid:
    cfg = cfg or {}
    sr = int(cfg.get("audio_events", {}).get("sample_rate", 16000))
    wav = decode_pcm(path, sr)
    if wav.size < sr // 2:
        return BeatGrid()

    # try librosa first (optional, more robust) — also grab the onset-strength
    # envelope so we know which beats are ACCENTED (the bass drops).
    try:
        import librosa
        onset_env = librosa.onset.onset_strength(y=wav, sr=sr)
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, units="frames")
        beats = librosa.frames_to_time(beat_frames, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        if len(beats) >= 2:
            bf = np.clip(np.asarray(beat_frames), 0, len(onset_env) - 1)
            strengths = _norm01(onset_env[bf]) if len(onset_env) else []
            return BeatGrid(bpm=bpm, beats=[round(float(t), 3) for t in beats],
                            strengths=[round(float(s), 3) for s in strengths])
    except Exception:  # noqa: BLE001
        pass

    return _onset_beats(wav, sr)


def _norm01(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return a
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.ones_like(a)


def strong_beats(grid: "BeatGrid", frac: float = 0.5) -> list[float]:
    """Return the ACCENTED beats (top `frac` by onset strength = the bass drops).
    Falls back to all beats when no per-beat strengths are available."""
    if not grid.beats:
        return []
    if not grid.strengths or len(grid.strengths) != len(grid.beats):
        return list(grid.beats)
    s = np.asarray(grid.strengths, dtype=np.float64)
    thr = float(np.quantile(s, max(0.0, min(1.0, 1.0 - frac))))
    strong = [t for t, v in zip(grid.beats, grid.strengths) if v >= thr]
    return strong or list(grid.beats)


def _onset_beats(wav: np.ndarray, sr: int) -> BeatGrid:
    """numpy onset-peak beat detector (good on percussive / click-like tracks)."""
    hop = max(1, int(sr * 0.016))            # ~16 ms resolution
    n = wav.size // hop
    if n < 4:
        return BeatGrid()
    frames = wav[:n * hop].reshape(n, hop)
    energy = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)
    flux = np.diff(energy, prepend=energy[0])
    flux[flux < 0] = 0.0                     # half-wave rectify (onsets only)
    if flux.max() <= 0:
        return BeatGrid()
    thr = float(flux.mean() + 1.0 * flux.std())
    min_gap = max(1, int(0.20 / 0.016))      # >=200 ms between beats
    peaks = []
    last = -min_gap
    for i in range(1, n - 1):
        if flux[i] >= thr and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1] \
                and (i - last) >= min_gap:
            peaks.append(i)
            last = i
    beats = [round(i * 0.016, 3) for i in peaks]
    if len(beats) < 2:
        return BeatGrid()
    # per-beat accent = onset flux at the peak (bass drops = strong onsets)
    strengths = _norm01([flux[i] for i in peaks]).tolist()
    diffs = np.diff(beats)
    bpm = float(60.0 / np.median(diffs)) if len(diffs) else 0.0
    return BeatGrid(bpm=round(bpm, 1), beats=beats,
                    strengths=[round(float(s), 3) for s in strengths])


def snap_to_beats(times, beats, max_shift: float = 0.12):
    """Snap each time to the nearest beat within `max_shift` seconds."""
    if not beats:
        return list(times)
    b = np.asarray(beats, dtype=np.float64)
    out = []
    for t in times:
        j = int(np.argmin(np.abs(b - t)))
        out.append(float(b[j]) if abs(b[j] - t) <= max_shift else float(t))
    return out


def beat_align_plan(plan, music_path: str, cfg: dict | None = None):
    """Snap an EditPlan's slow-mo beat onsets (and cut_in) to musical beats.

    The AI-directed slow-mo lands on the nearest ACCENTED beat (bass drop) within
    `beat_drop_max_shift`, so the highlight peaks on the drop; the cut_in snaps to
    the nearest ordinary beat. Returns (plan, changed). Safe no-op if no beats.
    """
    cfg = cfg or {}
    grid = detect_beats(music_path, cfg)
    if not grid:
        return plan, False
    a = cfg.get("edit", {}).get("audio", {})
    max_shift = float(a.get("beat_snap_max_shift", 0.12))
    drop_shift = float(a.get("beat_drop_max_shift", 0.35))
    drops = strong_beats(grid, float(a.get("beat_drop_frac", 0.5)))
    changed = False
    for b in getattr(plan, "slowmo_beats", []):
        # prefer a bass drop (wider tolerance); else the nearest ordinary beat
        new_start = snap_to_beats([b.start], drops, drop_shift)[0]
        if abs(new_start - b.start) <= 1e-3:
            new_start = snap_to_beats([b.start], grid.beats, max_shift)[0]
        if abs(new_start - b.start) > 1e-3:
            dur = b.end - b.start
            b.start = new_start
            b.end = new_start + dur
            changed = True
    new_cut = snap_to_beats([getattr(plan, "cut_in", 0.0)], grid.beats, max_shift)[0]
    if abs(new_cut - getattr(plan, "cut_in", 0.0)) > 1e-3:
        plan.cut_in = new_cut
        changed = True
    if changed:
        log.info(f"[music] beat-aligned plan to {grid.bpm:.0f} BPM "
                 f"({len(grid.beats)} beats, {len(drops)} drops)")
    return plan, changed
