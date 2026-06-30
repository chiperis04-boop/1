"""Audio-event perception — crowd-roar / spike detection for the Scout.

A goal/big chance is almost always accompanied by a sustained crowd roar; a
foul/whistle by a short spike. Detecting these gives the Scout a high-recall
semantic cue (and the Director audio context) without any heavy model.

Backends:
  * deterministic (default): decode the audio with ffmpeg, compute a per-window
    RMS "excitement" curve with numpy, and flag sustained energetic windows as
    `roar` and short loud transients as `spike`. No extra dependencies.
  * panns (optional): if `panns_inference` (CNN14) is installed it can be used
    for labelled audio tagging; we degrade to the deterministic detector
    otherwise. (Hook kept minimal; the deterministic path is what runs on CPU.)
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import numpy as np

from ..utils.io import get_logger

log = get_logger()


@dataclass
class AudioEvent:
    t: float                 # peak time (s)
    start: float
    end: float
    label: str               # roar | spike
    score: float             # 0..1 relative intensity

    def to_dict(self) -> dict:
        return {"t": round(self.t, 2), "start": round(self.start, 2),
                "end": round(self.end, 2), "label": self.label,
                "score": round(self.score, 3)}


@dataclass
class AudioAnalysis:
    sr: float = 16000.0
    hop: float = 0.5
    times: list[float] = field(default_factory=list)
    curve: list[float] = field(default_factory=list)     # 0..1 excitement
    events: list[AudioEvent] = field(default_factory=list)

    def curve_at(self, t: float) -> float:
        if not self.times:
            return 0.0
        i = int(min(max(t / self.hop, 0), len(self.curve) - 1))
        return self.curve[i]


def decode_pcm(path: str, sr: int = 16000) -> np.ndarray:
    """Decode `path` to a mono float32 [-1,1] waveform via ffmpeg (empty on fail)."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path,
             "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sr), "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not proc.stdout:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:  # noqa: BLE001
        return np.zeros(0, dtype=np.float32)


def analyze_audio(path: str, cfg: dict | None = None) -> AudioAnalysis:
    """Return an excitement curve + roar/spike events for a clip's audio."""
    cfg = cfg or {}
    a = cfg.get("audio_events", {}) if isinstance(cfg, dict) else {}
    sr = int(a.get("sample_rate", 16000))
    hop = float(a.get("hop_seconds", 0.5))

    wav = decode_pcm(path, sr)
    if wav.size == 0:
        return AudioAnalysis(sr=sr, hop=hop)

    win = max(1, int(sr * hop))
    n = wav.size // win
    if n == 0:
        return AudioAnalysis(sr=sr, hop=hop)
    frames = wav[:n * win].reshape(n, win)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)
    times = [i * hop for i in range(n)]

    # normalised 0..1 excitement curve
    peak = float(rms.max()) or 1.0
    curve = (rms / peak).astype(np.float64)

    events = _detect_events(rms, times, hop, a)
    return AudioAnalysis(sr=sr, hop=hop, times=times,
                         curve=[round(float(c), 4) for c in curve], events=events)


def _detect_events(rms: np.ndarray, times: list[float], hop: float,
                   a: dict) -> list[AudioEvent]:
    med = float(np.median(rms))
    mad = float(np.median(np.abs(rms - med))) or (float(rms.std()) or 1e-6)
    # robust z-score
    z = (rms - med) / (1.4826 * mad + 1e-9)
    roar_z = float(a.get("roar_z", 3.0))
    spike_z = float(a.get("spike_z", 5.0))
    min_roar = float(a.get("roar_min_seconds", 1.0))
    peak = float(rms.max()) or 1.0

    events: list[AudioEvent] = []
    # sustained roar: contiguous windows with z >= roar_z lasting >= min_roar
    i, nwin = 0, len(rms)
    min_len = max(1, int(round(min_roar / hop)))
    while i < nwin:
        if z[i] >= roar_z:
            j = i
            while j < nwin and z[j] >= roar_z:
                j += 1
            if (j - i) >= min_len:
                seg = rms[i:j]
                k = i + int(np.argmax(seg))
                events.append(AudioEvent(
                    t=times[k], start=times[i], end=times[min(j, nwin - 1)],
                    label="roar", score=min(1.0, float(rms[k] / peak))))
            i = j
        else:
            i += 1

    # short spikes (whistle/impact): single windows above spike_z not in a roar
    covered = np.zeros(nwin, dtype=bool)
    for ev in events:
        s = int(ev.start / hop); e = int(ev.end / hop)
        covered[s:e + 1] = True
    for idx in range(nwin):
        if not covered[idx] and z[idx] >= spike_z:
            events.append(AudioEvent(t=times[idx], start=times[idx],
                                     end=times[idx] + hop, label="spike",
                                     score=min(1.0, float(rms[idx] / peak))))
    events.sort(key=lambda e: e.t)
    return events
