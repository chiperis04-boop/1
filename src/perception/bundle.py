"""PerceptionBundle — the structured, multimodal context the Director reasons
over: shots, keyframes (JPEG), the commentary transcript, a compact detection
summary, an audio excitement curve/events and the scoreboard score.

Heavy perception (ASR, audio analysis) is lazy and guarded: if faster-whisper /
librosa are missing or disabled, those fields are simply empty and the bundle is
still usable. This keeps the plumbing CPU-testable without GPU models.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..utils.io import get_logger
from .shots import Shot, segment_shots

log = get_logger()


@dataclass
class PerceptionBundle:
    clip_path: str
    fps: float = 30.0
    duration: float = 0.0
    shots: list[Shot] = field(default_factory=list)
    keyframes: list[bytes] = field(default_factory=list)   # JPEG bytes
    keyframe_times: list[float] = field(default_factory=list)
    transcript: list[dict] = field(default_factory=list)   # {start,end,text}
    audio_curve: list[float] = field(default_factory=list)
    audio_events: list[dict] = field(default_factory=list)  # {t,label}
    detections_summary: str = ""
    score: dict | None = None

    def transcript_text(self) -> str:
        return " ".join(seg.get("text", "").strip() for seg in self.transcript).strip()


def build_bundle(clip_path: str, cfg: dict | None = None, shots=None,
                 frames=None, window=None, peak_t: float | None = None
                 ) -> PerceptionBundle:
    """Assemble a PerceptionBundle for one event clip.

    `shots`  : precomputed shots (else segmented here)
    `frames` : tracking FrameTrack list (for the detection summary + motion peak)
    `window` : EventWindow (for score/kind context)
    `peak_t` : decisive-beat time (s) to sample keyframes more densely around
    """
    cfg = cfg or {}
    d = cfg.get("director", {})
    fps, duration = _probe(clip_path)
    shots = shots if shots is not None else segment_shots(clip_path, cfg)

    if peak_t is None and frames:
        from ..detection.director import _peak_ball_speed_time
        try:
            peak_t = _peak_ball_speed_time(_FramesView(frames, fps))
        except Exception:  # noqa: BLE001
            peak_t = None

    times = _keyframe_times(duration,
                            max_frames=int(d.get("max_frames", 16)),
                            peak_t=peak_t)
    keyframes = _grab_jpegs(clip_path, times)

    summary = _detection_summary(shots, frames, peak_t)
    score = None
    if window is not None and getattr(window, "score_after", None):
        score = {"before": getattr(window, "score_before", None),
                 "after": window.score_after,
                 "minute": getattr(window, "minute", None)}

    bundle = PerceptionBundle(
        clip_path=clip_path, fps=fps, duration=duration, shots=shots,
        keyframes=keyframes, keyframe_times=times,
        detections_summary=summary, score=score,
    )

    if d.get("use_asr", False):
        bundle.transcript = _transcribe(clip_path, cfg)
    return bundle


# --------------------------------------------------------------------------- #
class _FramesView:
    """Adapt a raw FrameTrack list to the (.frames/.fps) shape some helpers want."""
    def __init__(self, frames, fps):
        self.frames = frames
        self.fps = fps


def _probe(clip_path: str) -> tuple[float, float]:
    try:
        import cv2
        cap = cv2.VideoCapture(clip_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps > 0:
            return float(fps), (n / fps if n else 0.0)
    except Exception:  # noqa: BLE001
        pass
    from ..edit import ff
    return 30.0, ff.duration(clip_path)


def _keyframe_times(duration: float, max_frames: int,
                    peak_t: float | None) -> list[float]:
    if duration <= 0:
        return [0.0]
    n = max(4, min(max_frames, 24))
    # uniform spread ...
    times = [duration * (i + 0.5) / n for i in range(n)]
    # ... plus a denser cluster around the decisive beat
    if peak_t is not None and 0 <= peak_t <= duration:
        for off in (-0.6, -0.3, 0.0, 0.3, 0.6):
            t = peak_t + off
            if 0 <= t <= duration:
                times.append(t)
    return sorted(set(round(t, 3) for t in times))


def _grab_jpegs(clip_path: str, times: list[float]) -> list[bytes]:
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return []
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out: list[bytes] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            continue
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2:
            out.append(buf.tobytes())
    cap.release()
    return out


def _detection_summary(shots, frames, peak_t) -> str:
    bits = [f"{len(shots)} shot(s)"]
    cuts = [round(s.start, 1) for s in (shots or [])[1:]]
    if cuts:
        bits.append("cuts at " + ", ".join(f"{c}s" for c in cuts))
    n_replay = sum(1 for s in (shots or []) if getattr(s, "is_replay", False))
    if n_replay:
        bits.append(f"{n_replay} replay-like shot(s)")
    if frames:
        with_ball = sum(1 for ft in frames if getattr(ft, "ball", None))
        avg_players = (sum(len(getattr(ft, "players", [])) for ft in frames)
                       / max(1, len(frames)))
        bits.append(f"ball visible {100 * with_ball // max(1, len(frames))}% of frames")
        bits.append(f"~{avg_players:.1f} players/frame")
    if peak_t is not None:
        bits.append(f"ball-speed peak ~{peak_t:.1f}s")
    return "; ".join(bits)


def _transcribe(clip_path: str, cfg: dict) -> list[dict]:
    """Commentary ASR via faster-whisper (optional). Empty on any failure."""
    try:
        from faster_whisper import WhisperModel
    except Exception:  # noqa: BLE001
        log.info("[perception] faster-whisper unavailable; no transcript")
        return []
    try:
        d = cfg.get("director", {})
        model = WhisperModel(d.get("asr_model", "base"),
                             device="auto", compute_type="default")
        segs, _ = model.transcribe(clip_path, vad_filter=True)
        return [{"start": float(s.start), "end": float(s.end),
                 "text": s.text.strip()} for s in segs]
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[perception] ASR failed: {exc}")
        return []
