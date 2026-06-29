"""Commentary keyword detector.

Transcribes the match audio with faster-whisper and scans the transcript for
excitement keywords ("goal", "what a strike", "golazo", "saved", "penalty",
"red card"...). Each keyword hit emits a signal whose type hint (goal/chance/
card) helps the fusion stage classify the moment.

Transcribing a 90-minute match with the 'small' model on a GPU takes a few
minutes; lower `model_size` for speed or raise it for accuracy.
"""
from __future__ import annotations

from ..utils.io import get_logger
from .types import Signal

log = get_logger()


def detect_commentary(audio_path: str, cfg: dict) -> list[Signal]:
    c = cfg["detect"]["commentary"]
    if not c.get("enabled", True) or not audio_path:
        return []

    from faster_whisper import WhisperModel

    device = cfg["vision"]["device"]
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(c["model_size"], device=device, compute_type=compute)

    segments, _ = model.transcribe(audio_path, vad_filter=True)

    keymap = c["keywords"]
    signals: list[Signal] = []
    for seg in segments:
        text = (seg.text or "").lower()
        for kind, words in keymap.items():
            for w in words:
                if w in text:
                    signals.append(Signal(
                        t=float(seg.start), source="commentary",
                        strength=0.7 if kind == "goal" else 0.5,
                        meta={"kind_hint": kind, "phrase": w,
                              "text": seg.text.strip()[:120]},
                    ))
                    break  # one hit per kind per segment

    log.info(f"[commentary] {len(signals)} keyword hits")
    return signals
