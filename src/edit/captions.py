"""Generate word-grouped captions for a single clip from its audio.

Re-uses faster-whisper on the short clip (fast) and groups words into short
lines (TikTok style) with start/end timestamps relative to the clip.
"""
from __future__ import annotations

from ..utils.io import get_logger, resolve_device

log = get_logger()


def caption_clip(clip_path: str, cfg: dict) -> list[dict]:
    if not cfg["edit"]["captions"]["enabled"]:
        return []
    try:
        from faster_whisper import WhisperModel
    except Exception:
        log.warning("[captions] faster-whisper unavailable; skipping")
        return []

    device = resolve_device(cfg["vision"]["device"])
    compute = "float16" if device == "cuda" else "int8"
    max_words = cfg["edit"]["captions"]["max_words_per_line"]

    # Captions are cosmetic: a transcription failure (no model, CPU OOM, bad
    # audio) must never fail the clip — degrade to no captions instead.
    try:
        model = WhisperModel(cfg["detect"]["commentary"]["model_size"],
                             device=device, compute_type=compute)
        segments, _ = model.transcribe(clip_path, word_timestamps=True,
                                        vad_filter=True)

        lines: list[dict] = []
        buf: list = []
        for seg in segments:
            for word in (seg.words or []):
                buf.append(word)
                if len(buf) >= max_words:
                    lines.append({
                        "text": "".join(w.word for w in buf).strip(),
                        "start": buf[0].start, "end": buf[-1].end,
                    })
                    buf = []
        if buf:
            lines.append({"text": "".join(w.word for w in buf).strip(),
                          "start": buf[0].start, "end": buf[-1].end})
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[captions] transcription unavailable ({exc}); "
                    "rendering without captions")
        return []

    log.info(f"[captions] {len(lines)} caption lines (device={device})")
    return lines
