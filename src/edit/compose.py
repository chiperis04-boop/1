"""Audio bed + captions + signature colour grade.

Final per-clip A/V treatment before branding:
  * picks a music track (round-robin from assets/music)
  * mixes original commentary/crowd + ducked music + optional crowd-roar SFX
  * burns word-grouped captions (TikTok style)
  * applies the channel colour grade

All FFmpeg calls go through `ff.run` (visible failures). Audio presence is
checked so the audio map target always exists (fixes B4); captions are written
to a sidecar textfile to avoid drawtext inline-escaping pitfalls (fixes M2).
"""
from __future__ import annotations

from pathlib import Path

from ..utils.io import get_logger
from . import ff

log = get_logger()

_MUSIC_IDX = 0


def compose_clip(clip_path: str, out_path: str, cfg: dict, branding: dict,
                 captions: list[dict] | None = None, add_music: bool = True) -> str:
    e = cfg["edit"]
    # In compilation mode the music is laid once over the whole reel, so
    # per-segment music is suppressed (add_music=False).
    music = _pick_music(e["audio"]["music_dir"]) if add_music else None
    crowd = e["audio"].get("crowd_sfx")
    encoder = ff.pick_encoder(cfg["render"]["encoder"])

    inputs = ["-i", clip_path]
    filt = []

    # ensure the base clip has an audio stream to mix against
    src_has_audio = ff.has_audio(clip_path)
    if not src_has_audio:
        inputs += ["-f", "lavfi", "-i",
                   "anullsrc=channel_layout=stereo:sample_rate=48000"]
    base_audio = "1:a" if not src_has_audio else "0:a"
    amix_parts = [f"[{base_audio}]"]
    next_idx = 2 if not src_has_audio else 1

    music_vol = e["audio"].get("music_volume", 0.35)
    if music:
        inputs += ["-stream_loop", "-1", "-i", music]
        filt.append(f"[{next_idx}:a]volume={music_vol}[mus]")
        amix_parts.append("[mus]")
        next_idx += 1

    if crowd and Path(crowd).exists():
        inputs += ["-i", crowd]
        filt.append(f"[{next_idx}:a]volume=0.6[crowd]")
        amix_parts.append("[crowd]")
        next_idx += 1

    if len(amix_parts) > 1:
        filt.append("".join(amix_parts) +
                    f"amix=inputs={len(amix_parts)}:duration=first:"
                    f"dropout_transition=2,dynaudnorm[aout]")
        amap = "[aout]"
    else:
        amap = base_audio

    # ---- video: grade (+ captions) ----
    vf = _grade_filter()
    cap_files: list[str] = []
    if cfg["edit"]["captions"]["enabled"] and captions:
        cap_filter, cap_files = _caption_drawtext(captions, cfg, out_path)
        if cap_filter:
            vf += "," + cap_filter
    filt.append(f"[0:v]{vf}[vout]")

    ff.run([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filt),
        "-map", "[vout]", "-map", amap,
        *ff.venc_args(encoder), "-c:a", "aac", "-b:a", "192k",
        "-shortest", out_path,
    ], desc="compose")

    for f in cap_files:
        try:
            Path(f).unlink()
        except OSError:
            pass
    log.info(f"[compose] music+grade+captions -> {Path(out_path).name}")
    return out_path


# --------------------------------------------------------------------------- #
def _pick_music(music_dir: str):
    global _MUSIC_IDX
    d = Path(music_dir)
    if not d.exists():
        return None
    tracks = sorted(str(p) for p in d.glob("*")
                    if p.suffix.lower() in {".mp3", ".wav", ".m4a", ".aac"})
    if not tracks:
        return None
    track = tracks[_MUSIC_IDX % len(tracks)]
    _MUSIC_IDX += 1
    return track


def _grade_filter() -> str:
    return "eq=contrast=1.06:saturation=1.12:gamma=0.98,curves=preset=lighter"


def _caption_drawtext(captions: list[dict], cfg: dict, out_path: str):
    """Return (filter_string, [sidecar_files]).

    Each caption line is written to its own UTF-8 textfile and referenced via
    drawtext `textfile=`, which avoids fragile inline escaping of punctuation.
    """
    font = cfg["edit"]["captions"]["font"]
    fontsize = cfg["edit"]["captions"].get("fontsize", 58)
    parts, files = [], []
    for i, c in enumerate(captions):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        tf = out_path.replace(".mp4", f"_cap{i:03d}.txt")
        Path(tf).write_text(text, encoding="utf-8")
        files.append(tf)
        start, end = float(c["start"]), float(c["end"])
        fontclause = f"fontfile={font}:" if font and Path(font).exists() else ""
        parts.append(
            f"drawtext={fontclause}textfile={tf}:expansion=none:"
            f"fontcolor=white:fontsize={fontsize}:borderw=4:bordercolor=black:"
            f"x=(w-text_w)/2:y=h*0.72:"
            f"enable='between(t,{start:.2f},{end:.2f})'"
        )
    return (",".join(parts), files) if parts else ("", [])
