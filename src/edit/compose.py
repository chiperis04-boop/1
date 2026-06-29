"""Audio bed + captions + signature colour grade.

Final per-clip A/V treatment before branding:
  * picks a music track (round-robin from assets/music)
  * mixes original commentary/crowd + ducked music + optional crowd-roar SFX
  * burns word-grouped captions (TikTok style) via the Pillow HUD engine
  * applies the channel colour grade

Captions are rendered with `overlay_render` (Pillow) instead of ffmpeg
`drawtext`, so they get real typography (Inter, stroke + shadow, a soft contrast
pill) and sit inside the safe zone. Audio is mixed + graded in one pass, then
the caption overlays are alpha-composited on top.
"""
from __future__ import annotations

from pathlib import Path

from ..utils.io import get_logger
from . import ff
from . import overlay_render as ovr

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

    # ---- video: colour grade (captions are composited in a 2nd pass) ----
    vf = _grade_filter()
    filt.append(f"[0:v]{vf}[vout]")

    graded = out_path.replace(".mp4", "_graded.mp4") if (
        cfg["edit"]["captions"]["enabled"] and captions) else out_path
    ff.run([
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filt),
        "-map", "[vout]", "-map", amap,
        *ff.venc_args(encoder), "-c:a", "aac", "-b:a", "192k",
        "-shortest", graded,
    ], desc="compose")

    # ---- captions: Pillow overlays composited over the graded clip ----
    if cfg["edit"]["captions"]["enabled"] and captions:
        overlays, pngs = _caption_overlays(captions, cfg, branding, out_path)
        if overlays:
            ovr.composite(graded, overlays, out_path, encoder)
        else:
            Path(graded).replace(out_path)
        ovr.cleanup(pngs)
        try:
            Path(graded).unlink()
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


def _caption_overlays(captions: list[dict], cfg: dict, branding: dict,
                      out_path: str):
    """Render each caption line to a full-frame RGBA PNG (Pillow) and return
    (overlays, png_paths) for alpha compositing."""
    w = cfg["_active_profile"]["width"]
    h = cfg["_active_profile"]["height"]
    font = (cfg["edit"]["captions"].get("font")
            or branding.get("channel", {}).get("font", ""))
    zones = ovr.zones_from_cfg(cfg)
    overlays, pngs = [], []
    for i, c in enumerate(captions):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        png = out_path.replace(".mp4", f"_cap{i:03d}.png")
        ov = ovr.caption_overlay(text, w, h, font, png, zones,
                                 start=float(c["start"]), end=float(c["end"]))
        if ov:
            overlays.append(ov)
            pngs.append(png)
    return overlays, pngs
