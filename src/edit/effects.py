"""Cinematic beats: slow-motion on the key moment and a freeze-frame zoom.

Both are built with FFmpeg and go through `ff.run` so failures surface with the
log tail. All outputs stay at the active profile WxH/fps with a valid audio
track, so downstream concatenation is safe.
"""
from __future__ import annotations

from pathlib import Path

from ..utils.io import get_logger
from . import ff

log = get_logger()

_MIN_CLIP_FOR_SLOWMO = 4.0   # seconds
_EDGE_GUARD = 0.4            # keep effects away from the very edges


def apply_slowmo(clip_path: str, key_t: float, out_path: str, cfg: dict) -> str:
    e = cfg["edit"]["effects"]
    if not e.get("slowmo_on_key", True):
        return clip_path

    dur = ff.duration(clip_path)
    if dur < _MIN_CLIP_FOR_SLOWMO:
        log.info("[effects] clip too short for slow-mo; skipping")
        return clip_path

    factor = float(e["slowmo_factor"])           # 0.4 => 40% speed
    window = float(e.get("slowmo_window", 2.5))
    # clamp the slowed window inside the clip with an edge guard (fixes M1)
    start = min(max(_EDGE_GUARD, key_t - window / 2), dur - _EDGE_GUARD - 0.1)
    end = min(start + window, dur - _EDGE_GUARD)
    if end - start < 0.5:
        log.info("[effects] no room for slow-mo window; skipping")
        return clip_path

    pts = 1.0 / factor
    atempo = max(0.5, min(2.0, factor))          # atempo valid range 0.5..2.0
    encoder = ff.pick_encoder(cfg["render"]["encoder"])

    vf = (
        f"[0:v]trim=0:{start:.3f},setpts=PTS-STARTPTS[v0];"
        f"[0:v]trim={start:.3f}:{end:.3f},setpts={pts:.3f}*(PTS-STARTPTS)[v1];"
        f"[0:v]trim={end:.3f},setpts=PTS-STARTPTS[v2];"
        f"[v0][v1][v2]concat=n=3:v=1:a=0[v]"
    )
    af = (
        f"[0:a]atrim=0:{start:.3f},asetpts=PTS-STARTPTS[a0];"
        f"[0:a]atrim={start:.3f}:{end:.3f},asetpts=PTS-STARTPTS,atempo={atempo:.3f}[a1];"
        f"[0:a]atrim={end:.3f},asetpts=PTS-STARTPTS[a2];"
        f"[a0][a1][a2]concat=n=3:v=0:a=1[a]"
    )
    ff.run([
        "ffmpeg", "-y", "-i", clip_path,
        "-filter_complex", vf + ";" + af,
        "-map", "[v]", "-map", "[a]",
        *ff.venc_args(encoder), "-c:a", "aac", "-b:a", "192k", out_path,
    ], desc="slowmo")
    log.info(f"[effects] slow-mo x{factor} around {key_t:.1f}s")
    return out_path


def freeze_zoom_intro(clip_path: str, key_t: float, out_path: str, cfg: dict) -> str:
    """Prepend a short freeze-frame-with-zoom call-out of the key-beat frame.

    The freeze still is rendered directly at the active profile WxH (fixes B1)
    and gets a silent stereo track so it concatenates cleanly with the body
    (fixes B2).
    """
    e = cfg["edit"]["effects"]
    if not e.get("freeze_zoom", True):
        return clip_path

    w = cfg["_active_profile"]["width"]
    h = cfg["_active_profile"]["height"]
    fps = cfg["render"]["fps"]
    scale = float(e["freeze_zoom_scale"])
    encoder = ff.pick_encoder(cfg["render"]["encoder"])

    dur_clip = ff.duration(clip_path)
    grab_t = min(max(0.0, key_t), max(0.0, dur_clip - 0.1))

    freeze_png = out_path.replace(".mp4", "_frame.png")
    ff.run(["ffmpeg", "-y", "-ss", f"{grab_t:.3f}", "-i", clip_path,
            "-frames:v", "1", freeze_png], desc="grab freeze frame")

    freeze_clip = out_path.replace(".mp4", "_freeze.mp4")
    dur = 1.2
    nframes = int(dur * fps)
    # zoom on a still rendered straight to profile size, with silent audio
    ff.run([
        "ffmpeg", "-y",
        "-loop", "1", "-t", f"{dur}", "-i", freeze_png,
        "-f", "lavfi", "-t", f"{dur}", "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-filter_complex",
        f"[0:v]scale={w*2}:{h*2},"
        f"zoompan=z='min(zoom+0.0015,{scale})':d={nframes}:fps={fps}:s={w}x{h},"
        f"format=yuv420p,setsar=1[v]",
        "-map", "[v]", "-map", "1:a",
        *ff.venc_args(encoder), "-c:a", "aac", "-b:a", "192k", "-shortest",
        freeze_clip,
    ], desc="freeze zoom still")

    # standardize the body too, then concat via filter (robust to tiny diffs)
    body = out_path.replace(".mp4", "_body.mp4")
    ff.standardize(clip_path, body, w, h, fps, encoder)

    ff.run([
        "ffmpeg", "-y", "-i", freeze_clip, "-i", body,
        "-filter_complex",
        "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]",
        "-map", "[v]", "-map", "[a]",
        *ff.venc_args(encoder), "-c:a", "aac", "-b:a", "192k", out_path,
    ], desc="concat freeze+body")
    log.info("[effects] prepended freeze-zoom call-out")
    return out_path
