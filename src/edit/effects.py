"""Cinematic beats: slow-motion on the key moment and a freeze-frame zoom.

Both are built with FFmpeg and go through `ff.run` so failures surface with the
log tail. All outputs stay at the active profile WxH/fps with a valid audio
track, so downstream concatenation is safe.

Slow-motion quality (fixes the 03_goal.mp4 replay artefacts):
  * **Smooth video** — the slowed segment is motion-interpolated
    (`minterpolate`) so it generates true in-between frames instead of stuttering
    on duplicated ones. Falls back to plain `setpts` if interpolation fails.
  * **Clean audio** — the match audio under the slow-mo is *never* naively
    stretched (the cause of the low-pitched robotic drone). It is time-stretched
    with a chained, pitch-preserving `atempo` so it stays in sync with the
    slowed video, and by default **ducked** so the music bed carries the beat.
    Modes: ``duck`` (default) | ``mute`` | ``pitch_preserve``.
"""
from __future__ import annotations

from pathlib import Path

from ..utils.io import get_logger
from . import ff

log = get_logger()

_MIN_CLIP_FOR_SLOWMO = 4.0   # seconds
_EDGE_GUARD = 0.4            # keep effects away from the very edges


def _atempo_chain(factor: float) -> str:
    """A pitch-preserving atempo filter chain whose product == factor.

    A single `atempo` only accepts 0.5..2.0; for stronger slow-mo (e.g. 0.4) we
    cascade stages (0.4 -> atempo=0.5,atempo=0.8) so the audio stretches to match
    the slowed video *without* the pitch dropping into a robotic drone.
    """
    f = max(0.05, float(factor))
    stages: list[float] = []
    while f < 0.5:
        stages.append(0.5)
        f /= 0.5
    while f > 2.0:
        stages.append(2.0)
        f /= 2.0
    stages.append(f)
    return ",".join(f"atempo={s:.4f}" for s in stages)


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

    sm = e.get("slowmo", {}) if isinstance(e.get("slowmo"), dict) else {}
    audio_mode = sm.get("audio_mode", "duck")    # duck | mute | pitch_preserve
    duck_level = float(sm.get("duck_level", 0.15))
    interpolate = bool(sm.get("interpolate", True))

    pts = 1.0 / factor
    fps = int(cfg["render"]["fps"])
    encoder = ff.pick_encoder(cfg["render"]["encoder"])

    # ---- video: slowed middle segment, motion-interpolated for smoothness ----
    mid_v = f"setpts={pts:.5f}*(PTS-STARTPTS)"
    if interpolate:
        mid_v += (f",minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:"
                  f"me_mode=bidir:vsbmc=1")
    vf = (
        f"[0:v]trim=0:{start:.3f},setpts=PTS-STARTPTS[v0];"
        f"[0:v]trim={start:.3f}:{end:.3f},{mid_v}[v1];"
        f"[0:v]trim={end:.3f},setpts=PTS-STARTPTS[v2];"
        f"[v0][v1][v2]concat=n=3:v=1:a=0[v]"
    )

    # ---- audio: keep duration in sync; never drop the pitch ----
    chain = _atempo_chain(factor)
    if audio_mode == "pitch_preserve":
        mid_a = f"asetpts=PTS-STARTPTS,{chain}"
    elif audio_mode == "mute":
        mid_a = f"asetpts=PTS-STARTPTS,{chain},volume=0.0"
    else:  # duck (default): pitch-correct + quiet so the music bed carries it
        mid_a = f"asetpts=PTS-STARTPTS,{chain},volume={duck_level:.3f}"
    af = (
        f"[0:a]atrim=0:{start:.3f},asetpts=PTS-STARTPTS[a0];"
        f"[0:a]atrim={start:.3f}:{end:.3f},{mid_a}[a1];"
        f"[0:a]atrim={end:.3f},asetpts=PTS-STARTPTS[a2];"
        f"[a0][a1][a2]concat=n=3:v=0:a=1[a]"
    )

    cmd = [
        "ffmpeg", "-y", "-i", clip_path,
        "-filter_complex", vf + ";" + af,
        "-map", "[v]", "-map", "[a]",
        *ff.venc_args(encoder), "-c:a", "aac", "-b:a", "192k", out_path,
    ]
    try:
        ff.run(cmd, desc="slowmo")
    except ff.FFmpegError:
        if not interpolate:
            raise
        # motion interpolation can fail on pathological content — degrade to a
        # plain (still pitch-correct) slow-mo rather than failing the clip.
        log.warning("[effects] minterpolate failed; retrying without it")
        vf_plain = vf.replace(
            f",minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc:"
            f"me_mode=bidir:vsbmc=1", "")
        ff.run([
            "ffmpeg", "-y", "-i", clip_path,
            "-filter_complex", vf_plain + ";" + af,
            "-map", "[v]", "-map", "[a]",
            *ff.venc_args(encoder), "-c:a", "aac", "-b:a", "192k", out_path,
        ], desc="slowmo (no interp)")
    log.info(f"[effects] slow-mo x{factor} around {key_t:.1f}s "
             f"(audio={audio_mode}, interp={interpolate})")
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
