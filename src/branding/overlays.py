"""Branding overlays — the layer that makes every clip unmistakably *yours*.

Driven by config/branding.yaml and rendered with the Pillow-based HUD engine
(`src/edit/overlay_render.py`) — no more ffmpeg `drawtext`. Every element is a
designed widget (rounded translucent card, channel-accent bar, drop-shadow
typography) placed inside the configured **safe zones** so it never overlaps the
broadcaster's own scoreboard / logos.

Elements and their timing (frame-accurate, fixes the static "GOAL - 2'" bug):
  * hook question — only the first ~1.6s
  * lower-third badge — "GOAL  67'" using the *correct* OCR minute, faded in
    after the hook so the two never stack
  * stats card — shot/sprint/speed/beaten on a card, top-right safe area
  * watermark — persistent, low-opacity

Intro/outro stings are real clips when provided, else generated Pillow title
cards. Concatenation still goes through `ff.standardize` so stream params match.
"""
from __future__ import annotations

from pathlib import Path

from ..detect.types import Moment
from ..utils.io import get_logger
from ..edit import ff
from ..edit import overlay_render as ovr

log = get_logger()

_HOOK_IDX: dict[str, int] = {}


def apply_branding(clip_path: str, moment: Moment, stats: dict, out_path: str,
                   cfg: dict, branding: dict, with_intro_outro: bool = True) -> str:
    """Burn the channel overlays. When `with_intro_outro` is False (compilation
    segment mode) the intro/outro stings are skipped — they are added once to
    the assembled reel instead."""
    w = cfg["_active_profile"]["width"]
    h = cfg["_active_profile"]["height"]
    encoder = ff.pick_encoder(cfg["render"]["encoder"])
    font = branding["channel"].get("font", "")
    accent = branding["channel"].get("primary_color", [0, 220, 255])
    zones = ovr.zones_from_cfg(cfg)
    clip_dur = ff.duration(clip_path) or 1e9

    overlays: list[ovr.Overlay] = []
    pngs: list[str] = []

    def png(tag: str) -> str:
        p = out_path.replace(".mp4", f"_{tag}.png")
        pngs.append(p)
        return p

    # ---- hook (first 1.6s only) ----
    hook_end = 1.6
    if branding.get("hook", {}).get("enabled"):
        hook = _pick_hook(branding, moment.kind)
        ov = ovr.hook_overlay(hook, w, h, font, png("hook"), zones,
                              accent=accent, start=0.0, end=min(hook_end, clip_dur))
        if ov:
            overlays.append(ov)

    # ---- lower third: GOAL  67'  (correct minute, after the hook) ----
    lt = branding.get("lower_third", {})
    if lt.get("enabled"):
        label = moment.kind.upper()
        if lt.get("show_minute") and moment.minute is not None:
            label += f"   {moment.minute}'"
        ov = ovr.lower_third_overlay(label, w, h, font, png("lt"), zones,
                                     accent=accent,
                                     start=min(hook_end + 0.1, clip_dur),
                                     end=clip_dur)
        if ov:
            overlays.append(ov)

    # ---- stats card ----
    so = branding.get("stats_overlay", {})
    if so.get("enabled") and stats:
        approx = "" if stats.get("metric") else "~"   # mark pixel estimates
        lines = []
        if so.get("show_shot_distance") and stats.get("shot_distance_m"):
            lines.append(f"Shot: {approx}{stats['shot_distance_m']:.0f} m")
        if so.get("show_sprint_distance") and stats.get("sprint_distance_m"):
            lines.append(f"Sprint: {approx}{stats['sprint_distance_m']:.0f} m")
        if so.get("show_top_speed") and stats.get("top_speed_kmh"):
            lines.append(f"Top speed: {stats['top_speed_kmh']:.0f} km/h")
        if so.get("show_players_beaten") and stats.get("players_beaten") is not None:
            lines.append(f"Beaten: {stats['players_beaten']}")
        ov = ovr.stats_card_overlay(lines, w, h, font, png("stats"), zones,
                                    accent=accent,
                                    start=min(hook_end + 0.3, clip_dur),
                                    end=clip_dur)
        if ov:
            overlays.append(ov)

    # ---- watermark ----
    wm = branding.get("watermark", {})
    if wm.get("enabled"):
        ov = ovr.watermark_overlay(wm.get("text", ""), w, h, font,
                                   png("wm"), zones,
                                   opacity=wm.get("opacity", 0.6))
        if ov:
            overlays.append(ov)

    branded = out_path.replace(".mp4", "_branded.mp4")
    ovr.composite(clip_path, overlays, branded, encoder)
    ovr.cleanup(pngs)

    if not with_intro_outro:
        Path(branded).replace(out_path)
        log.info(f"[branding] {moment.kind} (segment) -> {Path(out_path).name}")
        return out_path

    final = _add_intro_outro(branded, out_path, cfg, branding, encoder)
    log.info(f"[branding] {moment.kind} m={moment.minute} -> {Path(final).name}")
    return final


# --------------------------------------------------------------------------- #
def _pick_hook(branding: dict, kind: str):
    templates = branding["hook"].get("templates", {})
    pool = templates.get(kind) or templates.get("goal") or []
    if not pool:
        return None
    i = _HOOK_IDX.get(kind, 0)
    _HOOK_IDX[kind] = i + 1
    return pool[i % len(pool)]


def _add_intro_outro(clip_path: str, out_path: str, cfg: dict, branding: dict,
                     encoder: str) -> str:
    w = cfg["_active_profile"]["width"]
    h = cfg["_active_profile"]["height"]
    fps = cfg["render"]["fps"]
    parts: list[str] = []

    intro = branding.get("intro", {})
    if intro.get("enabled"):
        ip = intro.get("clip")
        if ip and Path(ip).exists():
            parts.append(ff.standardize(ip, out_path + ".intro.mp4", w, h, fps, encoder))
        else:
            parts.append(_text_card(intro.get("fallback_text", ""), w, h, fps,
                                    out_path + ".intro.mp4", branding, encoder))

    # body — already standardized geometry, but normalise for identical params
    parts.append(ff.standardize(clip_path, out_path + ".body.mp4", w, h, fps, encoder))

    outro = branding.get("outro", {})
    if outro.get("enabled"):
        op = outro.get("clip")
        if op and Path(op).exists():
            parts.append(ff.standardize(op, out_path + ".outro.mp4", w, h, fps, encoder))
        else:
            parts.append(_text_card(outro.get("cta_text", ""), w, h, fps,
                                    out_path + ".outro.mp4", branding, encoder))

    if len(parts) == 1:
        Path(parts[0]).replace(out_path)
        return out_path

    listfile = out_path + ".list.txt"
    with open(listfile, "w") as fh:
        for p in parts:
            fh.write(f"file '{Path(p).resolve()}'\n")
    # all parts are standardized identically -> safe stream-copy concat
    ff.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
        "-c", "copy", out_path,
    ], desc="concat intro+body+outro")
    return out_path


def _text_card(text: str, w: int, h: int, fps: int, out: str, branding: dict,
               encoder: str) -> str:
    """A generated intro/outro title card, rendered with Pillow (no drawtext)
    and given a silent stereo track so it concatenates cleanly."""
    font = branding["channel"].get("font", "")
    accent = branding["channel"].get("primary_color", [0, 220, 255])
    png = out.replace(".mp4", "_card.png")
    ovr.text_card_png(text, w, h, font, png, accent=accent)
    dur = 1.0
    ff.run([
        "ffmpeg", "-y",
        "-loop", "1", "-t", f"{dur}", "-i", png,
        "-f", "lavfi", "-t", f"{dur}", "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-vf", f"fps={fps},format=yuv420p,setsar=1",
        "-map", "0:v", "-map", "1:a",
        *ff.venc_args(encoder), "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-ac", "2", "-shortest", out,
    ], desc="text card")
    ovr.cleanup([png])
    return out
