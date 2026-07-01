"""Branding overlays — the layer that makes every clip unmistakably *yours*.

Driven by config/branding.yaml:
  * hook question (first ~1.6s, big centred text)
  * lower-third badge: moment type + match minute
  * stats overlay: shot/sprint distance, players beaten
  * persistent watermark
  * intro sting + outro CTA (provided clips or generated cards)

Every concatenated segment is forced through `ff.standardize` so the parts share
an identical stream layout (fixes the earlier concat/audio-mismatch bug). Text is
escaped via `ff.esc_drawtext`.
"""
from __future__ import annotations

from pathlib import Path

from ..detect.types import Moment
from ..utils.io import get_logger
from ..edit import ff

log = get_logger()

_HOOK_IDX: dict[str, int] = {}


def apply_branding(clip_path: str, moment: Moment, stats: dict, out_path: str,
                   cfg: dict, branding: dict, with_intro_outro: bool = True,
                   composer_typography: bool = False) -> str:
    """Burn the channel overlays. When `with_intro_outro` is False (compilation
    segment mode) the intro/outro stings are skipped — they are added once to
    the assembled reel instead.

    `composer_typography=True` selects the v2 MINIMAL overlay set: the Composer
    already burned the event hook (from the Director's EditPlan) and the
    reaction-gated stat cards, so branding must NOT draw a second static hook or
    the big stats overlay (that was the overlapping-hook / UI-overload look).
    In this mode branding only adds a compact lower-third + optional watermark +
    intro/outro, so the finished clip carries ONE clean set of overlays."""
    font = branding["channel"].get("font", "")
    fontclause = f"fontfile={font}:" if font and Path(font).exists() else ""
    encoder = ff.pick_encoder(cfg["render"]["encoder"])
    filters: list[str] = []
    sidecars: list[str] = []

    def add(text: str, opts: str):
        """Write text to a sidecar file and build a drawtext using textfile=,
        which avoids all inline-escaping pitfalls (emoji, quotes, %, :)."""
        if not text:
            return
        tf = out_path.replace(".mp4", f"_t{len(sidecars):02d}.txt")
        Path(tf).write_text(text, encoding="utf-8")
        sidecars.append(tf)
        filters.append(f"drawtext={fontclause}textfile={tf}:expansion=none:{opts}")

    for text, opts in _overlay_specs(moment, stats, branding, composer_typography):
        add(text, opts)

    vf = ",".join(filters) if filters else "null"
    branded = out_path.replace(".mp4", "_branded.mp4")
    ff.run([
        "ffmpeg", "-y", "-i", clip_path, "-vf", vf,
        *ff.venc_args(encoder), "-c:a", "aac", "-b:a", "192k", branded,
    ], desc="branding overlays")

    for f in sidecars:
        try:
            Path(f).unlink()
        except OSError:
            pass

    if not with_intro_outro:
        Path(branded).replace(out_path)
        log.info(f"[branding] {moment.kind} (segment) -> {Path(out_path).name}")
        return out_path

    final = _add_intro_outro(branded, out_path, cfg, branding, encoder)
    log.info(f"[branding] {moment.kind} -> {Path(final).name}")
    return final


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def _overlay_specs(moment: Moment, stats: dict, branding: dict,
                   composer_typography: bool) -> list[tuple[str, str]]:
    """Return the ordered (text, drawtext-opts) overlays to burn — the single
    source of truth for WHICH overlays appear (pure, so it is unit-testable).

    In `composer_typography` (v2) mode the Composer already burns the
    event-driven hook (from the Director's EditPlan) and the reaction-gated stat
    cards, so this returns NEITHER a second static hook NOR the big stats block
    — only a compact, edge-safe lower-third + watermark. That is what keeps the
    finished clip to ONE clean overlay set instead of the overlapping-hook /
    POSSESSION-over-player overload."""
    specs: list[tuple[str, str]] = []

    # ---- hook (first 1.6s) — v1 only (v2 hook comes from the Composer) ----
    if branding.get("hook", {}).get("enabled") and not composer_typography:
        hook = _pick_hook(branding, moment.kind)
        if hook:
            specs.append((hook,
                          "fontcolor=white:fontsize=64:borderw=5:bordercolor=black:"
                          "x=(w-text_w)/2:y=h*0.12:enable='between(t,0,1.6)'"))

    # ---- lower third: GOAL - 67' — both modes, inside the safe margin ----
    lt = branding.get("lower_third", {})
    if lt.get("enabled"):
        label = moment.kind.upper()
        # only show the minute when it is actually known (>0). OCR-detected goals
        # without a kick-off mapping have minute 0 -> "GOAL - 0'" looks broken, so
        # we drop it rather than print a wrong clock.
        if lt.get("show_minute") and moment.minute:
            label += f"  -  {moment.minute}'"
        fs = 38 if composer_typography else 46
        specs.append((label,
                      f"fontcolor=white:fontsize={fs}:box=1:boxcolor=black@0.55:"
                      f"boxborderw=14:x=w*0.06:y=h*0.80"))

    # ---- stats overlay — v1 only (v2 uses reaction-gated Composer stat cards) --
    so = branding.get("stats_overlay", {})
    if so.get("enabled") and stats and not composer_typography:
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
        for i, line in enumerate(lines):
            specs.append((line,
                          "fontcolor=white:fontsize=38:box=1:boxcolor=black@0.45:"
                          f"boxborderw=12:x=w-text_w-50:y=h*0.30+{i * 70}"))

    # ---- watermark — both modes (channel identity), respects its enabled flag -
    wm = branding.get("watermark", {})
    if wm.get("enabled") and wm.get("text"):
        specs.append((wm["text"],
                      f"fontcolor=white@{wm.get('opacity', 0.6)}:fontsize=34:"
                      "x=w-text_w-30:y=40"))
    return specs


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
    font = branding["channel"].get("font", "")
    fontclause = f"fontfile={font}:" if font and Path(font).exists() else ""
    dur = 1.0
    ff.run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:d={dur}:r={fps}",
        "-f", "lavfi", "-t", f"{dur}", "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-vf", f"drawtext={fontclause}text='{ff.esc_drawtext(text)}':expansion=none:"
               f"fontcolor=white:"
               f"fontsize=80:x=(w-text_w)/2:y=(h-text_h)/2,format=yuv420p,setsar=1",
        "-map", "0:v", "-map", "1:a",
        *ff.venc_args(encoder), "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-ac", "2", "-shortest", out,
    ], desc="text card")
    return out
