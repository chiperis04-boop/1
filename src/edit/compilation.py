"""Duration-targeted compilation (reel) builder.

For 30-60s outputs a single moment is too short, so we stitch several rendered
*segments* (each = telestrated + reframed + effects + grade/captions + per-moment
lower-third, but WITHOUT per-segment music or intro/outro) into one reel:

    [intro] seg1 | seg2 | seg3 ... [outro]   + one continuous music bed

`select_for_duration` chooses how many segments to include to land inside the
target window. `build_compilation` standardises every segment (so concatenation
is safe), concatenates, lays a single ducked music bed across the whole thing,
then adds one intro/outro via the existing branding helper.
"""
from __future__ import annotations

from pathlib import Path

from ..utils.io import get_logger
from . import ff
from .compose import _pick_music
from ..branding.overlays import _add_intro_outro

log = get_logger()


def select_for_duration(items: list[dict], target: float, dur_max: float,
                        per_moment_max: float) -> list[dict]:
    """Pick items (each {'path','duration','confidence'}) to fill ~target seconds
    without exceeding dur_max. Highest-confidence first, then restored to
    chronological order. `per_moment_max` is informational (segments are already
    trimmed upstream)."""
    ordered = sorted(items, key=lambda x: x.get("confidence", 0), reverse=True)
    chosen, total = [], 0.0
    for it in ordered:
        d = min(it["duration"], per_moment_max) if per_moment_max else it["duration"]
        if total + d > dur_max and chosen:
            continue
        chosen.append(it)
        total += d
        if total >= target:
            break
    chosen.sort(key=lambda x: x.get("order", 0))
    log.info(f"[compilation] selected {len(chosen)} segments ~{total:.1f}s "
             f"(target {target:.0f}s, max {dur_max:.0f}s)")
    return chosen


def build_compilation(segments: list[str], out_path: str, cfg: dict,
                      branding: dict) -> str:
    """Assemble standardized segments into a finished reel."""
    if not segments:
        raise ValueError("no segments to compile")

    w = cfg["_active_profile"]["width"]
    h = cfg["_active_profile"]["height"]
    fps = cfg["render"]["fps"]
    encoder = ff.pick_encoder(cfg["render"]["encoder"])
    work = Path(out_path).parent

    # 1) standardize every segment so stream params match exactly
    std = []
    for i, seg in enumerate(segments):
        s = str(work / f"_seg{i:02d}.mp4")
        ff.standardize(seg, s, w, h, fps, encoder)
        std.append(s)

    # 2) concat (safe stream-copy after standardize)
    body = str(work / "_reel_body.mp4")
    listfile = str(work / "_reel.txt")
    with open(listfile, "w") as fh:
        for p in std:
            fh.write(f"file '{Path(p).resolve()}'\n")
    ff.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
            "-c", "copy", body], desc="concat segments")

    # 3) one continuous music bed across the whole reel
    bedded = _add_music_bed(body, str(work / "_reel_music.mp4"), cfg, encoder)

    # 4) single intro + outro
    final = _add_intro_outro(bedded, out_path, cfg, branding, encoder)
    log.info(f"[compilation] reel -> {Path(final).name} "
             f"({ff.duration(final):.1f}s)")
    return final


# --------------------------------------------------------------------------- #
def _add_music_bed(body: str, out: str, cfg: dict, encoder: str) -> str:
    music = _pick_music(cfg["edit"]["audio"]["music_dir"])
    if not music:
        return body
    vol = cfg["edit"]["audio"].get("music_volume", 0.35)
    # mix the reel's own audio (commentary/crowd) with a looped, ducked track
    ff.run([
        "ffmpeg", "-y", "-i", body, "-stream_loop", "-1", "-i", music,
        "-filter_complex",
        f"[1:a]volume={vol}[mus];"
        f"[0:a][mus]amix=inputs=2:duration=first:dropout_transition=2,"
        f"dynaudnorm[aout]",
        "-map", "0:v", "-map", "[aout]",
        *ff.venc_args(encoder), "-c:a", "aac", "-b:a", "192k",
        "-shortest", out,
    ], desc="music bed")
    return out
