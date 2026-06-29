"""Clip extraction.

Cuts each fused Moment out of the *original* (full-quality) source. We re-encode
(not stream-copy) so cut points are frame-accurate, which downstream tracking
and reframing rely on. Failures surface via `ff.run`.
"""
from __future__ import annotations

from pathlib import Path

from ..detect.types import Moment
from ..utils.io import ensure_dir, get_logger
from . import ff

log = get_logger()


def extract_clips(src: str, moments: list[Moment], workdir: str, cfg: dict) -> list[str]:
    clip_dir = ensure_dir(Path(workdir) / "clips")
    encoder = ff.pick_encoder(cfg["render"]["encoder"])
    fps = cfg["render"]["fps"]
    hwaccel = ff.pick_hwaccel(cfg.get("ingest", {}).get("hwaccel", "auto"))
    decode_args = ["-hwaccel", hwaccel] if hwaccel else []
    paths: list[str] = []

    for i, m in enumerate(moments):
        out = str(clip_dir / f"clip_{i:02d}_{m.kind}.mp4")
        dur = max(0.5, m.end - m.start)
        ff.run([
            "ffmpeg", "-y",
            *decode_args,
            "-ss", f"{m.start:.3f}", "-i", src, "-t", f"{dur:.3f}",
            "-r", str(fps),
            *ff.venc_args(encoder),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            out,
        ], desc=f"cut clip {i}")
        paths.append(out)
        log.info(f"[clipper] {m.kind} @ {m.t:.1f}s ({dur:.1f}s) -> {Path(out).name}")

    return paths
