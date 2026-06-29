"""Stage 1 — Ingest.

Probes the source match, extracts a 16 kHz mono audio track for analysis and
(optionally) creates a downscaled proxy video so detection runs faster. The
original file is always kept for the final high-quality render.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils.io import ensure_dir, ffprobe, get_logger
from .edit import ff

log = get_logger()


@dataclass
class MediaInfo:
    src_path: str
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool
    audio_path: str | None
    proxy_path: str | None


def ingest(src: str | Path, workdir: str | Path, cfg: dict) -> MediaInfo:
    src = Path(src)
    workdir = ensure_dir(workdir)
    info = ffprobe(src)
    log.info(
        f"[ingest] {src.name}: {info['duration']:.0f}s "
        f"{info['width']}x{info['height']} @ {info['fps']:.2f}fps "
        f"audio={info['has_audio']}"
    )

    audio_path = None
    if info["has_audio"]:
        audio_path = str(workdir / "audio_16k.wav")
        sr = cfg["ingest"]["audio_sample_rate"]
        ff.run([
            "ffmpeg", "-y", "-i", str(src),
            "-vn", "-ac", "1", "-ar", str(sr), "-f", "wav", audio_path,
        ], desc="extract audio")
        log.info(f"[ingest] extracted analysis audio -> {audio_path}")
    else:
        log.warning("[ingest] source has no audio track; audio detection disabled")

    # downscaled proxy for cheap frame analysis (OCR/scene). Detection only.
    # On a GPU box this decodes with NVDEC and encodes with NVENC, which makes
    # building the proxy from a multi-GB 1080p match a few minutes instead of
    # ~15-20 on CPU. Both degrade gracefully to software.
    proxy_path = str(workdir / "proxy.mp4")
    target_h = cfg["ingest"]["analysis_height"]
    hwaccel = ff.pick_hwaccel(cfg["ingest"].get("hwaccel", "auto"))
    proxy_encoder = ff.pick_encoder(cfg["ingest"].get("proxy_encoder", "libx264"))

    decode_args = ["-hwaccel", hwaccel] if hwaccel else []

    if proxy_encoder == "h264_nvenc":
        enc_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "30"]
    else:
        enc_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28"]

    ff.run([
        "ffmpeg", "-y", *decode_args, "-i", str(src),
        "-vf", f"scale=-2:{target_h}", "-an",
        *enc_args, proxy_path,
    ], desc="build proxy")
    log.info(f"[ingest] built {target_h}p proxy ({proxy_encoder}, "
             f"hwaccel={hwaccel}) -> {proxy_path}")

    return MediaInfo(
        src_path=str(src),
        duration=info["duration"],
        fps=info["fps"],
        width=info["width"],
        height=info["height"],
        has_audio=info["has_audio"],
        audio_path=audio_path,
        proxy_path=proxy_path,
    )
