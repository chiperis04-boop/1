"""Central FFmpeg/FFprobe helpers.

Every montage stage goes through here so behaviour is consistent and failures
are *visible* (we capture stderr instead of discarding it). Also provides:

  * ensure_tools()  — fail fast with a clear message if ffmpeg is missing
  * has_audio()     — ffprobe stream inspection
  * pick_encoder()  — probe NVENC once, fall back to libx264
  * standardize()   — force a clip to exactly 1 video + 1 stereo AAC@48k track at
                      a target WxH/fps (synthesising silence if needed). This is
                      the single source of truth that makes concatenation safe.
  * esc_drawtext()  — escape arbitrary text for the drawtext filter.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from ..utils.io import get_logger

log = get_logger()


class FFmpegError(RuntimeError):
    pass


def ensure_tools() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise FFmpegError(
            f"Required tool(s) not found on PATH: {', '.join(missing)}. "
            f"Install ffmpeg (which includes ffprobe)."
        )


def run(cmd: list[str], desc: str = "") -> None:
    """Run an ffmpeg command, raising FFmpegError with the relevant error lines
    on failure (filters out the noisy encoder-statistics block)."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        lines = proc.stderr.decode("utf-8", "ignore").strip().splitlines()
        keywords = ("error", "invalid", "no such", "unable", "failed",
                    "not found", "cannot", "does not", "conversion failed")
        flagged = [ln for ln in lines if any(k in ln.lower() for k in keywords)]
        # show flagged error lines (deduped, capped) plus a short tail
        seen, picked = set(), []
        for ln in flagged:
            if ln not in seen:
                seen.add(ln)
                picked.append(ln)
        detail = "\n".join(picked[-12:] or lines[-12:])
        raise FFmpegError(
            f"ffmpeg failed{(' (' + desc + ')') if desc else ''} "
            f"(exit {proc.returncode}):\n{detail}"
        )


def probe(path: str | Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {path}: {out.stderr.strip()}")
    return json.loads(out.stdout)


def has_audio(path: str | Path) -> bool:
    try:
        data = probe(path)
    except FFmpegError:
        return False
    return any(s.get("codec_type") == "audio" for s in data.get("streams", []))


def duration(path: str | Path) -> float:
    try:
        return float(probe(path)["format"].get("duration", 0.0))
    except (FFmpegError, KeyError, ValueError):
        return 0.0


@lru_cache(maxsize=1)
def _nvenc_available() -> bool:
    try:
        enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True).stdout
    except Exception:
        return False
    if "h264_nvenc" not in enc:
        return False
    # actually try a 1-frame encode — presence in the list isn't enough
    try:
        run([
            "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.1",
            "-c:v", "h264_nvenc", "-f", "null", "-",
        ], desc="nvenc probe")
        return True
    except FFmpegError:
        return False


def pick_encoder(requested: str) -> str:
    """Return a working video encoder, downgrading nvenc->libx264 if needed."""
    if requested == "h264_nvenc" and not _nvenc_available():
        log.warning("[ff] h264_nvenc unavailable; falling back to libx264")
        return "libx264"
    return requested


@lru_cache(maxsize=4)
def _hwaccel_works(method: str) -> bool:
    """Probe whether an ffmpeg hardware device can actually be created.
    `-hwaccel auto` is unsafe: on headless/misconfigured hosts ffmpeg may try
    CUDA/VAAPI and *abort* when the libs are half-present. We instead explicitly
    init the device and check it succeeds."""
    if not method or method == "none":
        return False
    try:
        run(["ffmpeg", "-hide_banner", "-init_hw_device", f"{method}=dev",
             "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
             "-frames:v", "1", "-f", "null", "-"], desc=f"{method} probe")
        return True
    except FFmpegError:
        return False


def pick_hwaccel(requested: str) -> str:
    """Resolve a *safe* hwaccel decode flag value, or '' for software.
    'auto' tries CUDA (NVDEC) and falls back to software if unavailable."""
    if not requested or requested == "none":
        return ""
    candidates = ["cuda", "vaapi"] if requested == "auto" else [requested]
    for m in candidates:
        if _hwaccel_works(m):
            return m
    if requested != "auto":
        log.warning(f"[ff] hwaccel '{requested}' unavailable; using software decode")
    return ""


def venc_args(encoder: str, intermediate: bool = False) -> list[str]:
    """Video-encoder args.

    `intermediate=True` returns near-visually-lossless settings for INTERNAL
    passes (graphics/reframe/slow-mo), so chaining several render stages no
    longer softens the image through repeated lossy generations. The final,
    delivered encode uses the normal (smaller) quality settings.
    """
    if intermediate:
        if encoder == "h264_nvenc":
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "14",
                    "-pix_fmt", "yuv420p"]
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "12",
                "-pix_fmt", "yuv420p"]
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p"]


def standardize(src: str, out: str, w: int, h: int, fps: int,
                encoder: str = "libx264", trim: float | None = None) -> str:
    """Force `src` to exactly 1 video + 1 stereo AAC@48k audio track at WxH/fps.

    Pads/letterboxes to preserve aspect, and synthesises a silent audio track
    when the source has none. After this, clips are safe to concatenate.
    `trim` (seconds), when given, caps the output duration (used for beat-aligned
    compilation cut points)."""
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
          f"fps={fps},format=yuv420p,setsar=1")

    cmd = ["ffmpeg", "-y", "-i", src]
    if not has_audio(src):
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        amap = "1:a"
    else:
        amap = "0:a"

    cmd += ["-vf", vf, "-map", "0:v:0", "-map", amap,
            *venc_args(encoder),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    if trim and trim > 0:
        cmd += ["-t", f"{float(trim):.3f}"]
    cmd += ["-shortest", out]
    run(cmd, desc=f"standardize {Path(src).name}")
    return out


def esc_drawtext(text: str) -> str:
    """Escape text for use inside a drawtext `text='...'` value."""
    if text is None:
        return ""
    # order matters: backslash first
    out = text.replace("\\", "\\\\")
    for ch in ("'", ":", "%"):
        out = out.replace(ch, "\\" + ch)
    # characters that terminate a filter / option
    for ch in ("[", "]", ",", ";"):
        out = out.replace(ch, "\\" + ch)
    return out



def mux_audio(video_only: str, audio_src: str, out: str,
              encoder: str = "libx264", intermediate: bool = False) -> str:
    """Combine a (possibly silent) video file with the audio from `audio_src`.

    Re-encodes video to H.264 for downstream compatibility and synthesises a
    silent stereo track when `audio_src` has none, so the result always has a
    valid audio stream.
    """
    cmd = ["ffmpeg", "-y", "-i", video_only]
    if has_audio(audio_src):
        cmd += ["-i", audio_src, "-map", "0:v:0", "-map", "1:a:0"]
    else:
        cmd += ["-f", "lavfi", "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000",
                "-map", "0:v:0", "-map", "1:a"]
    cmd += [*venc_args(encoder, intermediate), "-c:a", "aac", "-b:a", "192k",
            "-ar", "48000", "-ac", "2", "-shortest", out]
    run(cmd, desc="mux audio")
    return out


def minterpolate_expr(fps: int, quality: str = "mci") -> str:
    """Motion-interpolation filter for smooth slow-motion.

    `mci` (motion-compensated interpolation) synthesises genuinely new
    in-between frames so a stretched segment plays fluidly instead of stepping
    through duplicated frames. `blend` is a cheaper cross-fade fallback.
    """
    if quality == "blend":
        return f"minterpolate=fps={int(fps)}:mi_mode=blend"
    return (f"minterpolate=fps={int(fps)}:mi_mode=mci:mc_mode=aobmc:"
            f"me_mode=bidir:vsbmc=1")


class RawFrameSink:
    """Pipe raw BGR frames straight into a single ffmpeg encode (+ audio mux).

    Replaces the old "cv2.VideoWriter(mp4v) -> ff.mux_audio()" pattern, which
    compressed twice (lossy MPEG-4 part-2 intermediate, then re-encode to
    H.264). Here frames are encoded ONCE with the project's H.264 args and the
    audio is muxed from `audio_src` in the same pass, so there is no
    generational quality loss and no extra transcode.

    Usage:
        sink = ff.RawFrameSink(out, w, h, fps, encoder, audio_src=clip)
        for frame in frames:   # HxWx3 uint8, BGR, contiguous
            sink.write(frame)
        sink.close()
    """

    def __init__(self, out_path: str, width: int, height: int, fps: float,
                 encoder: str = "libx264", audio_src: str | None = None,
                 intermediate: bool = False):
        self.out_path = out_path
        self.width = int(width)
        self.height = int(height)
        self._stderr = tempfile.TemporaryFile()
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}", "-r", f"{fps}",
            "-i", "pipe:0",
        ]
        has_aud = bool(audio_src) and has_audio(audio_src)
        if has_aud:
            cmd += ["-i", audio_src, "-map", "0:v:0", "-map", "1:a:0"]
        else:
            cmd += ["-f", "lavfi", "-i",
                    "anullsrc=channel_layout=stereo:sample_rate=48000",
                    "-map", "0:v:0", "-map", "1:a"]
        cmd += [*venc_args(encoder, intermediate), "-c:a", "aac", "-b:a", "192k",
                "-ar", "48000", "-ac", "2", "-shortest", out_path]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL,
                                      stderr=self._stderr)

    def write(self, frame_bgr) -> None:
        # frame must be HxWx3 uint8 BGR matching (height, width)
        try:
            self._proc.stdin.write(memoryview(frame_bgr.tobytes()))
        except BrokenPipeError:
            self._raise_from_stderr()

    def close(self) -> str:
        try:
            self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        rc = self._proc.wait()
        if rc != 0:
            self._raise_from_stderr(rc)
        self._stderr.close()
        return self.out_path

    def _raise_from_stderr(self, rc: int | None = None) -> None:
        self._stderr.seek(0)
        tail = self._stderr.read().decode("utf-8", "ignore").strip().splitlines()
        self._stderr.close()
        raise FFmpegError(
            f"ffmpeg raw-frame encode failed (exit {rc}):\n"
            + "\n".join(tail[-12:])
        )
