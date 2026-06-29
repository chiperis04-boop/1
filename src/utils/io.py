"""Shared IO helpers: config loading, logging, ffprobe, paths, JSON state."""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml
from rich.logging import RichHandler

# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "fhs") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = RichHandler(rich_tracebacks=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
    return logger


log = get_logger()


# --------------------------------------------------------------------------- #
# device resolution (graceful CPU fallback)
# --------------------------------------------------------------------------- #
def resolve_device(requested: str | None = "cuda") -> str:
    """Return ``'cuda'`` only when a usable CUDA device is actually present,
    otherwise ``'cpu'``.

    Config defaults to ``device: cuda`` for the L40S target, but every model
    backend (faster-whisper/ctranslate2, Ultralytics YOLO, EasyOCR) will *hard
    crash* if asked for CUDA on a host without a working driver
    (e.g. "CUDA driver version is insufficient for CUDA runtime version").
    Routing through this helper lets the GPU stages degrade to CPU instead of
    aborting the run, so a dry run / CPU box still produces clips."""
    req = (requested or "cpu").lower()
    if req != "cuda":
        return req
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_branding(path: str | Path = "config/branding.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# ffprobe
# --------------------------------------------------------------------------- #
def ffprobe(path: str | Path) -> dict[str, Any]:
    """Return basic media info: duration, fps, width, height, has_audio."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)

    info: dict[str, Any] = {"duration": float(data["format"].get("duration", 0.0))}
    has_audio = False
    for st in data.get("streams", []):
        if st.get("codec_type") == "video" and "width" in st:
            info["width"] = int(st["width"])
            info["height"] = int(st["height"])
            # fps may be like "30000/1001"
            num, _, den = st.get("avg_frame_rate", "30/1").partition("/")
            den = den or "1"
            info["fps"] = (float(num) / float(den)) if float(den) else 30.0
        if st.get("codec_type") == "audio":
            has_audio = True
    info["has_audio"] = has_audio
    return info


# --------------------------------------------------------------------------- #
# state persistence (resumable pipeline)
# --------------------------------------------------------------------------- #
def _default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    raise TypeError(f"not serializable: {type(o)}")


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
