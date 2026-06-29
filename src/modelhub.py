"""Automatic model provisioning.

Downloads every model the pipeline needs from public Hugging Face repos (no
manual Roboflow step). Honors a Hugging Face token from the environment
(`HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`) when present — useful for higher rate
limits or gated repos — but the default repos are public, so it also works with
no token.

Models (all AGPL-3.0, football-specific):
  * player/goalkeeper/referee/ball detector  — martinjolif/yolo-football-player-detection
  * dedicated ball detector (small-object)   — martinjolif/yolo-football-ball-detection
  * pitch keypoint (pose) for homography      — martinjolif/yolo-football-pitch-detection

`ensure_models(cfg)` is idempotent: it copies each model to the path the config
expects (under models/, typically a persistent Volume on Modal) and skips files
that already exist. It is safe to call on every run and right before a video is
processed, so telestration works on upload with zero manual setup.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .utils.io import get_logger

log = get_logger()

# target-config-path  ->  (hf_repo_id, filename_in_repo)
_REGISTRY = {
    "player": ("martinjolif/yolo-football-player-detection",
               "yolo-football-player-detection.pt"),
    "ball":   ("martinjolif/yolo-football-ball-detection",
               "yolo-football-ball-detection.pt"),
    "pitch":  ("martinjolif/yolo-football-pitch-detection",
               "yolo-football-pitch-detection.pt"),
}


def _hf_token() -> str | None:
    return (os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            or None)


def _download(repo: str, filename: str, dest: str) -> bool:
    """Fetch one file from HF Hub and copy it to `dest`. Returns success."""
    dest_p = Path(dest)
    if dest_p.exists() and dest_p.stat().st_size > 0:
        return True
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[models] huggingface_hub unavailable: {exc}")
        return False
    try:
        dest_p.parent.mkdir(parents=True, exist_ok=True)
        cached = hf_hub_download(repo_id=repo, filename=filename, token=_hf_token())
        shutil.copy(cached, dest_p)
        log.info(f"[models] downloaded {repo}/{filename} -> {dest}")
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning(f"[models] failed to fetch {repo}/{filename}: {exc}")
        return False


def ensure_models(cfg: dict, include_pitch: bool | None = None) -> dict:
    """Ensure all required model files are present. Returns a status dict.

    Downloads the player + ball detectors always (when vision is enabled), and
    the pitch model when pitch calibration is enabled (or include_pitch=True).
    Falls back to the generic COCO YOLO for the player model if the football
    model can't be fetched, so the pipeline still runs.
    """
    if not cfg.get("models", {}).get("auto_download", True):
        return {"skipped": True}

    v = cfg.get("vision", {})
    status: dict[str, bool] = {}

    # player detector (required for any telestration)
    player_path = v.get("player_model", "models/football-player.pt")
    repo, fn = _REGISTRY["player"]
    status["player"] = _download(repo, fn, player_path)
    if not status["player"]:
        # last-resort: generic COCO model auto-downloads via ultralytics
        log.warning("[models] using generic yolov8x.pt as player fallback")
        v["player_model"] = "yolov8x.pt"

    # dedicated ball detector (optional but recommended)
    ball_path = v.get("ball_model")
    if ball_path:
        repo, fn = _REGISTRY["ball"]
        status["ball"] = _download(repo, fn, ball_path)
        if not status["ball"]:
            v["ball_model"] = None      # fall back to player model's ball class

    # pitch keypoints (only if calibration is on / explicitly requested)
    want_pitch = (include_pitch if include_pitch is not None
                  else v.get("pitch", {}).get("enabled", False))
    if want_pitch:
        pitch_path = v.get("pitch", {}).get("model", "models/football-pitch.pt")
        repo, fn = _REGISTRY["pitch"]
        status["pitch"] = _download(repo, fn, pitch_path)

    return status


if __name__ == "__main__":  # `python -m src.modelhub` — fetch everything
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.utils.io import load_config
    st = ensure_models(load_config(), include_pitch=True)
    log.info(f"[models] done: {st}")
