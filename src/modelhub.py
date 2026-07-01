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


def ensure_models(cfg: dict, include_pitch: bool | None = None,
                  include_seg: bool | None = None) -> dict:
    """Ensure all required model files are present. Returns a status dict.

    Downloads the player + ball detectors always (when vision is enabled), the
    pitch model when pitch calibration is enabled (or include_pitch=True), and
    the YOLO*-seg model when under-player occlusion is enabled (or
    include_seg=True). Falls back to the generic COCO YOLO for the player model
    if the football model can't be fetched, so the pipeline still runs.
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
        # last-resort: prefer the prefetched COCO model on the Volume, else let
        # ultralytics auto-download yolov8x.pt at runtime.
        fallback = ("models/yolov8x.pt" if Path("models/yolov8x.pt").exists()
                    else "yolov8x.pt")
        log.warning(f"[models] using generic {fallback} as player fallback")
        v["player_model"] = fallback

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

    # player segmentation for under-player occlusion (opt-in)
    want_seg = (include_seg if include_seg is not None
                else cfg.get("telestration", {}).get("occlusion", False))
    if want_seg:
        status["seg"] = _ensure_seg(cfg)

    return status


def _ensure_seg(cfg: dict) -> bool:
    """Ensure the YOLO*-seg occlusion model is available. If it is a bare
    Ultralytics name (e.g. 'yolo11x-seg.pt') let Ultralytics manage the cache;
    if it is a repo path under models/ try an HF fetch. Best-effort."""
    tele = cfg.get("telestration", {})
    path = tele.get("occlusion_model", "yolo11x-seg.pt")
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return True
    # bare weight name -> Ultralytics auto-downloads on first YOLO(...) load
    if "/" not in path and p.suffix == ".pt":
        try:
            from ultralytics import YOLO
            YOLO(path)                      # triggers the cached download
            log.info(f"[models] seg model '{path}' cached via Ultralytics")
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning(f"[models] seg model '{path}' prefetch failed: {exc}")
            return False
    log.info(f"[models] seg model '{path}' not present; occlusion will fall back")
    return False


def ocr_storage_dir() -> str:
    """Directory EasyOCR caches its detection/recognition weights in.

    Defaults under models/ so on Modal it lands on the persistent Volume instead
    of ephemeral container storage. Overridable via FHS_OCR_DIR."""
    return os.environ.get("FHS_OCR_DIR", "models/easyocr")


def prefetch_runtime_models(cfg: dict) -> dict:
    """Pre-download the models that otherwise load *lazily at first render*:
    faster-whisper (commentary + captions), EasyOCR (scoreboard), and the
    generic COCO YOLO used as the player fallback.

    Without this, the first real render on a fresh container pays a large
    download (and re-pays it on every cold start, since those caches are
    ephemeral). Running it in setup_models writes them to the models Volume.
    Every step is best-effort: a failure is logged, not fatal."""
    status: dict[str, bool] = {}

    # ---- faster-whisper (Systran/faster-whisper-<size>) ----
    model_size = cfg.get("detect", {}).get("commentary", {}).get("model_size", "small")
    try:
        from faster_whisper import WhisperModel
        # device=cpu/int8 just to trigger the download into the HF cache
        # (HF_HOME); the GPU container reloads from the same cache at runtime.
        WhisperModel(model_size, device="cpu", compute_type="int8")
        status["whisper"] = True
        log.info(f"[models] faster-whisper '{model_size}' cached")
    except Exception as exc:  # noqa: BLE001
        status["whisper"] = False
        log.warning(f"[models] faster-whisper '{model_size}' prefetch failed: {exc}")

    # ---- EasyOCR (scoreboard) ----
    ocr = cfg.get("detect", {}).get("scoreboard_ocr", {})
    if ocr.get("enabled", True):
        try:
            import easyocr
            langs = ocr.get("languages", ["en"])
            store = ocr_storage_dir()
            Path(store).mkdir(parents=True, exist_ok=True)
            easyocr.Reader(langs, gpu=False, model_storage_directory=store,
                           download_enabled=True, verbose=False)
            status["easyocr"] = True
            log.info(f"[models] EasyOCR {langs} cached -> {store}")
        except Exception as exc:  # noqa: BLE001
            status["easyocr"] = False
            log.warning(f"[models] EasyOCR prefetch failed: {exc}")

    # ---- generic COCO YOLO fallback (only used if the football model is down) ----
    try:
        from ultralytics import YOLO
        dest = Path("models/yolov8x.pt")
        if not (dest.exists() and dest.stat().st_size > 0):
            m = YOLO("yolov8x.pt")                       # downloads to CWD/cache
            ckpt = getattr(m, "ckpt_path", None) or "yolov8x.pt"
            ckpt_p = Path(ckpt)
            if ckpt_p.exists() and ckpt_p.resolve() != dest.resolve():
                shutil.copy(ckpt_p, dest)
                # don't leave the download littering the working dir
                if ckpt_p.parent.resolve() == Path.cwd().resolve():
                    ckpt_p.unlink(missing_ok=True)
        status["yolov8x_fallback"] = dest.exists()
        log.info(f"[models] COCO fallback yolov8x.pt cached -> {dest}")
    except Exception as exc:  # noqa: BLE001
        status["yolov8x_fallback"] = False
        log.warning(f"[models] yolov8x fallback prefetch failed: {exc}")

    return status


def report_models(root: str = "models") -> None:
    """Log a manifest (path + human size) of everything in the models dir, so an
    operator can *see* that the downloads actually happened."""
    base = Path(root)
    if not base.exists():
        log.warning(f"[models] directory '{root}' does not exist")
        return
    total = 0
    log.info(f"[models] manifest for '{root}':")
    for p in sorted(base.rglob("*")):
        if p.is_file():
            sz = p.stat().st_size
            total += sz
            log.info(f"[models]   {sz/1e6:8.1f} MB  {p.relative_to(base)}")
    log.info(f"[models] total: {total/1e6:.1f} MB")


def ensure_all_models(cfg: dict) -> dict:
    """One call that fetches *everything* for 100% offline operation:
    YOLO player/ball/pitch + faster-whisper + EasyOCR + COCO fallback."""
    status = ensure_models(cfg, include_pitch=True)
    status.update(prefetch_runtime_models(cfg))
    return status


if __name__ == "__main__":  # `python -m src.modelhub` — fetch everything
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.utils.io import load_config
    st = ensure_models(load_config(), include_pitch=True)
    log.info(f"[models] done: {st}")
