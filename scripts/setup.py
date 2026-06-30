#!/usr/bin/env python3
"""One-command automatic setup: dependencies + PyTorch + ALL model weights.

Run it with the interpreter you want the project to use:

    py -m scripts.setup            # Windows (Python launcher)
    python -m scripts.setup        # Linux / macOS
    py scripts/setup.py            # also works (no package needed)

What it does, in order (each step is best-effort + clearly reported):
  1. PyTorch + torchvision  — auto-picks the CUDA wheel if an NVIDIA GPU is
     detected (via `nvidia-smi`), otherwise the CPU wheel. Deterministic index
     URLs so it behaves the same on Windows/Linux/macOS.
  2. requirements.txt       — everything else (ultralytics, supervision, kornia,
     trackers, easyocr, moviepy, gradio, ...).
  3. ffmpeg check           — system binary; auto-installs via the OS package
     manager when possible, else prints exact instructions.
  4. ALL model weights      — YOLO player/ball/pitch + faster-whisper + EasyOCR
     + COCO fallback, via src.modelhub.ensure_all_models() into ./models.
  5. Doctor                 — verifies torch/CUDA, ffmpeg and the model manifest.

IMPORTANT: this script imports ONLY the Python standard library at top level, so
it runs on a bare interpreter before any dependency exists. Project modules are
imported lazily, only after their dependencies have been installed.

Flags:
  --cpu              force the CPU PyTorch build (skip GPU autodetect)
  --cuda cu121|cu118 force a specific CUDA wheel
  --skip-torch       don't touch torch/torchvision (already installed)
  --skip-deps        don't install requirements.txt
  --skip-models      don't download model weights
  --skip-ffmpeg      don't check/install ffmpeg
  --no-pitch         don't download the pitch-keypoint model
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/{tag}"


# --------------------------------------------------------------------------- #
# tiny console helpers (no rich dependency — this runs before installs)
# --------------------------------------------------------------------------- #
def _c(code: str, text: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def step(msg):  print(_c("1;36", f"\n==> {msg}"))
def ok(msg):    print(_c("1;32", f"  OK  {msg}"))
def warn(msg):  print(_c("1;33", f"  !!  {msg}"))
def err(msg):   print(_c("1;31", f" ERR  {msg}"))
def info(msg):  print(f"      {msg}")


def run(cmd: list[str], **kw) -> int:
    info("$ " + " ".join(cmd))
    return subprocess.call(cmd, **kw)


def pip(*args: str) -> int:
    return run([sys.executable, "-m", "pip", *args])


# --------------------------------------------------------------------------- #
# GPU detection
# --------------------------------------------------------------------------- #
def detect_cuda_tag() -> str | None:
    """Return a torch CUDA wheel tag ('cu121'/'cu118') if an NVIDIA GPU is
    visible, else None. Reads the driver's max CUDA version from nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, timeout=20)
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if not m:
        return "cu121"                       # GPU present but unparsable -> safe default
    major, minor = int(m.group(1)), int(m.group(2))
    if major >= 12:
        return "cu121"
    if major == 11:
        return "cu118"
    return "cu118"                            # very old driver; best effort


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #
def install_torch(args) -> bool:
    step("PyTorch + torchvision")
    if args.skip_torch:
        warn("skipped (--skip-torch)")
        return True
    tag = None
    if args.cpu:
        info("forcing CPU build (--cpu)")
    elif args.cuda:
        tag = args.cuda
        info(f"forcing CUDA build (--cuda {tag})")
    else:
        tag = detect_cuda_tag()
        info(f"GPU autodetect: {'NVIDIA / ' + tag if tag else 'none -> CPU build'}")
    index = TORCH_CUDA_INDEX.format(tag=tag) if tag else TORCH_CPU_INDEX
    code = pip("install", "torch", "torchvision", "--index-url", index)
    if code == 0:
        ok(f"torch installed ({'CUDA ' + tag if tag else 'CPU'})")
        return True
    err("torch install failed — see pip output above")
    return False


def install_requirements(args) -> bool:
    step("Project dependencies (requirements.txt)")
    if args.skip_deps:
        warn("skipped (--skip-deps)")
        return True
    pip("install", "--upgrade", "pip")
    code = pip("install", "-r", str(ROOT / "requirements.txt"))
    if code == 0:
        ok("requirements installed")
        return True
    err("requirements install failed")
    return False


def ensure_ffmpeg(args) -> bool:
    step("ffmpeg (system binary)")
    if args.skip_ffmpeg:
        warn("skipped (--skip-ffmpeg)")
        return True
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        ok("ffmpeg + ffprobe found on PATH")
        return True
    warn("ffmpeg not found — attempting automatic install")
    system = platform.system()
    attempts: list[list[str]] = []
    if system == "Windows":
        if shutil.which("winget"):
            attempts.append(["winget", "install", "-e", "--id", "Gyan.FFmpeg",
                             "--accept-source-agreements", "--accept-package-agreements"])
        if shutil.which("choco"):
            attempts.append(["choco", "install", "ffmpeg", "-y"])
    elif system == "Darwin":
        if shutil.which("brew"):
            attempts.append(["brew", "install", "ffmpeg"])
    else:  # Linux
        if shutil.which("apt-get"):
            attempts.append(["sudo", "apt-get", "install", "-y", "ffmpeg"])
        elif shutil.which("dnf"):
            attempts.append(["sudo", "dnf", "install", "-y", "ffmpeg"])
    for cmd in attempts:
        if run(cmd) == 0 and shutil.which("ffmpeg"):
            ok("ffmpeg installed")
            return True
    err("could not auto-install ffmpeg. Install it manually and re-run:")
    if system == "Windows":
        info("winget install Gyan.FFmpeg   (or download from https://www.gyan.dev/ffmpeg/builds/)")
    elif system == "Darwin":
        info("brew install ffmpeg")
    else:
        info("sudo apt-get install -y ffmpeg")
    info("(then make sure `ffmpeg` and `ffprobe` are on your PATH)")
    return False


def download_models(args) -> bool:
    step("Model weights (YOLO player/ball/pitch + whisper + EasyOCR + COCO)")
    if args.skip_models:
        warn("skipped (--skip-models)")
        return True
    try:
        sys.path.insert(0, str(ROOT))
        from src.modelhub import (ensure_models, prefetch_runtime_models,  # noqa: E402
                                  report_models)
        from src.utils.io import load_config                               # noqa: E402
    except Exception as exc:  # noqa: BLE001
        err(f"cannot import project model tools (deps not installed?): {exc}")
        return False

    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    info("downloading detectors (player/ball" +
         ("/pitch" if not args.no_pitch else "") + ") ...")
    status = ensure_models(cfg, include_pitch=not args.no_pitch)
    info("downloading runtime weights (faster-whisper, EasyOCR, COCO fallback) ...")
    status.update(prefetch_runtime_models(cfg))

    report_models(str(ROOT / "models"))
    core_ok = bool(status.get("player"))
    if core_ok:
        ok(f"models ready: {status}")
    else:
        warn(f"core player model not fetched (will fall back to COCO yolov8x): {status}")
    return True            # non-fatal: pipeline degrades gracefully


def doctor() -> None:
    step("Doctor — verifying the environment")
    # torch / cuda
    try:
        import torch
        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "CPU"
        ok(f"torch {torch.__version__} | CUDA available: {cuda} | device: {name}")
    except Exception as exc:  # noqa: BLE001
        warn(f"torch not importable: {exc}")
    # ffmpeg
    if shutil.which("ffmpeg"):
        ok("ffmpeg on PATH")
    else:
        warn("ffmpeg NOT on PATH")
    # models dir
    mdir = ROOT / "models"
    pts = list(mdir.rglob("*.pt")) if mdir.exists() else []
    if pts:
        ok(f"{len(pts)} model file(s) in ./models")
    else:
        warn("no .pt model files found in ./models yet")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Automatic setup for AI Football "
                                             "Highlight Studio")
    ap.add_argument("--cpu", action="store_true", help="force CPU torch build")
    ap.add_argument("--cuda", choices=["cu121", "cu118"],
                    help="force a specific CUDA torch wheel")
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--skip-deps", action="store_true")
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-ffmpeg", action="store_true")
    ap.add_argument("--no-pitch", action="store_true",
                    help="don't download the pitch-keypoint model")
    args = ap.parse_args()

    os.chdir(ROOT)
    print(_c("1;35", "AI Football Highlight Studio — automatic setup"))
    info(f"python : {sys.version.split()[0]} ({sys.executable})")
    info(f"os     : {platform.system()} {platform.release()}")
    info(f"root   : {ROOT}")

    results: dict[str, bool] = {}
    results["torch"] = install_torch(args)
    if results["torch"]:
        results["deps"] = install_requirements(args)
    else:
        warn("skipping requirements until torch is installed")
        results["deps"] = False
    results["ffmpeg"] = ensure_ffmpeg(args)
    if results["deps"]:
        results["models"] = download_models(args)
    else:
        warn("skipping model download until dependencies are installed")
        results["models"] = False
    doctor()

    step("Summary")
    for k, v in results.items():
        (ok if v else err)(f"{k}: {'ready' if v else 'needs attention'}")
    all_ok = all(results.values())
    if all_ok:
        print(_c("1;32", "\nAll set. Next:"))
        info("py -m src.pipeline list-profiles")
        info("py -m src.pipeline studio input\\match.mp4 --profile tiktok --limit 1")
    else:
        print(_c("1;33", "\nSetup finished with warnings — see the lines marked "
                         "!! / ERR above and re-run after fixing."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
