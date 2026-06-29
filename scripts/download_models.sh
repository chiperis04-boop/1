#!/usr/bin/env bash
# Download / prepare the open-source models used by the pipeline.
# Re-run safe: skips files that already exist.
# POSIX-portable: works under bash and dash/sh (Modal may invoke either).
set -eu
# enable pipefail only when the shell supports it (bash/zsh); no-op under dash
(set -o pipefail) 2>/dev/null && set -o pipefail || true

SCRIPT="${BASH_SOURCE:-$0}"
ROOT="$(cd "$(dirname "$SCRIPT")/.." && pwd)"
MODELS="$ROOT/models"
mkdir -p "$MODELS"

echo "==> Models directory: $MODELS"

# --------------------------------------------------------------------------- #
# 1) Generic YOLO (COCO) — detects 'person' (player) and 'sports ball'.
#    Works out of the box. Good baseline before a football-specific model.
# --------------------------------------------------------------------------- #
if [ ! -f "$MODELS/yolov8x.pt" ]; then
  echo "==> Fetching YOLOv8x (generic COCO)..."
  python - <<'PY'
from ultralytics import YOLO
YOLO("yolov8x.pt")  # auto-downloads to the ultralytics cache
import shutil, os
from ultralytics.utils import SETTINGS
# copy from cache into ./models for a self-contained project
src = None
for root, _, files in os.walk(os.path.expanduser("~")):
    if "yolov8x.pt" in files:
        src = os.path.join(root, "yolov8x.pt"); break
if src:
    shutil.copy(src, os.path.join("models", "yolov8x.pt"))
    print("copied", src)
PY
fi

# --------------------------------------------------------------------------- #
# 2) Football-specific weights (RECOMMENDED).
#    Roboflow Universe hosts community football player+ball+referee models.
#    Set ROBOFLOW_API_KEY and the project slug, or drop your own .pt in models/
#    as yolov8x-football.pt (the name config/config.yaml expects).
# --------------------------------------------------------------------------- #
if [ ! -f "$MODELS/yolov8x-football.pt" ]; then
  if [ -n "${ROBOFLOW_API_KEY:-}" ] && [ -n "${ROBOFLOW_PROJECT:-}" ]; then
    echo "==> Downloading football model from Roboflow ($ROBOFLOW_PROJECT)..."
    python - <<PY
import os
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
proj = os.environ["ROBOFLOW_PROJECT"]   # e.g. "workspace/football-players-detection"
ws, name = proj.split("/")
version = int(os.environ.get("ROBOFLOW_VERSION", "1"))
model = rf.workspace(ws).project(name).version(version).model
print("Roboflow model ready; export weights and place at models/yolov8x-football.pt")
PY
  else
    echo "!! No football model found and ROBOFLOW_API_KEY/ROBOFLOW_PROJECT not set."
    echo "   Falling back to generic yolov8x.pt."
    echo "   To use a dedicated model, see docs/MODELS.md, then place it at:"
    echo "     $MODELS/yolov8x-football.pt"
    if [ -f "$MODELS/yolov8x.pt" ]; then
      cp "$MODELS/yolov8x.pt" "$MODELS/yolov8x-football.pt"
    fi
  fi
fi

# --------------------------------------------------------------------------- #
# 3) Pitch keypoint model (OPTIONAL — enables metric stats + radar view).
#    Used by src/vision/pitch.py. Train from the roboflow/sports
#    football-field-detection dataset, or use a community mirror, then place at
#    models/pitch-keypoints.pt and set vision.pitch.enabled: true.
#    Mirror example: https://huggingface.co/Simon9/football-field-detection-roboflow
# --------------------------------------------------------------------------- #
if [ ! -f "$MODELS/pitch-keypoints.pt" ]; then
  echo "!! Pitch keypoint model not present (optional, enables metric stats)."
  echo "   roboflow/sports 'football-field-detection' -> models/pitch-keypoints.pt"
fi

# --------------------------------------------------------------------------- #
# 4) Dedicated ball model (OPTIONAL — much better small-ball detection).
#    From roboflow/sports football-ball-detection. Place at models/ball.pt and
#    point vision.ball_model at it.
# --------------------------------------------------------------------------- #
if [ ! -f "$MODELS/ball.pt" ]; then
  echo "!! Dedicated ball model not present (optional, recommended)."
  echo "   roboflow/sports 'football-ball-detection' -> models/ball.pt"
fi

echo "==> Tip: install roboflow/sports tooling (player/ball/pitch/team utils):"
echo "       pip install git+https://github.com/roboflow/sports.git"

# --------------------------------------------------------------------------- #
# 5) faster-whisper model is fetched automatically on first run (cached).
#    EasyOCR weights likewise download on first use.
# --------------------------------------------------------------------------- #
echo "==> Done. faster-whisper + EasyOCR weights download lazily on first run."
