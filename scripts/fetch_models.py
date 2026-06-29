#!/usr/bin/env python3
"""Download EVERY model + asset the pipeline needs for 100% offline operation.

Fetched into models/ (a persistent Volume on Modal):
  * YOLO player / ball / pitch detectors      (public HF repos, martinjolif/*)
  * faster-whisper model                       (commentary + captions)
  * EasyOCR detection/recognition weights      (scoreboard OCR)
  * generic COCO yolov8x.pt                     (player-model fallback)

Public Hugging Face repos — no auth required, but HF_TOKEN is honored if set.
Run once (e.g. on Modal via setup_models) to populate the models Volume:

    python scripts/fetch_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.modelhub import ensure_all_models, report_models   # noqa: E402
from src.utils.io import load_config, get_logger             # noqa: E402

log = get_logger()

if __name__ == "__main__":
    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    status = ensure_all_models(cfg)
    log.info(f"[fetch_models] result: {status}")
    report_models(str(ROOT / "models"))

    # The football YOLO models are what telestration truly needs; whisper/ocr
    # degrade gracefully. Fail the run only if a core detector is missing AND we
    # didn't deliberately skip (auto_download: false).
    if status.get("skipped"):
        sys.exit(0)
    core_ok = status.get("player", False)
    sys.exit(0 if core_ok else 1)
