#!/usr/bin/env python3
"""Download every model the pipeline needs (player/ball/pitch) into models/.

Public Hugging Face repos — no auth required, but HF_TOKEN is honored if set.
Run once (e.g. on Modal via setup_models) to populate the models Volume:

    python scripts/fetch_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.modelhub import ensure_models          # noqa: E402
from src.utils.io import load_config, get_logger  # noqa: E402

log = get_logger()

if __name__ == "__main__":
    cfg = load_config(str(ROOT / "config" / "config.yaml"))
    status = ensure_models(cfg, include_pitch=True)   # fetch pitch too
    log.info(f"[fetch_models] result: {status}")
    ok = all(v for k, v in status.items() if k != "skipped")
    sys.exit(0 if ok or status.get("skipped") else 1)
