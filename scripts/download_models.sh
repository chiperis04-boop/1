#!/usr/bin/env bash
# Download all models the pipeline needs (player/ball/pitch) from public
# Hugging Face repos into ./models. Idempotent. Honors HF_TOKEN if set.
# POSIX-portable (works under bash and dash/sh).
set -eu
(set -o pipefail) 2>/dev/null && set -o pipefail || true

SCRIPT="${BASH_SOURCE:-$0}"
ROOT="$(cd "$(dirname "$SCRIPT")/.." && pwd)"
cd "$ROOT"

echo "==> Fetching football models (player / ball / pitch) from Hugging Face..."
python scripts/fetch_models.py

echo "==> Done. faster-whisper + EasyOCR weights download lazily on first run."
