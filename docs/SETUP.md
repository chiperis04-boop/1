# Setup — GPU server

Tested target: Ubuntu 22.04, NVIDIA GPU (>=8 GB VRAM recommended), CUDA 12.1.

## 1. System dependencies

```bash
sudo apt update
sudo apt install -y ffmpeg git python3.10 python3.10-venv build-essential
# verify NVENC is available in your ffmpeg build (optional, for fast encode):
ffmpeg -hide_banner -encoders | grep nvenc
nvidia-smi   # confirm the driver + GPU are visible
```

If `ffmpeg` has no `h264_nvenc`, set `render.encoder: libx264` in `config/config.yaml`.

## 2. Python environment

```bash
cd football-highlight-studio
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
```

Install the **CUDA build of PyTorch first**, matched to your CUDA version
(see https://pytorch.org). Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then the rest:

```bash
pip install -r requirements.txt
```

Verify the GPU is used by torch:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Models

```bash
bash scripts/download_models.sh
```

This fetches a generic YOLOv8x (detects players + ball out of the box). For much
better accuracy, install a football-specific model — see [MODELS.md](MODELS.md).

`faster-whisper` and `EasyOCR` weights download automatically on first run.

## 4. Assets

Place a font, music and a crowd SFX (see `assets/README.md`). Minimum to run:

- `assets/fonts/Inter-Bold.ttf`  (any .ttf; update path in config if different)

Music/SFX/intro are optional — the pipeline degrades gracefully without them.

## 5. Run

```bash
# put a full match here
cp /path/to/match.mp4 input/

# detect-only first (sanity check the moments it finds)
python -m src.pipeline detect input/match.mp4

# full render to vertical TikTok clips
python -m src.pipeline run input/match.mp4 --profile tiktok
```

Outputs land in `output/match/NN_goal.mp4` with a matching `.txt` caption file.

## 6. Performance notes
- Detection (audio/scene/OCR/commentary) runs on the **downscaled proxy** and is
  cheap. Commentary transcription is the slowest detect step; lower
  `detect.commentary.model_size` to `base` for speed.
- The heavy GPU cost is **per-clip tracking** (YOLO every frame). Because we only
  track the short highlight clips — not the whole match — a 90-min match with
  ~20 moments processes in minutes on a modern GPU, not hours.
- Increase throughput by running clips in parallel (one process per GPU/stream).

### GPU acceleration knobs (e.g. NVIDIA L40S)
Building the analysis proxy from a multi-GB 1080p match is the main fixed cost.
On a CUDA GPU the pipeline uses **NVDEC to decode + NVENC to encode** the proxy
and hardware decode for clip cutting:
- `ingest.hwaccel: auto` — probes CUDA/VAAPI and uses it if the device actually
  initialises, else silently falls back to software (so it never crashes on a
  headless/misconfigured box). Force with `cuda`, disable with `none`.
- `ingest.proxy_encoder: h264_nvenc` — GPU proxy encode, auto-falls back to libx264.
- `detect.scene.frame_skip: 2` — process every Nth proxy frame; 2-3x faster scene
  detection on a full match with negligible accuracy loss.
- `render.encoder: h264_nvenc` — GPU encode for all clip/reel renders.

Rough end-to-end for a 2h23m 1080p match on an L40S with these on: ~45-75 min
(vision on, ~20 moments); ~40-80 min vision-off dry run. These are estimates —
measure on your first run.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `h264_nvenc not found` | set `render.encoder: libx264` (auto-handled, but you can force) |
| ffmpeg aborts on hwaccel | set `ingest.hwaccel: none` (the auto-probe should prevent this) |
| OCR finds no score | set `detect.scoreboard_ocr.roi` manually (see PIPELINE.md) |
| Too many/few moments | tune `detect.audio.zscore_threshold` and `fusion.min_confidence` |
| Reframe jitters | raise `edit.reframe.smoothing` toward 0.95 |
| Telestration on wrong player | the key-player heuristic mis-voted; see PIPELINE.md |
