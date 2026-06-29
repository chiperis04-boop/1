# Football Highlight Studio

A fully local, GPU-accelerated, **open-source** pipeline that turns a **full match video**
into finished short-form highlight clips with **player telestration** (arrows, spotlights,
zones), music, sound design, captions and a **consistent branded look** that sets your
channel apart from generic goal compilations.

Everything runs on your own server. No paid APIs are required.

---

## What it does (end to end)

```
full_match.mp4
   │
   ▼
[1] INGEST        probe + normalize + extract audio
   │
   ▼
[2] DETECT        find candidate moments using 4 cheap signals:
   │                • audio energy spikes (crowd roar + commentator)
   │                • scene cuts / replays (PySceneDetect)
   │                • scoreboard OCR (score change = goal, ~100% precise)
   │                • commentary keywords (WhisperX transcript)
   │
   ▼
[3] FUSE          merge signals → ranked moments with confidence + type
   │                (goal / chance / save / skill / card)
   │
   ▼
[4] CLIP          cut each moment with smart pre/post padding
   │
   ▼
[5] VISION (GPU)  YOLO detects players + ball, ByteTrack tracks them
   │                across the clip
   │
   ▼
[6] TELESTRATE    draw arrows, spotlight the scorer, highlight runs/zones,
   │                ball trail, freeze-frame call-outs
   │
   ▼
[7] EDIT          action-aware 9:16 reframe, slow-mo on the key beat,
   │                freeze-zoom, music + crowd SFX mix, captions
   │
   ▼
[8] BRAND         intro sting, lower-thirds, hook question, stat overlay,
   │                signature color grade, outro CTA
   │
   ▼
finished_clip_9x16.mp4   (+ caption text + hashtags for posting)
```

---

## The unique elements that differentiate the channel

Plain goal cuts are a commodity. This pipeline bakes in a recognizable, repeatable style:

1. **Analyst-grade telestration** — automatic motion arrows, a moving spotlight under the
   scorer, highlighted free space and defensive errors. Few automated channels do this.
2. **Freeze-frame + zoom call-out** right before the decisive action ("watch this run →"),
   then it plays out. Highly shareable.
3. **Auto mini-stats** derived from tracking: shot distance, sprint distance, players beaten.
4. **Signature intro/outro sting + fixed color grade + font** = instant brand recognition.
5. **Hook question on screen** in the first second ("Spot the defender's mistake 👀") to
   drive comments and watch-time.
6. **"15-second breakdown" format** — a micro tactical analysis, not just a goal, which is
   what separates you from thousands of raw-cut channels.

All of these are produced automatically from config — no manual editing.

---

## Open-source stack

| Stage | Tool | License | GPU |
|---|---|---|---|
| Container / encode | FFmpeg | LGPL/GPL | optional (NVENC) |
| Audio energy | librosa, soundfile, numpy | ISC/BSD | no |
| Scene / replay detect | PySceneDetect | BSD-3 | no |
| Scoreboard OCR | EasyOCR | Apache-2.0 | yes (faster) |
| Commentary transcript | WhisperX / faster-whisper | BSD/MIT | yes |
| Player + ball detection | Ultralytics YOLO (v8/11) | AGPL-3.0 | yes |
| Tracking + annotation | Roboflow `supervision` (ByteTrack) | MIT | yes |
| Ball tracking (optional) | TrackNetV3 | MIT | yes |
| Editing / compositing | MoviePy + OpenCV | MIT / Apache | yes |
| Orchestration | Python 3.10+, Typer CLI | — | — |

> **License note:** Ultralytics YOLO is **AGPL-3.0**. For a personal/commercial channel that
> is usually fine, but if you ever redistribute the software you must comply with AGPL (or buy
> an Ultralytics commercial license, or swap to an Apache/MIT detector — see `docs/MODELS.md`).

---

## Quick start

### Easiest — one file, one command (self-bootstrapping)

Upload **`studio.py`** to your GPU server and run:

```bash
python3 studio.py
```

On first launch it fetches the project (if needed), creates a venv, installs all
dependencies, ensures `ffmpeg` is present (downloads a static build if missing),
downloads the models, and opens the **WebUI** at `http://<server>:7860`.

For a CUDA build of PyTorch, point it at the matching wheel index:

```bash
FHS_TORCH_INDEX=https://download.pytorch.org/whl/cu121 python3 studio.py
```

You can also drive the CLI through the same launcher:

```bash
python3 studio.py run input/match.mp4 --profile tiktok
python3 studio.py detect input/match.mp4
```

### Deploy on Modal (serverless GPU)

Running on [modal.com](https://modal.com)? Use **`modal_app.py`** (not `studio.py` —
Modal bakes deps into a container image instead of bootstrapping a venv):

```bash
pip install modal && modal token new
modal run modal_app.py::setup_models     # one-time: fill the models Volume
modal deploy modal_app.py                # prints your WebUI URL
```

Full guide + caveats (GPU choice, Volumes, big uploads, cost): **[docs/DEPLOY_MODAL.md](docs/DEPLOY_MODAL.md)**.

### Manual (full control)

```bash
# 0. system deps (Ubuntu/Debian GPU server)
sudo apt update && sudo apt install -y ffmpeg python3.10 python3.10-venv git

cd football-highlight-studio
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
bash scripts/download_models.sh

# launch the WebUI ...
python -m src.pipeline webui
# ... or the CLI
python -m src.pipeline run input/match.mp4 --profile tiktok
```

Finished clips land in `output/<match-name>/` (and are browsable in the WebUI).

See **[docs/SETUP.md](docs/SETUP.md)** for server/CUDA details,
**[docs/PIPELINE.md](docs/PIPELINE.md)** for how each stage works,
**[docs/AUDIT.md](docs/AUDIT.md)** for the v0.1→v0.2 correctness audit, and
**[docs/SKILLS.md](docs/SKILLS.md)** for the acceptance criteria.

---

## Project layout

```
football-highlight-studio/
├── studio.py                # ⭐ single-file self-bootstrapping launcher
├── README.md
├── requirements.txt
├── Makefile
├── config/
│   ├── config.yaml          # detection + edit tuning
│   └── branding.yaml        # your channel identity
├── scripts/
│   └── download_models.sh
├── app/
│   └── webui.py             # Gradio WebUI (upload + manage)
├── src/
│   ├── runner.py            # importable pipeline API (used by CLI + WebUI)
│   ├── pipeline.py          # CLI (typer)
│   ├── ingest.py
│   ├── detect/              # audio / scene / ocr / commentary / fusion
│   ├── vision/              # YOLO detect+track / telestration / stats
│   ├── edit/                # ff helpers / clipper / reframe / effects / compose / captions
│   ├── branding/            # overlays
│   └── utils/               # io + logging helpers
├── tests/
│   └── test_montage.py      # synthetic end-to-end montage test (no GPU)
├── docs/                    # SETUP / PIPELINE / MODELS / AUDIT / SKILLS
├── assets/                  # music, sfx, fonts, intro/outro (you provide)
├── models/                  # downloaded weights
├── input/                   # drop full matches here
└── output/                  # finished clips
```
