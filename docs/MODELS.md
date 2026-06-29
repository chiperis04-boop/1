# Models & licensing

## Detection model (players + ball)

The pipeline expects a YOLO `.pt` at `models/yolov8x-football.pt`
(set by `vision.player_model` in `config/config.yaml`).

### Option A — generic COCO model (zero setup)
`scripts/download_models.sh` fetches `yolov8x.pt`. It detects `person` (class 0,
used as player) and `sports ball` (class 32). Works immediately; ball detection
on a tiny, fast-moving football is the weak point.

### Option B — football-specific model (recommended)
Community football detectors (player / ball / referee / goalkeeper) are on
**Roboflow Universe** and give far better ball + role detection. Two ways:

1. **Roboflow API** — set env vars then run the download script:
   ```bash
   export ROBOFLOW_API_KEY=...           # free account
   export ROBOFLOW_PROJECT=workspace/football-players-detection
   export ROBOFLOW_VERSION=1
   bash scripts/download_models.sh
   ```
   Export the trained weights and place them at `models/yolov8x-football.pt`.

2. **Bring your own** — train/fine-tune YOLOv8/11 on a football dataset and drop
   the `.pt` in `models/`. Update the `vision.classes` mapping if your label
   indices differ (e.g. `player: [1]`, `ball: [0]`).

### Dedicated ball tracking (optional, best ball trail)
For a clean ball trail, add **TrackNetV3** (badminton/tennis/football ball
tracker) or a ball-only YOLO and point `vision.ball_model` at it. The tracker
will prefer that model's ball detections over the generic class.

## Transcription
`faster-whisper` downloads the chosen `model_size` automatically on first run
and caches it. `small` is a good speed/accuracy balance for commentary; use
`medium`/`large-v3` for noisy multi-language broadcasts.

## OCR
`EasyOCR` downloads its detector/recogniser weights on first use for the
languages in `detect.scoreboard_ocr.languages`.

## Licensing — read before distributing

| Component | License | Implication |
|---|---|---|
| **Ultralytics YOLO** | **AGPL-3.0** | Fine for private/internal use and for content you produce. If you *distribute the software* or run it as a network service for others, AGPL applies to your code — or buy an Ultralytics commercial licence, or swap to an Apache/MIT detector (e.g. RT-DETR variants, YOLOX). |
| supervision | MIT | permissive |
| faster-whisper / CTranslate2 | MIT | permissive |
| EasyOCR | Apache-2.0 | permissive |
| PySceneDetect | BSD-3 | permissive |
| FFmpeg | LGPL/GPL | depends on build flags |
| Roboflow models | per-model | check each dataset/model licence on Universe |

**Content rights are separate from code licensing.** Match footage, broadcast
graphics, music and player likenesses have their own rights. Using full
broadcast matches may infringe the rightsholder's copyright regardless of the
tools. Use footage you have the right to edit (your own, licensed, or
rights-cleared), and license any music you add.
