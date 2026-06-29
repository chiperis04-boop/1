# Roadmap — closing the CV limitations with football-specific OSS

The v0.2 limitations are not dead-ends — the football computer-vision community
has purpose-built, open tools for each one. This document maps every limitation
to concrete projects, the integration approach in *this* codebase, and the
licensing/effort reality.

> Sources are linked inline. Content from third-party docs is paraphrased.

---

## The flagship dependency: `roboflow/sports`

[`roboflow/sports`](https://github.com/roboflow/sports) is the single most useful
project for us. Notably, the challenges it explicitly targets are *exactly* our
limitations: ball tracking, jersey-number reading, player tracking, player
re-identification, and camera calibration for speed/distance stats
([source](https://github.com/roboflow/sports)). It ships reference datasets +
models for:

- soccer **player** detection
- soccer **ball** detection (dedicated small-object model)
- soccer **pitch keypoint** detection (for homography / calibration)

plus notebooks for **team classification** and **radar (top-down) views**
([Roboflow tracking notebook](https://github.com/roboflow/notebooks/blob/main/notebooks/how-to-track-football-players.ipynb),
[calibration write-up](https://blog.roboflow.com/camera-calibration-sports-computer-vision/)).

Install: `pip install git+https://github.com/roboflow/sports.git`. Datasets/
weights live on Roboflow Universe; pitch-keypoint mirrors also exist on
Hugging Face (e.g. [Simon9/football-field-detection-roboflow](https://huggingface.co/Simon9/football-field-detection-roboflow)).

---

## Limitation 1 — Generic COCO model tracks the ball poorly

**Fix:** swap the generic detector for **football-specific weights** and add a
small-object strategy for the ball.

| Lever | What to use |
|---|---|
| Player/ball/ref/keeper classes | roboflow/sports `football-players-detection` + `football-ball-detection` models (YOLOv8). Community broadcast-tuned repos confirm the 4-class TV-feed setup works ([example](https://github.com/probablyabdullah/Football-Tracking-with-YOLOv5-Bytetrack)). |
| Tiny fast ball | dedicated ball model at higher `imgsz`, or **sliced/tiled inference (SAHI)** so the ball isn't lost at downscaled resolution; optionally **TrackNetV3** for trajectory. |
| Detection backbone | YOLOv8/11 today; **RF-DETR** is a strong low-latency alternative that handles dense occlusion well ([Roboflow RF-DETR tracker](https://blog.roboflow.com/american-football-player-tracker/)). |

**Integration:** already supported via `vision.player_model` / `vision.ball_model`.
This roadmap adds `vision.ball.slice_inference` (SAHI) and documents the model
sources in `scripts/download_models.sh` + `docs/MODELS.md`. **Effort: low.**

---

## Limitation 2 — Stats are pixel estimates, not metric

**Fix:** **camera calibration via pitch-keypoint homography.** Detect known pitch
landmarks per frame, solve a homography mapping image → a real pitch template
(105 × 68 m), then measure distance/speed in metres. Roboflow's calibration
work and recent papers do exactly this; keypoint-exploitation + geometric
constraints markedly improve calibration accuracy
([arXiv 2410.07401](https://arxiv.org/html/2410.07401v1)).

**Integration (scaffolded now):**
- `src/vision/pitch.py` — `PitchEstimator` runs a pitch-keypoint model, builds a
  smoothed per-frame homography to a `PITCH_TEMPLATE` (metres).
- `src/vision/stats.py` — when a homography is available, sprint distance, top
  speed and shot distance are computed in **metres / km·h⁻¹**; otherwise it
  falls back to the old pixel estimate and labels the numbers as approximate.

**Effort: medium** (needs the pitch-keypoint model + a robust visible-subset
homography solve). **Risk:** broadcast cuts/zoom change calibration constantly —
mitigated by per-frame solve + temporal smoothing + RANSAC.

---

## Limitation 3 — Protagonist mis-vote under occlusion

**Fix:** combine **team classification + possession + re-identification** instead
of "nearest player to the ball".

| Lever | What to use |
|---|---|
| Team assignment | roboflow/sports team-classification approach: crop players, embed (SigLIP) + cluster into two teams (+ keeper/ref). A lightweight HSV jersey-histogram clustering works offline with no extra models. |
| Possession | the team of the player nearest the ball at the key beat = attacking team; restrict the protagonist search to that team. |
| Re-ID across occlusion | appearance embeddings to re-link broken ByteTrack IDs (roboflow/sports lists player re-identification as a core task, [source](https://github.com/roboflow/sports)). |
| Disambiguation | optional **jersey-number OCR** to lock identity. |

**Integration (scaffolded now):** `src/vision/teams.py` provides a default HSV
team classifier and a `pick_key_player(track, ...)` that is team/possession aware
and temporally smoothed. SigLIP embeddings + jersey OCR are optional upgrades.
**Effort: medium.**

---

## Limitation 4 — Detection precision/recall depends on hand-tuned thresholds

**Fix:** add a learned **action-spotting** model that localises events (goal,
shot, card, corner…) directly from video, used as a high-confidence signal in
fusion — turning "loud crowd ≈ something" into "model says GOAL at 41:12".

| Option | Notes |
|---|---|
| [`SoccerNet/sn-spotting`](https://github.com/SoccerNet/sn-spotting) | reference task, baselines, the SoccerNet dataset/labels. |
| [`oslactionspotting`](https://pypi.org/project/oslactionspotting/) | a pip-installable library that unifies action-spotting algorithms — easiest to integrate. |
| [`recokick/ball-action-spotting`](https://github.com/recokick/ball-action-spotting) | 1st-place SoccerNet Ball Action Spotting 2023 solution. |
| [`arturxe2/ASTRA`](https://github.com/arturxe2/ASTRA) | transformer action spotter (ACM MMSports 2023). |

**Integration (scaffolded now):** `src/detect/action_spotting.py` exposes a
detector that, when a model is configured, emits `Signal(source="action_spotting",
kind_hint=...)`; `fusion.weights.action_spotting` is the highest weight so a
model hit dominates. Without a model it's a graceful no-op and the existing
audio/OCR/commentary signals carry on. **Effort: high** (model + features
pipeline) — hence a hook now, full wiring documented as the next milestone.

---

## Bonus capabilities these tools unlock (free differentiation)

- **Radar / top-down tactical view** (from the same homography) — a recognisable
  channel signature; roboflow/sports + the tracking notebook demonstrate it.
- **Possession %, pass maps, heatmaps** — once players are in pitch coordinates.
- **Auto jersey numbers** on the lower-third for the scorer.

---

## Suggested build order

1. **L1 models** (low effort, big quality jump): football player+ball weights.
2. **L3 team classifier** (medium): fixes the most visible telestration error.
3. **L2 homography** (medium): unlocks real stats *and* the radar view.
4. **L4 action spotting** (high): best detection accuracy; do last.

## Licensing reality (carry-over)
- roboflow/sports code: check the repo's `LICENSE`; Roboflow Universe **models/
  datasets carry per-asset licences** — verify each before commercial use.
- Ultralytics YOLO remains **AGPL-3.0** (see `docs/MODELS.md`).
- SoccerNet data has its own terms (research-oriented); confirm before shipping.
- As always, **broadcast footage / music / likeness rights are separate** and the
  user's responsibility.
