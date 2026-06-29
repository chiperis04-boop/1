# Pipeline reference

This explains each stage, the file that implements it, and how to tune it.

```
ingest → detect → fuse → clip → track → telestrate → reframe → effects → compose → brand
```

## 1. Ingest — `src/ingest.py`
Probes the source, extracts 16 kHz mono audio for analysis, and builds a
downscaled proxy (`analysis_height`, default 540p) used by the cheap detectors.
The **original file** is always used for the final cut, so quality is preserved.

## 2. Detect — `src/detect/`
Four independent detectors emit `Signal(t, source, strength, meta)`:

| Detector | File | Signal | Notes |
|---|---|---|---|
| Audio energy | `audio_energy.py` | loudness z-score spikes | CPU; crowd roar = goals/chances |
| Scene/replay | `scene.py` | clusters of short cuts | replays follow goals |
| Scoreboard OCR | `scoreboard_ocr.py` | score change | **most precise goal signal** |
| Commentary | `commentary.py` | keyword hits | gives a type hint (goal/chance/card) |

### Tuning the audio detector
`detect.audio.zscore_threshold` (default 2.2): lower → more (noisier) moments,
higher → only the loudest peaks. `min_gap_seconds` merges nearby spikes.

### Setting the scoreboard ROI
OCR auto-crops the top-left strip by default. If your broadcast graphic sits
elsewhere, set fractional coordinates `[x1,y1,x2,y2]` in
`detect.scoreboard_ocr.roi`, e.g. a top-center bug:
```yaml
scoreboard_ocr:
  roi: [0.30, 0.02, 0.70, 0.14]
```
Only `A-B → A+1-B` / `A-B+1` increments count as goals (filters OCR noise).

## 3. Fuse — `src/detect/fusion.py`
Signals within an 8s window collapse into one `Moment`. Confidence is a weighted
blend (`fusion.weights`); the **kind** is inferred (a score change ⇒ `goal`, a
"saved" commentary hit ⇒ `save`, etc.). Moments below `min_confidence` are
dropped; the top `max_moments` are kept and padded (`clip.*_seconds`).

## 4. Clip — `src/edit/clipper.py`
Frame-accurate FFmpeg cut from the original, re-encoded so downstream stages get
exact boundaries. Goals get longer buildup/celebration padding.

## 5. Track — `src/vision/detect_track.py`
YOLO (Ultralytics) detects players + ball per frame; ByteTrack (via
`supervision`) assigns stable IDs. Ball gaps are interpolated. The **key player**
(protagonist) is the one most often nearest the ball in the final third of the
clip — this drives the spotlight, arrow and stats.

> If the wrong player is highlighted, the heuristic mis-voted (e.g. ball
> occluded). Options: provide a dedicated ball model (`vision.ball_model`),
> raise `imgsz`, or extend `_pick_key_player` with team-colour filtering.

## 6. Telestrate — `src/vision/telestration.py`
OpenCV draws (all toggleable in `telestration`):
- **spotlight** glow under the key player
- **motion arrow** tracing their recent path
- **ball trail** fading polyline
- **highlight zone** shaded free space ahead

Audio is remuxed back after drawing.

## 7. Reframe — `src/edit/reframe.py`
Action-aware crop to `9:16`/`1:1`. The crop centre follows ball → key player →
player centroid, then is exponentially smoothed (`reframe.smoothing`) so the
virtual camera glides. `mode: center` or `letterbox` disable tracking.

## 8. Effects — `src/edit/effects.py`
- **slow-mo** around the key beat (`setpts`/`atempo`)
- **freeze-zoom** call-out prepended before the action (the "watch this" beat)

## 9. Compose — `src/edit/compose.py`
Mixes original audio + ducked music (round-robin from `assets/music`) + a crowd
SFX swell, applies the broadcast colour grade, and burns TikTok-style captions
generated per-clip by `src/edit/captions.py` (faster-whisper word timestamps).

## 10. Brand — `src/branding/overlays.py`
Hook question (first ~1.5s), lower-third (`GOAL · 67'`), stats overlay
(`src/vision/stats.py`), watermark, and intro/outro stings (provided clips or
generated text cards). This layer is what makes outputs recognisably *yours*.

## Resumability
`output/<match>/work/moments.json` caches detection. Delete it to re-detect.
Per-clip intermediates live in `output/<match>/work/clip_NN/`.

## Extending
- **New detector:** emit `Signal`s and add to `_detect_and_fuse` + a `fusion.weights` entry.
- **New overlay:** add a drawtext/overlay filter in `overlays.py`.
- **Team/jersey filtering, pitch homography (metric stats), per-team colour
  telestration** are natural next additions — all build on the existing `TrackResult`.
