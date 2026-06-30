# Audit Final 3 — Deploy stabilization (v1 Classic + v2 Studio)

Scope: fix the WebUI deploy blocker on Modal and audit **both** pipelines —
v1 "Classic" (`src/runner.py`) and v2 "Studio" (`src/studio_pipeline.py`) — for
correctness, reliability and fitness for the stated goal.

Branch: `feat/v2-studio-pipeline` (PR #3). All changes here build on that branch.

## Status legend
- ✓ **Verified** — exercised locally (no GPU) and passing.
- ○ **Implemented / GPU-verify** — code is correct on read-through and wired
  consistently, but its runtime path needs an NVIDIA GPU (YOLO/track/OCR/whisper)
  and is only fully provable on the L40S.
- ⚠️ **Risk** — works but has a caveat worth knowing.
- ✗ **Missing** — not present / not done.

## How this was verified (no-GPU environment)
- Python 3.11.15 venv with the **exact pinned web stack** + numpy/pyyaml/rich.
- ffmpeg/ffprobe 7.x (GPL static, with `drawtext`/libfreetype) on PATH.
- The GPU stages (YOLO detect, BoT-SORT+CMC, EasyOCR jersey/scoreboard,
  faster-whisper, pitch homography, K-means team colours on real frames) are
  **not** runnable here and are marked ○ accordingly.

---

## A. The WebUI blocker — root cause & holistic fix  ✓ Verified

### Symptom
`GET /` on Modal raised `TypeError: unhashable type: 'dict'` deep in
`jinja2 environment cache` (called from `gradio/routes.py` →
`templates.TemplateResponse(...)`). This was the *second* Gradio-stack crash in
a row (the first was `gradio_client` `"argument of type 'bool' is not iterable"`
in `get_api_info`).

### Root cause (one cause, two symptoms)
`gradio==4.44.1` (Sept 2024) declares only `fastapi<1.0` and puts **no upper
bound** on `starlette` / `pydantic`. Left unpinned, pip resolves *much newer*
transitive deps:
- **starlette 1.x** changed the `TemplateResponse(name, context)` positional
  call order Gradio 4.44.1 still uses → Jinja received a **dict** where it
  expected a template-name string and tried to hash it → `unhashable type: 'dict'`.
- **newer pydantic** emits JSON schemas with **boolean** `additionalProperties`,
  which `gradio_client==1.3.0` can't walk → `"argument of type 'bool' is not iterable"`.

Both are version-contract breaks, not logic bugs. Patching them one-by-one
(monkeypatch after monkeypatch) is an anti-pattern.

### Decision: pin the whole web stack to the Gradio-4.44.1 era (coherent lock)
Rather than chase 5.x (which would require re-validating the entire `Blocks`
UI + streaming generators + `mount_gradio_app`), we lock the proven 4.44.1-era
set, which is internally consistent and fixes **both** bug classes at the source:

```
gradio==4.44.1
gradio-client==1.3.0
fastapi==0.115.0
starlette==0.38.6      # last 0.x keeping gradio 4's TemplateResponse call order
pydantic==2.9.2        # schema shape gradio_client 1.3.0 can parse
pydantic-core==2.23.4  # matched to pydantic 2.9.2
orjson==3.10.7
tomlkit==0.12.0        # exact pin gradio 4.44.1 requires
markupsafe>=2.0,<3
jinja2>=3.1,<4
uvicorn>=0.23,<0.31
websockets>=10.0,<13.0 # gradio_client 1.3.0 cap
huggingface-hub>=0.23,<1.0  # keeps HfFolder that gradio imports (0.36.2 resolved)
pillow>=10.0,<11       # gradio 4.44.1 requires pillow<11
```

`app/webui.py` keeps a small defensive shim (`_patch_gradio_client_schema_bug`)
that degrades a non-dict / bool schema node to `"Any"/"bool"` instead of
raising. With the pins above it is a **no-op** (verified `get_api_info()` works
without it) — kept only as belt-and-suspenders, not as the fix.

### Verification (local, real HTTP)
- Minimal `Blocks` app (Radio engine switch + slider + streaming generator)
  mounted via `mount_gradio_app` + `FastAPI` TestClient: **GET / = 200**,
  **GET /config = 200**, `get_api_info()` OK.
- The **actual** `app.webui.build()` UI (the real form, both engines) mounted
  identically to `modal_app.web`: **GET / = 200**, **GET /config = 200**,
  `get_api_info()` OK.
- Installed versions confirmed: gradio 4.44.1, gradio_client 1.3.0,
  fastapi 0.115.0, starlette 0.38.6, pydantic 2.9.2, jinja2 3.1.6,
  markupsafe 2.1.5, huggingface_hub 0.36.2 (`HfFolder` present).

> DoD #1 (GET / = 200, UI loads, no crash) — **met locally**; on Modal it then
> depends only on the image building, which uses this same `requirements.txt`.

---

## B. Dependency reproducibility  ✓ Verified (resolve) / ⚠️ heavy stack ranged

- `uv pip compile requirements.txt` (py3.11) resolves the **entire** tree with
  **no conflicts** (exit 0), and crucially the heavy CV/ML stack does **not**
  force-upgrade the pinned web stack (gradio/starlette/pydantic/jinja2 held).
- The new **v2 packages are now pinned** for reproducibility (were `>=`):
  `kornia==0.8.3`, `trackers==2.5.0`, `google-generativeai==0.8.6`,
  `moviepy==2.2.1` (de-duplicated: the old `moviepy>=1.0.3` line was removed).
- ⚠️ The core ML libs (`torch`, `ultralytics`, `supervision`, `easyocr`,
  `faster-whisper`, `opencv-python`, `scikit-learn`, `numpy`) remain
  lower-bounded by design — pinning exact CUDA wheels is environment-specific
  and Modal installs `torch`/`torchvision` separately in the image. Resolver
  currently lands on numpy 2.4.x / torch 2.12.x / opencv 4.13 / ultralytics 8.4.

> DoD #5 (UI + new v2 packages pinned coherently & reproducibly) — **met** for
> the fragile/blueprint set; heavy ML libs intentionally ranged (documented).

---

## C. Config contract  ✓ Verified
Every config key the code reads exists in `config/config.yaml`:
- `detect.scout.*` (goal/pre/post seconds, verify_radius, verified_goal_boost,
  merge_gap) ✓
- `director.*` (backend, model, base_url, sample_fps, max_frames, temperature,
  hooks) ✓
- `tracking.*` (backend, **gmc_method**, track thresholds, track_buffer,
  match_thresh, with_reid, **kalman_process_noise/measurement_noise**,
  smoothing) ✓
- `graphics.*` (enabled, homography_solver) ✓
- `possession.*` (radius_m, min_frames, bridge_frames) ✓
- `vision.jerseys.*` (enabled, sample_every_frames, min_box_height_px,
  min_confidence, min_reads) ✓
- `vision.teams.*` (enabled, method, n_teams) ✓
- `telestration.team_halos` ✓
Optional keys read with safe defaults (not required in the file):
`telestration.trail_length` (default 30), `render.typography_engine`
(default `pillow`).

---

## D. WebUI ↔ pipeline contract  ✓ Verified
- `run_btn.click(inputs=[...])` has **18** inputs; `render_job(...)` has **18**
  positional params — exact match.
- Outputs: 5 (`logbox, clip_dd, preview, caption, files`); every `yield` in
  `render_job` (early-return, start, both progress branches, error, final)
  produces a **5-tuple** — consistent.
- The single `render_job` handles **both** engines via `is_v2 =
  engine.startswith("studio")`, mapping the 5 v2-only toggles into a `run_studio`
  override dict, and the v1 toggles into a `run_pipeline` override dict. Progress
  streaming handles both shapes: v1 `Progress` object (`kind == "p"`) and v2
  `(stage, pct, msg)` (`kind == "pv2"`).

---

## E. v1 "Classic" pipeline  ✓ Verified (CPU e2e)
`tests/test_montage.py` and `tests/test_audit.py` pass end-to-end on CPU with a
synthetic match:
- reframe (letterbox) → slow-mo → freeze-zoom → compose (grade+captions+music)
  → branding (intro/outro) → compilation reel. Every output is **1 H.264 video +
  1 AAC stereo** at the exact profile geometry (1080×1920). ✓
- `ff.pick_encoder("h264_nvenc")` correctly falls back to `libx264` with no GPU. ✓
- `ff.standardize` synthesizes a silent stereo AAC track for video-only input,
  making concat safe. ✓
- Runner isolates per-clip failures (records `stage_failed` + `error`) instead of
  aborting the batch. ✓
- Vision/telestration stages (YOLO + ByteTrack via supervision) ○ GPU-verify.

> DoD #4 (each mp4 = 1 H.264 + 1 AAC stereo @ profile WxH/fps) — **verified** for
> the v1 montage/branding/compilation chain on CPU.

---

## F. v2 "Studio" pipeline

### F.1 Logic verified without a GPU  ✓ Verified
New `tests/test_studio.py` (13 tests, all green) covers the pure decision logic:
- **Audio-safe slow-mo**: `_atempo_chain` chains `atempo` for factors < 0.5
  (e.g. 0.4 → `atempo=0.5,atempo=0.8`), every stage stays within ffmpeg's legal
  `[0.5, 2.0]`, and the product equals the requested factor (tested 0.4, 0.1,
  0.6, 1.0). ✓
- **Cameraman smoothing**: constant-velocity `_kalman_1d` reduces MSE-to-truth
  and frame-to-frame jitter on a noisy camera pan; `_kalman_smooth` degrades to
  the EMA fallback on a bad cfg value without crashing; `_focus_points` fuses
  hero+ball, falls back to ball-only / players-centroid / last-known. ✓
- **Possession**: per-frame nearest-holder → confirmed runs with `min_frames`
  gating and short-dropout **bridging** into one run; far ball yields no false
  run; team possession share computed. ✓
- **Jersey number parsing**: `#NN` / `no. NN` / `number NN` / `player N`
  extracted from a Director description; prose → `None`; ambiguous number maps
  to the **highest-confidence** track. ✓
- **Hero resolution order**: jersey-number match **>** team-aware
  nearest-to-ball **>** plain geometric — verified in that exact priority. ✓
- **Team colours**: per-track club-colour lookup with caller default on unknown;
  `pick_key_player` restricts the protagonist to the attacking team. ✓
- **Scout dedupe**: clustered detections (action-spotting + OCR + replay on one
  goal) merge into a single window, keeping the goal label, `verified` flag,
  unioned sources and the score string. ✓

### F.2 Orchestration wiring  ✓ Verified (read-through)
`run_studio` per-clip chain is internally consistent (every call signature
traced): `ingest → scout → clip → [track_only → build_plan → director →
homography(opt) → analytics → replan → draw_graphics(original space) → render
(CMC crop to 9:16) → finish(slow-mo+typography) → branding]`. CropPlan is
duck-type compatible with the analytics/teams/jersey/possession readers
(`.frames[*].idx/.players/.ball`, `.fps`, `.hero_id`). Graphics are drawn in
**original** pixel space **before** the crop, exactly as the coordinate-space
note requires. Each per-clip failure is caught and recorded
(`stage_failed`/`error`); the batch continues. ✓

### F.3 GPU-only runtime paths  ○ Implemented / GPU-verify
These read correctly and degrade gracefully (CPU/`resolve_device`, try/except
around model loads) but need the L40S to prove:
- YOLO player/ball detection + **BoT-SORT with GMC/CMC** (`tracking.gmc_method`
  written into a generated `botsort.yaml`; roboflow `trackers` alt backend).
- Telestration drawing (team-colour foot ellipses, jersey-# label, ball trail,
  live POSSESSION plate).
- EasyOCR jersey-number reading (and scoreboard OCR for scout verification).
- faster-whisper commentary/captions.
- Pitch homography (cv2 RANSAC default; kornia `find_homography_dlt` /
  `find_homography_lines_dlt` solvers wired — note: the real Kornia function is
  `find_homography_lines_dlt`, correctly used).
- K-means team-colour clustering on real torso crops.

> DoD #2/#3 (upload → stream → downloadable clips; team-colour halos, POSSESSION
> plate, jersey hero-lock with geometric fallback, non-jittery CMC reframe) —
> logic ✓ verified; full pixel output is ○ **verify on GPU (L40S)**.

---

## G. Auto-installer  ✓ Verified (no-GPU paths)
- `python -m scripts.setup --help` works on a bare interpreter (only stdlib at
  top level; project modules imported lazily). ✓
- `detect_cuda_tag()` returns `None` cleanly when `nvidia-smi` is absent (→ CPU
  torch wheel), and parses the driver's max CUDA version when present. ✓
- ffmpeg auto-install attempts per-OS package managers, else prints exact manual
  instructions; model download is best-effort and non-fatal (graceful
  degradation to COCO YOLO fallback). ✓
- The full GPU install (torch CUDA wheel + weights + doctor) is ○ GPU-verify.

> DoD #7 (`python -m scripts.setup` one-command install + doctor) — control-flow
> and CPU/`--help`/autodetect paths verified; GPU install path ○ verify on L40S.

---

## H. Fixes applied in this pass
1. `requirements.txt`: pinned the v2 blueprint packages
   (`kornia/trackers/google-generativeai/moviepy`) and removed the duplicate
   `moviepy>=1.0.3` line — full tree still resolves with no conflicts.
2. `src/render/composer.py`: removed a stray `from moviepy import editor as mpy`
   inside `_typography_moviepy` that would `ImportError` on **MoviePy 2.x**
   (`moviepy.editor` was removed in 2.x). The default typography engine is
   Pillow, so this was latent, but the optional `typography_engine="moviepy"`
   path now works on whichever MoviePy is installed (the `_moviepy_*` shims
   already try 2.x then 1.x).
3. Added `tests/test_studio.py` — 13 GPU-free unit tests for the v2 logic.

---

## I. Honesty / compliance notes
- **Licensing**: Ultralytics YOLO is **AGPL-3.0**, and the auto-downloaded
  football weights (martinjolif/*) are AGPL-3.0. Operators deploying this
  commercially must comply with AGPL (or obtain a commercial Ultralytics
  license). Rights to the input match footage are the operator's responsibility.
- **Jersey numbers on wide shots are unreliable** — and the code is honest about
  it: numbers are only trusted with `min_reads`/`min_confidence` aggregation, and
  hero selection **degrades to the geometric (nearest-to-ball) pick** when no
  number is read. The audit confirms this fallback order in `tests/test_studio.py`.
- **Stats are estimates**: possession is metric only when a pitch homography is
  available (`metric=True`); otherwise it is a pixel-proximity approximation
  (`metric=False`) and should be labelled as such in any on-screen stat.
- **Errors are visible**: `ff.run` surfaces ffmpeg stderr (no silent swallowing);
  per-clip failures are recorded with the failing stage rather than hidden.

---

## J. Definition-of-Done summary
| # | Criterion | Status |
|---|-----------|--------|
| 1 | `modal deploy` web up, GET / = 200, UI loads | ✓ verified locally (real GET /), Modal build uses same pins |
| 2 | Upload → stream progress → downloadable clips/reel (both engines) | v1 ✓ CPU e2e; v2 ○ GPU-verify; UI streaming contract ✓ |
| 3 | v2: auto models, classes-by-name, team-colour halos, POSSESSION, jersey hero-lock+fallback, smooth CMC | logic ✓; pixel output ○ GPU-verify |
| 4 | Each mp4 = 1 H.264 + 1 AAC stereo @ profile WxH/fps | ✓ verified (v1 chain + standardize) |
| 5 | Dependencies pinned coherently & reproducibly | ✓ web + v2 pinned; heavy ML ranged (documented) |
| 6 | Tests green; clean compile | ✓ montage + audit + studio pass; compileall clean |
| 7 | `scripts.setup` one-command install + doctor | ✓ CPU/help/autodetect; GPU install ○ verify |
| 8 | Honest audit with GPU boundaries | ✓ this document |

## K. Time estimate (unchanged, honest)
A 2h23m 1080p match on an L40S with NVDEC/NVENC: ~45–75 min. **v2 is more
expensive than v1** because tracking + analytics run per-clip (BoT-SORT+CMC,
team K-means, jersey OCR, possession) on top of the v1 montage cost.


---

## L. Quality-polish pass (output "looks good") — ✓ Verified on CPU

After the deploy/audit pass, a second round addressed *visual & audio polish*
(the difference between "technically correct" and "looks professional"). All of
the following are CPU-testable and are covered by `tests/test_polish.py` (5
tests) plus an added zero-phase test in `tests/test_studio.py`.

1. **Motion-interpolated slow-motion.** Slow-mo previously stretched timestamps
   (`setpts`) only, so at 0.4× it stepped through duplicated frames (judder).
   Both engines now apply `minterpolate` (motion-compensated) to the slowed
   window, config-gated (`edit.effects.slowmo_interpolate`, `…_quality`) with a
   graceful fallback to plain stretch if interpolation fails. *Verified:* the
   slowed window goes from **180 → 288 unique frames** with interpolation on.

2. **Zero-lag camera smoothing.** The virtual-camera path used a causal filter
   that lagged the action (crop trailed the ball). Since the whole clip is known
   offline:
   - v2 Cameraman: forward-only Kalman → full **RTS forward-backward smoother**.
   - v1 reframe: causal EMA → **zero-phase forward+backward EMA**.
   *Verified:* on a symmetric bump the RTS peak stays on the true peak (±3
   frames) while the causal filter visibly lags later.

3. **No more `mp4v` double-compression.** Every cv2 render
   (`reframe`, `cameraman.render`, `composer.draw_graphics`,
   `composer._typography_pillow`) wrote a lossy MPEG-4 part-2 intermediate that
   was then re-encoded to H.264 (two lossy generations). New `ff.RawFrameSink`
   pipes raw BGR frames into **one** H.264 encode with the audio muxed in the
   same pass — one generation, and faster. The upscaled 9:16 crop now uses
   **Lanczos** instead of `INTER_AREA` (which is down-sampling-only). *Verified:*
   RawFrameSink yields a valid 1-video(h264)+1-audio mp4, and surfaces
   `FFmpegError` on a failed encode (errors not swallowed).

4. **Broadcast-grade audio.** Music was a static gain + `dynaudnorm`. Now:
   - **Sidechain ducking** (`sidechaincompress`) pulls the music down whenever
     the commentary is present (`edit.audio.duck_under_commentary`).
   - **EBU R128 `loudnorm`** brings the final mix to the social target
     (`edit.audio.loudnorm`, `loudness_target_lufs: -14`), pinned to 48 kHz
     stereo so downstream concat stays safe. *Verified:* compose builds a valid
     duck+loudnorm filtergraph → 1 video + 48 kHz stereo audio.

5. **Caption safe-zone.** Burned captions moved from `y=h*0.72` (inside the
   bottom UI band of TikTok/Reels/Shorts) to a configurable `captions.safe_y`
   (default **0.62**); v2 stat plates moved from `0.66 → 0.58`. *Verified:* the
   drawtext expression honours `safe_y` and no longer uses `0.72`.

### Honest limits of the polish pass
- `minterpolate` is **CPU-heavy** and can introduce warping artifacts on fast,
  low-contrast motion; it's why it's config-gated with a plain-stretch fallback.
  Tune `slowmo_interpolation_quality: blend` for a cheaper, artifact-free (but
  softer) result. Real per-match cost should be measured on the L40S.
- Zero-phase smoothing improves tracking but the **crop size is still fixed**
  (no adaptive zoom): very wide team shots are still cropped to a narrow column.
- These tests prove the **filtergraphs/encoders are valid and behave as intended**
  on synthetic clips. The full v2 telestration pixels (halos/trail/plates) and
  real-match framing quality remain ○ **verify on GPU**.
- Output quality still depends most on **moment selection** (which clips are
  cut) — a heuristic stage that must be validated on real footage.
