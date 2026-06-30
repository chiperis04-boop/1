# Football Highlight Studio — Audit & Stabilization Report (FINAL_3)

Date: 2026-06-30
Auditor roles applied: Release/Dependency, Modal/Serverless, AV/FFmpeg, CV, App/UX,
Reliability/SRE, Honesty/Compliance.
Scope requested: (A) fix the WebUI deploy blocker at the root for **both** engines,
(B) full audit of **both** pipelines (v1 "Classic" and v2 "Studio"), (C) commit/push
with tests + compile.

Status legend:
- ✓ **Verified** — checked in this sandbox (no GPU) and passing.
- ○ **Implemented / GPU-verify** — correct by inspection; the runtime path needs an
  NVIDIA L40S (YOLO/tracking/telestration/OCR/Whisper/homography) or a live Modal
  deploy to confirm end-to-end.
- ⚠️ **Risk** — works, but has an operational caveat worth knowing.
- ✗ **Missing** — not present in the repository.

---

## Part 0 — CRITICAL FINDING: the v2 "Studio" pipeline does not exist in this repo

The task brief describes a second, **v2 "Studio"** pipeline (Scout → Director →
Cameraman → Composer + football analytics) and lists modules to audit. **None of
those modules exist** in the repository — neither in the local working tree nor on
`origin/main` (verified via `git ls-tree -r origin/main`). The repo is **v1
"Classic" only**.

Absent vs. the brief (all ✗ **Missing**):

| Brief claims | Reality in repo |
|---|---|
| `src/studio_pipeline.py` (run_studio orchestrator) | ✗ not present |
| `src/detection/scout.py`, `src/detection/director.py` | ✗ (only `src/detect/` v1 exists) |
| `src/tracking/cameraman.py` (BoT-SORT + GMC/CMC, Kalman 9:16) | ✗ not present |
| `src/graphics/homography.py` (cv2/kornia, under-player) | ✗ not present |
| `src/render/composer.py` (slow-mo/typography) | ✗ not present |
| `src/vision/possession.py`, `jerseys.py`, `analytics.py` | ✗ not present |
| `scripts/setup.py` (`python -m scripts.setup`) | ✗ (only `scripts/fetch_models.py` + `download_models.sh`) |
| WebUI engine switcher (v1/v2) + 5 v2 options, v2 default | ✗ `app/webui.py` has a single v1 form (13 inputs) |
| Deps: `kornia`, roboflow `trackers`, `google-generativeai`, BoT-SORT config | ✗ not in `requirements.txt`; no `botsort.yaml` |
| New config keys: `detect.scout.*`, `director.*`, `tracking.*` (gmc_method, kalman_*), `graphics.*`, `possession.*`, `vision.jerseys.*`, `telestration.team_halos` | ✗ none present in `config/config.yaml` |

**Consequence for the Definition of Done.** Every v2-specific acceptance criterion —
engine switcher in the UI, team-coloured halos, POSSESSION plate, jersey-number hero
binding with geometric fallback, BoT-SORT+CMC reframe, the v2 `(stage,pct,msg)`
progress contract, `python -m scripts.setup` — **cannot be satisfied by this codebase
because the code is not here.** I did **not** fabricate ~10 GPU-only CV modules and
claim they work; per the brief's own honesty constraint ("do not pass unverified as
verified"), the correct status is ✗ Missing, documented plainly.

What this report therefore covers truthfully: the **v1** pipeline (which exists and
runs), the **WebUI blocker** (which applies to v1 and is fixed/verified), plus the
concrete defects found and fixed this pass.

> If the intent was for the auditor to *build* v2, that is a separate, large
> implementation effort (new tracking/homography/analytics/render subsystems, new
> deps, new config surface, a UI engine switch) that is mostly GPU-verifiable only.
> It should be scoped as its own task; this pass stabilizes and audits what ships.

---

## Part A — WebUI blocker: status = already fixed at the root, re-verified today

### The brief's blocker (`TypeError: unhashable type: 'dict'`)
This was already root-caused and fixed in a prior pass (see `AUDIT_FINAL_2.md`): the
web stack floated to releases newer than gradio 4.44.1 was built against, so
Starlette 1.x's changed `TemplateResponse` signature made jinja2 try to hash the
context dict. The fix is a **coherent version lock** in `requirements.txt`
(`gradio==4.44.1`, `gradio-client==1.3.0`, `fastapi==0.115.0`, `starlette==0.38.6`,
`pydantic==2.9.2`/`pydantic-core==2.23.4`, `orjson`, `tomlkit`, `markupsafe<3`,
`jinja2<4`, `uvicorn<0.31`, `websockets<13`, `pillow<11`, `huggingface-hub<1.0`).

### Independent re-verification (✓ Verified, no GPU) — done 2026-06-30
In a clean Python 3.11 venv I installed the pinned web stack, imported the **real**
`app.webui.build()`, mounted it exactly as `modal_app.web()` does
(`mount_gradio_app(FastAPI(), demo, "/")`), and issued live requests via a
`TestClient`:

| Check | Result |
|---|---|
| `GET /` (the route that used to 500) | **200**, 51,339 bytes of UI HTML |
| `GET /config` | **200** |
| `GET /info` (exercises `gradio_client` schema walk — bug #3 path) | **200** |
| Input-count contract: `run_btn.click` inputs vs `render_job` params | **13 == 13** ✓ |

I also **reproduced historical bug #2** along the way: leaving `huggingface-hub`
unpinned resolved it to 1.x, which removed `HfFolder`, and gradio's
`oauth.py` import crashed with `ImportError: cannot import name 'HfFolder'`. Adding
`huggingface-hub>=0.23,<1.0` (resolved 0.36.2, which still ships `HfFolder`) fixes
it. This confirms that pin is **load-bearing**, not incidental.

The `_patch_gradio_client_schema_bug()` guard in `app/webui.py` is a documented,
harmless no-op when the pins hold; it does not mask the version contract.

> Note: there is only one engine, so "both engines load" reduces to "the v1 UI
> loads", which is ✓ Verified. There is no v2 form to break.

| Item | Status |
|---|---|
| Root cause (starlette/pydantic/hub drift vs gradio 4.44.1) | ✓ Verified |
| Coherent lock applied; `GET /`=200, `/config`=200, `/info`=200 | ✓ Verified (re-run today) |
| `huggingface-hub<1.0` pin necessity reproduced | ✓ Verified |
| `modal deploy` actually serves on L40S | ○ deploy-verify (no Modal account/GPU here) |

---

## Part B — Defects found and fixed in this pass

### B1. Slow-mo A/V desync at `slowmo_factor < 0.5` (FFmpeg/AV) — ✓ Fixed & Verified
`src/edit/effects.py::apply_slowmo` slowed the **video** window by
`setpts = (1/factor)*PTS` but **clamped** the audio to a single
`atempo = max(0.5, factor)`. `atempo` only accepts 0.5–2.0, so the default
`slowmo_factor = 0.4` stretched video ×2.5 while audio stretched only ×2.0. The
slowed audio segment ended up ~`0.5 × window` short, and since the three video and
three audio sub-segments are concatenated independently, **everything after the
slow-mo beat drifted out of sync** for the rest of the clip.

Measured before the fix (synthetic 8 s clip, factor 0.4, window 2.5 s):
`video = 11.733 s`, `audio = 10.481 s` → **1.252 s desync**.

Fix: a proper **`atempo` chain** (`_atempo_chain`) whose product equals the target
factor, each stage inside the valid 0.5–2.0 range — e.g. `0.4 → atempo=0.5,atempo=0.8`;
`0.2 → 0.5,0.5,0.8`. The slowed audio now stretches by exactly `1/factor`, matching
the video. This is the "audio-safe slow-mo / atempo chain for factors < 0.5" the
brief calls out (it just lives in v1, the only pipeline present).

Measured after the fix:
- factor 0.4: desync **1.252 s → 0.015 s**
- factor 0.2: desync **~0.086 s** (residual is `atempo` resample granularity; was on
  the order of seconds before).

Chain math unit-checked for `{0.4, 0.25, 0.2, 0.5, 0.6, 1.0}` (product == factor).
`test_montage.py` (which uses factor 0.4) still passes.

> Why the tests didn't catch it: `test_montage` asserts stream **count** and
> **geometry**, not per-stream **duration parity**. Worth adding a duration-parity
> assertion in a future pass.

### B2. `numpy` left unpinned vs a numpy-1.x CV stack (Dependency) — ✓ Fixed & Verified
`requirements.txt` had `numpy>=1.24` with no ceiling. A fresh resolve pulls
**numpy 2.4.x**, while the rest of the locked stack (librosa→numba, opencv-python 4.9,
ultralytics 8.2, ctranslate2/faster-whisper, and the CUDA torch baked by
`modal_app.py`) targets the numpy-1.x ABI — a real resolver/runtime risk on a cold
Modal image (numba in particular tracks numpy ABI closely). Pinned to
`numpy>=1.24,<2.0`. Re-verified the UI still serves `GET / = 200` on numpy 1.26.4.

---

## Part C — Full audit of the v1 pipeline against the goals

### 1. Output contract — every mp4 = 1 H.264 video + 1 AAC stereo @ profile WxH/fps
`ff.standardize()` is the single choke point (scale+pad to WxH, force fps,
`format=yuv420p`, `setsar=1`, synthesise silent stereo AAC@48k when no audio). Tests
assert exactly `1 video + 1 audio @ 1080×1920` on per-clip outputs **and** the reel.
✓ **Verified**.

### 2. Safe concatenation (effects + compilation)
Slow-mo / freeze-zoom / intro-outro / reel assembly standardize every part to
identical stream params, then concat with `-c copy`. ✓ **Verified** (montage stitches
a 2-segment reel with intro+outro).

### 3. FFmpeg correctness & visible failures
All ffmpeg calls go through `ff.run`, which captures stderr and raises `FFmpegError`
with the flagged error lines — no swallowed failures. `pick_encoder` does a real
1-frame NVENC probe → libx264 fallback; `pick_hwaccel` explicitly inits the hw device
so `-hwaccel auto` can't abort on a half-configured host. Fallback path ✓ **Verified**
(logs `h264_nvenc unavailable; falling back to libx264`). NVENC/NVDEC active path
○ GPU-verify.

### 4. Detection → fusion
Audio energy / scene-replay / scoreboard OCR / commentary / optional action-spotting,
fused by weighted windowed clustering with confidence cap, type classification and
`max_moments`. Fusion logic ✓ **Verified** (audit test fuses mocked signals →
moments → clips). Real detectors (OCR/Whisper heavy) ○ GPU-verify.
⚠️ **Risk**: in `_detect_and_fuse`, only `detect_commentary` is individually isolated;
a hard failure in `detect_audio`/`detect_scenes`/`detect_scoreboard` would propagate
out of `run_pipeline` (the WebUI catches and shows it, so it's visible, not silent).
Per-detector isolation is a reasonable future hardening.

### 5. CV: resolve-by-name, telestration, tracking, teams
`_resolve_classes` maps YOLO `model.names` **by name** (`*ball*`→ball,
`*player*`/`*keeper*`→player, referees excluded), falling back to config indices — so
any `martinjolif/*` model works regardless of index order. ✓ by inspection;
○ GPU-verify on a real model. v1 tracking is **ByteTrack via supervision** (not
BoT-SORT+CMC — that is a v2 claim, ✗ Missing). Telestration, HSV team K-means and the
possession-aware key-player pick are implemented and structurally sound; GPU+models
required. ○ GPU-verify.
Honesty: pixel-space stats are flagged `metric=False` and prefixed `~` in the overlay;
metric stats only when homography coverage is sufficient. ✓ Good practice.

### 6. Models — automatic download, resolve-by-name
`modelhub.ensure_models` pulls player/ball/pitch from public HF repos
(`martinjolif/*`), is idempotent, honors `HF_TOKEN`, and falls back to generic
`yolov8x.pt` if the football model can't be fetched. `run_pipeline` calls it before
processing when `vision.enabled`. Logic ✓ by inspection; download + Volume persistence
○ deploy-verify.

### 7. App/UX — runner contract & progress streaming
`render_job`'s overrides map 1:1 onto keys the runner reads; it streams a progress log
via generator+queue, isolates the worker thread's exception into `holder["error"]`,
and yields clip dropdown / preview / caption / files. Input-count contract
**13 == 13** ✓ **Verified today**. v1 progress is the `Progress` object (not the v2
`(stage,pct,msg)` tuple — v2 absent).

### 8. Reliability / SRE
Per-clip failures isolated in `_process_clip` (records `stage_failed` + `error`,
continues the batch); `result.json` written; encoder/hwaccel/device degrade
gracefully (`resolve_device` → cpu when CUDA absent). ✓ **Verified** (audit test runs
CPU-only end-to-end; captions/commentary degrade with warnings, not crashes).

### 9. Config contract
Cross-checked every `cfg[...]` access across `src/` against `config/config.yaml`:
`ingest, detect.{audio,scene,scoreboard_ocr,commentary,action_spotting}, fusion, clip,
models, vision.{…,pitch,teams}, telestration, edit.{reframe,effects,audio,captions},
render.{encoder,fps,profiles,compilation}` — **no missing keys**; `_active_profile`
injected at runtime. ✓ **Verified**. The brief's v2 keys
(`detect.scout/director/tracking/graphics/possession/vision.jerseys/telestration.team_halos`)
are absent from both code and config — consistent (✗ Missing, no orphan keys).

### 10. Honesty / compliance
README documents Ultralytics **AGPL-3.0** and that footage/music/likeness rights are
the operator's responsibility. ⚠️ Minor: the README stack table still lists
"WhisperX / TrackNetV3" though `requirements.txt` ships faster-whisper and WhisperX is
commented out — cosmetic doc drift, not a code issue.

### 11. Modal deploy specifics
`GPU="L40S"` matches target; `web` timeout is 2 h (a streamed long render isn't killed
mid-way); image bakes ffmpeg + OpenCV libs + `fonts-dejavu-core`, installs CUDA torch
then `requirements.txt`; models/output on Volumes; HF secret optional. Structure ✓ by
inspection. ⚠️ **deploy-verify**: Modal SDK calls (`@modal.concurrent`,
`@modal.asgi_app`, `Image.add_local_dir`, `scaledown_window`) and full-image
dependency resolution (torch + easyocr + ultralytics + librosa together) can only be
confirmed against a live Modal account + GPU. The web-stack subset resolves and serves
locally; the heavy CV subset was not co-installed here.

---

## What changed in this pass
| File | Change |
|---|---|
| `src/edit/effects.py` | New `_atempo_chain()`; `apply_slowmo` uses it → fixes A/V desync for `factor < 0.5` (B1) |
| `requirements.txt` | Pin `numpy>=1.24,<2.0` for CV-stack ABI compatibility/reproducibility (B2) |
| `docs/AUDIT_FINAL_3.md` | This report (incl. the v2-absent finding) |

## Test evidence (CPU sandbox, ffmpeg with drawtext present)
- `python -m compileall -q src app studio.py modal_app.py scripts tests` → **clean**.
- `python tests/test_montage.py` → **ALL MONTAGE CHECKS PASSED** (AV layout, slow-mo,
  freeze-zoom, compose, branding intro/outro, 2-segment reel).
- `python tests/test_audit.py` → **AUDIT TESTS PASSED** (encoder fallback, silent-audio
  synthesis, per_clip + compilation runners, captions degrade to CPU).
- Real `app.webui` mounted: `GET /`=200, `/config`=200, `/info`=200; `render_job`
  13 params == 13 click inputs.
- Slow-mo A/V parity: factor 0.4 desync 1.252 s → **0.015 s** after fix.

## Definition-of-Done status (honest)
**v1 (present):**
1. WebUI serves without crashing; `GET /`=200, UI loads — ✓ Verified locally; ○ confirm on `modal deploy`.
2. Upload → pipeline → streaming progress → downloadable clips/reel — ✓ runner+UI contract verified; ○ end-to-end on GPU.
3. Auto model download + resolve-by-name + telestration — ○ GPU-verify (logic verified).
4. Each mp4 = 1 H.264 + 1 AAC stereo @ WxH/fps — ✓ Verified.
5. UI dependency tree pinned coherently & reproducibly — ✓ Verified (+ numpy cap added).
6. Tests green; compile clean — ✓ Verified.
7. Audio-safe slow-mo (atempo chain) — ✓ Verified (fixed this pass).
8. Honest audit with GPU boundaries — ✓ this document.

**v2 (absent) — cannot be met by this repo:**
- Engine switcher UI / team-colour halos / POSSESSION plate / jersey-number hero with
  geometric fallback / BoT-SORT+CMC reframe / `(stage,pct,msg)` progress /
  `python -m scripts.setup` / v2 dependency lock (kornia, trackers,
  google-generativeai) — ✗ **Missing** (no code present). Needs a dedicated
  implementation task.
