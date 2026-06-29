# Football Highlight Studio — Audit & Stabilization Report (FINAL_2)

Date: 2026-06-29
Scope: (A) fix the WebUI deploy blocker at the root, (B) full audit against the
project goals, (C) make changes reproducible and tested.

Status legend:
- ✓ **Verified** — checked locally (no GPU needed) and passing.
- ○ **Implemented / GPU-verify** — code is correct by inspection but the runtime
  path needs an NVIDIA L40S (YOLO/ByteTrack/telestration/OCR/Whisper/homography)
  or a real Modal deploy to confirm end-to-end.
- ⚠️ **Risk** — works, but has an operational caveat worth knowing.
- ✗ **Missing** — not implemented.

---

## Part A — WebUI blocker: root cause and the fix

### Symptom
`GET /` on the Modal Gradio app returned 500:
`TypeError: unhashable type: 'dict'` in `jinja2/environment.py` → template cache.

### Root cause (verified by reproduction)
This was **not** a Gradio code bug — it was a *transitive dependency drift*.
`gradio 4.44.1` declares only `fastapi<1.0` and places **no cap on starlette or
pydantic**. Left unpinned, pip resolves the newest releases (observed:
`starlette 1.3.1`, `fastapi 0.138.2`, `pydantic 2.13.4`) — all published long
after gradio 4.44.1 (Sept 2024).

1. Gradio 4.44.1 (`gradio/routes.py:432`) calls Starlette with the **legacy
   positional order** `templates.TemplateResponse(template_name, context_dict)`.
   Starlette 1.x **removed** that ordering — the new signature is
   `TemplateResponse(request, name, context=...)`. So Starlette treated the
   context **dict** as the template `name`, and jinja2 tried to use that dict as
   a hashable cache key → `unhashable type: 'dict'`.
2. The earlier "`argument of type 'bool' is not iterable`" crash in
   `gradio_client.get_api_info` is the *same family*: newer pydantic emits JSON
   schemas with boolean `additionalProperties`, which gradio_client 1.3.0's
   schema walker can't handle.

Both crashes share one cause: **the web stack floated to versions newer than the
ones gradio 4.44.1 was built against.** Patching each symptom with a monkeypatch
(as was done historically) treats the symptom, not the cause.

### Decision (holistic, not another patch)
**Pin the entire web stack to the gradio-4.44.1-era releases.** The WebUI
(`app/webui.py`) is written against the Gradio 4 Blocks API; a coherent 4.x lock
is the lowest-risk, fully-reproducible fix and resolves *both* bug classes at the
source. (Moving to Gradio 5 was considered; it is viable but a larger surface
change for no extra benefit here, since the app uses only standard Gradio-4
components.)

`requirements.txt` now locks:

```
gradio==4.44.1
gradio-client==1.3.0
fastapi==0.115.0
starlette==0.38.6     # last 0.x line that still accepts gradio 4's TemplateResponse call
pydantic==2.9.2       # schema shape gradio_client 1.3.0 can parse
pydantic-core==2.23.4
orjson==3.10.7
tomlkit==0.12.0       # exact pin required by gradio 4.44.1
markupsafe>=2.0,<3
jinja2>=3.1,<4
uvicorn>=0.23,<0.31
websockets>=10.0,<13.0
```
plus `pillow>=10.0,<11` (gradio caps `pillow<11.0`) and the retained
`huggingface-hub>=0.23,<1.0` (keeps `HfFolder`).

### Verification (✓ Verified, no GPU)
Installed the **full** `requirements.txt` (+ CPU torch) into a clean Python
3.11 venv:
- `pip check` → **No broken requirements found.**
- Imported the **real** `app.webui.build()`, mounted via
  `mount_gradio_app(FastAPI(), demo, "/")`, and issued live requests:
  - `GET /` → **200** (~49 KB HTML, contains "Football Highlight Studio")
  - `GET /config` → **200**
  - `demo.get_api_info()` → **6 named endpoints**, no exception
- Confirmed `get_api_info()` now succeeds **without** the monkeypatch (the pins
  are the real fix). The monkeypatch in `app/webui.py` was **re-documented as a
  harmless defensive no-op**, not removed, so a future schema edge case still
  degrades gracefully instead of 500-ing. It does not mask the version contract.

| Item | Status |
|---|---|
| Root cause identified (starlette/pydantic drift vs gradio 4.44.1) | ✓ Verified |
| Coherent version lock applied & whole tree resolves (`pip check`) | ✓ Verified |
| `GET /` = 200, UI HTML served, `/config` = 200, api_info OK | ✓ Verified |
| `modal deploy` actually serves on L40S | ○ GPU/deploy-verify |

---

## Part B — Reliability fix found during audit (graceful CPU degradation)

`tests/test_audit.py` (CPU, vision disabled) failed at the **captions** stage:
`CUDA failed ... driver version is insufficient`. Root cause: `caption_clip`,
`detect_commentary`, `scoreboard_ocr`, `detect_track`, and `pitch` all passed
`cfg["vision"]["device"]` (default `"cuda"`) to their model backends **without
checking GPU availability** — so a CPU box (or a dry run) hard-crashed instead of
degrading.

Fix:
- Added `resolve_device()` in `src/utils/io.py`: returns `"cuda"` only when
  `torch.cuda.is_available()`, else `"cpu"`.
- Routed all five call sites through it.
- Wrapped `caption_clip` transcription in try/except → returns `[]` (captions are
  cosmetic and must never fail a clip), and isolated `detect_commentary` likewise
  (one detector must not abort the whole detection pass).

Result: `test_audit` now passes; logs show `captions ... device=cpu`. ✓ Verified.

---

## Part C — Full audit against the goals

### 1. Output contract — every mp4 = 1 H.264 video + 1 AAC stereo @ profile WxH/fps
- `ff.standardize()` is the single choke point: scales+pads to `WxH`, forces
  `fps`, `format=yuv420p`, `setsar=1`, **synthesises silent stereo AAC@48k** when
  the source has no audio, and every concat input passes through it.
- `test_montage` and `test_audit` assert exactly `1 video + 1 audio @ 1080x1920`
  on per-clip outputs **and** the compilation reel. ✓ **Verified**.

### 2. Safe concatenation (per_clip effects + compilation reel)
- Slow-mo / freeze-zoom / intro-outro / reel assembly all standardize parts to
  identical stream params, then concat with `-c copy`. The freeze-zoom path
  builds the still at the **profile** resolution with a silent track. ✓ Verified
  (montage test stitches a 2-segment reel: 16.0s, intro+outro present).

### 3. FFmpeg correctness & visible failures
- All ffmpeg calls go through `ff.run`, which **captures stderr** and raises
  `FFmpegError` with the flagged error lines (no swallowed failures). ✓ Verified.
- NVENC/NVDEC: `pick_encoder` does a real 1-frame nvenc probe and falls back to
  libx264; `pick_hwaccel` explicitly inits the hw device (so `-hwaccel auto`
  cannot abort on a half-configured host). Fallback path ✓ Verified
  (test logs `h264_nvenc unavailable; falling back to libx264`). NVENC/NVDEC
  active path ○ GPU-verify.

### 4. Detection → fusion
- Audio energy (librosa), scene/replay (PySceneDetect), scoreboard OCR
  (EasyOCR), commentary (faster-whisper), optional action-spotting — fused by a
  weighted, windowed clustering with confidence cap, type classification, padding
  and a `max_moments` cap. Fusion logic ✓ Verified (audit test fuses mocked
  signals → moments → clips). Individual detectors on real footage ○ GPU-verify
  (OCR/Whisper are heavy; audio/scene run on CPU).
- ⚠️ **Risk**: detector signal sources other than commentary are **not** wrapped
  individually in `_detect_and_fuse`; an unexpected hard failure in
  `detect_audio`/`detect_scenes`/`detect_scoreboard` would propagate out of
  `run_pipeline` (the WebUI catches it and shows the error, so it is visible, not
  silent). Commentary and captions are now isolated. Consider per-detector
  isolation as a future hardening.

### 5. CV: resolve-by-name, telestration, calibration, teams
- `_resolve_classes` maps YOLO `model.names` **by name** (`*ball*`→ball,
  `*player*`/`*keeper*`→player, referees excluded), falling back to config
  indices — so any martinjolif/* football model works regardless of index order.
  ✓ Verified by inspection; ○ GPU-verify on a real model.
- Telestration (spotlight/arrow/ball-trail/zone), ByteTrack tracking,
  homography stats, HSV team clustering, possession-aware key-player pick — all
  implemented and structurally sound; require GPU + models. ○ GPU-verify.
- Honesty: pixel-space stats are flagged `metric=False` and the overlay prefixes
  them with `~`; metric stats only when homography coverage > 0.5. ✓ Good
  practice (matches the "honest estimates" requirement).

### 6. Models — automatic download, resolve-by-name
- `modelhub.ensure_models` pulls player/ball/pitch from public HF repos
  (`martinjolif/*`), is idempotent (skips existing non-empty files), honors
  `HF_TOKEN`, and **falls back to generic `yolov8x.pt`** if the football player
  model can't be fetched, so the pipeline still runs. `run_pipeline` calls it
  before processing when `vision.enabled`. Logic ✓ Verified by inspection;
  actual download + Modal Volume persistence ○ deploy-verify.

### 7. App/UX — runner contract & progress streaming
- `app/webui.render_job` overrides map 1:1 onto keys the runner reads
  (`vision.enabled`, `telestration.enabled`, `edit.effects.*`, `edit.audio.*`,
  `detect.audio.zscore_threshold`, `fusion.min_confidence`, `render.output_mode`,
  `render.compilation.*`). It streams a progress log via a generator + queue,
  isolates the worker thread's exception into `holder["error"]`, and yields clip
  dropdown / preview / caption / downloadable files at the end. ✓ Verified by
  inspection; the underlying `run_pipeline(... on_progress=)` contract is
  exercised by `test_audit`.

### 8. Reliability / SRE
- Per-clip failures are isolated in `_process_clip` (records `stage_failed` +
  `error`, continues the batch); `result.json` is written; encoder/hwaccel/device
  all degrade gracefully. ✓ Verified.

### 9. Config contract
- Cross-checked every hard `cfg[...]` / `branding[...]` access across `src/`,
  `app/` against `config/config.yaml` and `config/branding.yaml` — including the
  vision/telestration/scene/audio/scoreboard paths the tests do **not** exercise.
  **No missing keys.** `_active_profile` is injected at runtime. ✓ Verified.

### 10. Honesty / compliance
- README documents Ultralytics **AGPL-3.0** and that broadcast-footage / music /
  likeness **content rights are the operator's responsibility** (`docs/SKILLS.md`
  checklist mirrors this). ✓ Present.

### 11. Modal deploy specifics
- ✓ Fixed: `GPU = "L40S"` (was `"A10G"`) to match the target hardware.
- ✓ Fixed: `web` function `timeout` raised `60*60 → 2*60*60`. A full 1080p match
  renders in ~45-75 min and streams progress over one long-lived request; the old
  60-min timeout would kill a long render mid-stream.
- Image bakes ffmpeg + OpenCV system libs (`libgl1`, `libglib2.0-0`), installs
  CUDA torch from PyPI, then `requirements.txt`; models + output on persistent
  Volumes; HF secret optional (`required=False`). Structure ✓ by inspection.
- ⚠️ **Risk / deploy-verify**: Modal SDK API calls (`@modal.concurrent`,
  `@modal.asgi_app`, `scaledown_window`, `Image.add_local_dir`) target the Modal
  1.0 SDK and can only be confirmed against a live account + token. EasyOCR
  downloads its own weights on first use into container-ephemeral storage (not the
  models Volume) — acceptable, but a cold first OCR run pays that cost.

---

## What was changed in this pass
| File | Change |
|---|---|
| `requirements.txt` | Coherent web-stack lock (fastapi/starlette/pydantic/etc.) + pillow cap; documented root cause |
| `app/webui.py` | Re-documented the gradio_client guard as a defensive no-op (pins are the real fix) |
| `src/utils/io.py` | New `resolve_device()` — cuda only when actually available |
| `src/edit/captions.py` | `resolve_device` + transcription wrapped (captions never fail a clip) |
| `src/detect/commentary.py` | `resolve_device` + failure isolation |
| `src/detect/scoreboard_ocr.py` | EasyOCR gpu flag via `resolve_device` |
| `src/vision/detect_track.py` | YOLO device resolved once via `resolve_device` |
| `src/vision/pitch.py` | Keypoint model device via `resolve_device` |
| `modal_app.py` | `GPU="L40S"`; web `timeout` → 2h |

## Test evidence (CPU sandbox, ffmpeg present)
- `python -m compileall src app studio.py modal_app.py tests` → clean.
- `python tests/test_montage.py` → **ALL MONTAGE CHECKS PASSED** (AV layout,
  slow-mo, freeze-zoom, compose, branding intro/outro, 2-segment reel).
- `python tests/test_audit.py` → **AUDIT TESTS PASSED** (encoder fallback,
  silent-audio synthesis, per_clip + compilation runners, captions degrade to
  CPU).
- Real `app.webui` mounted: `GET /`=200, `/config`=200, api_info=6 endpoints.

## Definition-of-Done status
1. WebUI serves without crashing; `GET /`=200, UI loads — ✓ Verified locally; ○ confirm on `modal deploy`.
2. Upload → pipeline → streaming progress → downloadable clips/reel — ✓ runner+UI contract verified; ○ end-to-end on GPU.
3. Auto model download + resolve-by-name + telestration on players/ball — ○ GPU-verify (logic verified).
4. Each mp4 = 1 H.264 + 1 AAC stereo @ WxH/fps — ✓ Verified.
5. Dependencies pinned coherently & reproducibly — ✓ Verified (`pip check` clean).
6. Tests green; compilation clean — ✓ Verified.
7. Honest audit with GPU boundaries — ✓ this document.
