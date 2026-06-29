# Project log — Football Highlight Studio

A consolidated record of what was built, why, and how to run it. Newest first.

## v0.4 — L40S hardware acceleration
- **NVDEC decode + NVENC proxy encode** in `ingest` (`ingest.hwaccel`,
  `ingest.proxy_encoder`); hardware decode in `clipper` for cutting the source.
- **Scene-detect `frame_skip`** (`detect.scene.frame_skip`) — 2-3x faster pass
  over a full match.
- **`ff.pick_hwaccel()`** probes that a CUDA/VAAPI device actually initialises
  and falls back to software, fixing a real crash where `-hwaccel auto` aborts on
  headless/misconfigured hosts.
- Verified: ingest on a synthetic AC3 5.1 source; montage + audit tests green.
- Estimate for a 2h23m 1080p match on an L40S: **~45-75 min** (vision on).

## v0.3 — Deployment + duration-targeted output
- **`modal_app.py` + `docs/DEPLOY_MODAL.md`** — serverless GPU deploy on Modal
  (container Image, GPU function serving the Gradio UI via ASGI, persistent
  Volumes, `setup_models` + headless `process`).
- **Compilation mode** (`render.output_mode: compilation`) — stitches top moments
  into one 30-60s reel with a single intro/outro, continuous music bed and
  per-moment lower-thirds. `src/edit/compilation.py`.
- **Football-CV integrations** (`docs/ROADMAP_FOOTBALL_CV.md`): pitch homography
  for metric stats (`vision/pitch.py`), team classification + possession-aware
  protagonist (`vision/teams.py`), action-spotting hook (`detect/action_spotting.py`).

## v0.2 — Turnkey + WebUI + audited montage
- **`studio.py`** single-file self-bootstrapping launcher (venv, deps, ffmpeg,
  models, then WebUI) for plain VMs.
- **`app/webui.py`** Gradio UI: upload, profile, mode + features, live progress,
  preview, downloads, library management.
- **`src/runner.py`** importable API (progress callbacks, per-clip error
  isolation) shared by CLI and WebUI.
- **`src/edit/ff.py`** central ffmpeg layer (visible errors, NVENC fallback,
  stream `standardize`, `mux_audio`, drawtext escaping).
- Audit + fixes of real montage bugs (freeze-zoom size, concat audio mismatch,
  drawtext `%`/`:`/emoji via `expansion=none` + textfile sidecars). See
  `docs/AUDIT.md` and `docs/AUDIT_FINAL.md`.

## v0.1 — Core pipeline
- Full-match → moment detection (audio energy, scene/replay, scoreboard OCR,
  commentary) → fusion → clip → YOLO+ByteTrack tracking → telestration
  (spotlight/arrows/ball trail) → action-aware 9:16 reframe → slow-mo/freeze →
  music/captions/grade → branded overlays. Docs: `PIPELINE.md`, `MODELS.md`,
  `SETUP.md`, `SKILLS.md`.

---

## How to run (two paths)

### Modal (your plan — serverless GPU)
```bash
pip install modal && modal token new
modal run modal_app.py::setup_models      # one-time: populate models Volume
modal deploy modal_app.py                 # prints your WebUI URL
```
Open the URL → upload match → pick profile + mode (per_clip / compilation) →
Render. Details + caveats: `docs/DEPLOY_MODAL.md`.

### Plain GPU VM
```bash
python3 studio.py                         # bootstraps everything, opens WebUI
# or headless:
python3 studio.py run input/match.mp4 --profile tiktok
```

## Output
- `per_clip` mode → several finished vertical clips (`output/<match>/NN_kind.mp4`)
  each with a `.txt` caption/hashtags.
- `compilation` mode → one 30-60s reel (`output/<match>/reel_01.mp4`) + caption.

## Tests (no GPU needed)
```bash
python -m compileall -q src app studio.py tests modal_app.py
python tests/test_montage.py     # full montage + compilation render chain
python tests/test_audit.py       # encoder fallback, AV standardize, runner e2e
```

## Honest status
Montage / compilation / turnkey logic and runner orchestration are verified by
execution on synthetic clips. GPU "vision" stages (YOLO/tracking/telestration/
OCR/whisper/homography) are implemented and contract-checked but require a GPU +
models — validate on first real run (recommend a `--no-vision` dry run first).
Licensing: Ultralytics YOLO is AGPL-3.0; broadcast footage / music / likeness
rights are the operator's responsibility (see `docs/MODELS.md`).
