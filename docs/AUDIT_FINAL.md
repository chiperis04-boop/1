# Final audit — Football Highlight Studio

Scope: verify the project against the goals stated across the build, measure how
fully each is met, and state honestly what is verified here vs only verifiable on
the target GPU server.

Environment of this audit: CPU-only sandbox, Python 3.x, **no GPU, no models, no
real match footage**, static `ffmpeg`/`ffprobe` (BtbN build, `drawtext` present).
Therefore montage / compilation / turnkey logic is **executed and verified**;
the GPU "vision" stages are **reviewed statically** (they need a GPU + models).

---

## Status legend
- ✅ **Verified** — exercised and asserted here.
- 🟢 **Implemented (GPU-verify)** — code complete + import/contract-checked, but
  only executable on the GPU server.
- ⚠️ **Partial / risk** — works with a caveat worth knowing.

---

## Goal alignment (G1–G7)

| Goal | Status | Evidence / notes |
|---|---|---|
| **G1** Full match → local auto-detection of moments | 🟢 Implemented (GPU-verify) | Detectors (audio/scene/OCR/commentary/action-spotting) + `fusion` implemented; **fusion→clipper→render orchestration executed end-to-end** with mocked detectors (`test_audit.py`). Real detectors need librosa/scenedetect/easyocr/faster-whisper on the server. |
| **G2** Player telestration (arrows/spotlight/trail) | 🟢 Implemented (GPU-verify) | `vision/telestration.py` + YOLO/ByteTrack tracking; geometry-preserving contract documented. Needs GPU + football model. |
| **G3** Music / SFX / captions + branded unique elements | ✅ Verified | `test_montage.py` + `test_audit.py` render grade+captions(emoji/`%`/`:`)+music bed+hook/lower-third/stats/watermark+intro/outro, all → 1 video + 1 audio @1080×1920. |
| **G4** Turnkey: one `studio.py` → deps+models+WebUI | 🟢 Implemented (GPU-verify) | `studio.py` stdlib-only bootstrap (venv, re-exec, deps, ffmpeg static download, model fetch, self-fetch sources, CLI passthrough) reviewed line-by-line; WebUI↔runner wiring **AST-verified** (12 inputs == 12 params). Full install not run in sandbox (would pull torch). |
| **G5** 30–60s videos (compilation) | ✅ Verified | `compilation.py` + runner `compilation` mode executed: reel assembled from segments + single intro/outro + continuous music bed, duration in target window, caption written. |
| **G6** Football-specific OSS to close limitations | 🟢 Implemented (GPU-verify) | Pitch homography (`pitch.py`, metric stats), team classification + possession-aware protagonist (`teams.py`), action-spotting hook (`action_spotting.py`, highest fusion weight). Optional + graceful no-op when disabled (verified imports + disabled-returns). Roadmap in `ROADMAP_FOOTBALL_CV.md`. |
| **G7** Reliability / honesty (no crash on stream; visible errors; correct files) | ✅ Verified | Per-clip stage-isolated errors; `ff.run` surfaces real ffmpeg error lines; `standardize` guarantees uniform streams (verified on video-only input → synth audio); NVENC→libx264 fallback verified; pixel stats flagged `~` approximate. |

**Verdict:** the project **meets its goals**. Everything that can be executed
without a GPU is verified working (montage, branding, compilation, encoder
fallback, runner orchestration, turnkey logic by review). The remaining items are
**fully implemented** and gated only by GPU/model availability on the deployment
server — not by missing functionality.

---

## Checks performed

### Static / contract (task 1)
- `compileall` of `src app studio.py tests` → clean.
- Both YAML configs parse.
- **Config contract:** ~70 `config.yaml` keys + ~17 `branding.yaml` keys the code
  reads are all present (0 missing); 4 render profiles valid.
- No dead references to removed helpers (`_mux_audio`/`_normalize`/`_run`).
- WebUI `render_job` (12 params) exactly matches `click(inputs=[…])` (12), in order.

### Edge cases (task 2)
- `pick_encoder("h264_nvenc")` → `libx264` on a GPU-less host (probe-based).
- `standardize` on a **video-only** clip produces a valid silent stereo track.
- Missing font and missing music degrade gracefully (covered by montage test).
- Captions/overlays with emoji, `%`, `:`, apostrophes render (textfile + `expansion=none`).

### End-to-end runner (task 3)
- Real `run_pipeline` with `ingest`+detectors mocked, vision disabled:
  - **per_clip:** 2 clips rendered @1080×1920, `result.json` written.
  - **compilation:** reel @1080×1920 (~12s for the short fixture) + caption.
- Exercises the real `fusion → clipper → reframe → compose → branding →
  compilation` path on synthetic media.

---

## Defect found & fixed during this audit

| Sev | Issue | Fix |
|---|---|---|
| **Major** | `detect/scoreboard_ocr.py` imported `cv2` at module top, which (via `runner`'s top-level import) coupled the **entire pipeline import to OpenCV** — breaking the CPU / `--no-vision` path on hosts without opencv. | Made the `cv2` import lazy (inside `detect_scoreboard`), consistent with `reframe.py`/`detect_track.py`. Re-verified: runner now imports and runs with no OpenCV present. |

No other defects found. (Earlier blocker/major montage bugs were already fixed
and are covered by `docs/AUDIT.md` + the passing `test_montage.py`.)

---

## Only verifiable on the GPU server (honest gaps)
These are implemented but **not executed** in this sandbox; validate on first
real run:
1. YOLO player/ball detection + ByteTrack tracking quality (needs football model).
2. Telestration overlays following the correct player.
3. Scoreboard OCR reading the real broadcast score graphic / ROI.
4. faster-whisper commentary transcription accuracy/speed.
5. Pitch-keypoint homography coverage → metric stat accuracy.
6. Team classification correctness on real kits.
7. `studio.py` full first-run install (torch + requirements + model downloads).
8. NVENC acceleration actually engaging on a GPU box.

Recommended first-server smoke test: run `--no-vision` on a real match to confirm
detection→cut→reel, then enable vision features one at a time per
`docs/ROADMAP_FOOTBALL_CV.md` build order.

---

## Test commands
```bash
python -m compileall -q src app studio.py tests
python tests/test_montage.py     # montage + compilation render chain
python tests/test_audit.py       # encoder fallback, AV standardize, runner e2e
```

**Audit result: PASS** — goals met; deploy-ready for GPU-server validation.
