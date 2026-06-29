# Skills & acceptance criteria

This is the contract the project is built and judged against. It captures the
engineering skills the build relies on, the editing logic in practical terms,
and the concrete, testable criteria that define "done".

---

## A. Engineering skills applied

1. **Media engineering (FFmpeg/AV):** frame-accurate cutting, filtergraphs
   (`trim/setpts/atempo/zoompan/drawtext/amix`), stream standardisation, safe
   concatenation, NVENC vs libx264 selection.
2. **Computer vision:** object detection (YOLO), multi-object tracking
   (ByteTrack), interpolation, heuristic subject selection, drawing overlays.
3. **Audio analysis:** RMS loudness envelopes, rolling baselines, peak picking.
4. **ASR:** transcription with word timings for captions + keyword spotting.
5. **OCR:** ROI-based scoreboard reading and score-change detection.
6. **Systems / packaging:** self-bootstrapping installer, venv management,
   dependency resolution, model provisioning, graceful degradation.
7. **App engineering:** an importable runner API, a streaming WebUI, structured
   job results, and resumable state.
8. **Reliability practices:** visible errors (no swallowed stderr), per-stage
   isolation, capability probing, and automated tests on synthetic media.

---

## B. How the montage actually works (practical walkthrough)

1. **Find the action cheaply.** The whole match is *not* watched frame-by-frame.
   A downscaled proxy + the audio track are scanned by four cheap detectors.
   Crowd/commentator loudness spikes, replay cut-bursts, scoreboard score
   changes and excited commentary keywords each vote on "something happened
   here".
2. **Agree on moments.** Votes within ~8 s merge into one moment; a weighted sum
   gives a confidence and the moment type (a score change ⇒ goal, etc.). Only
   the strongest are kept.
3. **Cut with story padding.** Each moment is cut from the *original* file with
   buildup before and celebration after (goals get more).
4. **Understand the clip.** On the short clip only, YOLO+ByteTrack find and
   follow players and the ball; the protagonist is the player most often nearest
   the ball at the decisive beat.
5. **Draw the analysis.** Spotlight under the protagonist, a motion arrow along
   their path, a ball trail, a highlighted space — the "analyst" look.
6. **Make it vertical without losing the action.** A virtual camera crops to
   9:16 and glides to keep the ball/protagonist centred.
7. **Add the beat.** Slow-mo on the key moment; a freeze-zoom "watch this"
   call-out is prepended.
8. **Sound + captions + grade.** Music (ducked) + crowd swell are mixed under
   the original audio; word-level captions are burned; a broadcast colour grade
   is applied.
9. **Brand it.** Hook question, lower-third (GOAL · minute), auto-stats,
   watermark, intro/outro — identical across every clip so the channel is
   recognisable.
10. **Deliver + verify.** Output is a vertical MP4 plus a caption/hashtag file;
    every render passes through standardised encoding so files are valid and
    uniform.

---

## C. Acceptance criteria (Definition of Done)

### C1. Turnkey deployment
- [ ] Uploading `studio.py` alone to a clean server and running
      `python3 studio.py` results in a reachable WebUI **without any manual pip
      or ffmpeg steps** (assuming network access).
- [ ] Missing `ffmpeg` is auto-resolved (static build) or a clear message is
      shown.
- [ ] Re-running is fast (bootstrap is skipped via marker files).
- [ ] A CUDA torch build is selectable via `FHS_TORCH_INDEX`.

### C2. WebUI usability
- [ ] Upload a match, pick a platform profile, toggle features, start a job.
- [ ] Progress streams live (stage + percent + per-clip counter).
- [ ] Rendered clips are previewable in-browser, with caption text and download.
- [ ] A library view lists past renders and supports preview + delete.
- [ ] A clip failure does not abort the batch; the failing stage + error are
      surfaced.

### C3. Detection quality
- [ ] On a match with on-screen score, every score change is detected (OCR) and
      classified as a goal.
- [ ] Audio sensitivity and min-confidence are adjustable from the UI.
- [ ] Detection results are cached and reusable (`moments.json`).

### C4. Render correctness (the bar that v0.1 failed)
- [ ] Every produced MP4 has exactly one H.264 video + one AAC stereo audio
      track at the chosen profile's WxH/fps.
- [ ] Intro/body/outro concatenation never drops audio or fails on stream
      mismatch.
- [ ] Freeze-zoom and slow-mo never crash on clips where the key beat is near an
      edge or the clip is short.
- [ ] Captions/hooks with punctuation/emoji do not break the filtergraph.
- [ ] NVENC is used when available and silently falls back to libx264 otherwise.

### C5. Telestration (GPU)
- [ ] With a football detection model, players and ball are tracked on each clip
      and overlays follow the protagonist.
- [ ] Telestration preserves clip geometry so reframing stays correct.
- [ ] Telestration can be disabled for a fast, CPU-only dry run.

### C6. Honesty / safety
- [ ] Docs state the licensing reality (Ultralytics AGPL) and that **content
      rights** (broadcast footage, music, likenesses) are the user's
      responsibility.
- [ ] Auto-stats are labelled as estimates, not metric truth.

---

## D. Verification performed in this repo
- `python -m compileall src app studio.py` — byte-compiles clean.
- `tests/test_montage.py` — builds synthetic clips with ffmpeg `lavfi` and runs
  the **montage chain** (reframe-letterbox → slow-mo → freeze-zoom → compose →
  branding with intro/outro) end-to-end, asserting each output has 1 video + 1
  audio stream at the target geometry. This validates everything except the GPU
  vision stages, which require a GPU + model and are covered by C5 on the
  target server.

## E. Known limitations (carried forward honestly)
- Generic COCO detection tracks a small football poorly; a football-specific
  model is recommended (see `docs/MODELS.md`).
- Stats are pixel-based estimates (no pitch homography yet).
- Protagonist selection can mis-vote under heavy occlusion.
- Detection thresholds trade precision/recall — human review in the WebUI before
  publishing is the intended workflow.
