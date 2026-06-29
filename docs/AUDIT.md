# Audit — v0.1 → v0.2 hardening

A code-level audit of the first scaffold. Findings are grouped by severity, each
with the concrete fix applied in v0.2. The goal: move from "reads well" to
"actually renders correct files unattended".

## Methodology
- Static read of every module.
- Trace of the per-clip ffmpeg chain (cut → track → telestrate → reframe →
  slowmo → freeze → compose → brand) checking stream consistency at each hand-off.
- Synthetic end-to-end test of the **montage chain** (everything except the GPU
  vision stages) using ffmpeg `lavfi` test sources — see `tests/test_montage.py`.

---

## BLOCKER bugs (would crash or corrupt output)

### B1. Freeze-zoom `zoompan` size expression is invalid
`effects.freeze_zoom_intro` used `zoompan=...:s=iw/2xih/2`. `zoompan`'s `s`
takes a concrete `WxH` (or named size), not an `iw/ih` expression — ffmpeg
errors out. **Fix:** render the freeze still directly at the active profile
`WxH`, drop the `scale=8000` hack, use `zoompan=...:s=<W>x<H>:fps=<fps>`.

### B2. Concatenation mixes clips with and without audio
The freeze clip (B1) and the generated intro/outro/`_text_card` paths produced
videos whose audio-stream presence/params didn't match the body clip. The
`concat` demuxer requires **identical stream layout & codec params** across
parts; mismatch => failure or dropped audio. **Fix:** every segment is forced
through a single `standardize()` step that guarantees exactly *1 video + 1
stereo AAC audio @ 48 kHz* at profile `WxH/fps`, and a silent track is synthised
when a source has none.

### B3. `_normalize` advertised a silent track but never mapped it
It added an `anullsrc` input then mapped only `0:a:0?`, so audio-less inputs
stayed audio-less (feeding B2). **Fix:** replaced by `standardize()` which maps
the synthesised silence when the source has no audio (decided via `ffprobe`).

### B4. `compose_clip` maps `0:a` unconditionally
If an upstream stage dropped audio, `-map 0:a` aborts the render. **Fix:**
`has_audio()` check; when absent, a silent bed is added before mixing so the
map target always exists.

---

## MAJOR bugs (wrong/ugly output, not a crash)

### M1. Slow-mo segments can be empty at clip edges
When `key_t` is near 0 or near the end, `trim=0:0` / out-of-range trims create
empty concat segments → stutter or failure. **Fix:** clamp the slow window to
`[0.3, dur-0.3]`, skip the effect entirely if the clip is too short, and verify
segment durations before building the graph.

### M2. drawtext escaping is incomplete
Captions/hooks could contain `,` `[` `]` `=` `;` `\` `'` `%` `:` — several break
the filtergraph. **Fix:** a single `esc_drawtext()` that escapes the full set,
and captions are written via a `textfile=` + sidecar file when they contain
risky characters, avoiding inline-escaping pitfalls.

### M3. Reframe assumes track coords match the video it crops
`track_clip` runs on the original cut, but reframe was applied to the
*telestrated* video. That happens to share dimensions, but the contract was
implicit. **Fix:** documented + asserted that telestration preserves geometry;
reframe takes explicit `src_w/src_h` from the track and validates against the
input it crops.

### M4. Stage ordering: captions generated after slow-mo, but burned by compose
Caption timestamps come from transcribing the (already time-warped) clip, which
is correct — but `compose` then re-derives nothing, so timing is fine. Kept, but
documented; word-timing now comes from the *post-effects* clip explicitly.

---

## MINOR / robustness

- **R1.** No global error isolation per clip beyond a bare `except`; a failure
  produced a silent `None`. **Fix:** structured result with `status`,
  `error`, and which stage failed, surfaced in the WebUI.
- **R2.** `random` imported but unused in `compose.py`. Removed.
- **R3.** No check that `ffmpeg`/`ffprobe` exist before running. **Fix:**
  `ff.ensure_tools()` called at startup with a clear message.
- **R4.** NVENC encoder hard-fails on boxes without it. **Fix:** auto-probe
  NVENC at startup; fall back to `libx264` and log the downgrade.
- **R5.** Detection had no resume granularity below "all moments". Acceptable
  for now; per-clip outputs are skipped if already present.
- **R6.** `_print_moments_obj` was dead code. Removed.
- **R7.** Subprocesses swallowed stderr (`DEVNULL`), making failures opaque.
  **Fix:** `ff.run()` captures stderr and raises `FFmpegError` with the tail of
  the log.

---

## Design gaps addressed in v0.2
- **No importable API:** added `src/runner.py` so the WebUI and CLI share one
  code path with a progress callback.
- **No UI:** added `app/webui.py` (Gradio) for upload, configuration, live
  progress, preview and library management.
- **Not turnkey:** added `studio.py`, a single self-bootstrapping launcher that
  builds a venv, installs dependencies, downloads models and starts the WebUI.
- **No quality gates / acceptance criteria:** added `docs/SKILLS.md`.

## Known remaining limitations (honest)
- Vision accuracy depends on the detector; the generic COCO model tracks a small
  football poorly — a football-specific model is strongly recommended (MODELS.md).
- Stats (shot/sprint distance) are pixel-based estimates, not homography-metric.
- The "key player" heuristic can mis-vote on occlusion; team-colour filtering is
  a planned improvement.
- Auto-detection trades precision/recall via thresholds; review in the WebUI
  before publishing is expected, not optional.
