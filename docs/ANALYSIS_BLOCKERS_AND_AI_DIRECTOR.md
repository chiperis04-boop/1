# Blockers, visual-artifact risks & the frame-aware AI Director

This is a *design/analysis* document (not verified code). It does two things:

1. A complete, honest inventory of what stands between the current pipeline and
   the goal (broadcast-quality vertical highlights) — including every known
   source of **visual artifacts**, organised by stage and severity.
2. A concrete architecture to replace the "blind assembly line" with a
   **frame-aware AI Director** that actually *watches* the footage, decides the
   full edit, and then *reviews its own rendered output* — instead of template
   code that never understands the image.

Status tags: 🔴 blocks the goal / causes obvious artifacts · 🟠 quality risk ·
🟡 minor · 🧠 "blind vs sees-the-frame" root-cause marker.

---

## 0. Root cause (why limitations keep appearing)

The v2 chain is `Scout → clip → Cameraman → Director → analytics → Composer`.
Every editorial decision is made **before anyone looks at what the shot
actually shows**, from proxies for understanding:

| Decision | Made by | What it actually uses | 🧠 |
|---|---|---|---|
| Is this a highlight? which moment? | Scout | audio energy + scene cuts + (opt) action heuristic + OCR | blind to *meaning* |
| Clip in/out points | clipper | fixed seconds around an anchor | blind to play start/stop |
| Which player is the hero | analytics | jersey-OCR → nearest-to-ball → geometry | partly sees, mostly geometry |
| How to frame (crop) | Cameraman | centroid of hero+ball, fixed crop width | blind to shot type |
| When to slow-mo | Director(heuristic) | peak *ball speed* | blind to the decisive *event* |
| Hook / caption | Director(heuristic) | a lookup table by kind | blind to what happened |

The Director — the one module meant to "understand" — **defaults to
`heuristic` (offline, no vision)**, and even on an LLM backend it (a) sees only
~1 fps JPEGs, (b) gets no audio/commentary, (c) returns just 6 fields, and (d)
controls almost none of the decisions above (not moment selection, not cut
points, not framing, not replays, not pacing). So it is a narrow advisor, not a
director. **This is the thing to fix to reach the goal.**

---

## 1. Scout / moment selection — 🔴 the single biggest quality lever

Output quality is dominated by *which* moments are cut and *where* they start
and end. Today this is signal-heuristic, not semantic.

- 🔴 🧠 **False positives / negatives.** Loud crowd ≠ goal; a quiet goal has no
  spike. Fouls, throw-ins, and replays get clipped; real chances get missed.
  There is no model that *understands* "a goal/chance/save happened."
- 🔴 **Clip boundaries are fixed time windows** around an anchor, not the
  natural start (build-up) / end (celebration) of the play. Clips routinely
  start or end mid-action.
- 🔴 **Broadcast replays.** TV inserts its own slow-mo replays right after a
  goal. Scout can (a) clip the replay as a *second* event (duplicate), or (b)
  include replay footage *inside* a clip → a hard cut to a different angle with
  burned-in broadcast graphics, mid-clip aspect/zoom change → very visible.
- 🔴 **Scene cuts inside one clip.** A single window often spans a broadcast cut
  (wide → close-up → crowd → wide). The per-clip tracker and the single smoothed
  crop path then average across *different scenes* → the vertical crop lurches,
  track IDs reset, homography becomes invalid. (PySceneDetect is already in the
  stack — `src/detect/scene.py` — but the v2 reframe path does not segment shots.)

## 2. Cameraman / vertical reframe — 🔴/🟠 the main on-screen artifact source

- 🔴 **No shot-boundary awareness** (see 1) → the crop glides *across* a cut,
  producing a lurch/whip that reads as broken.
- 🟠 **Fixed crop width, no zoom.** A 9:16 column from a wide tactical shot loses
  most of the pitch (players become specks); a close-up is fine. A real director
  picks a *zoom level per shot* (punch-in on close-ups, letterbox-with-context
  on tactical wides). Current code can only do one behaviour.
- 🟠 **Tracking ID instability on wide shots.** Players are tiny → BoT-SORT
  switches IDs → the "hero" jumps to another player → crop snaps. CMC stabilises
  *camera* motion, not ID continuity.
- 🟠 **Finite smoothing bandwidth vs fast balls.** Even zero-phase smoothing has
  a cutoff; on a long ball/cross the ball can still leave the narrow column for a
  moment.
- 🟡 **Broadcast overlays** (score bug, logos) sit at screen edges and get
  cropped awkwardly or remain half-visible in vertical.

## 3. Director — 🔴 too narrow & blind to realise the vision

- 🔴 🧠 **Default = heuristic (no vision).** Decisions come from geometry only.
- 🔴 **Even with a VLM the input is impoverished:** ~1 fps frames, **no audio**,
  no temporal/continuity signal, no detector/tracker context fused in.
- 🔴 **Tiny output surface.** 6 fields. It does not decide moment curation, cut
  in/out, per-shot framing/zoom, replay use, multi-beat slow-mo, pacing,
  transitions, music, or caption content beyond a hook.
- 🟠 **Timestamp precision.** `slomo_trigger_timestamp` is "seconds from clip
  start" but the model saw 1 fps → ±1 s error → slow-mo can fire off the actual
  beat. The heuristic uses *peak ball speed*, which is often the clearance/pass,
  not the *save* or the *skill* — semantically wrong instant.
- 🔴 🧠 **No review of the result.** The Director never watches the rendered clip
  to check it actually looks right — the explicit thing you want.

## 4. Homography / telestration — 🟠 flicker & warp

- 🟠 **Homography needs visible pitch lines;** on close-ups/crowd it fails →
  tactical overlays misplaced or dropped. **Per-frame homographies are blended
  linearly**, which is not geometrically valid → overlays can drift/warp.
- 🟠 **Graphics flicker.** Halos/numbers/possession plate are drawn from
  *per-frame* detections with no temporal smoothing → they blink on/off as
  detections flicker. A hero halo should persist across short gaps.
- 🟡 **Foot-ellipse uses bbox bottom;** wrong under occlusion/odd poses.

## 5. Composer / slow-mo / typography / audio — 🟠

- 🟠 **`minterpolate` warping** on fast, low-contrast motion (already gated +
  fallback, but real cost/quality must be measured on GPU).
- 🟠 **Slow-mo targets the wrong instant** when driven by the heuristic (peak
  ball speed), per §3.
- 🟡 **Typography is static** (no motion design), fixed 2 s hook regardless of
  clip; plates can overlap the halo/subject.
- 🟡 **Music is not synced** to cuts or the slow-mo beat; no transition design.

## 6. Cross-cutting / encoding — 🔴 cumulative softness (often missed)

- 🔴 **Generational re-encode.** The v2 per-clip path re-encodes H.264 **~5–6
  times**: graphics → crop-render → slow-mo → typography → branding intro/outro
  concat → (optional) compilation. Each pass is lossy; the image visibly softens
  and chroma degrades by the final file. This compounds with every feature added
  and is a prime cause of "it looks a bit mushy."
- 🟠 **`yuv420p` chroma subsampling** on thin bright graphics (ball trail, #
  labels) → colour fringing/edge crawl on saturated lines.
- 🟠 **Fixed `fps: 30`.** A 25/50 fps broadcast source gets frame-rate-converted
  → judder unless handled.
- 🟡 **No colour-space tagging** (BT.709) → possible shifts across players/encoders.

---

## 7. Target architecture — the frame-aware AI Director

Principle: **a VLM makes the editorial decisions from rich perception; cheap
deterministic CV executes them; a critic pass watches the output and iterates.**
Use the model where understanding is needed (per clip, dozens of calls), never
per-frame.

```
                 ┌──────────────── PERCEPTION (give the model eyes+ears) ───────────────┐
 full match ──▶  shot-segmentation (PySceneDetect)  → per-shot keyframes (dense at beat) │
                 commentary ASR (faster-whisper)     → transcript w/ timestamps          │
                 detections+tracks (YOLO+BoT-SORT)   → compact per-shot summary           │
                 scoreboard OCR + audio energy       → score/Δ, excitement curve          │
                 └──────────────────────────────────────────────────────────────────────┘
                                            │ (compact multimodal context per candidate)
                                            ▼
                 ┌──────────────── DIRECTOR (vision LLM) → EditPlan (rich JSON) ──────────┐
                 │  • curate: keep/drop + importance per candidate (NOT just heuristic)   │
                 │  • cut in/out aligned to shot boundaries & play start/stop             │
                 │  • per-SHOT framing: {mode: punch_in|crop|letterbox, zoom, subject}    │
                 │  • replays: mark broadcast replays to drop or reuse                    │
                 │  • slow-mo BEATS: [{start,end,factor}] at the true decisive instants   │
                 │  • hero: description + WHICH shot/frames to read the jersey from        │
                 │  • hook / lower-third / caption / hashtags / pacing / music energy     │
                 └────────────────────────────────────────────────────────────────────────┘
                                            │ (deterministic)
                                            ▼
                 ┌──────────────── EXECUTOR (existing renderer, generalised) ─────────────┐
                 │  per-shot crop plan (reset at every cut) · multi-beat slow-mo ·         │
                 │  temporally-smoothed graphics · single-pass compositing where possible │
                 └────────────────────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                 ┌──────────────── CRITIC / QA (watch the OUTPUT, iterate) ───────────────┐
                 │  programmatic checks: subject in-frame %, text in safe-area, crop-jump  │
                 │  at cuts, black-bar/letterbox sanity, loudness, duration               │
                 │  + optional VLM "does this look good?" on output keyframes              │
                 │  → bounded revisions (<=N) feeding corrections back to the Director     │
                 └────────────────────────────────────────────────────────────────────────┘
```

### 7.1 What changes vs today
- **Scout becomes a *candidate generator*, not the judge.** It proposes windows;
  the Director (with vision + ASR) *curates* and sets precise in/out. This kills
  most false positives and mid-action cuts.
- **Shot segmentation is first-class.** Tracking, crop planning and homography
  run **per shot**, and the crop resets at each cut → no lurch across cuts;
  replays become explicit shots the Director can drop.
- **The Director's output is an `EditPlan`** the renderer obeys, covering the
  full decision set (framing/zoom per shot, multi-beat slow-mo, pacing, text).
- **A review loop** finally satisfies "watch and realise the video": cheap
  deterministic QA first (in-frame %, safe-area, crop-jump, loudness), then an
  optional VLM look at output keyframes, with ≤N bounded revisions.

### 7.2 Perception payload (how the model "sees" without per-frame cost)
Per candidate event, send the VLM: 6–12 keyframes biased toward the decisive
beat + 1 montage thumbnail per shot, the ASR transcript slice, a one-line
detection/track summary ("2 shots, 1 cut at 4.2 s, ball xspeed peak at 6.1 s,
score 1→2"), and the OCR score. This is compact, cheap, and gives real context.

### 7.3 Hero re-identification (make jersey reading actually work)
Today wide-shot OCR is unreliable → geometry fallback. Improvement: the Director
names the hero **and the shot/timestamp of the best close-up**; jersey OCR runs
only on those **high-res crops** (not wide frames), with multi-read voting; still
falls back to team-aware geometry. This is the honest path to "lock onto #10."

### 7.4 Backends & cost
- Decisions: a real vision LLM — Gemini 1.5/2.x, GPT-4o, or **local MiniCPM-V /
  Qwen2-VL** via the existing OpenAI-compatible path. ~1 call/clip × dozens of
  clips = affordable.
- Keep `heuristic` as the **offline fallback**, but it should be clearly labelled
  as "blind mode" — not the default for quality output.

---

## 8. Prioritised roadmap

**P0 — correctness & artifact removal (do first; mostly deterministic CV):**
1. Shot-boundary segmentation in the v2 path; per-shot tracking + crop reset.
2. Replay detection → drop (or isolate) broadcast replays.
3. Cut the generational re-encode: composite graphics+typography in fewer
   passes, raise intermediate quality (visually-lossless), encode final once.
4. Temporal smoothing/persistence of telestration (no flicker).

**P1 — the frame-aware Director (the core of your vision):**
5. `EditPlan` schema + perception packaging (shot keyframes + ASR + detection
   summary) + VLM decisions for curation, cut in/out, per-shot framing, slow-mo
   beats, hero+jersey-shot, hook/caption/pacing.
6. Generalise the executor to honour per-shot framing/zoom and multi-beat
   slow-mo.

**P2 — the review/critic loop ("watch & realise"):**
7. Programmatic QA (subject-in-frame %, safe-area, crop-jump at cuts, loudness,
   duration) → bounded auto-revisions.
8. Optional VLM critique of output keyframes feeding back to the Director.

**Honest caveats:** P0 is implementable & CPU-testable here. P1/P2 need a vision
model and real match footage on the L40S to validate; quality and per-match cost
must be measured, not assumed. None of P1/P2 is verified yet — this section is a
design, explicitly not a claim of working behaviour.
