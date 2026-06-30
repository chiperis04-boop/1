# Implementation plan — agentic, frame-aware highlight Studio (v3)

Goal: replace the "script + presets" pipeline with **AI agents that watch the
footage, understand what happens, decide the whole edit, and review their own
output** — then deterministic code executes their decisions. Constraints from
the operator: **cost is irrelevant, models must be open-source, one L40S (48 GB)
running 24/7.** That lets us run real local vision/audio models per shot and
iterate with a critic loop.

This is a build plan, phased and verifiable. Each phase lists: modules, the
open-source model, the data contract, what is CPU-testable vs GPU-only, and an
exit criterion. Nothing here is "done" yet — it is the path to make it correct.

---

## 0. Model stack (all open-source, fits one L40S 48 GB)

| Role | Model (open-source) | Serving | ~VRAM | Notes |
|---|---|---|---|---|
| Director / Critic (vision-LLM) | **Qwen2-VL-7B-Instruct** (AWQ) or **MiniCPM-V-2.6 8B** | vLLM, OpenAI-compatible | 10–18 GB | already reachable via `director._call_openai` (base_url) |
| Commentary ASR | **faster-whisper large-v3** | in-proc (CTranslate2) | ~3 GB | already in stack (`edit/captions.py`) |
| Detection + tracking | **Ultralytics YOLO + BoT-SORT (GMC)** | in-proc | ~2–3 GB | already in stack (`tracking/cameraman.py`) |
| Pitch keypoints / homography | martinjolif pitch model + cv2/kornia | in-proc | ~1 GB | already in stack (`graphics/homography.py`) |
| Appearance Re-ID (hero across shots) | **OSNet (torchreid)** | in-proc | ~0.5 GB | NEW — stabilises hero ID across cuts |
| Crowd/whistle audio events | **PANNs (CNN14)** or YAMNet | in-proc | ~0.3 GB | NEW — "goal roar"/whistle cue for Scout |

Everything co-resides on one L40S. If the operator later wants a bigger brain,
**Qwen2-VL-72B-AWQ** (~40 GB) fits one L40S alone (move YOLO/whisper to a second
process/GPU) or 2×L40S — the OpenAI-compatible client makes the swap a config
change, no code change.

Serving: a small **vLLM** container exposes the VLM at an OpenAI-compatible
endpoint; `director.base_url` / `OPENAI_BASE_URL` already point at it. Weights
live on a persistent Modal Volume so the 24/7 server has zero cold-download.

---

## 1. Agentic architecture (perception → plan → execute → critique → revise)

```
 PERCEPTION (tools the agents can call / are fed)
   shots = segment_shots(clip)          # PySceneDetect content detector
   asr   = transcribe(clip)             # faster-whisper large-v3, word timestamps
   det   = detect_track_per_shot(clip)  # YOLO+BoT-SORT, per shot (no cross-cut IDs)
   reid  = appearance_embeddings(det)   # OSNet, link the hero across shots
   audio = audio_events(clip)           # PANNs roar/whistle + energy curve
   ocr   = scoreboard(clip)             # score + Δ
        │  -> PerceptionBundle (compact, multimodal)
        ▼
 DIRECTOR AGENT (vision-LLM, tool-aware, reasoning)
   input : keyframes (dense at beats) + per-shot thumbs + ASR text + det/audio/ocr summary
   output: EditPlan (rich, validated JSON)  — the single source of truth
        ▼
 EXECUTOR (deterministic; today's renderer, generalised)
   per-shot crop plans · multi-beat slow-mo · temporally-smoothed graphics ·
   single high-quality composite · final encode once
        ▼
 CRITIC AGENT + programmatic QA  (watch the OUTPUT)
   deterministic checks first (cheap), then VLM "does this look right?" on output frames
   → if it fails: structured feedback → Director revises EditPlan → re-execute (≤N loops)
```

The Director and Critic are **agents** (they reason over real perception and can
request more frames / a re-cut), not a single templated call. The Executor stays
dumb and deterministic so renders are reproducible.

---

## 2. Data contracts (the API between brain and hands)

### 2.1 `PerceptionBundle` (`src/perception/bundle.py`)
```python
@dataclass
class Shot:        # one continuous broadcast camera take
    idx: int; start: float; end: float
    kind: str           # wide|medium|closeup|crowd|replay|graphic (VLM-labelled)
    is_replay: bool
    keyframes: list[bytes]          # JPEGs, dense near motion peaks
@dataclass
class PerceptionBundle:
    clip_path: str; fps: float; duration: float
    shots: list[Shot]
    transcript: list[dict]          # {start,end,text} from ASR
    audio_curve: list[float]        # excitement/energy per second
    audio_events: list[dict]        # {t, label: roar|whistle|...}
    detections_summary: str         # compact text the VLM can read
    score: dict | None              # {before, after, minute} from OCR
```

### 2.2 `EditPlan` (`src/agents/editplan.py`) — replaces the 6-field manifest
```python
@dataclass
class ShotEdit:
    shot_idx: int
    keep: bool                      # drop replays/irrelevant shots
    framing: str                    # punch_in|crop_follow|letterbox_wide
    zoom: float                     # 1.0=full crop height .. >1 tighter
    subject: str                    # hero|ball|wide
@dataclass
class SlowmoBeat:
    start: float; end: float; factor: float
@dataclass
class EditPlan:
    keep_clip: bool                 # the Director can veto a false-positive
    importance: float               # 0..1 for ranking/compilation
    event: str                      # goal|chance|save|skill|card
    cut_in: float; cut_out: float   # precise, aligned to shot/play
    shots: list[ShotEdit]
    hero_description: str
    hero_jersey_shot: int | None    # which shot to read the number from (close-up)
    slowmo_beats: list[SlowmoBeat]
    hook_text: str; lower_third: str; caption: str; hashtags: list[str]
    pacing: str                     # punchy|cinematic
    music_energy: str               # low|build|high
    source: str                     # which backend produced it
    # + coerce()/validate() like EditingManifest, with safe heuristic defaults
```
Backward-compat: `EditingManifest` is derived from `EditPlan` so the current
Composer keeps working during the migration.

---

## 3. Phases

### Phase P0 — correctness & artifact removal (deterministic, CPU-testable) 🔴
Kills the visible artifacts independent of any LLM. **Do first.**

1. **Shot segmentation in v2** — `src/perception/shots.py` wrapping PySceneDetect
   (already a dependency). Studio runs tracking + crop planning **per shot**, and
   the crop **resets at each cut** (no lurch across cuts).
   - Files: `studio_pipeline._process` (loop over shots), `cameraman.build_plan`
     (per-shot), new `shots.py`.
2. **Replay handling** — flag broadcast replays (duplicate of a just-seen play +
   the broadcast's own slow-mo/graphic signature) and drop them by default.
3. **Cut the generational re-encode** — compose graphics + typography in a single
   pass where possible; use a visually-lossless intermediate (`-crf 12` / FFV1)
   between unavoidable stages; **encode the final once**. Target ≤2 lossy
   generations (was ~5–6).
4. **Telestration temporal smoothing** — persist hero halo / track positions
   across short detection gaps; smooth graphic positions (no flicker).

Exit: on a synthetic multi-shot clip, the crop does not jump at cuts; replays are
dropped; final file is one clean encode; halos don't flicker. All verifiable on
CPU with new tests in `tests/test_shots.py` / extended `tests/test_polish.py`.

### Phase P1 — the frame-aware Director (the core) 🧠
5. **Perception builder** — `src/perception/build.py` assembles a
   `PerceptionBundle`: `shots.py` + `transcribe()` (faster-whisper) + per-shot
   keyframes + detection summary + audio events (PANNs) + OCR.
6. **LLM client** — `src/agents/llm_client.py`: thin OpenAI-compatible client
   (works with vLLM/Ollama) with vision messages, JSON-mode, retries, schema
   validation. (Generalises today's `director._call_openai`.)
7. **Director agent** — `src/agents/director_agent.py`: given a
   `PerceptionBundle`, returns a validated `EditPlan`. It *curates* (keep/drop +
   importance), sets precise `cut_in/out` on shot boundaries, per-shot framing,
   slow-mo beats at the true decisive instants (cross-checked against the audio
   roar / ball-speed peak), and the hero + the close-up shot to read the number.
8. **Executor generalisation** — Cameraman honours per-shot `framing/zoom`;
   Composer does **multi-beat** slow-mo; hero jersey OCR runs only on the
   Director-chosen close-up shot (multi-read voting) → team-aware geometry
   fallback (honest about wide-shot unreliability).

Exit: with the local VLM up, the Director's `EditPlan` drives a clip end-to-end;
moment curation removes obvious false positives; slow-mo lands on the event.
GPU-only verification on the L40S with real footage.

### Phase P2 — the review / critic loop ("watch & realise") 🧠
9. **Programmatic QA** — `src/qa/checks.py` on the rendered output: subject-in-
   frame %, captions inside the social safe-area, crop-jump magnitude at cuts,
   black-bar/letterbox sanity, loudness (−14 LUFS ±1), duration vs target. Cheap,
   deterministic, CPU-testable.
10. **Critic agent** — `src/agents/critic.py`: samples OUTPUT keyframes, asks the
    VLM "does this look like a pro vertical highlight? what's wrong?", returns
    structured issues. Director consumes QA + critic feedback and revises the
    `EditPlan`; re-execute up to **N≤3** loops, keeping the best by QA score.

Exit: the loop measurably improves QA scores; failures (subject lost, text in UI
band, crop jump) auto-correct. QA is CPU-tested; the VLM critic is GPU-verified.

### Phase P3 — smarter perception (raises the ceiling) 🟠
11. **Real action understanding** — fuse a SoccerNet-style action-spotting cue +
    PANNs crowd-roar + ASR keywords ("goal", player names) so Scout becomes a
    high-recall *candidate generator* and the Director the precise judge.
12. **Cross-shot hero Re-ID** — OSNet appearance embeddings link the hero through
    cuts/replays so framing & graphics stay on the right player.
13. **Music intelligence** — beat-detect the bed (librosa) and align cuts /
    slow-mo onset to the beat; pick music energy from `EditPlan.music_energy`.

---

## 4. Modal / serving (24/7, no cold starts)

- **vLLM web app** (`modal_app.py` new function `vlm`): serves Qwen2-VL/MiniCPM-V
  at an OpenAI-compatible endpoint; `keep_warm=1`, weights on a persistent
  Volume. `director.base_url` points at it.
- **Worker function** (`studio`/`process`): YOLO + BoT-SORT + faster-whisper +
  OSNet + PANNs in one GPU container; calls the vLLM endpoint for decisions.
- All weights pre-baked into the image or on Volume so the always-on server has
  zero download latency. Health/doctor step in `scripts/setup.py` extended to
  ping the vLLM endpoint.

---

## 5. Testing strategy (honest boundaries)

- **CPU here (synthetic clips, real ffmpeg):** shot segmentation, per-shot crop
  reset, replay-drop logic, re-encode-count reduction, graphics smoothing,
  `EditPlan` schema validation/coercion, QA checks, executor honouring a
  hand-written `EditPlan`. New: `tests/test_shots.py`, `tests/test_editplan.py`,
  `tests/test_qa.py`; extend `test_studio.py`/`test_polish.py`.
- **GPU only (L40S, real match):** VLM Director/Critic quality, ASR accuracy,
  YOLO/BoT-SORT/Re-ID, homography on real lines, jersey OCR on close-ups, and the
  end-to-end "does it look good" judgement. These are marked ○ until run on the
  L40S — we will not claim them verified from CPU.

---

## 6. Risks & fallbacks (no over-promising)

- **VLM latency/quality:** mitigated by per-clip (not per-frame) calls, JSON-mode
  + schema coercion, and the heuristic as an explicit *blind-mode* fallback.
- **minterpolate / VLM hallucination:** every agent output is validated; if the
  plan is malformed or QA fails after N loops, fall back to the deterministic
  plan (current behaviour) so a clip never fails to render.
- **Re-ID / OCR on wide shots stays unreliable** — by design we read numbers only
  on Director-chosen close-ups and degrade to geometry; we keep labelling stats
  as estimates.

---

## 7. Sequencing (recommended order)

1. P0.1 shot segmentation + per-shot crop reset  ← biggest artifact win, CPU-testable
2. P0.3 re-encode reduction  ← biggest "softness" win
3. P0.2 replay-drop, P0.4 graphics smoothing
4. P1.5–6 perception + LLM client  → P1.7 Director agent → P1.8 executor
5. P2.9 QA → P2.10 critic loop
6. P3 perception upgrades (action spotting, Re-ID, music sync)

P0 is fully implementable & verifiable in this environment now. P1–P3 need the
L40S + local models to validate, but all the plumbing (schemas, client, executor,
QA) is built and unit-tested on CPU first so that GPU bring-up is configuration,
not new code.
